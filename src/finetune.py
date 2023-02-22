# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs a summarization fine-tuning task on a tokenized dataset.

Runs a summarization fine-tuning task on a tokenized dataset.
"""
import json
import os

from absl import app
from absl import flags
from absl import logging
from absl.flags import argparse_flags
from datasets import load_from_disk
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from torch import float32
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import TrainerCallback
from transformers import TrainerControl
from transformers import TrainerState
from transformers import TrainingArguments
from transformers import utils as transformer_utils
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'tokenized_dataset_path', None, 'Path to tokenized dataset.'
)
flags.DEFINE_string('gcs_output', None, 'GCS path to model and checkpoints')
flags.DEFINE_string('model_checkpoint', 't5-small', 'Model checkpoint name')
flags.DEFINE_string('deepspeed', None, 'deepspeed config file name')
flags.DEFINE_integer('max_input_length', 256,
                     'Filter documents larger than this size')
flags.DEFINE_integer('batch_size', 64, 'Training mini-batch size')
flags.DEFINE_integer('epochs', 2, 'Number of training epochs')
flags.DEFINE_boolean('fp16', False, 'Use fp16 mode')
flags.DEFINE_boolean('bf16', True, 'Use bfloat16 mode')
flags.DEFINE_string('fp16_opt_level', 'O3', 'fb16 optimization level')
flags.DEFINE_string('fp16_backend', 'cuda_amp', 'fb16 backend name')
flags.DEFINE_integer('local_rank', -1, 'Distributed local process rank.')


class GCSSaveCallback(TrainerCallback):
  """A [`TrainerCallback`] that handles checkpoints.
  """

  def on_save(self, args: TrainingArguments, state: TrainerState,
              control: TrainerControl, **kwargs):
    checkpoint_folder = f'checkpoint-{state.global_step}'
    local_output_dir = os.path.join(args.output_dir, checkpoint_folder)
    if not os.path.exists(local_output_dir):
      logging.error(
          'Check point called for a non existing checkpoint %s',
          local_output_dir,
      )
      return
    gcs_root, gcs = utils.gcs_path(FLAGS.gcs_output)
    gcs_output_dir = os.path.join(gcs_root, local_output_dir)
    logging.info('Uploading %s....', gcs_output_dir)
    gcs.put(local_output_dir, gcs_output_dir, recursive=True)
    return None


def train(argv):
  """Main fine-tuning method.

  Args:
    argv: unused.
  """
  del argv

  logging.info('Loading Dataset....')
  path, _ = utils.gcs_path(path=FLAGS.tokenized_dataset_path, gcs_prefix='gs://')
  tokenized_datasets = load_from_disk(path)

  model_name = FLAGS.model_checkpoint.split('/')[-1]
  path, fs = utils.gcs_path(FLAGS.model_checkpoint)
  if fs:
    logging.info('Downloading model....')
    fs.get(path, 'model', recursive=True)
    model_checkpoint = './model'
  else:
    model_checkpoint = FLAGS.model_checkpoint

  tokenizer = AutoTokenizer.from_pretrained(
      model_checkpoint, model_max_length=FLAGS.max_input_length
  )
  
  rouge_score = evaluate.load('rouge')
  nltk.download('punkt')

  logging.info('Loading model....')
  if FLAGS.fp16:
    logging.info('Using float16....')
  model = AutoModelForSeq2SeqLM.from_pretrained(
      model_checkpoint, torch_dtype=float32)
  # Show the training loss with every epoch
  logging_steps = len(tokenized_datasets['train']) // FLAGS.batch_size

  output_dir = f'{model_name}_checkpoints'

  args = Seq2SeqTrainingArguments(
      output_dir=output_dir,
      evaluation_strategy='epoch',
      learning_rate=5.6e-5,
      per_device_train_batch_size=FLAGS.batch_size,
      per_device_eval_batch_size=FLAGS.batch_size,
      weight_decay=0.01,
      save_total_limit=10,
      num_train_epochs=FLAGS.epochs,
      predict_with_generate=True,
      fp16=FLAGS.fp16,
      bf16=FLAGS.bf16,
      half_precision_backend=FLAGS.fp16_backend,
      fp16_opt_level=FLAGS.fp16_opt_level,
      logging_steps=logging_steps,
      dataloader_num_workers=1,
      dataloader_drop_last=True,
      dataloader_pin_memory=True,
      deepspeed=FLAGS.deepspeed,
      disable_tqdm=True)

  def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = [
        '\n'.join(sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        '\n'.join(sent_tokenize(label.strip())) for label in decoded_labels
    ]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}

  data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

  gcs_save_callback = GCSSaveCallback()
  trainer = Seq2SeqTrainer(
      model,
      args,
      train_dataset=tokenized_datasets['train'],
      eval_dataset=tokenized_datasets['validation'],
      data_collator=data_collator,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics,
      callbacks=[gcs_save_callback],
  )
  logging.info('Training model...')
  trainer.train()

  logging.info('Evaluating model...')
  metrics = trainer.evaluate()

  output_dir = f'{model_name}_final'
  logging.info('Saving model....')
  trainer.save_model(output_dir)

  if os.path.exists(output_dir) and args.should_save:

    logging.info('Saving metrics...')
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
      json.dump(metrics, f, indent=2)

    if FLAGS.gcs_output:
      src = output_dir
      dst, gcs = utils.gcs_path(FLAGS.gcs_output)
      logging.info('Uploading model from %s to %s...', src, dst)
      gcs.put(src, dst, recursive=True)
      gcs.rename(os.path.join(dst, output_dir), dst, recursive=True)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  transformer_utils.logging.disable_progress_bar()

  parser = argparse_flags.ArgumentParser(allow_abbrev=False)
  app.run(train, flags_parser=parser.parse_known_args)

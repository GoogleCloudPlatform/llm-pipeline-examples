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
"""Tokenizes datasets in preparation for training/tuning.

Tokenizes datasets in preparation for training/tuning.
"""

import os
from absl import app
from absl import flags
from absl import logging
from absl.flags import argparse_flags

from datasets import DatasetDict
from datasets import load_from_disk
from transformers import AutoTokenizer
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('model_checkpoint', 't5-small', 'Model checkpoint name')
flags.DEFINE_string(
    'dataset_path',
    None,
    'Path to download dataset from.',
    required=True)
flags.DEFINE_string(
    'tokenized_dataset_path',
    None,
    'Path to save preprocessed dataset.',
    required=True
)
flags.DEFINE_string(
    'document_column', 'article', 'Name of dataset column contianing document.'
)
flags.DEFINE_string(
    'summary_column', 'highlights', 'Name of dataset column contianing summary.'
)
flags.DEFINE_integer(
    'max_input_length', 256, 'Filter documents larger than this size'
)
flags.DEFINE_integer(
    'max_target_length', 30, 'Filter titles larger than this size'
)


def process(examples, tokenizer):
  """Returns tokenized examples using tokenizer.

  Args:
    examples: Pre tokenizations input examples.
    tokenizer: Tokenizer to use.

  Returns: Tokenized examples.

  """
  model_inputs = tokenizer(
      text=examples[FLAGS.document_column],
      max_length=FLAGS.max_input_length,
      truncation=True)
  labels = tokenizer(
      text_target=examples[FLAGS.summary_column],
      max_length=FLAGS.max_target_length,
      truncation=True)

  model_inputs['labels'] = labels['input_ids']
  return model_inputs


def main(argv):
  del argv

  path, _ = utils.gcs_path(path=FLAGS.dataset_path, gcs_prefix='gs://')
  raw_datasets = load_from_disk(path)

  path, fs = utils.gcs_path(path=FLAGS.model_checkpoint)
  if fs:
    logging.info('Downloading model....')
    fs.get(path, 'model', recursive=True)
    model_checkpoint = './model'
  else:
    model_checkpoint = FLAGS.model_checkpoint

  logging.info('Preprocessing Dataset....')
  tokenizer = AutoTokenizer.from_pretrained(
      model_checkpoint, model_max_length=FLAGS.max_input_length
  )
  model_name = FLAGS.model_checkpoint.split('/')[-1]
  tokenized_datasets = DatasetDict()

  for key in raw_datasets:
    fingerprint = '{}_{}_{}_{}'.format(
        key, model_name, FLAGS.max_input_length, FLAGS.max_target_length
    )
    tokenized_datasets[key] = raw_datasets[key].map(
        process,
        batched=True,
        fn_kwargs={'tokenizer': tokenizer},
        new_fingerprint=fingerprint)
  tokenized_datasets = tokenized_datasets.remove_columns(
      raw_datasets['train'].column_names)

  logging.info('Saving preprocessed Dataset....')
  path, _ = utils.gcs_path(path=FLAGS.tokenized_dataset_path, gcs_prefix='gs://')
  tokenized_datasets.save_to_disk(path)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  parser = argparse_flags.ArgumentParser(allow_abbrev=False)
  app.run(main, flags_parser=parser.parse_known_args)

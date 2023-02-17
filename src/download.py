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
"""Downloads CNN Dailymail dataset.

Downloads CNN Dailymail dataset.
"""

from absl import app
from absl import flags
from absl import logging
from absl.flags import argparse_flags
import threading

from datasets import load_dataset
from utils import gcs_path
from huggingface_hub import Repository
import evaluate
import nltk
import os
from transformers import utils as transformer_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', None, 'Name of dataset to download.')
flags.DEFINE_string('subset', None, 'subset of dataset.')
flags.DEFINE_string('dataset_path', None, 'Path to download dataset to.')
flags.DEFINE_string('model_checkpoint', 't5-small', 'Model checkpoint name')
flags.DEFINE_string('workspace_path', None, 'Path to download workspace data to.')

def download_workspace():
  ws, ws_fs = gcs_path(FLAGS.workspace_path)

  nltk.download('punkt')
  logging.info('Saving nltk_data....')
  ws_fs.put('./nltk_data', os.path.join(ws, 'nltk_data'), recursive=True)


  model_path, model_fs = gcs_path(FLAGS.model_checkpoint)
  if model_fs:
    logging.info('Copying Model....')
    model_fs.cp(model_path, ws, recursive=True)
  else:
    logging.info('Downloading Model....')
    Repository(local_dir="./model", clone_from=FLAGS.model_checkpoint)
    logging.info('Saving model....')
    ws_fs.put('./model', os.path.join(ws, 'model'), recursive=True)

  evaluate.load('rouge')
  logging.info('Saving huggingface data....')
  dirs_to_upload = ['evaluate', 'modules', 'hub']
  ws = os.path.join(ws,'huggingface')
  for dir in dirs_to_upload:
    logging.info('Saving huggingface/%s ....', dir)
    ws_fs.put(
      os.path.join('.cache/huggingface', dir),
      os.path.join(ws,dir),
      recursive=True)

def download_dataset():
  src, src_fs = gcs_path(FLAGS.dataset)
  dst, _ = gcs_path(path=FLAGS.dataset_path, gcs_prefix='gs://')
  if src_fs:
    logging.info('Copying Dataset....')
    src_fs.cp(src, dst, recursive=True)
  else:
    logging.info('Downloading Dataset....')
    datasets = load_dataset(FLAGS.dataset, FLAGS.subset)
    logging.info('Saving Dataset to %s....', FLAGS.dataset_path)
    datasets.save_to_disk(dst)

def main(argv):
  del argv

  t1 = threading.Thread(target=download_workspace)
  t2 = threading.Thread(target=download_dataset)

  t1.start()
  t2.start()

  t1.join()
  t2.join()
  

if __name__ == '__main__':
  transformer_utils.logging.disable_progress_bar()
  logging.set_verbosity(logging.INFO)
  parser = argparse_flags.ArgumentParser(allow_abbrev=False)
  app.run(main, flags_parser=parser.parse_known_args)

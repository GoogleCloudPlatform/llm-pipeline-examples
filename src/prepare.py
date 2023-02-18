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
"""Initializes global libraries that are not multiprocess friendly.

Initializes global libraries that are not multiprocess friendly.
"""
from absl import app
from absl import flags
from absl import logging
from utils import gcs_path

import os
FLAGS = flags.FLAGS

flags.DEFINE_string('workspace_path', None, 'Path to download workspace data from.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  src, fs = gcs_path(FLAGS.workspace_path)

  logging.info('Downloading nltk_data....')
  fs.get(os.path.join(src, 'nltk_data'), 'nltk_data', recursive=True)

  if fs.exists(os.path.join(src, 'model')):
    logging.info('Downloading model....')
    fs.get(os.path.join(src, 'model'), 'model', recursive=True)


  logging.info('Downloading huggingface data....')
  fs.get(os.path.join(src, 'huggingface'), '.cache/huggingface', recursive=True)
  
if __name__ == '__main__':
  app.run(main)

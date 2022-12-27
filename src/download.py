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

from datasets import load_dataset
from utils import gcs_path

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', None, 'Name of dataset to download.')
flags.DEFINE_string('version', None, 'Version of dataset.')
flags.DEFINE_string('download_path', None, 'Path to download dataset to.')


def main(argv):
  del argv

  src, src_fs = gcs_path(FLAGS.dataset)
  dst, dst_fs = gcs_path(FLAGS.download_path)
  if src_fs:
    logging.info('Copying Dataset....')
    src_fs.cp(src, dst, recursive=True)
  else:
    logging.info('Downloading Dataset....')
    datasets = load_dataset(FLAGS.dataset, FLAGS.version)
    logging.info('Saving Dataset....')
    datasets.save_to_disk(dst, fs=dst_fs)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  parser = argparse_flags.ArgumentParser(allow_abbrev=False)
  app.run(main, flags_parser=parser.parse_known_args)

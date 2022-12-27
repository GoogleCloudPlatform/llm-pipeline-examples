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
import evaluate
import nltk
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import utils as transformer_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('model_checkpoint', 't5-small', 'Model checkpoint name')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  evaluate.load('rouge')
  nltk.download('punkt')
  AutoTokenizer.from_pretrained(FLAGS.model_checkpoint)
  AutoModelForSeq2SeqLM.from_pretrained(FLAGS.model_checkpoint)

if __name__ == '__main__':
  transformer_utils.logging.disable_progress_bar()
  app.run(main)

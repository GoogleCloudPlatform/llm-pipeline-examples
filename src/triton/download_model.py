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
"""Download a model from Vertex AI to a local folder
"""
import argparse
import os
from typing import List, Tuple

from absl import app as absl_app
from absl import flags
from absl import logging
from absl.flags import argparse_flags

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", None, "Path of HF model to load.")
flags.DEFINE_string("destination_path", None, "Relative path to download the model to.")

def init_model():
  """Initializes the model using deep speed."""
  os.environ["TOKENIZERS_PARALLELISM"] = "false"

  # distributed setup
  local_rank = int(os.getenv("LOCAL_RANK", "0"))
  world_size = int(os.getenv("WORLD_SIZE", "1"))

  model_path = os.environ.get("AIP_STORAGE_URI", FLAGS.model_path)
  dst = os.environ.get("TRITON_MODEL_PATH", FLAGS.destination_path)

  logging.info("Model path: %s", model_path)
  if model_path.startswith("gs://"):
    src = model_path.replace("gs://", "")
    if (len(dst) == 0):
      dst = src.split("/")[-1] + "/"
    dst = dst.split("/")[-1] + "/"
    if local_rank == 0:
      gcs = gcsfs.GCSFileSystem()
      logging.info("Downloading model from %s", model_path)
      for f in gcs.ls(src):
        if gcs.isfile(f):
          logging.info("Downloading %s", f)
          gcs.get(f, dst)
 
 def main(argv):
  """Main server method.

  Args:
    argv: unused.
  """
  del argv

  init_model()


if __name__ == "__main__":
  absl_app.run(main, flags_parser=parse_flags)
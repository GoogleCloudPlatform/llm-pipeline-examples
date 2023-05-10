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
"""Simple Flask prediction for a model.

Simple Flask prediction for a model.
"""

import argparse
import os
import subprocess
from typing import List, Tuple

from absl import app as absl_app
from absl import flags
from absl import logging
from absl.flags import argparse_flags
from flask import Flask, send_from_directory
from flask import request
from utils import timer
import gcsfs
from triton_processor import T5TritonProcessor

app = Flask(__name__, root_path=os.path.join(os.getcwd(), "app/"))
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path",
    "t5-base",
    (
        "path to the FT converted model on GCS or local filesystem. For example"
        " '/all_models/t5-v1_1-base'"
    ),
)
flags.DEFINE_integer(
    "port", 5000, "port to expose this server on. Default is '5000'."
)
flags.DEFINE_integer(
    "triton_port",
    8000,
    "local triton server port to proxy requests to. Default is '8000'.",
)
flags.DEFINE_string(
    "triton_host",
    "localhost",
    (
        "Optional. Separate host to route triton requests to. Default is"
        " 'localhost'."
    ),
)
flags.DEFINE_string(
    "hf_model_path",
    "t5-base",
    (
        "path to the source model on HuggingFace. For example"
        " 'google/t5-v1_1-base'."
    ),
)


def download_model(model_path):
  """Downloads the model to the expected Triton directory."""
  os.environ["TOKENIZERS_PARALLELISM"] = "false"

  logging.info("Model path: %s", model_path)
  if model_path.startswith("gs://"):
    src = model_path.replace("gs://", "")
    dst = "/workspace/all_models/"
    gcs = gcsfs.GCSFileSystem()
    logging.info("Downloading model FROM %s", model_path)
    logging.info("Downloading model TO %s", dst)
    gcs.get(src, dst, recursive=True)
    model_path = dst + src.split("/")[-1] + "/"

  return model_path


@app.route("/health")
def health():
  return {"health": "ok"}


@app.route("/ui", methods=["GET"])
def ui():
  return send_from_directory("templates", "ui.html")


@app.route("/infer", methods=["POST"])
@timer
def infer():
  """Process a generic inference request.

  The task should be provided as the first chunk of text.

  Returns:
    Inferencing result object {'predictions': [..]}
  """
  logging.info("Request received.")
  logging.info("Passing %s to Triton", request.json["instances"])
  return_payload = []
  client = _get_triton_client()
  for req in request.json["instances"]:
    text_out = client.infer(text=req)
    return_payload.append(text_out)
  client.client.close()
  return {"predictions": return_payload}


def parse_flags(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
  """Parses command line arguments entry_point.

  Args:
    argv: Unparsed arguments.

  Returns:
    Tuple of an argparse result and remaining args.
  """
  parser = argparse_flags.ArgumentParser(allow_abbrev=False)
  return parser.parse_known_args(argv)


def main(argv):
  """Main server method.

  Args:
    argv: unused.
  """
  del argv
  app.host = os.environ.get("SERVER_HOST", "0.0.0.0")
  app.port = int(os.environ.get("AIP_HTTP_PORT", str(FLAGS.port)))

  model_path = os.environ.get("AIP_STORAGE_URI", FLAGS.model_path).rstrip("/")
  model_dir = download_model(model_path)

  subprocess.Popen(
      ["/opt/tritonserver/bin/tritonserver", f"--model-repository={model_dir}", "--allow-vertex-ai=false", "--allow-http=true", "--http-port=8000"]
  )

  # Check inside converted model for a config.json for tokenizer dictionary. If not found, use the passed flag.
  nested_model_dir = model_dir
  for _ in range(3):
    filepaths = os.listdir(nested_model_dir)
    
    try:
      filepaths.remove("config.pbtxt")
    except ValueError:
      pass
    
    nested_model_dir = os.path.join(nested_model_dir, filepaths[0])

  if os.path.exists(os.path.join(nested_model_dir, "config.json")):
    app.tokenizer_model_path = nested_model_dir
  else:
    app.tokenizer_model_path = FLAGS.hf_model_path

  app.run(app.host, app.port, debug=False)

def _get_triton_client():
  return T5TritonProcessor(
      app.tokenizer_model_path, FLAGS.triton_host, FLAGS.triton_port
  )

if __name__ == "__main__":
  absl_app.run(main, flags_parser=parse_flags)

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
from typing import List, Tuple

from absl import app as absl_app
from absl import flags
from absl import logging
from absl.flags import argparse_flags
from flask import Flask, send_from_directory
from flask import request
import gcsfs
import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

app = Flask(__name__, root_path=os.path.join(os.getcwd(), "app/"))
FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", None, "Path of HF model to load.")
flags.DEFINE_integer("port", 5000, "server port.")


def init_model():
  model_path = os.environ.get("AIP_STORAGE_URI", FLAGS.model_path)
  logging.info("Model path: %s", model_path)
  if model_path.startswith("gs://"):
    src = model_path.replace("gs://", "")
    dst = src.split("/")[-1] + "/"
    gcs = gcsfs.GCSFileSystem()
    logging.info("Downloading model from %s", model_path)
    for f in gcs.ls(src):
      if gcs.isfile(f):
        logging.info("Downloading %s", f)
        gcs.get(f, dst)
    model_path = dst
  
  # now a model can be loaded.
  logging.info("Loading local model from %s", model_path)
  app.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  app.generation_config = GenerationConfig.from_pretrained(model_path)
  logging.info("Model ready to serve")
  app.tokenizer = tokenizer


@app.route("/health")
def health():
  return {"health": "ok"}


@app.route("/ui", methods=["GET"])
def ui():
  return send_from_directory("templates", "ui.html")


@app.route("/infer", methods=["POST"])
def infer():
  """Process an inferencing request."""
  logging.info("Received request")
  inputs = app.tokenizer(
      request.json["instances"],
      return_tensors="pt",
      padding=True,
      truncation=True).to(app.model.device)
  logging.info("Encoded")
  config_overrides = request.json["config"]
  if config_overrides is None:
    config_overrides = {}
  logging.debug("Passing inferencing configuration: %s", config_overrides)
  outputs = app.model.generate(
    inputs["input_ids"],
    **config_overrides
  )

  logging.info("Generated.")
  text_out = map(lambda x: app.tokenizer.decode(x, skip_special_tokens=True),
                 outputs)
  logging.info("Decoded")
  return {"predictions": list(text_out)}


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
  app.host = os.environ.get("SERVER_HOST", "localhost")
  app.port = int(os.environ.get("AIP_HTTP_PORT", str(FLAGS.port)))

  init_model()
  app.run(app.host, app.port, debug=False)


if __name__ == "__main__":
  absl_app.run(main, flags_parser=parse_flags)

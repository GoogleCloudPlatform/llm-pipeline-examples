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
import deepspeed
from flask import Flask, send_from_directory
from flask import request
import gcsfs
import torch
from transformers import AutoConfig
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig

app = Flask(__name__, root_path=os.path.join(os.getcwd(), "app/"))
FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", None, "Path of HF model to load.")
flags.DEFINE_integer("port", 5000, "server port.")


def init_model():
  """Initializes the model using deep speed."""
  os.environ["TOKENIZERS_PARALLELISM"] = "false"

  # distributed setup
  local_rank = int(os.getenv("LOCAL_RANK", "0"))
  world_size = int(os.getenv("WORLD_SIZE", "1"))
  torch.cuda.set_device(local_rank)
  app.local_rank = local_rank
  app.world_size = world_size

  model_path = os.environ.get("AIP_STORAGE_URI", FLAGS.model_path)
  logging.info("Model path: %s", model_path)
  if model_path.startswith("gs://"):
    src = model_path.replace("gs://", "")
    dst = src.split("/")[-1] + "/"
    if local_rank == 0:
      gcs = gcsfs.GCSFileSystem()
      logging.info("Downloading model from %s", model_path)
      for f in gcs.ls(src):
        if gcs.isfile(f):
          logging.info("Downloading %s", f)
          gcs.get(f, dst)
    model_path = dst

  deepspeed.init_distributed()
  config = AutoConfig.from_pretrained(model_path)
  model_hidden_size = config.d_model

  train_batch_size = 1 * world_size

  ds_config = {
      "fp16": {
          "enabled": False
      },
      "bf16": {
          "enabled": False
      },
      "zero_optimization": {
          "stage":
              3,
          "offload_param": None,
          "overlap_comm":
              True,
          "contiguous_gradients":
              True,
          "reduce_bucket_size":
              model_hidden_size * model_hidden_size,
          "stage3_prefetch_bucket_size":
              0.9 * model_hidden_size * model_hidden_size,
          "stage3_param_persistence_threshold":
              10 * model_hidden_size
      },
      "steps_per_print": 2000,
      "train_batch_size": train_batch_size,
      "train_micro_batch_size_per_gpu": 1,
      "wall_clock_breakdown": False
  }
  # fmt: on

  dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

  # now a model can be loaded.
  logging.info("Loading local model from %s", model_path)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

  # initialise Deepspeed ZeRO and store only the engine object
  ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
  ds_engine.module.eval()  # inference

  tokenizer = AutoTokenizer.from_pretrained(model_path)
  logging.info("Model ready to serve")
  app.ds_engine = ds_engine
  app.tokenizer = tokenizer
  app.dschf = dschf


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
      truncation=True).to(device=app.local_rank)
  logging.info("Encoded")
  with torch.no_grad():
    outputs = app.ds_engine.module.generate(
        inputs["input_ids"], synced_gpus=True)
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
  if app.local_rank == 0:
    app.run(app.host, app.port, debug=False)
  else:
    t = app.tokenizer.encode("", return_tensors="pt").to(device=app.local_rank)
    while True:
      with torch.no_grad():
        app.ds_engine.module.generate(t, synced_gpus=True)


if __name__ == "__main__":
  absl_app.run(main, flags_parser=parse_flags)

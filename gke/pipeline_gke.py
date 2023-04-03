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
"""LLM Kubeflow Training Pipeline.

A Kubeflow Training pipeline for training LLM.
"""

from collections.abc import Sequence
import json
from os import path
import time
from typing import NamedTuple
from typing import List

from absl import app
from absl import flags
from absl import logging
from google.cloud.aiplatform import PipelineJob
from google.cloud.aiplatform import Endpoint

import kfp
import kfp.components as comp
from kfp.v2 import compiler
from kfp.v2 import dsl
from kfp.v2.dsl import component
from kfp.v2.dsl import Input
from kfp.v2.dsl import Model

from deploy_to_gke import deploy_to_gke

FLAGS = flags.FLAGS

flags.DEFINE_string("project", None, "Project ID to run pipeline.")
flags.DEFINE_string("pipeline_root", None, "GCS location for pipeline files.")
flags.DEFINE_string("config", None, "Pipeline configuration.")
flags.DEFINE_bool("enable_caching", True, "Whether to cache successful stages.")
flags.DEFINE_bool("override_deploy", False, "Overrides deployed endpoint model even if model metrics are worse than deployed model.")
flags.DEFINE_bool("verify", False, "Wait till success and do prediction.")
flags.DEFINE_bool("cleanup_endpoint", False, "Delete the endpoint after verifying the deployment - can be useful for test scenarios.")
flags.DEFINE_string("verify_payload", "predict_payload.json", "Payload sent to prediction endpoint for verification.")
flags.DEFINE_string("verify_result", "predict_result.json", "Expected result from verification.")
flags.DEFINE_string("image_tag", "release",
                    "Image tag for components base images")
flags.DEFINE_string("endpoint_name", '', "Name of the endpoint to deploy trained model to. Defaults to config.model_display_name.")
flags.DEFINE_string("model_location", '', "GCS url of the model to deploy to GKE.")
flags.DEFINE_bool("use_faster_transformer", False,
                  "Experimental flag to use FasterTransformer to convert the provided model into an optimized format. Currently only supported for the T5 model family.")
flags.mark_flag_as_required("project")
flags.mark_flag_as_required("pipeline_root")
flags.mark_flag_as_required("config")

download_component = comp.load_component_from_file("../components/download.yaml")
preprocess_component = comp.load_component_from_file(
    "../components/preprocess.yaml")
trainer_component = comp.load_component_from_file("../components/trainer.yaml")
convert_component = comp.load_component_from_file("../components/convert.yaml")

@kfp.dsl.pipeline(name="llm-pipeline")
def my_pipeline(
    model_display_name: str,
    deploy_gpu_type: str,
    deploy_gpu_count: int,
    gke_zone: str,
    gke_id: str,
    model_checkpoint_location: str,
):
  """Pipeline defintion function."""
# pylint: disable=unused-variable
  model_input = FLAGS.model_location
  if not model_input:
    model_input = model_checkpoint_location

  deploy_op = deploy_to_gke(
    project=FLAGS.project,
    model_display_name=model_display_name,
    serving_container_image_uri=(
        f"gcr.io/llm-containers/predict-triton:{FLAGS.image_tag}"
    ),
    model_storage_location=model_input,
    gpu_type=deploy_gpu_type,
    gpu_count=deploy_gpu_count,
    location=gke_zone,
    cluster_id=gke_id)

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not path.exists(FLAGS.config):
    raise app.UsageError("Config file doesn't exist.")

  with open(FLAGS.config, "r") as f:
    config = json.load(f)

  dest_path = "/tmp/pipeline.json"
  compiler.Compiler().compile(
      pipeline_func=my_pipeline,
      package_path=dest_path,
      pipeline_parameters=config)

  with open(dest_path, "r") as f:
    js = json.load(f)
    for _, v in js["pipelineSpec"]["deploymentSpec"]["executors"].items():
      v["container"]["image"] = f"{v['container']['image']}:{FLAGS.image_tag}"

  with open(dest_path, "w") as f:
    json.dump(js, f, indent=2)

  job = PipelineJob(
      display_name="gke-pipeline",
      template_path=dest_path,
      pipeline_root=FLAGS.pipeline_root,
      project=FLAGS.project,
      enable_caching=FLAGS.enable_caching)

  job.submit()

  # if FLAGS.verify:
  #   job.wait()

  #   endpoint = _get_endpoint(job, config["zone"])

  #   with open(FLAGS.verify_payload, "r") as f:
  #     payload = json.load(f)
    
  #   logging.info("Sending inference request...")
  #   result = endpoint.predict(list(payload["instances"]))

  #   if len(result.predictions) < 1:
  #     logging.error("No inferences returned")
  #     raise RuntimeError("Unexpected verification results")
    
  #   with open(FLAGS.verify_result) as f:
  #     expected_results = json.load(f)["predictions"][0]

  #   if result.predictions[0] != expected_results:
  #     logging.error("Unexpected inference results= [%s] expected= [%s]", result.predictions[0], expected_results)
  #     raise RuntimeError("Unexpected verification results")
    
  #   logging.info("Inference verified successfully!")

  # if FLAGS.cleanup_endpoint:
  #   job.wait()
    
  #   endpoint = _get_endpoint(job, config["zone"])

  #   logging.info(f"Deleting endpoint {endpoint.name}...")
  #   endpoint.delete(force=True)
  #   logging.info("Endpoint deleted.")

if __name__ == "__main__":
  app.run(main)

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
flags.DEFINE_bool("use_faster_transformer", False,
                  "Experimental flag to use FasterTransformer to convert the provided model into an optimized format. Currently only supported for the T5 model family.")
flags.mark_flag_as_required("project")
flags.mark_flag_as_required("pipeline_root")
flags.mark_flag_as_required("config")

download_component = comp.load_component_from_file("components/download.yaml")
preprocess_component = comp.load_component_from_file(
    "components/preprocess.yaml")
trainer_component = comp.load_component_from_file("components/trainer.yaml")
convert_component = comp.load_component_from_file("components/convert.yaml")

@component(base_image="gcr.io/llm-containers/deploy")
def should_deploy(
    project: str,
    model_display_name: str,
    model: Input[Model],
    override_deploy: bool, 
) -> str:
  """Deploys the model to Vertex AI Prediction."""
# pylint: disable=g-import-not-at-top, reimported, redefined-outer-name
  import google.cloud.aiplatform as aip
  import gcsfs
  import json
  import os

  existing_endpoints = aip.Endpoint.list(
      project=project,
      order_by="create_time",
      filter='display_name="{}"'.format(model_display_name))

  endpoint_active = False
  if existing_endpoints:
    endpoint_active = bool(existing_endpoints[0].traffic_split)

  gcs = gcsfs.GCSFileSystem()
  new_metrics = {}
  metrics_file = os.path.join(model.uri, "metrics.json")
  eval_metrics = ["eval_rouge1", "eval_rouge2", "eval_rougel"]
  if gcs.exists(metrics_file):
    with gcs.open(metrics_file, "r") as f:
      saved_metrics = json.load(f)
      for k, v in saved_metrics.items():
        if k.lower() in eval_metrics:
          new_metrics[k.lower()] = v

      print(f"New metrics: {new_metrics}")
  else:
    new_metrics = {}
    print("Warning! Model doesn't have metrics.")

  existing_models = aip.Model.list(
      project=project,
      order_by="create_time",
      filter='display_name="{}"'.format(model_display_name))

  if existing_models:
    parent_model = existing_models[0]
    if parent_model.labels:
      def are_better_metrics(a, b):
        for k in eval_metrics:
          if k not in a or k not in b:
            return False
          if a[k] <= b[k]:
            return False
        return True

      existing_metrics = {}
      for k in eval_metrics:
        if k in parent_model.labels:
          strv = parent_model.labels[k]
          v = float(strv.replace("_", ".")) if strv.replace(
              "_", "").isdigit() else .0
          existing_metrics[k] = v

      print(f"Current metrics: {existing_metrics}")
      if not are_better_metrics(
          new_metrics, existing_metrics) and endpoint_active:
        print("New model doesn't have better metrics.")
        if override_deploy:
          return "deploy"
        return "abort"
  return "deploy"


@component(base_image="gcr.io/llm-containers/deploy")
def deploy(
    project: str,
    model_display_name: str,
    serving_container_image_uri: str,
    model: Input[Model],
    machine_type: str,
    gpu_type: str,
    gpu_count: int,
    endpoint_name: str,
) -> NamedTuple(
    "Outputs",
    [
        ("model", str),
        ("endpoint", str),  # Return parameter.
    ],
):
  """Deploys the model to Vertex AI Predictin."""
# pylint: disable=g-import-not-at-top, reimported, redefined-outer-name
  import google.cloud.aiplatform as aip
  import gcsfs
  import json
  import os

  if not endpoint_name:
    endpoint_name = model_display_name

  existing_endpoints = aip.Endpoint.list(
      project=project,
      order_by="create_time",
      filter='display_name="{}"'.format(endpoint_name))

  if existing_endpoints:
    endpoint = existing_endpoints[0]
    endpoint.undeploy_all()
  else:
    endpoint = aip.Endpoint.create(
        project=project,
        display_name=endpoint_name,
    )

  existing_models = aip.Model.list(
      project=project,
      order_by="create_time",
      filter='display_name="{}"'.format(model_display_name))

  if existing_models:
    parent_model = existing_models[0]
    parent_model_resource_name = parent_model.resource_name
  else:
    parent_model_resource_name = None

  gcs = gcsfs.GCSFileSystem()
  new_metrics = None
  metrics_file = os.path.join(model.uri, "metrics.json")
  if gcs.exists(metrics_file):
    with gcs.open(metrics_file, "r") as f:
      new_metrics = json.load(f)
      print(f"New metrics: {new_metrics}")
  else:
    new_metrics = {}
    print("Warning! Model doesn't have metrics.")

  deployable_model = aip.Model.upload(
      project=project,
      display_name=model_display_name,
      artifact_uri=model.uri,
      serving_container_image_uri=serving_container_image_uri,
      parent_model=parent_model_resource_name,
      labels={
          k.lower(): str(v).replace(".", "_") for k, v in new_metrics.items()
      },
      serving_container_predict_route="/infer",
      serving_container_health_route="/health"
  )

  endpoint.deploy(
      model=deployable_model,
      deployed_model_display_name=model_display_name,
      machine_type=machine_type,
      accelerator_type=gpu_type,
      accelerator_count=gpu_count)

  return (deployable_model.name, endpoint.name)


@kfp.dsl.pipeline(name="llm-pipeline")
def my_pipeline(
    dataset: str,
    dataset_subset: str,
    document_column: str,
    summary_column: str,
    cluster_prefix: str,
    node_count: int,
    model_checkpoint: str,
    machine_type: str,
    gpu_count: int,
    batch_size: int,
    epochs: int,
    model_display_name: str,
    deploy_machine_type: str,
    deploy_gpu_type: str,
    deploy_gpu_count: int,
    gpu_type: str,
    zone: str,
    pipeline_node_memory_limit: str = "16G",
):
  """Pipeline defintion function."""
# pylint: disable=unused-variable
  download_op = download_component(
    dataset=dataset,
    subset=dataset_subset,
    model_checkpoint=model_checkpoint)

  preprocess_op = preprocess_component(
      model_checkpoint=model_checkpoint,
      document_column=document_column,
      summary_column=summary_column,
      raw_dataset=download_op.outputs["dataset_path"],
  )

  train_op = trainer_component(
      cluster_prefix=cluster_prefix,
      node_count=node_count,
      model_checkpoint=model_checkpoint,
      machine_type=machine_type,
      gpu_count=gpu_count,
      data=preprocess_op.outputs["output_dataset"],
      project=FLAGS.project,
      batch_size=batch_size,
      epochs=epochs,
      gpu_type=gpu_type,
      zone=zone,
      id=str(int(time.time())),
      image_tag=FLAGS.image_tag,
      workspace_path=download_op.outputs["workspace_path"]
  )

  should_deploy_op = should_deploy(
      project=FLAGS.project,
      model_display_name=model_display_name,
      model=train_op.outputs["model"],
      override_deploy=FLAGS.override_deploy)

  with dsl.Condition(should_deploy_op.output == "deploy", name="Deploy"):
    if FLAGS.use_faster_transformer:
      convert_op = convert_component(
        model_checkpoint=train_op.outputs["model"],
        gpu_number=deploy_gpu_count
      ).set_memory_limit(pipeline_node_memory_limit)

      deploy_op = deploy(
        project=FLAGS.project,
        model_display_name=model_display_name,
        serving_container_image_uri=(
            f"gcr.io/llm-containers/predict-triton:{FLAGS.image_tag}"
        ),
        model=convert_op.outputs["converted_model"],
        machine_type=deploy_machine_type,
        gpu_type=deploy_gpu_type,
        gpu_count=deploy_gpu_count,
        endpoint_name=FLAGS.endpoint_name)

    else:
      deploy_op = deploy(
        project=FLAGS.project,
        model_display_name=model_display_name,
        serving_container_image_uri=(
            f"gcr.io/llm-containers/predict:{FLAGS.image_tag}"
        ),
        model=train_op.outputs["model"],
        machine_type=deploy_machine_type,
        gpu_type=deploy_gpu_type,
        gpu_count=deploy_gpu_count,
        endpoint_name=FLAGS.endpoint_name)

def _get_endpoint_id(pipeline_job):
  """Returns the deploy endpoint name from a successful pipeline job."""

  for task in pipeline_job.task_details:
    if task.task_name == "deploy":
      endpoint = task.execution.metadata["output:endpoint"]
      logging.info("Endpoint %s found!", endpoint)
      return endpoint
  logging.error("No deploy task found :(. Task = %s",
    pipeline_job.task_details)
  raise RuntimeError("Unexpected deploy result format")

def _get_endpoint(pipeline_job, zone):
  """Returns the Endpoint object from a successful pipeline job."""
  endpoint_name=_get_endpoint_id(pipeline_job)
  region = zone[:zone.rfind("-")]
  logging.info("Region is %s", region)
  return Endpoint(endpoint_name, project=FLAGS.project, location=region)

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
      display_name="llm-pipeline",
      template_path=dest_path,
      pipeline_root=FLAGS.pipeline_root,
      project=FLAGS.project,
      enable_caching=FLAGS.enable_caching)

  job.submit()

  if FLAGS.verify:
    job.wait()

    endpoint = _get_endpoint(job, config["zone"])

    with open(FLAGS.verify_payload, "r") as f:
      payload = json.load(f)
    
    logging.info("Sending inference request...")
    result = endpoint.predict(list(payload["instances"]))

    if len(result.predictions) < 1:
      logging.error("No inferences returned")
      raise RuntimeError("Unexpected verification results")
    
    with open(FLAGS.verify_result) as f:
      expected_results = json.load(f)["predictions"][0]

    if result.predictions[0] != expected_results:
      logging.error("Unexpected inference results= [%s] expected= [%s]", result.predictions[0], expected_results)
      raise RuntimeError("Unexpected verification results")
    
    logging.info("Inference verified successfully!")

  if FLAGS.cleanup_endpoint:
    job.wait()
    
    endpoint = _get_endpoint(job, config["zone"])

    logging.info(f"Deleting endpoint {endpoint.name}...")
    endpoint.delete(force=True)
    logging.info("Endpoint deleted.")


if __name__ == "__main__":
  app.run(main)

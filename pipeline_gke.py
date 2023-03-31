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


@component(base_image="gcr.io/llm-containers/deploy-gke")
def deploy_to_gke(
    project: str,
    model_display_name: str,
    serving_container_image_uri: str,
    model: str,
    machine_type: str,
    gpu_type: str,
    gpu_count: int,
    endpoint_name: str,
    # New
    location: str,
    cluster_id: str
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
  from kubernetes import client

  ### GKE Demo
  import google.auth
  import google.auth.transport.requests
  import google.cloud.container
  from kubernetes import client
  from tempfile import NamedTemporaryFile
  import base64

  def get_k8s_client(project_id: str, location: str,
                   cluster_id: str) -> client.CoreV1Api:
    container_client = google.cloud.container.ClusterManagerClient()
    request = {
      "name": f"projects/{project_id}/locations/{location}/clusters/{cluster_id}"
    }
    response = container_client.get_cluster(request=request)
    creds, projects = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    configuration = client.Configuration()
    configuration.host = f'https://{response.endpoint}'
    with NamedTemporaryFile(delete=False) as ca_cert:
      ca_cert.write(base64.b64decode(response.master_auth.cluster_ca_certificate))
      configuration.ssl_ca_cert = ca_cert.name
    configuration.api_key_prefix['authorization'] = 'Bearer'
    configuration.api_key['authorization'] = creds.token

    return client.CoreV1Api(client.ApiClient(configuration))

  def create_deployment_object(model_display_name: str, gpu_type: str, gpu_count: int, image: str, model_location: str) -> client.V1Deployment:
    container = client.V1Container(
      name = model_display_name,
      image = image,
      ports=[client.V1ContainerPort(container_port=5000)],
      resources=client.V1ResourceRequirements(
        requests={"cpu": "100m", "memory": "200Mi"},
        limits={"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": f"{gpu_count}"}
      ),
      env=[client.V1EnvVar("AIP_STORAGE_URI", model_location)],
      liveness_probe= client.V1Probe(
        http_get=client.V1HTTPGetAction(path="/health", port=5000),
        initial_delay_seconds=10,
        period_seconds=3,
        failure_threshold=10
      )
    )

      # Create and configure a spec section
    template = client.V1PodTemplateSpec(
      metadata=client.V1ObjectMeta(labels={"app": model_display_name}),
      spec=client.V1PodSpec(
        containers=[container],
        node_selector={"cloud.google.com/gke-accelerator": gpu_type.lower().replace('_', '-')}
      )
    )

    # Create the specification of deployment
    spec = client.V1DeploymentSpec(
      replicas=1, template=template, selector={
        "matchLabels":
        {"app": model_display_name}})

    # Instantiate the deployment object
    return client.V1Deployment(
      api_version="apps/v1",
      kind="Deployment",
      metadata=client.V1ObjectMeta(name=model_display_name),
      spec=spec,
    )

  def create_service_object(model_display_name: str, target_port: int) -> client.V1Service:
    return client.V1Service(
      metadata=client.V1ObjectMeta(name=model_display_name),
      spec=client.V1ServiceSpec(
        type="ClusterIP",
        selector={"app": model_display_name},
        ports=[client.V1ServicePort(
          protocol="TCP",
          port=80,
          target_port=target_port)]
      )
    )

  k8s_client = get_k8s_client(project, location, cluster_id)
  api_client = client.AppsV1Api(k8s_client.api_client)
  dep = create_deployment_object(model_display_name, gpu_type, gpu_count, serving_container_image_uri, model)
  srv = create_service_object(model_display_name, 5000)
  deployment = api_client.create_namespaced_deployment("default", dep)
  service = k8s_client.create_namespaced_service("default", srv)

  return (model_display_name, service.spec.cluster_ip)

### End GKE Demo


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
  # download_op = download_component(
  #   dataset=dataset,
  #   subset=dataset_subset,
  #   model_checkpoint=model_checkpoint)

  # preprocess_op = preprocess_component(
  #     model_checkpoint=model_checkpoint,
  #     document_column=document_column,
  #     summary_column=summary_column,
  #     raw_dataset=download_op.outputs["dataset_path"],
  # )

  # train_op = trainer_component(
  #     cluster_prefix=cluster_prefix,
  #     node_count=node_count,
  #     model_checkpoint=model_checkpoint,
  #     machine_type=machine_type,
  #     gpu_count=gpu_count,
  #     data=preprocess_op.outputs["output_dataset"],
  #     project=FLAGS.project,
  #     batch_size=batch_size,
  #     epochs=epochs,
  #     gpu_type=gpu_type,
  #     zone=zone,
  #     id=str(int(time.time())),
  #     image_tag=FLAGS.image_tag,
  #     workspace_path=download_op.outputs["workspace_path"]
  # )

    # convert_op = convert_component(
    #   model_checkpoint=train_op.outputs["model"],
    #   gpu_number=deploy_gpu_count
    # ).set_memory_limit(pipeline_node_memory_limit)

  model_input = "gs://pirillo-bucket/pipeline_runs/237939871711/llm-pipeline-20230330101008/convert_-2783175366569623552/converted_model"
  deploy_op = deploy_to_gke(
    project=FLAGS.project,
    model_display_name=model_display_name,
    serving_container_image_uri=(
        f"gcr.io/llm-containers/predict-triton:{FLAGS.image_tag}"
    ),
    model=model_input,
    machine_type=deploy_machine_type,
    gpu_type=deploy_gpu_type,
    gpu_count=deploy_gpu_count,
    endpoint_name=FLAGS.endpoint_name,
    location="us-central1-c",
    cluster_id="v100testcluster")

  # else:
  #   deploy_op = deploy_to_gke(
  #     project=FLAGS.project,
  #     model_display_name=model_display_name,
  #     serving_container_image_uri=(
  #         f"gcr.io/llm-containers/predict:{FLAGS.image_tag}"
  #     ),
  #     model=train_op.outputs["model"],
  #     machine_type=deploy_machine_type,
  #     gpu_type=deploy_gpu_type,
  #     gpu_count=deploy_gpu_count,
  #     endpoint_name=FLAGS.endpoint_name)

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

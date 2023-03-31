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
"""Utility module for preprocessing and fine tuning.

Utility module for preprocessing and fine tuning.
"""

import functools
import gcsfs
import time

def gcs_path(path: str, gcs_prefix=''):
  if path.startswith('gs://'):
    fs = gcsfs.GCSFileSystem()
    return path.replace('gs://', gcs_prefix), fs
  if path.startswith('/gcs/'):
    fs = gcsfs.GCSFileSystem()
    return path.replace('/gcs/', gcs_prefix), fs
  return path, None

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time for call to {func.__name__}: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer

### GKE Demo
import google.auth
import google.auth.transport.requests
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
      requests={"cpu": "100m", "memory": "200Mi", "nvidia.com/gpu": f"{gpu_count}"},
      limits={"cpu": "4", "memory": "8Gi"}
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
      selector={"matchLabels": {"app": model_display_name}},
      ports=[client.V1ServicePort(
        protocol="TCP",
        port=80,
        target_port=target_port)]
    )
  )

### End GKE Demo

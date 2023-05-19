from typing import NamedTuple
from typing import List
from kfp.v2.dsl import component

@component(base_image="gcr.io/llm-containers/deploy-gke")
def deploy_to_gke(
    project: str,
    model_display_name: str,
    serving_container_image_uri: str,
    model_storage_location: str,
    gpu_type: str,
    gpu_count: int,
    location: str,
    cluster_id: str
) -> NamedTuple(
    "Outputs",
    [
        ("model", str),
        ("endpoint", str),  # Return parameter.
    ],
):
  """Deploys the model to a specified GKE cluster."""
# pylint: disable=g-import-not-at-top, reimported, redefined-outer-name
  k8s_client = get_k8s_client(project, location, cluster_id)
  api_client = client.AppsV1Api(k8s_client.api_client)
  dep = create_deployment_object(model_display_name, gpu_type, gpu_count, serving_container_image_uri, model_storage_location)
  srv = create_service_object(model_display_name, 5000)
  deployment = api_client.create_namespaced_deployment("default", dep)
  service = k8s_client.create_namespaced_service("default", srv)

  return (model_display_name, service.spec.cluster_ip)

import google.auth
import google.auth.transport.requests
import google.cloud.container
from kubernetes import client
from tempfile import NamedTemporaryFile
import base64

def get_k8s_client(project_id: str, location: str, cluster_id: str) -> client.CoreV1Api:
    container_client = google.cloud.container.ClusterManagerClient()
    request = {
        "name": f"projects/{project_id}/locations/{location}/clusters/{cluster_id}"
    }
    response = container_client.get_cluster(request=request)
    creds, _ = google.auth.default()
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
        ports=[client.V1ContainerPort(container_port=5000), client.V1ContainerPort(container_port=8000)],
        resources=client.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "200Mi"},
            limits={"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": f"{gpu_count}"}
        ),
        env=[client.V1EnvVar("AIP_STORAGE_URI", model_location)],
        liveness_probe= client.V1Probe(
            http_get=client.V1HTTPGetAction(path="/health", port=5000),
            initial_delay_seconds=20,
            period_seconds=10,
            failure_threshold=15
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

def create_service_object(model_display_name: str, target_ports: List[int]) -> client.V1Service:
    return client.V1Service(
    metadata=client.V1ObjectMeta(name=model_display_name),
    spec=client.V1ServiceSpec(
        type="ClusterIP",
        selector={"app": model_display_name},
        ports= map(lambda p : client.V1ServicePort(
                        protocol="TCP",
                        port=p,
                        target_port=p),
                    target_ports)
    )
)
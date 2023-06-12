# Running T5 on GKE

This document covers deploying HuggingFace T5 models to GKE, whether it is directly from HuggingFace or fine-tuned through the LLM Pipeline in this repo.

The output of running the commands described in this document will be a provisioned GKE cluster within your GCP project hosting the model you provide. The model will have an endpoint accessible from within your virtual cloud network, and external to the internet if your project's firewall is configured for it.

Support has been tested for the following T5 families available on huggingface.

* t5 (small, base, large, 11b)
* google/t5-v1_1 (small, base, large, xl, xxl)
* google/flan-t5 (small, base, large, xl, xxl)

### Pre-Requisites

These commands can be run from any terminal configured for gcloud, or through [Cloud Shell](https://cloud.google.com/shell/docs/launching-cloud-shell).

Start by enabling the required APIs within your GCP project.
`gcloud services enable container.googleapis.com storage.googleapis.com run.googleapis.com`

### Create an Environment file

An environment variable file containing the configuration for the GKE cluster and the model needs to be created. The full specification for the cluster configuration can be found [here](https://github.com/GoogleCloudPlatform/ai-infra-cluster-provisioning#configuration-for-users). A sample configuration is available in the repository at [llm-pipeline-examples/src/gke/sample_environment_config.yml](https://github.com/GoogleCloudPlatform/llm-pipeline-examples/blob/main/src/gke/cluster_config.yml)

Using the sample configuration will create a 2 node GKE cluster with a2-megagpu-16g VMs, and 4 nvidia-tesla-a100 GPUs on each node.

There are several variables that need to be set for the Model Deployment.

<table>
  <tr>
   <td><strong>Environment Variable Name</strong>
   </td>
   <td><strong>Required</strong>
   </td>
   <td><strong>Description</strong>
   </td>
   <td><strong>Example Value</strong>
   </td>
  </tr>
  <tr>
   <td><code>GPU_COUNT_PER_MODEL</code>
   </td>
   <td>Y
   </td>
   <td>Number of GPUs exposed to the pod, also used to set the parallelism when using FasterTransformer
   </td>
   <td><code>4</code>
   </td>
  </tr>
  <tr>
   <td><code>MODEL_SOURCE_PATH</code>
   </td>
   <td>Y
   </td>
   <td>GCS path or Huggingface repo pointing to the directory of the model to deploy.
<p>
Note: For a model fine tuned using the pipeline, look at the Model Artifact after the training step and use the URL property.
   </td>
   <td><code>gs://my-bucket/pipeline_runs/237939871711/llm-pipeline-20230328153111/train_5373485673388965888/Model/</code>

   or

   <p>google/t5-flan-xxl
   </td>
  </tr>
  <tr>
   <td><code>EXISTING_CLUSTER_ID</code>
   </td>
   <td>N
   </td>
   <td>Name of an existing cluster (in the corresponding Region and Project) to use instead of provisioning a new cluster.
   </td>
   <td><code>my-gke</code>
   </td>
  </tr>
  <tr>
   <td><code>KSA_NAME</code>
   </td>
   <td>N
   </td>
   <td>Name of the Kubernetes Service Account configured with access to the given GCS path. By default one will be provisioned as ‘aiinfra-gke-sa’
   </td>
   <td><code>my-other-ksa</code>
   </td>
  </tr>
  <tr>
   <td><code>MODEL_NAME</code>
   </td>
   <td>N
   </td>
   <td>Friendly name for the model, used in constructing the Kubernetes Resource names
   </td>
   <td><code>t5-flan</code>
   </td>
  </tr>
  <tr>
   <td><code>INFERENCING_IMAGE_TAG</code>
   </td>
   <td>N
   </td>
   <td>Image tag for the inference image. Default is ‘release’
   </td>
   <td><code>latest</code>
   </td>
  </tr>
  <tr>
   <td><code>USE_FASTER_TRANSFORMER</code>
   </td>
   <td>N
   </td>
   <td>Boolean to set when the FasterTransformer / Triton path should be enabled.
<p>
This controls whether a Conversion job is scheduled, and the inference image that will be deployed.
   </td>
   <td><code>true</code>
   </td>
  </tr>
  <tr>
   <td><code>CONVERTED_MODEL_UPLOAD_PATH</code>
   </td>
   <td>Y*
   </td>
   <td>Only required when USE_FASTER_TRANSFORMER is set.
<p>
A GCS path to upload the model after it is converted for FasterTransformer
   </td>
   <td><code>gs://my-bucket/converted_t5/1/Model</code>
   </td>
  </tr>
  <tr>
   <td><code>POD_MEMORY_LIMIT</code>
   </td>
   <td>N
   </td>
   <td>Sets the memory limit of pods for GKE in <a href="https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-memory">Kubernetes Memory resource format</a>. Defaults to “16Gi”.
   </td>
   <td><code>50Gi</code>
   </td>
  </tr>
</table>

### Running the image

The Cluster Provisioning + Deployment image is available at [gcr.io/llm-containers/gke-provision-deploy](gcr.io/llm-containers/gke-provision-deploy) .

Run the image using [`gcloud run deploy`](https://cloud.google.com/sdk/gcloud/reference/run/deploy).

`gcloud run deploy --env-vars-file src/gke/cluster_config.yml --image=gcr.io/llm-containers/gke-provision-deploy`

After the image finishes provisioning the cluster, the model will be converted (if necessary) and deployed to the cluster. The image will then terminate.


### Consuming the Deployed Model

A NodePort service on the cluster is automatically created during deployment. This nodeport allows a user to consume the model on a network that has access to the GKE node.

The IP address of the nodes can be found using the GKE dashboard on Pantheon or using gcloud -> kubectl.

For gcloud, retrieve the kubeconfig file and use kubectl commands to communicate with the cluster.

The cluster name will be `$NAME_PREFIX-gke`.

    $ gcloud containers clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT
    $ kubectl get nodes –output=wide	# Retrieve the Internal or External IP
    $ kubectl get svc # Retrieve the Port mapped to 5000 for basic consumption, 8000 for raw consumption
    $ curl --location 'http://$IP:$PORT/health'
    200 { "health": "ok" }

The image will also log these values at the end of a run.

Sample output:
```
From a machine on the same VPC as this cluster you can call http://10.128.0.29:32754/infer
***********
To deploy a sample notebook for experimenting with this deployed model, paste the following link into your browser:
https://pantheon.corp.google.com/vertex-ai/workbench/user-managed/deploy?download_url=https%3A%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fllm-pipeline-examples%main%2Fexamples%2Ft5-gke-sample-notebook.ipynb&project=<project_id>
Set the following parameters in the variables cell of the notebook:
host             = '10.128.0.29'
flask_node_port  = '32754'
triton_node_port = '31246'
payload = '<your_payload_goes_here>'

***********
```


### Available Endpoints


#### /health [GET]

Basic health endpoint serving as a Kubernetes Liveness probe.


#### /ui [GET]

Returns a basic UI for prompt engineering


#### /infer [POST]

Takes and returns the string version of an inference payload. Configured for the Vertex API, so payloads should be provided in the format of: \
{ “instances”: [“payload1”, “payload2” … ] }

Responses will be returned in Vertex format:

{ “predictions”: [“prediction1”, “prediction2” … ], “metrics”: [ {“metric1”: “value1”}, {“units”: “unit_measurement”}]


#### /v2/models/fastertransformer/infer [POST]

Only available on FasterTransformer image. A raw endpoint that directly communicates with Triton, taking the Triton tensor payload.


### Limitations of Image

These limitations are accurate as of June 1, 2023.


* Tokenizer within the FasterTransformer Predict image is based on the T5-base dictionary.
* FasterTransformer image only supports the T5 model family (t5, t5-v1_1, flan-t5). All sizes are supported.
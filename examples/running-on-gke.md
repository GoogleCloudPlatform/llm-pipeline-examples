## Deployment

The deployment to GKE is currently detached from the training pipeline. The model artifact that is a result of training will need to be manually deployed to the cluster using these instructions.


### Create an Environment file

&lt;TODO> / This section will need to change after CPT-0.7.0 release \
 \
An environment variable file containing the configuration for the GKE cluster and the model needs to be created. The full specification for the cluster configuration can be found [here](https://github.com/GoogleCloudPlatform/ai-infra-cluster-provisioning#configuration-for-users). A sample configuration is available in the repository at [llm-pipeline-examples/src/gke/sample_environment_config.list](https://github.com/GoogleCloudPlatform/llm-pipeline-examples/blob/main/src/gke/cluster_config.list)

There are also several variables that need to be set for the Model Deployment.


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
   <td>GCS path pointing to the directory of the model to deploy.
<p>
Note: For a model fine tuned using the pipeline, look at the Model Artifact after the training step and use the URL property.
   </td>
   <td><code>gs://my-bucket/pipeline_runs/237939871711/llm-pipeline-20230328153111/train_5373485673388965888/Model/</code>
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
   <td><code>gs://pirillo-sct-bucket/converted_t5/1/Model</code>
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

Run the image using the `docker run` command specifying the Environment File with `--env-file`.

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


* `Tokenizer within the Predict image is based on the T5-base dictionary.`
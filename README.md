# Training Large Language Models on Google Cloud

The challenges of training large language models are multiple. To start with, it
needs a large infrastructure of compute resources. Multiple machines with
multiple hardware accelerators such as GPUs and TPUs are needed to train a
single model. Getting the infrastructure ready for running is just the start of
the challenge. When training starts, it could take multiple days for training to
converge. This, besides the fact that we are training on a large number of
hardware, increases the probability of experiencing a failure during training.
If training fails, we need to restart and get the infrastructure ready again and
resume where we left off.

In addition to these problems, we face the common production ready machine
learning problems. Such as, reliable retraining, data storage, checkpoint
storage, model versioning, tracking model quality and deployment to production.

Google Cloud Platform is one of the largest cloud providers which provides
compute infrastructure suitable for training large language models. GCP is offering
 [A3](https://cloud.google.com/blog/products/compute/announcing-cloud-tpu-v5e-and-a3-gpus-in-ga)
 VMs, which are powered by Nvidia's latest [H100](https://www.nvidia.com/en-us/data-center/h100/)
  GPU.

There are multiple model pre-training frameworks. In these examples, we show how to 
pretrain a GPT model using [Megatron-Deepspeed](https://github.com/Microsoft/Megatron-Deepspeed).
We also show how to finetune a T5 model using the [Hugging Face transformer library](https://huggingface.co/docs/transformers/index).


## Quick Start Guide
### Prerequisites

1.  Make sure you have gcloud installed and that you are authenticated including
    application default authentication

    ```bash
    gcloud auth login
    gcloud auth application-default login
    ```

1.  **Enable Services** In your project, enable services needed to run the pipeline. You can do this
    by issuing the following command:

    ```bash
        export PROJECT_ID=<your project ID>
        gcloud services enable cloudfunctions compute.googleapis.com iam.googleapis.com cloudresourcemanager.googleapis.com --project=${PROJECT_ID}
    ```

1.  **Create Workspace** Create our workspace VM which we will use to process the data, create cluster
and launch training:

    ```bash
    gcloud compute instances create llm-processor     --project=${PROJECT_ID}     --zone=us-east4-c     --machine-type=e2-standard-4     --metadata=enable-oslogin=true     --scopes=https://www.googleapis.com/auth/cloud-platform     --create-disk=auto-delete=yes,boot=yes,device-name=llm-processor,image-project=cos-cloud,image-family=cos-stable,mode=rw,size=250,type=projects/${PROJECT_ID}/zones/us-central1-a/diskTypes/pd-balanced 
    ```

1. **Connect** Connect to the VM using SSH:
   
   ```bash
   gcloud compute ssh llm-processor --zone=us-east4-c --project=${PROJECT_ID}
   ```


1.  **Create Storage Bucket** Create a regional bucket in the same project. Make sure you choose to make
    it a regional bucket and choose the same region as where your pipeline will
    run. us-central1 recommended.

    ```bash
        export PROJECT_ID=<your project ID>
        export BUCKET_NAME=<your choice of a globally unique bucket ID>
        gcloud storage buckets create gs://$BUCKET_NAME --project=$PROJECT_ID --location=us-central1 --uniform-bucket-level-access
    ```


### Megatron-Deepspeed GPT Pretraining Instructions

    
The following instructions show how to train a GPT model using Megatron-Deepspeed 
on a 96 H100 GPU cluster. The intruction are simplified to use Goolge Cloud Compute Engine APIs
 only. In addition they use Google Cloud Strorage for storing the data:

* **Task**: LLM Pretraining
* **Implementation**: Megatron-Deepspped
* **Distributed Framework**: Deepspeed (ZeRO stage 3)
* **Infrastructure**: Google Cloud
* **Cluster Management**: AI Infra cluster provisioning tool
* **Storage**: Google Cloud Storage
* **Dataset**: Wikipedia
* **Model Architecture**: GPT 3
* **Model Size**: 176B


1.  **Download and Preprocess** Wikipedia dataset : 

    ```bash
        docker run gcr.io/llm-containers/gpt_preprocess:release ./preprocess.sh gs://$BUCKET_NAME
    ```    

    Warning: This could take hours to finish running.

2.  **Pretrain GPT 176B on an A3 VM cluster**

    ```bash
        sudo docker run -it gcr.io/llm-containers/batch:release $PROJECT_ID gcr.io/llm-containers/gpt_train:release gs://$BUCKET_NAME 0 0 0 ' {"data_file_name":"wiki_data_text_document", "tensor_parallel":4, "pipeline_parallel":12, "nlayers":70, "hidden":14336, "heads":112, "seq_len":2048, "train_steps":100, "eval_steps":10, "micro_batch":1, "gradient_acc_steps":128 }' '{ "name_prefix": "megatron-gpt", "zone": "us-central1-a", "node_count": 12, "machine_type": "a3-highgpu-8g", "gpu_type": "nvidia-h100-80gb", "gpu_count": 8 }'
    ```

### Huggingface T5 Finetuning Instructions
    
In this effort, we provide a fully functioning example of how can we use GCP
tools as well as the HuggingFace transformer library in conjunction with
deepseed to finetune a large language model (T5 XXL) for a text summarization
task. We encapsulate the code in container images that you can easily run on GCP.
Here is a summary of task and tooling we are going to use:


* **Task**: Text Summarization
* **Implementation**: Hugging Face Transformers library
* **Distributed Framework**: Deepspeed (ZeRO stage 3)
* **Infrastructure**: Google Cloud
* **Cluster Management**: AI Infra cluster provisioning tool
* **Storage**: Google Cloud Storage
* **Dataset**: CNN Dailymail
* **Model Architecture**: T5
* **Model Size**: 11B

1.  **Copy Data** Copy data and model checkpoint to GCS bucket.
    
    1. If you want to finetune the XXL T5 model (11B parameters), this can take up to 30 minutes to copy, you can use the following command:
    
        ```bash
            sudo docker run -it gcr.io/llm-containers/train:release python download.py --dataset=cnn_dailymail --subset=3.0.0 --dataset_path=gs://$BUCKET_NAME/dataset --model_checkpoint=google/t5-v1_1-xxl --workspace_path=gs://$BUCKET_NAME/workspace
        ```
    1. If you want to quickly test finetuning a small T5 model (50M parameters), you can use the following command:
    
        ```bash
            sudo docker run -it gcr.io/llm-containers/train:release python download.py --dataset=cnn_dailymail --subset=3.0.0 --dataset_path=gs://$BUCKET_NAME/dataset --model_checkpoint=t5-small --workspace_path=gs://$BUCKET_NAME/workspace
        ```

1.  **Preproces Data** Kick off data preprocessing:

    ```bash
    sudo docker run -it gcr.io/llm-containers/train:release python preprocess.py --model_checkpoint google/t5-v1_1-xxl --document_column article --summary_column highlights --dataset_path gs://$BUCKET_NAME/dataset --tokenized_dataset_path gs://$BUCKET_NAME/processed_dataset
    ```

1.  **Kick off training**

    1. To run a T5 XXL on 8 A3 VMs with H100 GPUs
    
        ```bash
        sudo docker run -it gcr.io/llm-containers/batch:release $PROJECT_ID gcr.io/llm-containers/train:release gs://$BUCKET_NAME/model 0 gs://$BUCKET_NAME/processed_dataset gs://$BUCKET_NAME/workspace '{ "model_checkpoint" : "google/t5-v1_1-xxl",  "batch_size" : 16,  "epochs" : 7}' ' { "name_prefix" : "t5node", "zone" : "us-east4-a",  "node_count" : 8,  "machine_type" : "a3-highgpu-8g", "gpu_type" : "nvidia-h100-80gb", "gpu_count" : 8 }'
        ```

    1. To run a T5 small on a single A100 GPU:

        ```bash
        sudo docker run -it gcr.io/llm-containers/batch:release $PROJECT_ID gcr.io/llm-containers/train:release gs://$BUCKET_NAME/model 0 gs://$BUCKET_NAME/processed_dataset gs://$BUCKET_NAME/workspace '{ "model_checkpoint" : "t5-small",  "batch_size" : 128,  "epochs" : 1}' ' { "name_prefix" : "t5node", "zone" : "us-east4-a",  "node_count" : 1,  "machine_type" : "a3-highgpu-8g", "gpu_type" : "nvidia-h100-80gb", "gpu_count" : 8 }'
        ```

    Make sure you have enough Quota for the VM and GPU types you
    select. You can learn more about Google Cloud quota from
    [here](https://cloud.google.com/compute/resource-usage#gpu_quota)

### Test your pipeline

1. Run one of the following command to deploy your model to Vertex AI:

    1. For the T5 small:
        ```bash
        sudo docker run -it gcr.io/llm-containers/deploy:release python deploy.py --project=${PROJECT_ID} --model_display_name=t5 --serving_container_image_uri=gcr.io/llm-containers/predict:release --model_path=gs://${BUCKET_NAME}/model --machine_type=n1-standard-32 --gpu_type=NVIDIA_TESLA_V100 --gpu_count=1 --region=us-central1
        ```

    1. For the T5 XXL:
        ```bash
        sudo docker run -it gcr.io/llm-containers/deploy:release python deploy.py --project=${PROJECT_ID} --model_display_name=t5 --serving_container_image_uri=gcr.io/llm-containers/predict:release --model_path=gs://${BUCKET_NAME}/model --machine_type=n1-standard-32 --gpu_type=NVIDIA_TESLA_V100 --gpu_count=1 --region=us-central1
        ```

    When it finishes deployment, it will output the following line:
    ```
    Endpoint model deployed. Resource name: projects/<project numbre>/locations/us-central1/endpoints/<endpoint id>
    ```

1. Create a json file with the following content or any article of your choice:

    ```json
    {
    "instances": [
        "Sandwiched between a second-hand bookstore and record shop in Cape Town's charmingly grungy suburb of Observatory is a blackboard reading 'Tapi Tapi -- Handcrafted, authentic African ice cream.' The parlor has become one of Cape Town's most talked about food establishments since opening in October 2020. And in its tiny kitchen, Jeff is creating ice cream flavors like no one else. Handwritten in black marker on the shiny kitchen counter are today's options: Salty kapenta dried fish (blitzed), toffee and scotch bonnet chile Sun-dried blackjack greens and caramel, Malted millet ,Hibiscus, cloves and anise. Using only flavors indigenous to the African continent, Guzha's ice cream has become the tool through which he is reframing the narrative around African food. 'This (is) ice cream for my identity, for other people's sake,' Jeff tells CNN. 'I think the (global) food story doesn't have much space for Africa ... unless we're looking at the generic idea of African food,' he adds. 'I'm not trying to appeal to the global universe -- I'm trying to help Black identities enjoy their culture on a more regular basis.'"
    ]
    }
    ```
    Save the file and give it a name. For this example, prediction.json

1. Do a prediction using the following command:

    ```bash
    curl \
        -X POST \
        -H "Authorization: Bearer $(gcloud auth print-access-token)" \
        -H "Content-Type: application/json" \
        https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/endpoints/432544312:predict \
        -d "@prediction.json"
    ```

### Expected Output

1. If you used a configurtion with the T5 small model (60M parameters), the output would be like:
```bash
{
  "predictions": [
    "'Tapi Tapi -- Handcrafted, authentic African ice cream' is a",
  ],
  "deployedModelId": "8744807401842016256",
  "model": "projects/649215667094/locations/us-central1/models/6720808805245911040",
  "modelDisplayName": "t5",
  "modelVersionId": "12"
}
```

2. If you use a configurtion with the T5 XXL (11B parameters), the output would be like:
```bash
{
  "predictions": [
    "Tapi Tapi is an ice cream parlor in Cape Town, South Africa.",
  ],
  "deployedModelId": "8744807401842016256",
  "model": "projects/649215667094/locations/us-central1/models/6720808805245911040",
  "modelDisplayName": "t5",
  "modelVersionId": "12"
}
```

## How it works

### Download

This is the ingestion step for the data. Currently, it uses huggingface.co
datasets library. To learn more about loading datasets using this library, check
out the library [reference](https://huggingface.co/docs/datasets/loading).
Eventually, the data is downloaded to Google Cloud Storage (GCS). The next steps
in the pipeline are expecting the data to be in GCS. This works well for
datasets in the multi GB order of magnitude. In our future work, we will present
how to process larger datasets using DataFlow. The full training scripts can be
found [here](src/download.py).

We package this as a pipeline component that produces the dataset on GCP. The
component takes the dataset and subset as input. These correspond to the ‘path’
and ‘name’ parameters passed directly to datasets.load_dataset. You can learn
more about loading datasets here:

![Download component](img/download.png)
![Download Parameters](img/download_params.png)

In the example, we use the
[CNN Dailymail dataset](https://huggingface.co/datasets/cnn_dailymail). The code
is packaged in a container available [here](https://gcr.io/llm-containers/train).

## Preprocessing

As part of the summarization task, we tokenize our dataset during the
preprocessing stage. The full script for tokenization can be found [here]. When
we are finished processing, we upload the tokenized dataset to the output GCS
path.

This component allows the use of dataset formats supported by
[datasets.load_dataset](https://huggingface.co/docs/datasets/v1.11.0/loading_datasets.html).
All the user needs to do is specify which column in the dataset contains the
document body and which column contains the document summary.

![Preprocess component](img/preprocess.png)
![Preprocess Parameters](img/preprocess_params.png)

### Fine Tuning

In this step of the pipeline we take the tokenized dataset and train a base
model to be finetuned on the dataset. For large language models, this is
typically the step that consumes most resources and takes a significant amount
of time. Depending on how much GPUs are used for training and how many epochs
you run the training for, this could vary from hours to days.

![Fine tuning component](img/train.png)
![Fine tuning Parameters](img/train_params.png)

The general workflow for finetuning is that we spawn a cluster of GPU VMs on
GCP. In the example shown we use 8 A2 plus matches with 8 A100 GPUs. We
preprovision them with [DLVM](https://cloud.google.com/deep-learning-vm) images
that include the necessary GPU drivers including NVidia Common Communication
Library ([NCCL](https://developer.nvidia.com/nccl)). We download our training
code which is packaged in a docker image. And use
[deepspeed](https://www.deepspeed.ai/) to launch and coordinate our training
across all the VMS. We use [fluentd](https://www.fluentd.org/) to collect logs
from the VMs and upload to
[Google Cloud Logging](https://cloud.google.com/logging). We save model
checkpoints to GCS including the final trained model.

![Training architecture](img/architecture.png)

Let’s look at the implementation of some of these parts:

#### Cluster Provisioning

To provision the cluster of VMs, we use a pre created container we call ‘batch
container’ . The container is available [here](https://gcr.io/llm-containers/batch). It
is based on the cluster provisioning tool container available
[here](https://gcr.io/llm-containers/cluster-provision-image). Creating VMs using the
tool is as simple as filling a few environment variables and calling an entry
point. It will automatically create the cluster with an image of choice and run
a starting command on all VMs.

When the job completes successfully, we call the entry point again requesting
the destruction of the cluster.

#### Training Container

All the VMs will be provisioned with the DLVM image with NVidia drivers
preinstalled. The VMs will download the training docker image and invoke the
training start command. The training container is available
[here](https://gcr.io/llm-containers/train) . And you can find the fine tuning scripts
[here](src/finetune.py).



The container image has some pre-installed packages and configuration. This
includes:

*   Transformer libraries and deepspeed with compiled ops
*   ssh server that allows deepspeed launcher to launch training inside the
    container
*   Google Cloud logging agent to collect logs from the container and publish to
    the cloud.
*   Training scripts

The head node will invoke the head script which:

*   Configures ssh to talk to all servers in the cluster
*   Configures fluentd to collect logs
*   Invokes deepspeed with correct config to run the training script using ZeRO
    stage 3

In turn, deepspeed would use its default ssh launcher to launch the training
scripts on all cluster containers.

After the training container provisions the cluster, it will start downloading 
the model weights and dataset and will kick off training after. It will show 
the following message:

```Waiting for training to start...```

This could take up to 20 minutes depending on the size of your model and data.

#### Saving to GCS

The training scripts use the Huggingface transformer library to finetune a
summarization task on the given dataset. To save the progress of training as it
goes, we create a training callback that uploads each checkpoint to GCS. The
call back looks like this:

```python
class GCSSaveCallback(TrainerCallback):
 """A [`TrainerCallback`] that handles checkpoints.
 """

 def on_save(self, args: TrainingArguments, state: TrainerState,
             control: TrainerControl, **kwargs):
   checkpoint_folder = f'checkpoint-{state.global_step}'
   local_output_dir = os.path.join(args.output_dir, checkpoint_folder)
   if not os.path.exists(local_output_dir):
     logging.error(
          'Check point called for a non existing checkpoint %s',
          local_output_dir,
     )
     return
   gcs_root, gcs = utils.gcs_path(FLAGS.gcs_output)
   gcs_output_dir = os.path.join(gcs_root, local_output_dir)
   logging.info('Uploading %s....', gcs_output_dir)
   gcs.put(local_output_dir, gcs_output_dir, recursive=True)
   return None
```

When the model is completely trained, we save the final copy to GCS.

#### Evaluation and metrics

When the model finishes training, we run our evaluation to collect the model
accuracy number. In this case, since it is a summarization task, we collect
ROUGE scores for the model. You can learn more about ROUGE scores
[here](https://towardsdatascience.com/the-ultimate-performance-metric-in-nlp-111df6c64460).
Once we have the final scores, we save them along the model to GCS. By saving
the model metrics along with the model, we are able to figure out the model
performance just by looking at the saved model without having to run evaluation
again as in the example below:

![Metrics](img/metrics.png)

### Model deployment

Now that our model is ready, we want to provide it as a service for authorized
users to call. This can serve as a backend to a frontend web application. We go
with a simple and quick solution to demo our model. 
For this purpose, we use
[Vertex AI Prediction](https://cloud.google.com/vertex-ai/docs/predictions/get-predictions)
which provides us with the quickest path to serve our model.


![Deployment component](img/deploy.png)
![Deployment Parameters](img/deploy_params.png)

We package the model serving code in a simple prediction container which is
available [here](https://gcr.io/llm-containers/predict). The container packs a Flask
server with a simple python script that uses deepspeed to perform prediction
from the model. It implements the necessary REST APIs required by Vertex AI
Prediction.


For larger models or
production class deployment, we can deploy directly to GKE and use
[Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server). See an [example with T5](examples/inferencing/running-on-gke.md).

## Current Pipeline Supported configurations

### Data

The pipeline only supports data that is loadable using
[load_dataset](https://huggingface.co/docs/datasets/v1.11.0/loading_datasets.htmlhttps://huggingface.co/docs/datasets/v1.11.0/loading_datasets.html).
It works well for data less than 100GB. For larger dataset, users will need to
write custom processing scripts on Dataflow.

### Compute

The pipeline VM types supported by GCP Compute Engine. For a full list, check
[GCE GPU Platforms](https://cloud.google.com/compute/docs/gpus). You can create
a cluster of any size as long as you have the quota for it.

### Models

For the summarization task, we support any sequence to sequence model in the
huggingface.co model repository. This includes T5, mT5, BART and Marian models.
For more information about sequence 2 sequence models, check
[here](https://huggingface.co/course/chapter1/7). The model can be of any size
as long as corresponding parameters (batch size, number of nodes, number of
GPUs, etc ..) can make the model fit into the compute cluster.

### Deployment Scale

Currently we support SKUs that are available for Vertex AI Prediction which can
be found
[here](https://cloud.google.com/vertex-ai/docs/predictions/configure-compute#machine-types).

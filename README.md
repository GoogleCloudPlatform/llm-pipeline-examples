# Running T5 using deepspeed as a Vertex AI pipeline

This project implements a T5 summarization fine tuning pipeline as a Kube Flow
Pipeline that is executable on Vertex AI Pipelines. It uses the CNN Daily Mail
dataset to finetuen T5 of various sizes and publishes it as a prediction
endpoint on Google Cloud. Follow the following instructions to run the pipeline
in your GCP project:

# Prerequisites

1. Make sure you have gcloud installed and that you are authenticated including application default authentication

    ```bash
    gcloud auth login
    gcloud auth application-default login
    ```

1. Install kfp and abseil packages

    ```bash
    pip install kfp absl-py google-cloud-aiplatform
    ```
# Instructions
Follow these instructions To run T5 training on a GPU cluster:

1.  In your project, enable services needed to run the pipeline. You can do this by issuing the following command:

    ```bash
        export PROJECT_ID=<your project ID>
        gcloud services enable aiplatform.googleapis.com cloudfunctions compute.googleapis.com iam.googleapis.com --project=${PROJECT_ID}
    ```

1.  Create a regional bucket in the same project. Make sure you choose to make it a regional bucket and choose the same region as where your pipeline will run. us-central1 recommended.

    ```bash
        export BUCKET_NAME=<your choice of a globally unique bucket ID>
        gcloud alpha storage buckets create gs://$BUCKET_NAME --project=$PROJECT_ID --location=us-central1 --uniform-bucket-level-access
    ```

1.  Clone this repo from the repository

    ```bash
    git clone <repostitory name goes here> 
    cd <destination folder>
    ```

1.  Run the following command:

    ```bash
    python3 pipeline.py --project=$PROJECT_ID --pipeline_root=gs://$BUCKET_NAME/pipeline_runs/ --config=configs/<config>
    ```

    Replace **\<config>** with one of the precreated configs below or create your own config:

    *  **small1vm1gpu.json** To create a single VM cluster with 1 A100 GPU and finetune T5 small on it.

    *  **small2vm16gpu.json** To create a 2 VM cluster with 16 A100 GPU each and finetune T5 small on it.

    *  **xxl2vm16gpu.json** To create a 2 VM cluster with 16 A100 GPU each and finetune T5 XXL on it. Caution: takes multiple days

    *  **xxl8vm16gpu.json** To create a 2 VM cluster with 16 A100 GPU each and finetune T5 XXL on it. Caution: takes multiple days

    Make sure you have enough Quota for the number of A2 VMs and A100 GPUs you select. An newly created project under google.com will typically have default quota to run **small1vm1gpu.json**.

    The tool displays a link to the pipeline after it finishes. Go to the link to watch the pipeline progress.

# Test your pipeline

1. After your pipeline completes successfully, expand the 'condition' node and click on the 'deploy' node inside. Under 'Output Parameters', copy the Value of 'endpoint'. Create an environment variable with the value:

    ```bash
    export ENDPOINT_ID="<The value you copied>"
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
        https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:predict \
        -d "@prediction.json"
    ```




#!/bin/bash -e
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
EXIT_CODE=0
VERIFY_PAYLOAD=0
echo "Called with options:"
echo $*

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Use this script to deploy an LLM to a GKE cluster."
      echo "Detailed documentation can be found at https://github.com/GoogleCloudPlatform/llm-pipeline-examples/pirillo/gke_samples/examples/running-on-gke.md"
      echo 
      echo "Options"
      echo "-h|--help) Display this menu."
      echo "-v|--verify) Setting this flag will use the -i and -o flags to validate the expected inferencing behavior of the deployed."
      echo "-i|--verify-input-payload=) Path to a file containing the inferencing input for verification. This will route to the Flask endpoint on the image."
      echo "-o|--verify-output-payload=) Path to a file containing the inferencing output for verification."
      echo "-p|--project=) ID of the project to use. Defaults to environment variable \$PROJECT_ID"
      echo "--name-prefix=) Name prefix for the cluster to create. Cluster will be named <name-prefix>-gke Defaults to environment variable \$NAME_PREFIX"
      echo "--converted-model-upload-path=) Only required when USE_FASTER_TRANSFORMER is set. A GCS path to upload the model after it is converted for FasterTransformer"
      echo "--cleanup) Deletes the model and cluster at the end of the run. Used for testing."
      exit 0
      ;;
    -v|--verify)
      shift
      VERIFY_PAYLOAD=1
      ;;
    -p)
      shift
      if test $# -gt 0; then
        export PROJECT_ID=$1
        shift
      else
        echo "No project id provided."
        exit 1
      fi
      ;;
    --project*)
      export PROJECT_ID=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    --name-prefix*)
      export NAME_PREFIX=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    --converted-model-upload-path*)
      export CONVERTED_MODEL_UPLOAD_PATH=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;; 
    -i)
      shift
      if test $# -gt 0; then
        export VERIFY_INPUT_PATH=$1
        shift
      else
        echo "No input payload provided."
        exit 1
      fi
      ;;
    --verify-input-payload*)
      export VERIFY_INPUT_PATH=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    -o)
      shift
      if test $# -gt 0; then
        export VERIFY_OUTPUT_PATH=$1
        shift
      else
        echo "No output payload provided."
        exit 1
      fi
      ;;
    --verify-output-payload*)
      export VERIFY_OUTPUT_PATH=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    --cleanup)
      export CLEANUP=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

_invoke_cluster_tool () {
  echo "Invoking cluster tool"
  echo PROJECT_ID $PROJECT_ID
  echo NAME_PREFIX $NAME_PREFIX
  echo ZONE $ZONE
  echo INSTANCE_COUNT $INSTANCE_COUNT
  echo GPU_COUNT $GPU_COUNT
  echo VM_TYPE $VM_TYPE
  echo ACCELERATOR_TYPE $ACCELERATOR_TYPE
  echo IMAGE_FAMILY_NAME $IMAGE_FAMILY_NAME
  echo IMAGE_NAME $IMAGE_NAME
  echo DISK_SIZE_GB $DISK_SIZE_GB
  echo DISK_TYPE $DISK_TYPE
  echo TERRAFORM_GCS_PATH $TERRAFORM_GCS_PATH
  echo VM_LOCALFILE_DEST_PATH $VM_LOCALFILE_DEST_PATH
  echo METADATA $METADATA
  echo LABELS $LABELS
  echo STARTUP_COMMAND $STARTUP_COMMAND
  echo ORCHESTRATOR_TYPE $ORCHESTRATOR_TYPE
  echo GCS_MOUNT_LIST $GCS_MOUNT_LIST
  echo NFS_FILESHARE_LIST $NFS_FILESHARE_LIST
  echo SHOW_PROXY_URL $SHOW_PROXY_URL
  echo MINIMIZE_TERRAFORM_LOGGING $MINIMIZE_TERRAFORM_LOGGING
  echo NETWORK_CONFIG $NETWORK_CONFIG
  echo ACTION $ACTION
  echo GKE_NODE_POOL_COUNT $GKE_NODE_POOL_COUNT
  echo GKE_NODE_COUNT_PER_NODE_POOL $GKE_NODE_COUNT_PER_NODE_POOL
  /usr/entrypoint.sh
}

if [[ -z $PROJECT_ID ]]; then
  echo "PROJECT_ID variable is not set."
  exit 1
fi
gcloud config set project $PROJECT_ID

if [[ -z $REGION ]]; then
  export REGION=${ZONE%-*}
fi
if [[ -z $INFERENCING_IMAGE_TAG ]]; then
  export INFERENCING_IMAGE_TAG=release
fi
if [[ -z $POD_MEMORY_LIMIT ]]; then
  export POD_MEMORY_LIMIT="16Gi"
fi
if [[ -z $KSA_NAME ]]; then
  export KSA_NAME="aiinfra-gke-sa"
fi
if [[ $USE_FASTER_TRANSFORMER -eq 1 ]]; then
  INFERENCE_IMAGE=gcr.io/llm-containers/predict-triton
  CONVERT_MODEL=1
else
  INFERENCE_IMAGE=gcr.io/llm-containers/predict
fi
if [[ -z $INFERENCING_IMAGE_URI ]]; then
  export INFERENCING_IMAGE_URI=$INFERENCE_IMAGE
fi
if [[ -z $GKE_VERSION ]]; then
  export GKE_VERSION="1.23"
fi

if [[ -z $EXISTING_CLUSTER_ID ]]; then

  export SERVICE_ACCOUNT=$(gcloud config get account)
  export OS_LOGIN_USER=$(gcloud iam service-accounts describe ${SERVICE_ACCOUNT} | grep uniqueId | sed -e "s/.* '\(.*\)'/sa_\1/")

  echo User is ${OS_LOGIN_USER}

  export ACTION=CREATE
  export METADATA="{install-unattended-upgrades=\"false\",enable-oslogin=\"TRUE\",jupyter-user=\"${OS_LOGIN_USER}\",install-nvidia-driver=\"True\"}"
  export SHOW_PROXY_URL=no
  export LABELS="{gcpllm=\"$CLUSTER_PREFIX\"}"
  export MINIMIZE_TERRAFORM_LOGGING=true
  _invoke_cluster_tool

  echo "Provisioning cluster..."
  export EXISTING_CLUSTER_ID=${NAME_PREFIX}-gke
  gcloud container clusters get-credentials $EXISTING_CLUSTER_ID --region $REGION --project $PROJECT_ID
else
  gcloud container clusters get-credentials $EXISTING_CLUSTER_ID --region $REGION --project $PROJECT_ID
  export PROJECT_NUMBER=$(gcloud projects list \
    --filter="${PROJECT_ID}" \
    --format="value(PROJECT_NUMBER)")
  echo "Using existing cluster $EXISTING_CLUSTER_ID"
  echo "Deploying Nvidia driver daemonset"
  kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
  echo "Creating service account binding"
  envsubst < specs/serviceAccount.yml | kubectl apply -f -
fi

if [[ $CONVERT_MODEL -eq 1 ]]; then
  if [[ -z $MODEL_SOURCE_PATH ]]; then
    echo "MODEL_SOURCE_PATH env var is not set."
    exit 1
  fi
  if [[ -z $CONVERTED_MODEL_UPLOAD_PATH ]]; then
    echo "CONVERTED_MODEL_UPLOAD_PATH env var is not set."
    exit 1
  fi
  if [[ -z $GPU_NUMBER ]]; then
    echo "GPU_NUMBER env var is not set."
    exit 1
  fi

  # Run convert image on cluster
  export CONVERT_JOB_ID=convert-$RANDOM
  envsubst < specs/convert.yml | kubectl apply -f -
  echo "Running 'convert' job on cluster."
  kubectl wait --for=condition=Complete --timeout=60m job/$CONVERT_JOB_ID

  export MODEL_SOURCE_PATH=$CONVERTED_MODEL_UPLOAD_PATH
fi

# Run predict image on cluster
echo "Deploying predict image to cluster"
export DEPLOYMENT_NAME=$MODEL_NAME-deployment
envsubst < specs/inference.yml | kubectl apply -f -
kubectl rollout status deployment/$DEPLOYMENT_NAME -n default

# Print urls to access the model
echo "Exposed Node IPs from node 0 on the cluster:"
ENDPOINTS=$(kubectl get nodes -o json | jq -r '.items[0].status.addresses[]')
echo $ENDPOINTS | jq -r '.'
INTERNAL_ENDPOINT=$(kubectl get nodes -o json | jq -r '.items[0].status.addresses[] | select(.type=="InternalIP") | .address | select(startswith("10."))')

echo "NodePort for Flask:"
FLASK_NODEPORT=$(kubectl get svc -o json | jq -r --arg NAME "$MODEL_NAME" '.items[].spec | select(.selector.app==$NAME) | .ports[] | select(.name=="flask")')
echo $FLASK_NODEPORT | jq -r '.'

FLASK_PORT=$(echo $FLASK_NODEPORT | jq -r '.nodePort')

echo "NodePort for Triton:"
TRITON_NODEPORT=$(kubectl get svc -o json | jq -r --arg NAME "$MODEL_NAME" '.items[].spec | select(.selector.app==$NAME) | .ports[] | select(.name=="triton")')
echo $TRITON_NODEPORT | jq -r '.'
TRITON_PORT=$(echo $TRITON_NODEPORT | jq -r '.nodePort')

printf "From a machine on the same VPC as this cluster you can call http://${INTERNAL_ENDPOINT}:${FLASK_PORT}/infer"

NOTEBOOK_URL=https://raw.githubusercontent.com/GoogleCloudPlatform/llm-pipeline-examples/main/examples/t5-gke-sample-notebook.ipynb
ENCODED_NOTEBOOK_URL=$(jq -rn --arg x $NOTEBOOK_URL  '$x|@uri')
DEPLOY_URL="https://console.cloud.google.com/vertex-ai/workbench/user-managed/deploy?download_url=${ENCODED_NOTEBOOK_URL}&project=${PROJECT_ID}"
SAMPLE_PAYLOAD=$(cat predict_payload.json | jq -c .)

printf "\n***********\n"
printf "To deploy a sample notebook for experimenting with this deployed model, paste the following link into your browser:\n"
printf "%s\n" "$DEPLOY_URL"
printf "Set the following parameters in the variables cell of the notebook:\n"
printf "host             = '$INTERNAL_ENDPOINT'\n"
printf "flask_node_port  = '$FLASK_PORT'\n"
printf "triton_node_port = '$TRITON_PORT'\n"
printf 'payload = """%s"""\n' "${SAMPLE_PAYLOAD}"
printf "\n***********\n"

if [[ $VERIFY_PAYLOAD -eq 1 ]]; then
  echo "Running inference validation using input: $VERIFY_INPUT_PATH and output: $PREDICT_OUTPUT"
  PREDICT_ENDPOINT="http://$INTERNAL_ENDPOINT:$FLASK_PORT/infer"
  echo "Calling predict endpoint: $PREDICT_ENDPOINT"

  echo Sending input: $(cat $VERIFY_INPUT_PATH)
  PREDICT_OUTPUT=$(curl \
    -X POST $PREDICT_ENDPOINT \
    --header 'Content-Type: application/json' \
    -d @$VERIFY_INPUT_PATH || :)

  echo Received output: $PREDICT_OUTPUT
  echo $PREDICT_OUTPUT | jq -rc > output.json

  diff <(jq -S . $VERIFY_OUTPUT_PATH) <(jq -S . output.json) > diff.txt || :
  if [[ $(wc -c diff.txt | awk '{print $1}') != 0 ]]; then
    echo "Predicted output does not match expected output."
    cat diff.txt
    EXIT_CODE=1
  else
    echo "Predicted output matches expected output."
    EXIT_CODE=0
  fi
fi

if [[ $CLEANUP -eq 1 ]]; then
  echo "Running clean-up."
  echo "Deleting uploaded model from path $CONVERTED_MODEL_UPLOAD_PATH"
  gsutil -m rm -r $CONVERTED_MODEL_UPLOAD_PATH || :
  echo "Deleting provisioned cluster $EXISTING_CLUSTER_ID"
  gcloud container clusters delete $EXISTING_CLUSTER_ID --region $REGION --quiet || :
fi

# Let logs flush before exit
sleep 5
exit $EXIT_CODE

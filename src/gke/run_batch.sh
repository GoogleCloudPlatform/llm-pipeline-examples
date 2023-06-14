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
  echo $1
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
    *)
      break
      ;;
  esac
done

echo "FLAGS SET:"
echo $PROJECT_ID
echo $VERIFY_PAYLOAD
echo $VERIFY_INPUT_PATH
echo $VERIFY_OUTPUT_PATH

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
fi

# Get kubeconfig for cluster
gcloud container clusters get-credentials $EXISTING_CLUSTER_ID --region $REGION --project $PROJECT_ID
if [[ $CONVERT_MODEL -eq 1 ]]; then
  # Run convert image on cluster
  export CONVERT_JOB_ID=convert-$RANDOM
  envsubst < specs/convert.yml | kubectl apply -f -
  echo "Running 'convert' job on cluster."
  CONVERT_POD_ID=$(kubectl get pods -l job-name=$CONVERT_JOB_ID -o=json | jq -r '.items[0].metadata.name')
  kubectl wait --for=condition=Ready --timeout=10m pod/$CONVERT_POD_ID
  kubectl logs $CONVERT_POD_ID -f
  kubectl wait --for=condition=Complete --timeout=60m job/$CONVERT_JOB_ID

  export MODEL_SOURCE_PATH=$CONVERTED_MODEL_UPLOAD_PATH
fi

# Run predict image on cluster
echo "Deploying predict image to cluster"
envsubst < specs/inference.yml | kubectl apply -f -
kubectl wait --for=condition=Ready --timeout=60m pod -l app=$MODEL_NAME

# Print urls to access the model
echo Exposed Node IPs from node 0 on the cluster:
ENDPOINTS=$(kubectl get nodes -o json | jq -r '.items[0].status.addresses[]')
echo $ENDPOINTS | jq -r '.'
INTERNAL_ENDPOINT=$(kubectl get nodes -o json | jq -r '.items[0].status.addresses[] | select(.type=="InternalIP") | .address | select(startswith("10."))')

echo NodePort for Flask:
FLASK_NODEPORT=$(kubectl get svc -o json | jq -r --arg NAME "$MODEL_NAME" '.items[].spec | select(.selector.app==$NAME) | .ports[] | select(.name=="flask")')
echo $FLASK_NODEPORT | jq -r '.'

FLASK_PORT=$(echo $FLASK_NODEPORT | jq -r '.nodePort')

echo NodePort for Triton:
TRITON_NODEPORT=$(kubectl get svc -o json | jq -r --arg NAME "$MODEL_NAME" '.items[].spec | select(.selector.app==$NAME) | .ports[] | select(.name=="triton")')
echo $TRITON_NODEPORT | jq -r '.'
TRITON_PORT=$(echo $TRITON_NODEPORT | jq -r '.nodePort')

printf "From a machine on the same VPC as this cluster you can call http://${INTERNAL_ENDPOINT}:${FLASK_PORT}/infer"

NOTEBOOK_URL=https://raw.githubusercontent.com/GoogleCloudPlatform/llm-pipeline-examples/main/examples/t5-gke-sample-notebook.ipynb
ENCODED_NOTEBOOK_URL=$(jq -rn --arg x $NOTEBOOK_URL  '$x|@uri')
DEPLOY_URL="https://pantheon.corp.google.com/vertex-ai/workbench/user-managed/deploy?download_url=${ENCODED_NOTEBOOK_URL}&project=${PROJECT_ID}"

printf "\n***********\n"
printf "To deploy a sample notebook for experimenting with this deployed model, paste the following link into your browser:\n"
printf "%s\n" "$DEPLOY_URL"
printf "Set the following parameters in the variables cell of the notebook:\n"
printf "host             = '$INTERNAL_ENDPOINT'\n"
printf "flask_node_port  = '$FLASK_PORT'\n"
printf "triton_node_port = '$TRITON_PORT'\n"
printf "payload = '<your_payload_goes_here>'\n"
printf "\n***********\n"

if [[ $VERIFY_PAYLOAD -eq 1]]
  PREDICT_ENDPOINT="http://$INTERNAL_ENDPOINT:$FLASK_PORT/infer"
  PREDICT_OUTPUT=$(curl \
    -X POST $PREDICT_ENDPOINT \
    --header 'Content-Type: application/json' \
    --data $(cat $VERIFY_INPUT_PATH) \
    | jq -rc )

  EXPECTED_OUTPUT=$(cat $VERIFY_OUTPUT_PATH | jq -rc)
  if [[ $PREDICT_OUTPUT -ne $EXPECTED_OUTPUT]]
    echo "Predicted output does not match expected output."
    echo "Predict output:"
    echo $PREDICT_OUTPUT
    echo
    echo "Expected output:"
    echo $EXPECTED_OUTPUT
    exit 1
  fi
fi

exit $EXIT_CODE

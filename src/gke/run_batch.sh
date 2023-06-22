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

_env_to_tf_var_gke () {
    echo "project_id = \"${PROJECT_ID:?}\""
    echo "resource_prefix = \"${NAME_PREFIX:?}\""
    echo "region = \"${REGION:- ${ZONE%-*}}\""

    {
        GKE_NODE_POOL_COUNT=${GKE_NODE_POOL_COUNT:- 1}
        GKE_NODE_COUNT_PER_NODE_POOL=${GKE_NODE_COUNT_PER_NODE_POOL:- 1}
        GKE_ENABLE_COMPACT_PLACEMENT=${GKE_ENABLE_COMPACT_PLACEMENT:- true}

        if [ -n "${GPU_COUNT}" ] && [ -n "${ACCELERATOR_TYPE}" ]; then
            GUEST_ACCELERATOR="{ count = ${GPU_COUNT}, type = \"${ACCELERATOR_TYPE}\" }"
        fi
        GUEST_ACCELERATOR="${GUEST_ACCELERATOR:- null}"

        NODE_POOLS=""
        while [ 0 -lt "${GKE_NODE_POOL_COUNT}" ]; do
            NODE_POOLS="${NODE_POOLS}{ \
                zone = \"${ZONE:?}\", \
                node_count = ${GKE_NODE_COUNT_PER_NODE_POOL:?}, \
                machine_type = \"${VM_TYPE:?}\", \
                guest_accelerator = ${GUEST_ACCELERATOR:?}, \
                enable_compact_placement = ${GKE_ENABLE_COMPACT_PLACEMENT:?}, \
            },"
            ((--GKE_NODE_POOL_COUNT))
        done

        echo "node_pools = [${NODE_POOLS}]"
    }
    return 0
}

_invoke_cluster_tool () {
  echo "Invoking cluster tool"
  echo '```tfvars'
  cat "${TFVARS_FILE}"
  echo '```'
  /root/aiinfra/scripts/entrypoint.sh \
      ${MINIMIZE_TERRAFORM_LOGGING:+ --quiet} \
      ${TERRAFORM_GCS_PATH:+ --backend-bucket ${TERRAFORM_GCS_PATH}} \
      ${ACTION,,} \
      gke \
      "${TFVARS_FILE}"
}

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
  export GKE_VERSION="1.26.3-gke.1000"
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
  export TFVARS_FILE=$(mktemp)
  _env_to_tf_var_gke >"${TFVARS_FILE}"
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

exit $EXIT_CODE

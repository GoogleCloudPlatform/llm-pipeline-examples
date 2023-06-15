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

_env_to_tf_var_mig () {
    TF_VAR_project_id="${PROJECT_ID:?}"
    TF_VAR_resource_prefix="${NAME_PREFIX:?}"
    TF_VAR_target_size="${INSTANCE_COUNT:?}"
    TF_VAR_zone="${ZONE:?}"

    if [ -n "${DISK_SIZE_GB}" ]; then TF_VAR_disk_size_gb="${DISK_SIZE_GB}"; fi
    if [ -n "${DISK_TYPE}" ]; then TF_VAR_disk_type="${DISK_TYPE}"; fi
    if [ -n "${GPU_COUNT}" ] && [ -n "${ACCELERATOR_TYPE}" ]; then
        TF_VAR_guest_accelerator="{count=${GPU_COUNT},type=\"${ACCELERATOR_TYPE}\"}"
    fi
    if [ -n "${LABELS}" ]; then TF_VAR_labels="${LABELS}"; fi
    if [ -n "${IMAGE_FAMILY_NAME}" ]; then
        TF_VAR_machine_image="{family=\"${IMAGE_FAMILY_NAME}\",name=null,project=\"ml-images\"}"
    elif [ -n "${IMAGE_NAME}" ]; then
        TF_VAR_machine_image="{family=null,name=\"${IMAGE_NAME}\",project=\"ml-images\"}"
    fi
    if [ -n "${VM_TYPE}" ]; then TF_VAR_machine_type="${VM_TYPE}"; fi

    if [ -n "${METADATA}" ]; then TF_VAR_metadata="${METADATA}"; fi
    if [ -n "${STARTUP_COMMAND}" ]; then TF_VAR_startup_script="${STARTUP_COMMAND}"; fi
    if [ -n "${NETWORK_CONFIG}" ]; then TF_VAR_network_config="${NETWORK_CONFIG}"; fi
}

_invoke_cluster_tool () {
  echo "Invoking cluster tool"
  printenv | grep 'TF_VAR_.*'

  /root/aiinfra/scripts/entrypoint.sh \
      ${MINIMIZE_TERRAFORM_LOGGING:+ --quiet} \
      ${TERRAFORM_GCS_PATH:+ --backend-bucket ${TERRAFORM_GCS_PATH}} \
      ${ACTION,,} \
      mig
}

export PROJECT=$1
export TRAIN_IMAGE=$2
export DATA_DIR=${3/\/gcs\//gs:\/\/}
export ID=$4
export DATA=${5/\/gcs\//gs:\/\/}
export WORKSPACE_PATH=${6/\/gcs\//gs:\/\/}
export TRAIN_CONFIG=$7
export CLUSTER_CONFIG=$8

echo ${TRAIN_CONFIG}
if [[ -n "${TRAIN_CONFIG}" ]]; then
  echo ${TRAIN_CONFIG} > train_config.json
  gsutil cp train_config.json ${DATA_DIR}/
else
  gsutil cp ${DATA_DIR}/train_config.json .
fi
source ./json_to_env.sh train_config.json

echo ${CLUSTER_CONFIG}
if [[ -n "${CLUSTER_CONFIG}" ]]; then
  echo ${CLUSTER_CONFIG} > cluster.json
  gsutil cp cluster.json ${DATA_DIR}/
else
  gsutil cp ${DATA_DIR}/cluster.json .
fi
source ./json_to_env.sh cluster.json

if [[ -n "${ID}" && "${ID}" != "0" ]]; then
  export JOB_ID=${NAME_PREFIX}-${ID}
else
  export JOB_ID=${NAME_PREFIX}
fi

shopt -s extglob
export REGION=${ZONE/%-+([a-z0-9])/}

export TRAIN_CMD="./train.sh ${DATA_DIR} ${DATA} ${WORKSPACE_PATH}"
export START="docker pull ${TRAIN_IMAGE}; nvidia-persistenced; docker run --name train_llm --security-opt apparmor=unconfined --cap-add SYS_ADMIN --device /dev/fuse --ipc host --network host --hostname \$(hostname) --gpus all -v /etc/ssh:/etc/ssh ${TRAIN_IMAGE} ${TRAIN_CMD}"
#gcloud compute resource-policies create group-placement ${JOB_ID}  --collocation COLLOCATED  --region ${REGION}  --project ${PROJECT}
#gcloud compute instance-templates create ${JOB_ID} --project=${PROJECT} --machine-type=${MACHINE_TYPE} --network-interface=network-tier=PREMIUM,network=default,address= --metadata=install-unattended-upgrades=false,enable-oslogin=TRUE,jupyter-user=${OS_LOGIN_USER},install-nvidia-driver=True,startup-script="${START}" --maintenance-policy=TERMINATE --provisioning-model=STANDARD --scopes=https://www.googleapis.com/auth/cloud-platform --accelerator=count=${GPU_COUNT},type=${GPU_TYPE} --create-disk=auto-delete=yes,boot=yes,device-name=gpu1,image=projects/ml-images/global/images/c2-deeplearning-pytorch-1-11-cu113-v20220701-debian-10,mode=rw,size=2000,type=pd-ssd --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any --resource-policies=${JOB_ID} --no-restart-on-failure
#gcloud compute instance-groups managed create ${JOB_ID} --project=${PROJECT} --base-instance-name=${JOB_ID} --size=${NODE_COUNT} --template=${JOB_ID} --zone=${ZONE} --list-managed-instances-results=PAGELESS

export JOB_FOUND=$(gsutil ls ${DATA_DIR}/machines.txt)

if [[ -z "${JOB_FOUND}" ]]; then
  echo "No machines.txt found. Trying to find a cluster with the spified prefix ${JOB_ID}"
  export JOB_FOUND=$(gcloud compute instances list | grep ${JOB_ID})
  if [[ -n "${JOB_FOUND}" ]]; then
    echo "Cluster found! Exporting machine list..."
    gcloud compute instances list | grep ${JOB_ID} | sed 's/\(\S\+\) .* \([0-9\.]\+\) \+\([0-9\.]\+\) \+RUNNING/\1 \2/' | sort > machines.txt
    gsutil cp machines.txt ${DATA_DIR}/
  fi
fi

if [[ -n "${JOB_FOUND}" ]]; then
  if [[ -z ${ID} || "${ID}" == "0" ]]; then
    echo "Restarting training on VMs..."
    gsutil rm ${DATA_DIR}/progress.txt || true
    gsutil cp ${DATA_DIR}/machines.txt .
    mkdir ~/.ssh
    gcloud compute ssh $(head -n 1 machines.txt | sed 's/\(\S\+\) .*/\1/') --zone=$ZONE --internal-ip --ssh-key-expire-after=1d --strict-host-key-checking=no --command="echo 'ssh is available'"

    while read -r machine ip;
    do
      echo $machine
      (gcloud compute ssh $machine --zone=$ZONE --internal-ip --ssh-key-expire-after=1d --strict-host-key-checking=no --command="bash -c -l 'sudo groupadd docker;sudo usermod -aG docker \$USER'" -- -n
      gcloud compute ssh $machine --zone=$ZONE --internal-ip --ssh-key-expire-after=1d --strict-host-key-checking=no --command="bash -c -l 'docker kill train_llm || true; docker container prune -f || true; $START ' >> log.txt 2>&1 &" -- -n) &
    done < machines.txt
    echo "Training initiated on all machines!"
  else
    echo "Cluster already exist! Skipping further retries since no override is requested..."
    exit 1
  fi
else
  echo "Provishioning cluster..."

  export GCS_PATH=${DATA_DIR}/deployment
  echo seed > deployment.txt
  gsutil cp deployment.txt $GCS_PATH/deployment.txt

  export SERVICE_ACCOUNT=$(gcloud config get account)
  export OS_LOGIN_USER=$(gcloud iam service-accounts describe ${SERVICE_ACCOUNT} | grep uniqueId | sed -e "s/.* '\(.*\)'/sa_\1/")
  echo User is ${OS_LOGIN_USER}

  export ACTION=CREATE
  export NAME_PREFIX=${JOB_ID}
  export INSTANCE_COUNT=${NODE_COUNT}
  export VM_TYPE=${MACHINE_TYPE}
  export ACCELERATOR_TYPE=${GPU_TYPE}
  export IMAGE_NAME=c0-deeplearning-common-cu113-v20221026-debian-10
  export TERRAFORM_GCS_PATH=${DATA_DIR}/deployment
  export METADATA="{install-unattended-upgrades=\"false\",enable-oslogin=\"TRUE\",jupyter-user=\"${OS_LOGIN_USER}\",install-nvidia-driver=\"True\"}"
  export STARTUP_COMMAND=${START}
  export PROJECT_ID=${PROJECT}
  export SHOW_PROXY_URL=no
  export LABELS="{gcpllm=\"$NAME_PREFIX\"}"
  export MINIMIZE_TERRAFORM_LOGGING=true
  #export DISK_SIZE_GB=1000

  _env_to_tf_var_mig
  _invoke_cluster_tool

  gcloud compute instances list | grep ${JOB_ID} | sed 's/\(\S\+\) .* \([0-9\.]\+\) \+\([0-9\.]\+\) \+RUNNING/\1 \2/' | sort > machines.txt
  gsutil cp machines.txt ${DATA_DIR}/
fi

echo "Waiting for training to start..."
(sleep 2400;echo check > check.txt) &

monitoring_started=false
export EXIT_CODE=
export LOG_START_TIME=$(date -Ins | sed -e "s/,/\./")
while [[ -z "$EXIT_CODE" ]]; do
  export RESULT=$(gsutil cat ${DATA_DIR}/progress.txt 2> /dev/null)
  if [[ "${RESULT}" == "succeeded" ]]; then
    echo "Training finished successfully!"
    export EXIT_CODE=0
    break
  fi
  if [[ "${RESULT}" == "failed" ]]; then
    echo "Training failed!"
    export EXIT_CODE=1
    break
  fi

  if [[ -e check.txt ]]; then
    rm check.txt
    
    if [[ "${RESULT}" != "started" ]]; then
      echo "Timeout! Training not started in 20 minutes"
      export EXIT_CODE=2
      break
    fi
  fi

  if [[ "${RESULT}" == "started" && "$monitoring_started" != true  ]]; then
    echo "Training started, start monitoring."
    (python3 training_cluster_monitor.py --project_id=${PROJECT} --model_output=${DATA_DIR}/model) &
    monitoring_started=true
  fi

  export LOG_END_TIME=$(date -Ins | sed -e "s/,/\./")
  sleep 15
  # Reading logs 15 seconds behind to give them a chance to be collected.
  gcloud logging read "log_name=projects/${PROJECT}/logs/deepspeed labels.\"compute.googleapis.com/resource_name\"=~\"${JOB_ID}\" timestamp>=\"${LOG_START_TIME}\" timestamp<\"${LOG_END_TIME}\"" --project=${PROJECT} | yq -r .textPayload | tac
  export LOG_START_TIME=${LOG_END_TIME}
  
done

# Only delete cluster when training succeeds. Otherwise, keep the cluster for investigation
if [[ "${EXIT_CODE}" == "0" ]]; then
  #gcloud compute instance-groups managed delete ${JOB_ID} --quiet --project=${PROJECT} --zone=${ZONE}
  #gcloud compute instance-templates delete ${JOB_ID} --quiet --project=${PROJECT}
  export ACTION=DESTROY
  _env_to_tf_var_mig
  _invoke_cluster_tool
fi

exit $EXIT_CODE

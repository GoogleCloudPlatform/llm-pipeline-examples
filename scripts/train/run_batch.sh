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

if [[ ${GPU_TYPE} == "nvidia-h100-80gb" ]]; then
  export USE_COS_IMAGE=1
  export TCPX=1
  export A3=1
fi

if [[ ${USE_COS_IMAGE} ]]; then
  echo "Using COS image"
  for (( i=0; i < ${GPU_COUNT}; i++ )); do
   export DOCKER_PARAMS="${DOCKER_PARAMS} --device /dev/nvidia${i}:/dev/nvidia${i}"
  done
  
  export DOCKER_PARAMS="${DOCKER_PARAMS} \
   --volume /var/lib/nvidia/lib64:/usr/local/nvidia/lib64 \
   --volume /var/lib/nvidia/bin:/usr/local/nvidia/bin \
   --volume /run/tcpx:/tmp \
   --volume /var/lib/tcpx:/usr/local/tcpx \
   --device /dev/nvidia-uvm:/dev/nvidia-uvm \
   --device /dev/nvidiactl:/dev/nvidiactl \
   --env NCCL_DEBUG=INFO \
   --env NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING,NET,VERSION \
   --env LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/usr/local/nvidia/lib64 \
   --cap-add=IPC_LOCK \
   --userns=host \
   -v /mnt/stateful_partition/etc/ssh:/mnt/stateful_partition/etc/ssh"

  export PRE_DOCKER_RUN="sudo sysctl -w net.ipv4.tcp_mtu_probing=0;"
  export VM_IMAGE=\"cos-105-17412-156-59\"
  export IMAGE_FAMILY=null
  export IMAGE_PROJECT=\"cos-cloud\"
else
  echo "Using DLVM image"
  export DOCKER_PARAMS="--gpus all"
  export PRE_DOCKER_RUN="nvidia-persistenced;"
  export VM_IMAGE=\"c0-deeplearning-common-cu113-v20221026-debian-10\"
  export IMAGE_FAMILY=null
  export IMAGE_PROJECT=\"ml-images\"
fi

export VM_IMAGE
export TRAIN_CMD="./train.sh ${DATA_DIR} ${DATA} ${WORKSPACE_PATH}"
if [[ ${TCPX} ]]; then
  export TRAIN_CMD="\
    sudo mkdir /usr/local/tcpx_exec; \
    sudo mount --bind /usr/local/tcpx_exec /usr/local/tcpx_exec; \
    sudo mount -o remount,exec /usr/local/tcpx_exec; \
    sudo cp -r /usr/local/tcpx/lib64 /usr/local/tcpx_exec; \
    sudo rm /lib/x86_64-linux-gnu/libnccl.so.2.18.1; \
    sudo rm /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so.0.0.0; \
    sudo rm /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so.0; \
    sudo chmod go+w /etc/ld.so.conf.d/nvidia.conf; \
    sudo echo /usr/local/tcpx_exec/lib64 >> /etc/ld.so.conf.d/nvidia.conf; \
    export NCCL_P2P_NET_CHUNKSIZE=524288; \
    export NCCL_P2P_PCI_CHUNKSIZE=524288; \
    export NCCL_P2P_NVL_CHUNKSIZE=1048576; \
    export NCCL_CHECK_POINTERS=0; \
    export NCCL_GRAPH_MIXING_SUPPORT=0; \
    export NCCL_PROTO=simple; \
    export NCCL_SOCKET_IFNAME=eth0; \
    export NCCL_CROSS_NIC=0; \
    export NCCL_ALGO=Ring; \
    export NCCL_NSOCKS_PERTHREAD=4; \
    export NCCL_SOCKET_NTHREADS=1; \
    export NCCL_MAX_NCHANNELS=8; \
    export NCCL_MIN_NCHANNELS=8; \
    export NCCL_DYNAMIC_CHUNK_SIZE=524288; \
    export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4; \
    export NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0; \
    export NCCL_NET_GDR_LEVEL=PIX; \
    export NCCL_P2P_PXN_LEVEL=0; \
    export NCCL_DEBUG=INFO; \
    export NCCL_DEBUG_SUBSYS=ENV; \
    export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=1000000; \
    export N_OP_SHARDS=8; \
    export CYCLIC=false; \
    export ACCELERATORS_PER_POD=8; \
    export NCCL_GPUDIRECTTCPX_FORCE_ACK=1; \
    ${TRAIN_CMD}"
fi
export START="docker pull ${TRAIN_IMAGE}; ${PRE_DOCKER_RUN} docker run --name train_llm \
 --security-opt \
 apparmor=unconfined \
 --cap-add SYS_ADMIN \
 --device /dev/fuse \
 --ipc host \
 --network host \
 --hostname \$(hostname) \
 ${DOCKER_PARAMS} \
 -v /etc/ssh:/etc/ssh \
 -v /var/tmp:/host/tmp \
 ${TRAIN_IMAGE} bash -c '${TRAIN_CMD}'"
echo "startup command = ${START}"
#gcloud compute resource-policies create group-placement ${JOB_ID}  --collocation COLLOCATED  --region ${REGION}  --project ${PROJECT}
#gcloud compute instance-templates create ${JOB_ID} --project=${PROJECT} --machine-type=${MACHINE_TYPE} --network-interface=network-tier=PREMIUM,network=default,address= --metadata=install-unattended-upgrades=false,enable-oslogin=TRUE,jupyter-user=${OS_LOGIN_USER},install-nvidia-driver=True,startup-script="${START}" --maintenance-policy=TERMINATE --provisioning-model=STANDARD --scopes=https://www.googleapis.com/auth/cloud-platform --accelerator=count=${GPU_COUNT},type=${GPU_TYPE} --create-disk=auto-delete=yes,boot=yes,device-name=gpu1,image=projects/ml-images/global/images/c2-deeplearning-pytorch-1-11-cu113-v20220701-debian-10,mode=rw,size=2000,type=pd-ssd --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any --resource-policies=${JOB_ID} --no-restart-on-failure
#gcloud compute instance-groups managed create ${JOB_ID} --project=${PROJECT} --base-instance-name=${JOB_ID} --size=${NODE_COUNT} --template=${JOB_ID} --zone=${ZONE} --list-managed-instances-results=PAGELESS

export JOB_FOUND=$(gsutil ls ${DATA_DIR}/machines.txt)

if [[ -z "${JOB_FOUND}" ]]; then
  echo "No machines.txt found. Trying to find a cluster with the spicified prefix ${JOB_ID}"
  export JOB_FOUND=$(gcloud compute instances list --project ${PROJECT}| grep ${JOB_ID})
  if [[ -n "${JOB_FOUND}" ]]; then
    echo "Cluster found! Exporting machine list..."
    gcloud compute instances list --project ${PROJECT}| grep ${JOB_ID} | sed 's/\(\S\+\) .* \([0-9\.]\+\)[0-9\.,]* \+\([0-9\.]\+\) \+RUNNING/\1 \2/' | sort | head -n ${NODE_COUNT} > machines.txt
    gsutil cp machines.txt ${DATA_DIR}/
  fi
fi

if [[ -n "${JOB_FOUND}" ]]; then
  if [[ -z ${ID} || "${ID}" == "0" ]]; then
    echo "Restarting training on VMs..."
    gsutil rm ${DATA_DIR}/progress.txt || true
    gsutil cp ${DATA_DIR}/machines.txt .
    mkdir ~/.ssh
    gcloud compute ssh $(head -n 1 machines.txt | sed 's/\(\S\+\) .*/\1/') --project ${PROJECT} --zone=$ZONE --internal-ip --ssh-key-expire-after=1d --strict-host-key-checking=no --command="echo 'ssh is available'"

    while read -r machine ip;
    do
      echo $machine
      (gcloud compute ssh $machine --project ${PROJECT} --zone=$ZONE --internal-ip --ssh-key-expire-after=1d --strict-host-key-checking=no --command="bash -c -l 'sudo groupadd docker;sudo usermod -aG docker \$USER'" -- -n
      gcloud compute ssh $machine --project ${PROJECT} --zone=$ZONE --internal-ip --ssh-key-expire-after=1d --strict-host-key-checking=no --command="bash -c -l 'docker kill train_llm || true; docker container prune -f || true; ${START//\'/\"} ' >> log.txt 2>&1 &" -- -n)
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

  sed -i -e "s/{project_id}/\"${PROJECT}\"/g" \
    -e "s/{region}/\"${REGION}\"/g" \
    -e "s/{cluster_prefix}/\"${JOB_ID}\"/g" \
    -e "s/{machine_type}/\"${MACHINE_TYPE}\"/g" \
    -e "s/{labels}/{gcpllm=\"$NAME_PREFIX\"}/g" \
    -e "s/{image_family}/${IMAGE_FAMILY}/g" \
    -e "s/{image_project}/${IMAGE_PROJECT}/g" \
    -e "s/{image_name}/${VM_IMAGE}/g" \
    -e "s/{node_count}/\"${NODE_COUNT}\"/g" \
    -e "s/{nodes_zone}/\"${ZONE}\"/g" \
    -e "s/{disk_size_gb}/2000/g" \
    -e "s/{maintenance_interval}/\"PERIODIC\"/g" \
    -e "s/{use_compact_placement_policy}/true/g" \
    /root/aiinfra/input/terraform.tfvars
    
    

  if [[ ${A3} ]]; then
    sed -i "s/{metadata}/{}/g" /root/aiinfra/input/terraform.tfvars
    echo "startup_script  = \"${START}\"" >> /root/aiinfra/input/terraform.tfvars
    export CLUSTER_TYPE=mig-cos
    export CPT_TEMPLATE=a3
    sed -i -e "s/cos-extensions install gpu -- --version=latest/cos-extensions install gpu -- --version=525.125.06/g" \
        -e "s/gpudirect-tcpx\\/tcpgpudmarxd/gpudirect-tcpx\\/tcpgpudmarxd-dev:v2.0.6/g" \
        -e "s/a3vm/a3vm --disable_quickack/g" \
        -e "s/nccl-plugin-gpudirecttcpx/nccl-plugin-gpudirecttcpx-dev:v3.1.5-v3-ns/g" \
        a3/terraform/modules/cluster/mig-cos/cloudinit/templates/aiinfra_startup_scripts.yaml.template
  else
    echo ${START} > start.sh
    gsutil cp start.sh ${DATA_DIR}/start.sh
    sed -i "s/{metadata}/{install-unattended-upgrades=\"false\",enable-oslogin=\"TRUE\",jupyter-user=\"${OS_LOGIN_USER}\",install-nvidia-driver=\"True\",startup-script-url=\"${DATA_DIR//\//\\\/}\\/start.sh\"}/g" \
    /root/aiinfra/input/terraform.tfvars
    export CLUSTER_TYPE=mig
    export CPT_TEMPLATE=a2
  fi

  cat /root/aiinfra/input/terraform.tfvars

  ./scripts/entrypoint.sh create ${CPT_TEMPLATE} ${CLUSTER_TYPE} -b ${DATA_DIR}/deployment -q 
  
  gcloud compute instances list --project ${PROJECT} | grep ${JOB_ID} | sed 's/\(\S\+\) .* \([0-9\.]\+\)[0-9\.,]* \+\([0-9\.]\+\) \+RUNNING/\1 \2/' | sort > machines.txt
  gsutil cp machines.txt ${DATA_DIR}/
  export CLUSTER_PROVISIONED=1
fi

echo "Waiting for training to start..."
(sleep 3600;echo check > check.txt) &

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
      echo "Timeout! Training not started in 60 minutes"
      export EXIT_CODE=2
      break
    fi
  fi

  if [[ "${RESULT}" == "started" && "$monitoring_started" != true  ]]; then
    echo "Training started!"
    #echo "Training started, start monitoring."
    #(python3 training_cluster_monitor.py --project_id=${PROJECT} --model_output=${DATA_DIR}/model) &
    monitoring_started=true
  fi

  export LOG_END_TIME=$(date -Ins | sed -e "s/,/\./")
  sleep 15
  # Reading logs 15 seconds behind to give them a chance to be collected.
  gcloud logging read "log_name=projects/${PROJECT}/logs/deepspeed labels.\"compute.googleapis.com/resource_name\"=~\"${JOB_ID}\" timestamp>=\"${LOG_START_TIME}\" timestamp<\"${LOG_END_TIME}\"" --project=${PROJECT} | yq -r .textPayload | tac
  export LOG_START_TIME=${LOG_END_TIME}
  
done

# Only delete cluster when training succeeds. Otherwise, keep the cluster for investigation
if [[ "${EXIT_CODE}" == "0"  && -n "${CLUSTER_PROVISIONED}" ]]; then
  #gcloud compute instance-groups managed delete ${JOB_ID} --quiet --project=${PROJECT} --zone=${ZONE}
  #gcloud compute instance-templates delete ${JOB_ID} --quiet --project=${PROJECT}
  ./scripts/entrypoint.sh destroy a3 ${CLUSTER_TYPE} -b ${DATA_DIR}/deployment -q
fi

exit $EXIT_CODE

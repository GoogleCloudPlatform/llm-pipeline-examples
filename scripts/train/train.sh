#!/bin/bash -ev

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


export MODEL_CHECKPOINT=$1
export DATA=$2
export MODEL_OUTPUT=$3
export ZONE=$4
export BATCH_SIZE=$5
export EPOCHS=$6
export GPU_COUNT=$7
export WORKSPACE_PATH=$8

export HOSTNAME=$(hostname)
shopt -s extglob
export CLUSTER_PREFIX=${HOSTNAME/%-+([a-z0-9])/-}

echo "Node ${HOSTNAME} model ${MODEL_CHECKPOINT} cluster ${CLUSTER_PREFIX}..."

export HEAD=$(gcloud compute instances list | grep ${CLUSTER_PREFIX} | sed 's/\(^\S\+\) .*/\1/' | sort | head -n 1)
echo "Head node found as ${HEAD}"

if [[ "$HEAD" == "$HOSTNAME" ]]; then
  sudo /usr/sbin/google-fluentd & 
  echo "Running ssh server..." >> /home/jupyter/deepspeed_output.log
  ./ssh_server.sh &

  ./update_env.sh >> /home/jupyter/deepspeed_output.log
  
  if [[ ! -d .ssh ]];
  then mkdir .ssh;
  fi
  echo "Configuring head node ..." >> /home/jupyter/deepspeed_output.log
  ./setup_head.sh ${CLUSTER_PREFIX} ${GPU_COUNT}
  export result=$?
  if [[ "$result" != "0" ]]; then 
    echo "Head setup failed"
    echo "Head setup failed" >> /home/jupyter/deepspeed_output.log
    exit $result
  fi
  if [[ -e done.txt ]]; then
    rm done.txt
  fi
  if [[ -e output.txt ]]; then
    rm output.txt
  fi
  echo started > progress.txt
  gsutil cp progress.txt ${MODEL_OUTPUT/\/gcs\//gs:\/\/}/progress.txt
  deepspeed --hostfile=deepspeed_hostfile finetune.py --deepspeed=deepspeed.json --model_checkpoint=${MODEL_CHECKPOINT} --batch_size=${BATCH_SIZE} --epochs=${EPOCHS} --tokenized_dataset_path=${DATA} --gcs_output=${MODEL_OUTPUT} &> /home/jupyter/deepspeed_output.log && echo succeeded > progress.txt || echo failed > progress.txt &
  tail --pid=$! -f /home/jupyter/deepspeed_output.log
  export RESULT=$(cat progress.txt)
  if [[ "${RESULT}" == "succeeded" ]]; then
    echo "Training finished successfully!"
    gsutil cp progress.txt ${MODEL_OUTPUT/\/gcs\//gs:\/\/}/progress.txt
    exit 0
  fi
  if [[ "${RESULT}" == "failed" ]]; then
    echo "Training failed!"
    gsutil cp progress.txt ${MODEL_OUTPUT/\/gcs\//gs:\/\/}/progress.txt
    exit 1
  fi
else
  ./update_env.sh
  ./ssh_server.sh
fi

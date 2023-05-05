#!/bin/bash -v

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
export TRAIN_CMD=$1

export HOME_DIR=/home/llm
export HOSTNAME=$(hostname)
shopt -s extglob

echo "Node ${HOSTNAME} model ${MODEL_CHECKPOINT}..."

gsutil cp ${MODEL_OUTPUT/\/gcs\//gs:\/\/}/machines.txt .
read -r HEAD HEAD_IP < machines.txt
echo "Head node found as ${HEAD}"

if [[ "$HEAD" == "$HOSTNAME" ]]; then
  sudo /usr/sbin/google-fluentd & 
  echo "Running ssh server..." >> ${HOME_DIR}/deepspeed_output.log
  ./ssh_server.sh &

  ./update_env.sh >> ${HOME_DIR}/deepspeed_output.log
  
  if [[ ! -d .ssh ]];
  then mkdir .ssh;
  fi
  echo "Configuring head node ..." >> ${HOME_DIR}/deepspeed_output.log
  ./setup_head.sh ${ZONE} ${GPU_COUNT}
  export result=$?
  if [[ "$result" != "0" ]]; then 
    echo "Head setup failed"
    echo "Head setup failed" >> ${HOME_DIR}/deepspeed_output.log
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
  ${TRAIN_CMD} 2>&1 | tee ${HOME_DIR}/deepspeed_output.log
  export RESULT=$?
  if [[ "${RESULT}" == "0" ]]; then
    echo "Training succeeded!" | tee progress.txt
    gsutil cp progress.txt ${MODEL_OUTPUT/\/gcs\//gs:\/\/}/progress.txt
    exit 0
  else
    echo "Training failed with code: ${RESULT}" | tee progress.txt
    gsutil cp progress.txt ${MODEL_OUTPUT/\/gcs\//gs:\/\/}/progress.txt
    exit 1
  fi
else
  ./update_env.sh
  ./ssh_server.sh
fi

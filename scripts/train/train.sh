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

echo "Preloading model and config from workspace ${WORKSPACE_PATH}..."
python3 -m prepare --workspace_path=${WORKSPACE_PATH}

export TRAIN_CMD="deepspeed --hostfile=deepspeed_hostfile finetune.py --deepspeed=deepspeed.json --model_checkpoint=${MODEL_CHECKPOINT} --batch_size=${BATCH_SIZE} --epochs=${EPOCHS} --tokenized_dataset_path=${DATA} --gcs_output=${MODEL_OUTPUT}"
./train_common.sh "${TRAIN_CMD}"

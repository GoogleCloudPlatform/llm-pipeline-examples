#!/bin/bash
export GCS_BUCKET=$1
export ZONE=$2
export NODE_COUNT=$3
export GPU_COUNT=$4
export DATA_FILE_NAME=$5
export MODEL_OUTPUT=gs://${GCS_BUCKET}

export BASE_PATH=~/data/

mkdir ${BASE_PATH}
gcsfuse ${GCS_BUCKET} ${BASE_PATH}

DATA_PATH=${BASE_PATH}/${DATA_FILE_NAME}
#wiki_data_text_document
DS_CONFIG=ds_config.json

# Hostfile path
HF=deepspeed_hostfile 

# Disabling tensor/pipeline parallelism
TP=$6
PP=$7

# HEADS ~= HIDDEN/128

# Model: BLOOM
NLAYERS=$8
HIDDEN=14336
HEADS=112
SEQ=2048


MICRO_BATCH=2
GRADIENT_ACC_STEPS=128

NODES=${NODE_COUNT}
GPN=${GPU_COUNT}

GLOBAL_BATCH=$(( ${GRADIENT_ACC_STEPS} * ${GPN} * ${MICRO_BATCH} * ${NODES}  / ${TP} / ${PP} ))

# Initial power scale for loss
SP=15

# Uncomment/comment one of the following blocks.

# For 1T model, start with microbatch=1, try to get 2 and 4.  If OOM w/ 4, use cpu-offloading

# Set to cpu for offloading to cpu for larger models
#OFFLOAD_DEVICE="cpu"
#CPU_OPTIM=" --cpu-optimizer"

# Set to none and empty string for no cpu offloading
OFFLOAD_DEVICE="none"  
CPU_OPTIM=" "

ZERO_STAGE=0
OUTPUT_DIR=ds_z_off-${OFFLOAD_DEVICE}_stage_${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_nodes${NODES}
mkdir -p $OUTPUT_DIR

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,
  "gradient_accumulation_steps": $GRADIENT_ACC_STEPS,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "gradient_clipping": 1.0,
  "bf16": {
    "enabled": true
  }, 
  "wall_clock_breakdown": true,
  "zero_allow_untested_optimizer": false,
  "aio": {
    "block_size": 1048576,
    "queue_depth": 16,
    "single_submit": false,
    "overlap_events": true,
    "thread_count": 2
  },
  "flops_profiler": {
      "enabled": false,
      "profile_step": 20,
      "module_depth": -1,
      "top_modules": 1,
      "detailed": false,
      "output_file": null
  }
}
EOT

export NCCL_DEBUG=warn
export CUDA_LAUNCH_BLOCKING=1

ds_args=" "
ds_args=" --deepspeed ${ds_args}"
#ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"



export TRAIN_CMD="deepspeed --force_multi --num_nodes=$NODES --hostfile $HF pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $HEADS \
    --seq-length $SEQ \
    --loss-scale $SP \
    --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters 1000 \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 40 \
    --eval-interval 1000 \
    --data-path $DATA_PATH \
    --num-workers 2 \
    --vocab-file $BASE_PATH/gpt2-vocab.json \
    --merge-file $BASE_PATH/gpt2-merges.txt \
    --save-interval 100 \
    --save ${BASE_PATH}/checkpoints
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --bf16 \
    --checkpoint-activations \
    --tensorboard-dir $OUTPUT_DIR \
    $CPU_OPTIM $ds_args \
    --exit-interval 5000"

export pybinc=$(python3 -m pybind11 --includes)
export pyinc=$(python3-config --extension-suffix)
sed -i -e 's/$(shell python3 -m pybind11 --includes)/'"${pybinc//\//\\\/}"'/' \
    -e 's/$(shell python3-config --extension-suffix)/'"${pyinc//\//\\\/}"'/' \
    megatron/data/Makefile 
cat megatron/data/Makefile 
sed -i -e 's/loss_scale = None/loss_scale = 0.0/' megatron/training.py
./train_common.sh "${TRAIN_CMD}"
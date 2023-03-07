#! /bin/bash
# Arg1 - NumGpus
# Arg2 - Model Path on HuggingFace
# Arg3 - DSM for GPU [See faster_transformer_install.sh for table]
# Arg4 - GCS path to upload the model
# Arg5 - Subdirectory converted model will be uploaded to
NUM_GPUS=$1
MODEL_PATH=$2
GPU_DSM=$3
GCS_UPLOAD_PATH=$4
TARGET_DIRECTORY=$5

# Parse model directory from directory or repo path
IFS="/"
MODEL_NAME=$MODEL_PATH
for x in $MODEL_PATH
do
    MODEL_NAME=$x
done

OUTPUT_PATH="$(pwd)"/$MODEL_NAME/fastertransformer

# Build fast_transformer
./faster_transformer_install.sh $GPU_DSM
cd FasterTransformer/build

# Download model
if [[ "$MODEL_PATH" == /gcs* ]] || [[ "$MODEL_PATH" == gs://* ]] ;
then
    # Download from GCS
    GCS_DOWNLOAD_PATH=${MODEL_OUTPUT/\/gcs\//gs:\/\/}
    gsutil cp -r $GCS_DOWNLOAD_PATH .
else
    # Download from HuggingFace
    apt-get update
    apt-get install git-lfs
    git lfs install
    git lfs clone https://huggingface.co/$MODEL_PATH
fi

# Convert model
pip install -r ../examples/pytorch/t5/requirement.txt

# Convert
python3 ../examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
        -saved_dir $OUTPUT_PATH/1/ \
        -in_file $MODEL_NAME/ \
        -inference_tensor_para_size $NUM_GPUS \
        -weight_data_type fp16

cp ../../all_models/t5/config.pbtxt $OUTPUT_PATH/config.pbtxt
sed -i "s/PLACEHOLDERNUMGPU/$NUM_GPUS/g" $OUTPUT_PATH/config.pbtxt
sed -i "s/PLACEHOLDERMODELNAME/$MODEL_NAME/g" $OUTPUT_PATH/config.pbtxt

cd ../..

# Upload converted model
GCS_PATH=${GCS_UPLOAD_PATH/\/gcs\//gs:\/\/} #/$TARGET_DIRECTORY

gsutil cp -r ./$MODEL_NAME "$GCS_PATH"
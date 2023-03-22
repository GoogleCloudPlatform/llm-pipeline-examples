#! /bin/bash
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
# Arg1 - NumGpus
# Arg2 - Model Path on HuggingFace or GCS
# Arg3 - GCS path to upload the model
NUM_GPUS=$1
MODEL_PATH=$2
GCS_UPLOAD_PATH=$3

# Parse model directory from directory or repo path
ORIG_IFS=$IFS
IFS="/"
MODEL_NAME=$MODEL_PATH
for x in $MODEL_PATH
do
    MODEL_NAME=$x
done
IFS=$ORIG_IFS

OUTPUT_PATH="$(pwd)/$MODEL_NAME/fastertransformer"
echo "Converted model will be uploaded to ${OUTPUT_PATH}"

cd FasterTransformer/build

# Download model
if [[ "$MODEL_PATH" == /gcs* ]] || [[ "$MODEL_PATH" == gs://* ]] ;
then
    # Download from GCS
    GCS_DOWNLOAD_PATH=${MODEL_PATH/\/gcs\//gs:\/\/}
    echo "Downloading model from ${GCS_DOWNLOAD_PATH}"
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

cp $MODEL_NAME/config.json $OUTPUT_PATH/1/$NUM_GPUS-gpu/config.json
cp $MODEL_NAME/tokenizer.json $OUTPUT_PATH/1/$NUM_GPUS-gpu/tokenizer.json
cd ../..

# Upload converted model
GCS_PATH=${GCS_UPLOAD_PATH/\/gcs\//gs:\/\/}

echo "Uploading model to ${GCS_PATH}"
gsutil cp -r ./$MODEL_NAME "$GCS_PATH"
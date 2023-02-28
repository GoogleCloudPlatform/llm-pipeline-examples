# Arg1 - NumGpus
# Arg2 - ModelName
# Arg3 - Output directory
# Install dependencies for t5 transformations
cd FasterTransformer/build
pip install -r ../examples/pytorch/t5/requirement.txt

echo  "Called with: NumGpus: " $1
echo  "Called with: ModelName: "$2
export NUM_GPUS=$1
export MODEL_NAME=$2

OUTPUT_PATH=$3/$MODEL_NAME/fastertransformer

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
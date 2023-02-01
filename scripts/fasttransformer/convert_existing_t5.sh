# Arg1 - NumGpus
# Arg2 - ModelName
# Install dependencies for t5 transformations
cd FasterTransformer/build
pip install -r ../examples/pytorch/t5/requirement.txt

# Convert
python3 ../examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
        -saved_dir ${WORKSPACE}/all_models/$2/fastertransformer/1/ \
        -in_file $2/ \
        -inference_tensor_para_size $1 \
        -weight_data_type fp32

rm -r $2

cd ../..
# Arg0 - NumGpus
# Arg1 - ModelName
# Install dependencies for t5 transformations
cd FasterTransformer/build
pip install -r ../examples/pytorch/t5/requirement.txt

# Download model
apt-get update
apt-get install git-lfs
git lfs install
git lfs clone https://huggingface.co/$1

# Convert
python3 ../examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
        -saved_dir ${WORKSPACE}/all_models/$1/fastertransformer/1/ \
        -in_file $1/ \
        -inference_tensor_para_size $0 \
        -weight_data_type fp32

cd ../..
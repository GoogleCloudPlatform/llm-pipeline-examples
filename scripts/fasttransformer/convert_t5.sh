# Install dependencies for t5 transformations
cd FasterTransformer/build
pip install -r ../examples/pytorch/t5/requirement.txt

# Download model
apt-get update
apt-get install git-lfs
git lfs install
git lfs clone https://huggingface.co/t5-base

# Convert
python3 ../examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
        -saved_dir t5-base/c-models \
        -in_file t5-base/ \
        -inference_tensor_para_size 1 \
        -weight_data_type fp32

# # Test conversion
# python3 ../examples/pytorch/t5/summarization.py  \
#         --ft_model_location t5-v1_1-base/c-models/ \
#         --hf_model_location t5-v1_1-base/ \
#         --test_ft \
#         --test_hf

cd ../..
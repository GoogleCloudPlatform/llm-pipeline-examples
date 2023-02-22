# Arg1 - Model path on HuggingFace
cd FasterTransformer/build

# Download model
apt-get update
apt-get install git-lfs
git lfs install
git lfs clone https://huggingface.co/$1

cd ../..
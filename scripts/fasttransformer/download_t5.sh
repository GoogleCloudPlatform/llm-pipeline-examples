# Arg1 - NumGpus
# Arg2 - ModelName
# Install dependencies for t5 transformations
cd FasterTransformer/build

# Download model
apt-get update
apt-get install git-lfs
git lfs install
git lfs clone https://huggingface.co/$2

cd ../..
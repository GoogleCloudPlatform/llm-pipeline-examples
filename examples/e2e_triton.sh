#! /bin/bash
### TODO
# 1. Parameterize NumGpu into:
# * Conversion Dockerfile
# * docker cp path
# * Image name/tag
# * Model name
# 2. Parameterize Model-Name (huggingface reference)
# * Convert dockerfile
# * docker cp path
# * Image name/tag
# * Model name
# 3. Deploy model to an endpoint
# * Add GPU Accelerator
# 4. Parameterize GPU being consumed
# * Use the Compute Units in Conversion script
# 5. Run triton-on-vertexai script after deployment
# * Need to set EndpointId in gcloud config
### Optimizations
# 1. Remove copying model to host, change copyModel.dockerfile to just copy directly from the existing container
# 2. Find a way to get the config.pbtxt without being hardcoded in the repo

# Arg0 - NumGpus
# Arg1 - Target GPU # Computes https://github.com/NVIDIA/FasterTransformer/blob/main/docs/t5_guide.md#build-the-project
# Arg3 - HuggingFace Storage Link

INTERMEDIATE_IMAGE_NAME=bakedt5image
TARGET_GPU_NUMBER=1
TARGET_NUM_COMPUTE=80
TARGET_MODEL_NAME=t5-base
DOCKER_BUILDKIT=1 docker build -f docker/download.Dockerfile \
     --build-arg NUMGPU=$TARGET_GPU_NUMBER \
     --build-arg NUMCOMPUTE=$TARGET_NUM_COMPUTE \
     --build-arg MODELNAME=$TARGET_MODEL_NAME \
     . -t $INTERMEDIATE_IMAGE_NAME:$TARGET_GPU_NUMBER \
     --progress=plain

DOCKER_BUILDKIT=1 docker build -f docker/convert.Dockerfile \
     --build-arg NUMGPU=$TARGET_GPU_NUMBER \
     --build-arg NUMCOMPUTE=$TARGET_NUM_COMPUTE \
     --build-arg MODELNAME=$TARGET_MODEL_NAME \
     --build-arg PARENT=$INTERMEDIATE_IMAGE_NAME:$TARGET_GPU_NUMBER
     . -t converted-$TARGET_MODEL_NAME:$TARGET_GPU_NUMBER \
     --progress=plain
docker image tag  converted-$TARGET_MODEL_NAME:$TARGET_GPU_NUMBER us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/$TARGET_MODEL_NAME-triton:$TARGET_GPU_NUMBER
docker image push us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/$TARGET_MODEL_NAME-triton:$TARGET_GPU_NUMBER
gcloud ai models upload --container-image-uri=us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/$TARGET_MODEL_NAME-triton:$TARGET_GPU_NUMBER --display-name=$TARGET_MODEL_NAME-ft

gcloud ai models upload --container-image-uri=us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/t5-triton:11b-4gpu --display-name=t511b

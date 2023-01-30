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

DOCKER_BUILDKIT=1 docker build . -f docker/convert.Dockerfile  -t convert
docker run --runtime=nvidia convert
docker container ls -lq
docker cp $(docker container ls -lq):/workspace/FasterTransformer/build/t5-base/c-models/4-gpu c-models/
docker build -f docker/copyModel.Dockerfile . -t t5triton4gpu
docker image tag  t5triton4gpu us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/t5-triton:4gpu
docker image push us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/t5-triton:4gpu
gcloud ai models upload --container-image-uri=us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/t5-triton:4gpu --display-name=t5_ft

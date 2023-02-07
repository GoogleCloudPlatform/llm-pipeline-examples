#! /bin/bash
### TODO
# 3. Deploy model to an endpoint
# * Add GPU Accelerator
# 4. Parameterize GPU being consumed
# * Use the Compute Units in Conversion script
# 5. Run triton-on-vertexai script after deployment
# * Need to set EndpointId in gcloud config
### Optimizations
# 2. Find a way to get the config.pbtxt without being hardcoded in the repo
# 3. Use example from Vertex+Triton to download model at deployment time https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/prediction/custom_prediction_routines/SDK_Triton_PyTorch_Local_Prediction.ipynb

# Arg0 - NumGpus
# Arg1 - Target GPU # Computes https://github.com/NVIDIA/FasterTransformer/blob/main/docs/t5_guide.md#build-the-project
# Arg3 - HuggingFace Storage Link

TARGET_GPU_NUMBER=4
TARGET_NUM_COMPUTE=80
TARGET_MODEL_NAME=google/t5-v1_1-base
TARGET_MODEL_PATH=t5-v1_1-base
TARGET_TAG=$TARGET_GPU_NUMBER-gpu
INTERMEDIATE_IMAGE=baked_$TARGET_MODEL_PATH:$TARGET_TAG
FINAL_IMAGE=converted-$TARGET_MODEL_PATH:$TARGET_TAG
GCR_FINAL_IMAGE=us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/$TARGET_MODEL_PATH-triton:$TARGET_TAG
DOCKER_BUILDKIT=1 docker build -f docker/download.Dockerfile \
     --build-arg NUMGPU=$TARGET_GPU_NUMBER \
     --build-arg NUMCOMPUTE=$TARGET_NUM_COMPUTE \
     --build-arg MODELNAME=$TARGET_MODEL_NAME \
     --build-arg MODELPATH=$TARGET_MODEL_PATH \
     . -t $INTERMEDIATE_IMAGE \
     --no-cache
     --progress=plain

DOCKER_BUILDKIT=1 docker build -f docker/convert.Dockerfile \
     --build-arg NUMGPU=$TARGET_GPU_NUMBER \
     --build-arg NUMCOMPUTE=$TARGET_NUM_COMPUTE \
     --build-arg MODELNAME=$TARGET_MODEL_NAME \
     --build-arg MODELPATH=$TARGET_MODEL_PATH \
     --build-arg PARENT=$INTERMEDIATE_IMAGE
     . -t $FINAL_IMAGE \
     --progress=plain
docker image tag  $FINAL_IMAGE $GCR_FINAL_IMAGE
docker image push $GCR_FINAL_IMAGE

MODEL_NAME=$TARGET_MODEL_PATH
gcloud ai models upload \
    --container-health-route=/v2/models/$MODEL_NAME \
    --container-predict-route=/v2/models/$MODEL_NAME/infer \
    --container-ports=8000 \
    --artifact-uri=gs://pirillo-sct-bucket/all_models/$MODEL_NAME \
    --container-image-uri=us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/vertex-triton-with-ft:22.07 \
    --display-name=$TARGET_MODEL_PATH-ft

gcloud ai endpoints deploy-model 2971025553785618432 \
    --region=us-central1  \
    --model=3904218455674454016 \
    --accelerator=count=4,type=nvidia-tesla-v100 \
    --machine-type=n1-standard-32 \
    --display-name=t5xxl-2207
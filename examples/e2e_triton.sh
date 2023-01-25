DOCKER_BUILDKIT=1 docker build . -f docker/convert.Dockerfile  -t convert
docker run --runtime=nvidia convert
docker container ls -lq
docker cp ${docker container ls -lq}:/workspace/FasterTransformer/build/t5-base/c-models/4-gpu c-models/
docker build -f copyModel.dockerfile . -t t5triton4gpu
docker image tag  t5triton4gpu us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/t5-triton:4gpu
#docker image push us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/t5-triton:4gpu
#gcloud ai models upload --container-image-uri=us-central1-docker.pkg.dev/supercomputer-testing/pirillo-gcr/t5-triton:4gpu --display-name=t5_ft


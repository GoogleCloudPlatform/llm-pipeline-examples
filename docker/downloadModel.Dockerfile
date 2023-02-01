# DOCKER_BUILDKIT=1 docker build --build-arg NUMGPU=4 --build-arg NUMCOMPUTE=80 --build-arg MODELNAME=t5-base -f docker/convertAndRunTriton.Dockerfile -t convertedtriton:latest
FROM chris113113/triton_with_ft:22.07
ARG NUMGPU=4
ARG NUMCOMPUTE=80
ARG MODELNAME="t5-base"
ENV MODEL_NAME_ENV $MODELNAME

# For --chmod to work the docker build must be run with `DOCKER_BUILDKIT=1 docker build`
COPY --chmod=777 scripts/fasttransformer/faster_transformer_install.sh .
COPY --chmod=777 scripts/fasttransformer/download_t5.sh .
COPY --chmod=777 scripts/fasttransformer/convert_existing_t5.sh .

RUN ./faster_transformer_install.sh $NUMCOMPUTE
RUN ./download_t5.sh $NUMGPU $MODELNAME

ENTRYPOINT /bin/bash
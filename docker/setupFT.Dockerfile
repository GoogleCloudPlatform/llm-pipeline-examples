# DOCKER_BUILDKIT=1 docker build --build-arg NUMGPU=4 --build-arg NUMCOMPUTE=80 --build-arg MODELNAME=t5-base -f docker/convertAndRunTriton.Dockerfile -t convertedtriton:latest
FROM nvcr.io/nvidia/pytorch:22.09-py3

ARG NUMCOMPUTE=70

RUN git clone https://github.com/NVIDIA/FasterTransformer.git
RUN mkdir -p FasterTransformer/build
RUN cd FasterTransformer/build && git submodule init && git submodule update

# For --chmod to work the docker build must be run with `DOCKER_BUILDKIT=1 docker build`
COPY --chmod=777 scripts/fasttransformer/faster_transformer_install.sh .
COPY --chmod=777 scripts/fasttransformer/download_t5.sh .
COPY --chmod=777 scripts/fasttransformer/convert_existing_t5.sh .
COPY --chmod=777 scripts/fasttransformer/download_and_convert_t5.sh .

RUN ./faster_transformer_install.sh $NUMCOMPUTE
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin:/root/google-cloud-sdk/bin

ENTRYPOINT /bin/bash
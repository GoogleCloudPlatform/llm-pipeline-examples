# DOCKER_BUILDKIT=1 docker build -f docker/convert.Dockerfile -t convert:latest
FROM nvcr.io/nvidia/pytorch:22.09-py3

RUN git clone https://github.com/NVIDIA/FasterTransformer.git
RUN mkdir -p FasterTransformer/build
RUN cd FasterTransformer/build && git submodule init && git submodule update

RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin:/root/google-cloud-sdk/bin

# For --chmod to work the docker build must be run with `DOCKER_BUILDKIT=1 docker build`
COPY --chmod=777 scripts/fasttransformer/faster_transformer_install.sh .
COPY --chmod=777 scripts/fasttransformer/download_t5.sh .
COPY --chmod=777 scripts/fasttransformer/convert_existing_t5.sh .
COPY --chmod=777 scripts/fasttransformer/all_in_one.sh .
COPY --chmod=777 examples/triton/t5/config.pbtxt ./all_models/t5/
# DOCKER_BUILDKIT=1 docker build -f docker/convert.Dockerfile -t convert:latest
# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nvidia/pytorch:22.09-py3

RUN git clone --depth 1 --branch release/v5.3_tag https://github.com/NVIDIA/FasterTransformer.git
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
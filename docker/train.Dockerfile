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
FROM nvcr.io/nvidia/pytorch:23.05-py3

ENV TORCH_CUDA_ARCH_LIST="7.0"
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
  && echo "deb http://packages.cloud.google.com/apt gcsfuse-focal main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
  && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update \
  && apt-get install --yes gcsfuse google-cloud-cli
  
RUN apt-get update
RUN apt install -yq openssh-server openssh-client ninja-build libaio-dev
RUN apt install -yq google-compute-engine-oslogin
RUN apt-get install -yq pdsh

RUN touch /tmp/deepspeed_output.log
RUN chmod 666 /tmp/deepspeed_output.log

COPY scripts/install.sh .
RUN ./install.sh
RUN pip install gcsfs

RUN chmod a+w /etc/sysctl.conf; echo "net.ipv4.ip_local_port_range = 50000 51000" >> /etc/sysctl.conf; chmod a-w /etc/sysctl.conf

RUN useradd -ms /bin/bash llm
RUN adduser llm sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
WORKDIR /home/llm
USER llm

COPY scripts/train/setup_head.sh .
COPY scripts/train/train_common.sh .
COPY scripts/train/train.sh .
COPY scripts/train/update_env.sh .
COPY scripts/train/ssh_server.sh .
COPY scripts/json_to_env.sh .

COPY src/deepspeed.json .
COPY src/finetune.py .
COPY src/preprocess.py .
COPY src/prepare.py .
COPY src/download.py .
COPY src/utils.py .

COPY configs/nccl_topo/a3_cos.xml .

COPY scripts/train/deepspeed-fluentd.conf /etc/google-fluentd/config.d/


EXPOSE 50000-51000
EXPOSE 29500
EXPOSE 1024-1039

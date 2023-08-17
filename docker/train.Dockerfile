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
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.2-0.py310:m109

RUN apt-get update
RUN apt install -yq openssh-server openssh-client ninja-build libaio-dev
RUN apt install -yq google-compute-engine-oslogin
RUN apt-get install -yq pdsh

RUN touch /tmp/deepspeed_output.log
RUN chmod 666 /tmp/deepspeed_output.log

COPY scripts/clean_up_torch_xla.sh .
COPY scripts/install.sh .
RUN pip3 uninstall -y torch torchvision
ENV DEBIAN_FRONTEND=noninteractive
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget --progress=dot:giga https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda=12.0.0-1
RUN apt-get -y --allow-change-held-packages remove libnccl2 libnccl-dev
RUN apt-get -y install libnccl2=2.18.1-1+cuda12.1 libnccl-dev=2.18.1-1+cuda12.1
RUN git clone --recursive https://github.com/pytorch/pytorch; \
    cd pytorch; \
    conda install cmake ninja; \
    pip install -y -r requirements.txt; \
    export _GLIBCXX_USE_CXX11_ABI=1; \
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}; \
    TORCH_CUDA_ARCH_LIST="8.0 9.0" NCCL_INCLUDE_DIR="/usr/include/" NCCL_LIB_DIR="/usr/lib/" USE_SYSTEM_NCCL=1 python setup.py install
RUN ./install.sh


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


RUN sudo chmod a+w /etc/sysctl.conf; echo "net.ipv4.ip_local_port_range = 50000 51000" >> /etc/sysctl.conf; sudo chmod a-w /etc/sysctl.conf

EXPOSE 50000-51000
EXPOSE 29500
EXPOSE 1024-1039

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
RUN wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run && chmod +x cuda_12.0.0_525.60.13_linux.run && sudo ./cuda_12.0.0_525.60.13_linux.run
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121
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

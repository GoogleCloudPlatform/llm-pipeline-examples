#! /bin/bash
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

# Needs to be run in below image.
# nvidia-docker run -ti --shm-size 5g --rm nvcr.io/nvidia/pytorch:22.09-py3 bash
git clone --depth 1 --branch release/v5.3_tag https://github.com/NVIDIA/FasterTransformer.git

# -DSM=xx should be replaced with the proper number of computes available for your GPU. 
# Table from https://github.com/NVIDIA/FasterTransformer/blob/main/docs/t5_guide.md
# GPU	compute capacity
# P40	60
# P4	61
# V100	70
# T4	75
# A100	80
# A30	80
# A10	86

# pytorch
mkdir -p FasterTransformer/build
cd FasterTransformer/build
git submodule init && git submodule update
cmake -DSM=60,61,70,75,80,86 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
make -j12
cd ../..
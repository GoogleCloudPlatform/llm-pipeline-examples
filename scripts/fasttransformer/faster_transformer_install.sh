# Arg1 - DSM

# Needs to be run in below image.
# nvidia-docker run -ti --shm-size 5g --rm nvcr.io/nvidia/pytorch:22.09-py3 bash
git clone https://github.com/NVIDIA/FasterTransformer.git
mkdir -p FasterTransformer/build
cd FasterTransformer/build
git submodule init && git submodule update

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
cmake -DSM=$1 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
# TensorRT
# cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_TRT=ON -DBUILD_MULTI_GPU=ON ..
# Tensorflow2
# cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_TF2=ON -DTF_PATH=/usr/local/lib/python3.8/dist-packages/tensorflow/ -DBUILD_MULTI_GPU=ON ..

make -j12
cd ../..
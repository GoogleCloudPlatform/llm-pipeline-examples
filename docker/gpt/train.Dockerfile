FROM gcr.io/llm-containers/train:latest

RUN pip install regex pybind11
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && git checkout 89cc215a49b0e99263a8184f17f17275879015aa && cd .. && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex/
RUN git clone https://github.com/microsoft/Megatron-DeepSpeed.git
RUN mv Megatron-DeepSpeed/* .

COPY scripts/gpt/ .
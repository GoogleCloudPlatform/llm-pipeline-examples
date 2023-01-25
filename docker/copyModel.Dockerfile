FROM chris113113/triton_with_ft:22.07
RUN export NUMGPU='4'

RUN export WORKSPACE=$(pwd)

# Local file location
COPY c-models/4-gpu/ $WORKSPACE/all_models/t5/fastertransformer/1/4-gpu/
COPY config.pbtxt $WORKSPACE/all_models/t5/fastertransformer/

ENTRYPOINT /opt/tritonserver/bin/tritonserver --model-repository=/workspace/all_models/t5/
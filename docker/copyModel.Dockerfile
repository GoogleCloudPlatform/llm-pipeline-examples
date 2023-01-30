FROM chris113113/triton_with_ft:22.07
ARG NUMGPU=4
RUN export ngpu=$NUMGPU
RUN export WORKSPACE=$(pwd)

# Local file location
COPY c-models/$NUMGPU-gpu/ $WORKSPACE/all_models/t5/fastertransformer/1/$NUMGPU-gpu/
COPY config.pbtxt $WORKSPACE/all_models/t5/fastertransformer/

RUN sed -i "/s/PLACEHOLDERNUMGPU/$NUMGPU/g" $WORKSPACE/all_models/t5/fastertransformer/config.pbtxt

ENTRYPOINT /opt/tritonserver/bin/tritonserver --model-repository=/workspace/all_models/t5/
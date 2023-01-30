FROM chris113113/triton_with_ft:22.07
ARG NUMGPU=4
ARG NUMCOMPUTE=80
ARG MODELNAME="t5-base"
RUN export WORKSPACE=$(pwd)

# Local file location
COPY config.pbtxt $WORKSPACE/all_models/$MODELNAME/fastertransformer/

RUN sed -i "/s/PLACEHOLDERNUMGPU/$NUMGPU/g" $WORKSPACE/all_models/$MODELNAME/fastertransformer/config.pbtxt
RUN ./convert.sh $NUMGPU $NUMCOMPUTE $MODELNAME

ENTRYPOINT /opt/tritonserver/bin/tritonserver --model-repository=/workspace/all_models/$MODELNAME/
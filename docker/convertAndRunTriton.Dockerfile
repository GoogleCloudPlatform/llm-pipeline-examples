# DOCKER_BUILDKIT=1 docker build --build-arg NUMGPU=4 --build-arg NUMCOMPUTE=80 --build-arg MODELNAME=t5-base -f docker/convertAndRunTriton.Dockerfile -t convertedtriton:latest
FROM chris113113/triton_with_ft:22.07
ARG NUMGPU=4
ARG NUMCOMPUTE=80
ARG MODELNAME="t5-base"
ENV MODEL_NAME_ENV $MODELNAME

# Local file location
COPY examples/config.pbtxt $WORKSPACE/all_models/$MODELNAME/fastertransformer/

# For --chmod to work the docker build must be run with `DOCKER_BUILDKIT=1 docker build`
COPY --chmod=777 scripts/fasttransformer/faster_transformer_install.sh .
COPY --chmod=777 scripts/fasttransformer/convert_t5.sh .
COPY --chmod=777 scripts/fasttransformer/convert.sh .

RUN sed -i "s/PLACEHOLDERNUMGPU/$NUMGPU/g" $WORKSPACE/all_models/$MODELNAME/fastertransformer/config.pbtxt
RUN sed -i "s/PLACEHOLDERMODELNAME/$MODELNAME/g" $WORKSPACE/all_models/$MODELNAME/fastertransformer/config.pbtxt

RUN ./faster_transformer_install.sh $NUMCOMPUTE
RUN ./convert_t5.sh $NUMGPU $MODELNAME

ENTRYPOINT /opt/tritonserver/bin/tritonserver --model-repository=/workspace/all_models/$MODEL_NAME_ENV/
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

ARG PARENT=bakedt511b
ARG NUMGPU=4
ARG NUMCOMPUTE=80
ARG MODELNAME="t5-base"

FROM PARENT

ENV MODEL_NAME_ENV $MODELNAME

# Local file location
COPY examples/config.pbtxt $WORKSPACE/all_models/$MODELNAME/fastertransformer/

# For --chmod to work the docker build must be run with `DOCKER_BUILDKIT=1 docker build`
COPY --chmod=777 scripts/fasttransformer/faster_transformer_install.sh .
COPY --chmod=777 scripts/fasttransformer/convert_t5.sh .

RUN sed -i "s/PLACEHOLDERNUMGPU/$NUMGPU/g" $WORKSPACE/all_models/$MODELNAME/fastertransformer/config.pbtxt
RUN sed -i "s/PLACEHOLDERMODELNAME/$MODELNAME/g" $WORKSPACE/all_models/$MODELNAME/fastertransformer/config.pbtxt

RUN ./convert_existing_t5.sh $NUMGPU $MODELNAME

ENTRYPOINT /opt/tritonserver/bin/tritonserver --model-repository=/workspace/all_models/$MODEL_NAME_ENV/
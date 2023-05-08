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


FROM gcr.io/llm-containers/ft-triton:22.09

RUN pip3 install absl-py flask gcsfs transformers tritonclient[http]

ADD src/predict_triton.py .
ADD src/triton_processor.py .
ADD src/utils.py .

ENV FLASK_APP=predict
ENV SERVER_HOST=0.0.0.0

ADD src/app ./app

ENTRYPOINT ["/bin/python3", "predict_triton.py"]
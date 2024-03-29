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
FROM us-docker.pkg.dev/gce-ai-infra/cluster-provision-dev/cluster-provision-image:v0.6.0

RUN apt-get -yq install jq python3-distutils python3-pip gettext-base

RUN pip3 install yq google-cloud-storage absl-py

WORKDIR /usr
ENV PATH=/usr:$PATH
COPY src/gke/run_batch.sh .
RUN chmod 755 run_batch.sh
COPY src/gke/specs specs/
ENTRYPOINT ["run_batch.sh"]

ADD predict_payload.json .
ADD predict_result.json .
ADD predict_result_flan.json .
ADD predict_result_triton.json .
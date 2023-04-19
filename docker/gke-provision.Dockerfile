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
FROM us-central1-docker.pkg.dev/supercomputer-testing/cluster-provision-repo/cluster-provision-image:latest

RUN apt-get -yq install jq python3-distutils python3-pip

RUN pip3 install yq google-cloud-storage absl-py

COPY gke/run_batch.sh .
ENTRYPOINT run_batch.sh
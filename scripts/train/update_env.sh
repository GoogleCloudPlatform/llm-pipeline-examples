# #!/bin/bash -ev


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

echo "Configuring user..."
export HOME_USER=llm
export SERVICE_ACCOUNT=$(gcloud config get account)
export OS_LOGIN_USER=$(gcloud iam service-accounts describe ${SERVICE_ACCOUNT} | grep uniqueId | sed -e "s/.* '\(.*\)'/sa_\1/")
sudo usermod -l ${OS_LOGIN_USER} ${HOME_USER}
sudo groupmod -n ${OS_LOGIN_USER} ${HOME_USER}

echo 'export PATH="/opt/conda/bin:/opt/conda/condabin:${PATH}";conda activate base 2> /dev/null' >> .profile

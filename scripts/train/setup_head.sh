#!/bin/bash -ev

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

export ZONE=$1
echo Zone is ${ZONE}
export USER=$(whoami)
echo User is ${USER}

if [[ ! -e machines.txt ]]; then
  echo "No machines.txt file found!"
  exit 1
fi

export slots=$2
if [[ "$slots" == "16" ]]; then
  export slots=15
fi
while read -r machine ip;
do
  echo ${machine}
  while true; do
    if gcloud compute ssh ${machine} --internal-ip --zone=${ZONE} --command="bash -l -c 'pip list | grep deepspeed'" --ssh-key-expire-after=1d --strict-host-key-checking=no -- -p 1022 -n; then
      break;
    fi
    echo "Waiting for ssh..."
    sleep 10
  done
  echo ${machine} address is ${ip} . Setting up user ${USER}
  if [[ ! -e .ssh/config || -z "$(cat .ssh/config | grep ${machine})" ]]; then
    echo "Host ${machine}
  Hostname ${ip}
  User ${USER}
  Port 1022
  IdentityFile ~/.ssh/google_compute_engine

" >> .ssh/config
  ssh ${machine} -n -o StrictHostKeyChecking=no echo "Done ssh."
  fi
  
  if [[ ! -e deepspeed_hostfile || -z "$(cat deepspeed_hostfile | grep ${machine})" ]]; then
    echo "${machine} slots=$slots" >> deepspeed_hostfile
  fi
done < machines.txt

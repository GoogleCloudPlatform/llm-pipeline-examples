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
"""Utility module for preprocessing and fine tuning.

Utility module for preprocessing and fine tuning.
"""

import gcsfs

def gcs_path(path: str, gcs_prefix=''):
  if path.startswith('gs://'):
    fs = gcsfs.GCSFileSystem()
    return path.replace('gs://', gcs_prefix), fs
  if path.startswith('/gcs/'):
    fs = gcsfs.GCSFileSystem()
    return path.replace('/gcs/', gcs_prefix), fs
  return path, None
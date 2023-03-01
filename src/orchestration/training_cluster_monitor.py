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
"""Monitor training clusters.

Restarts training cluster if detects training not progressing.
"""
import os
import re
import datetime
import time
import subprocess
import sys

from absl import app
from absl import flags
from absl import logging
from google.cloud import storage

FLAGS = flags.FLAGS


flags.DEFINE_string('project_id', '', 'GCP project id.')
flags.DEFINE_string('model_output', '', 'GCS prefix to model output files.')
flags.DEFINE_integer(
    'train_ttl_sec',
    10800,
    (
        'Restarts training cluster if latest checkpoint file is'
        ' older than train_ttl_sec.'
    ),
)
flags.DEFINE_integer(
    'polling_sec', 30, 'Interval of polling checkpoint status.'
)
flags.DEFINE_integer(
    'max_reloads', 5, 'Max reloads of traininng cluster upon no progress.'
)

parent_id = os.getppid()


def _is_process_running(pid):
  try:
    os.kill(pid, 0)
  except OSError:
    return False
  return True


def _parse_gcs_resource_name(resource_name):
  matches = re.match('gs://(.*?)/(.*)', resource_name)
  return matches.groups()


def _get_latest_checkpoint_file_utime(bucket_name, model_output_prefix):
  storage_client = storage.Client()
  blobs = storage_client.list_blobs(bucket_name, prefix=model_output_prefix)
  latest_time = None
  for item in blobs:
    if re.search(r'checkpoint-\d+', item.name) and (
        not latest_time or latest_time < item.updated
    ):
      latest_time = item.updated
  return latest_time


def _get_latest_file_utime(bucket_name, model_output_prefix):
  storage_client = storage.Client()
  blobs = storage_client.list_blobs(bucket_name, prefix=model_output_prefix)
  latest_time = None
  for item in blobs:
    if not latest_time or latest_time < item.updated:
      latest_time = item.updated
  return latest_time


def _should_restart_training_cluster(
    latest_checkpoint_time, train_ttl_sec, num_reloads, max_reloads
):
  return (
      datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
      - latest_checkpoint_time
  ).total_seconds() >= train_ttl_sec and num_reloads < max_reloads


def _restart_training_cluster():
  os.environ['ACTION'] = 'DESTROY'
  logging.info('Deleting cluster: %s', os.getenv('NAME_PREFIX'))
  subprocess.run(
      ['bash', '/usr/entrypoint.sh'], capture_output=True, check=False
  )

  os.environ['ACTION'] = 'CREATE'

  logging.info('Starting cluster: %s', os.environ['NAME_PREFIX'])
  subprocess.run(
      ['bash', '/usr/entrypoint.sh'], capture_output=True, check=False
  )


def _write_fail_message(bucket_name, model_output_prefix):
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(os.path.join(model_output_prefix, 'progress.txt'))
  with blob.open('w') as f:
    f.write('failed')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Project id: %s', FLAGS.project_id)
  logging.info('train_ttl_sec: %s', FLAGS.train_ttl_sec)
  logging.info('Checkpoint Path: %s', FLAGS.checkpoint_path)

  os.environ['GCLOUD_PROJECT'] = FLAGS.project_id

  bucket_name, model_output_prefix = _parse_gcs_resource_name(
      FLAGS.model_output
  )

  num_reloads = 0

  while True:
    if not _is_process_running(parent_id):
      sys.exit(0)
    latest_checkpoint_time = _get_latest_checkpoint_file_utime(
        bucket_name, model_output_prefix
    )

    if latest_checkpoint_time is None:
      latest_checkpoint_time = _get_latest_file_utime(
          bucket_name, model_output_prefix
      )
    logging.info('Latest checkpoint file time: %s', latest_checkpoint_time)

    if latest_checkpoint_time is not None and _should_restart_training_cluster(
        latest_checkpoint_time,
        FLAGS.train_ttl_sec,
        num_reloads,
        FLAGS.max_reloads,
    ):
      logging.info(
          'Restarts training cluster, latest checkpoint too old at: %s',
          latest_checkpoint_time,
      )
      _restart_training_cluster()
      num_reloads += 1
      if num_reloads == FLAGS.max_reloads:
        _write_fail_message(bucket_name, model_output_prefix)
    time.sleep(FLAGS.polling_sec)


if __name__ == '__main__':
  app.run(main)

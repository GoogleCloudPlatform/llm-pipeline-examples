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
"""Deploys model to Vertex AI Prediciton

Deploys model to Vertex AI Prediction
"""
import google.cloud.aiplatform as aip
import gcsfs
import json
import os
from absl import app
from absl import flags
from absl import logging
from absl.flags import argparse_flags

FLAGS = flags.FLAGS

flags.DEFINE_string('project', None, 'Deployment project ID')
flags.DEFINE_string('model_display_name', None, 'Name of the deployed model')
flags.DEFINE_string('serving_container_image_uri', None, 'Serving containter image URI')
flags.DEFINE_string('model_path', None, 'GSC path of model to deploy')
flags.DEFINE_string('machine_type', None, 'Deployment machine type')
flags.DEFINE_string('gpu_type', None, 'Deployment GPU type')
flags.DEFINE_integer('gpu_count', 1, 'Number of deployment GPUs')
flags.DEFINE_string('endpoint_name', None, 'Name of deployment endpoint')
flags.DEFINE_string('endpoint_id', '432544312','Endpoint ID')
flags.DEFINE_string('region', None, 'Deployment region')

def main(argv):
  del argv

  endpoint_name = FLAGS.endpoint_name
  if not endpoint_name:
    endpoint_name = FLAGS.model_display_name

  existing_endpoints = aip.Endpoint.list(
      project=FLAGS.project,
      order_by="create_time",
      filter='endpoint="{}"'.format(FLAGS.endpoint_id),
      location=FLAGS.region)

  if existing_endpoints:
    endpoint = existing_endpoints[0]
    endpoint.undeploy_all()
  else:
    endpoint = aip.Endpoint.create(
        project=FLAGS.project,
        display_name=endpoint_name,
        location=FLAGS.region,
        endpoint_id=FLAGS.endpoint_id
    )

  existing_models = aip.Model.list(
      project=FLAGS.project,
      order_by="create_time",
      filter='display_name="{}"'.format(FLAGS.model_display_name),
      location=FLAGS.region)

  if existing_models:
    parent_model = existing_models[0]
    parent_model_resource_name = parent_model.resource_name
  else:
    parent_model_resource_name = None

  gcs = gcsfs.GCSFileSystem()
  new_metrics = None
  metrics_file = os.path.join(FLAGS.model_path, "metrics.json")
  if gcs.exists(metrics_file):
    with gcs.open(metrics_file, "r") as f:
      new_metrics = json.load(f)
      print(f"New metrics: {new_metrics}")
  else:
    new_metrics = {}
    print("Model doesn't have metrics. This is expected if your training container doesn't produce metrics.json. Otherwise, review training output.")

  deployable_model = aip.Model.upload(
      project=FLAGS.project,
      display_name=FLAGS.model_display_name,
      artifact_uri=FLAGS.model_path,
      serving_container_image_uri=FLAGS.serving_container_image_uri,
      parent_model=parent_model_resource_name,
      labels={
          k.lower(): str(v).replace(".", "_") for k, v in new_metrics.items()
      },
      serving_container_predict_route="/infer",
      serving_container_health_route="/health",
      location=FLAGS.region,
      upload_request_timeout=1200
  )

  endpoint.deploy(
      model=deployable_model,
      deployed_model_display_name=FLAGS.model_display_name,
      machine_type=FLAGS.machine_type,
      accelerator_type=FLAGS.gpu_type,
      accelerator_count=FLAGS.gpu_count)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  parser = argparse_flags.ArgumentParser(allow_abbrev=False)
  app.run(main, flags_parser=parser.parse_known_args)

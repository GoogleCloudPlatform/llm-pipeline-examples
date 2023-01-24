import json

from google.api import httpbody_pb2
from google.cloud import aiplatform_v1
from google.cloud import aiplatform

f = open('gcloud_configuration.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)

aiplatform.init(
    # your Google Cloud Project ID or number
    # environment default used is not set
    project=data["project"],

    # the Vertex AI region you will use
    # defaults to us-central1
    location=data["location"],

    # Google Cloud Storage bucket in same region as location
    # used to stage artifacts
    staging_bucket=data["staging_bucket"]
)

endpoint = aiplatform.Endpoint(f'projects/{data["project"]}/locations/{data["location"]}/endpoints/{data["endpoint_id"]}')

with open('raw_request') as request_f:
    body = httpbody_pb2.HttpBody()
    response = endpoint.raw_predict(request_f.read(), {"Inference-Header-Content-Length": 480})

#with open('json_request.json') as request_f:
#    response = endpoint.predict(request_f.read())


print(response.content)
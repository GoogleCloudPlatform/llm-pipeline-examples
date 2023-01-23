from google.cloud import aiplatform

import json
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

print(data["sample_request"])
response = endpoint.raw_predict(data["sample_request"], {})

print(response.content)
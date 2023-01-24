import logging
import contextlib
try:
    from http.client import HTTPConnection # py3
except ImportError:
    from httplib import HTTPConnection # py2

def debug_requests_on():
    '''Switches on logging of the requests module.'''
    HTTPConnection.debuglevel = 1

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

def debug_requests_off():
    '''Switches off logging of the requests module, might be some side-effects'''
    HTTPConnection.debuglevel = 0

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.handlers = []
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.WARNING)
    requests_log.propagate = False

@contextlib.contextmanager
def debug_requests():
    '''Use with 'with'!'''
    debug_requests_on()
    yield
    debug_requests_off()

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

with open('raw_request') as request_f:
    with debug_requests():
        response = endpoint.raw_predict(request_f.read(), {'    ': 'test', 'Inference-Header-Content-Length': '480', "SOMEHEADERTEST": "SOMEVALUETEST"})

#with open('json_request.json') as request_f:
#    response = endpoint.predict(request_f.read())


print(response.content)
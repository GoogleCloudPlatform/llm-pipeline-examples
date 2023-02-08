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

from tritonlocal import T5TritonProcessor

text = "Sandwiched between a second-hand bookstore and record shop in Cape Town's charmingly grungy suburb of Observatory is a blackboard reading 'Tapi Tapi -- Handcrafted, authentic African ice cream.' The parlor has become one of Cape Town's most talked about food establishments since opening in October 2020. And in its tiny kitchen, Jeff is creating ice cream flavors like no one else. Handwritten in black marker on the shiny kitchen counter are today's options: Salty kapenta dried fish (blitzed), toffee and scotch bonnet chile Sun-dried blackjack greens and caramel, Malted millet ,Hibiscus, cloves and anise. Using only flavors indigenous to the African continent, Guzha's ice cream has become the tool through which he is reframing the narrative around African food. 'This (is) ice cream for my identity, for other people's sake,' Jeff tells CNN. 'I think the (global) food story doesn't have much space for Africa ... unless we're looking at the generic idea of African food,' he adds. 'I'm not trying to appeal to the global universe -- I'm trying to help Black identities enjoy their culture on a more regular basis.'"
processor = T5TritonProcessor()
request_body, headers = processor._get_payload(text)

with debug_requests():
    response = endpoint.raw_predict(request_body, headers)
    print(response.content)

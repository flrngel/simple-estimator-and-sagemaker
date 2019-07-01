import boto3
import json
import base64
from misc.convert_image import load_image, serialize_example

import base64

client = boto3.client('sagemaker-runtime')

endpoint_name = "<SageMaker_Endpoint>"
content_type = "application/json"

b64 = base64.b64encode(serialize_example(load_image('./misc/test.jpg')))
payload = {
    'inputs': {
        "images": [{
            'b64': b64.decode('utf-8')
        }]
    }
}


payload = json.dumps(payload)

response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType=content_type,
    Body=payload,
)

print(response)
print('--------------------------')
print(response['Body'].read())

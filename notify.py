import requests
import secrets

def send(header, description):
    URL = secrets.url
    data = {
        'value1': header,
        'value2': description
    }
    r = requests.post(URL, data)
    return r
import requests
url = "http://127.0.0.1:5001/invocations"
payload = {"inputs": [[5.1,3.5,1.4,0.2]]}
r = requests.post(url, json=payload, timeout=5)
print(r.status_code, r.text)

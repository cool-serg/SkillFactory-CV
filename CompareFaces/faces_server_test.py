import base64
import requests

with open("reference.jpg", "rb") as f:
    image_bytes = f.read()

image_base64 = base64.b64encode(image_bytes).decode("utf-8")

url = "http://127.0.0.1:5000/face-check"
payload = {
    "image_base64": image_base64,
    "timeout": 10
}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=payload, headers=headers)

print("Status code:", response.status_code)
print("Response:", response.text)

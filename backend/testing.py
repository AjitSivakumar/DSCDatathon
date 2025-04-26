import requests

query = {"query": "What are some good coffee places nearby?"}

response = requests.post("http://127.0.0.1:5000/ask", json=query)

print("Status code:", response.status_code)
print("Response JSON:", response.json())

import requests
import os

API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "mixtral-8x7b-32768"
URL = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Hello, are you working?"}],
    "temperature": 0
}

try:
    response = requests.post(URL, headers=headers, json=payload, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {str(e)}")

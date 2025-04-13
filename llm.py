import time
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def llm(model="gpt2", system_prompt=None, user_prompt=None, assistant_prompt=None, params=None):
    headers = {
        "Authorization": os.getenv('Authorization'),
    }

    # Combine prompts into a single string
    prompt = ""
    if system_prompt:
        prompt += f"System: {system_prompt[:100]}\n"  # Truncate system prompt
    if assistant_prompt:
        prompt += f"Assistant: {assistant_prompt[:100]}\n"  # Truncate assistant prompt
    if user_prompt:
        prompt += f"Human: {user_prompt[:500]}\n"  # Truncate user prompt
    prompt += "Assistant:"

    body = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 50,  # Reduced from 100
            "temperature": 0.7,
            **(params or {})
        }
    }

    response = requests.post(f"https://api-inference.huggingface.co/models/{model}", headers=headers, json=body)

    if response.status_code == 200:
        return response.json()[0].get("generated_text", "").split("Assistant:")[-1].strip()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")
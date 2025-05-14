import requests
from config.settings import API_KEY, LLAMA_API_URL
import json
import re

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def call_llama(prompt, temperature=0.5):
    """
    Make an API call to LLaMA with the given prompt.
    
    Args:
        prompt (str): The prompt to send to the LLaMA API
        temperature (float, optional): Temperature parameter for generation. Defaults to 0.5.
    
    Returns:
        str: The response from LLaMA
    """
    data = {
        "model": "llama3.1-70b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False
    }
    response = requests.post(LLAMA_API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def extract_json_from_response(response_text):
    """
    Extract JSON from LLaMA response text.
    
    Args:
        response_text (str): The text response from LLaMA
    
    Returns:
        dict: Extracted JSON data or default fallback if extraction fails
    """
    json_pattern = r'({[\s\S]*})'
    json_match = re.search(json_pattern, response_text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(0).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            cleaned_json = re.sub(r'[\n\r\t]', ' ', json_str)
            cleaned_json = re.sub(r',\s*}', '}', cleaned_json)
            try:
                return json.loads(cleaned_json)
            except:
                return None
    return None
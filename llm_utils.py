from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Constants
PRIMARY_MODEL = 'gpt-4o'
MINI_MODEL = "gpt-3.5-turbo"
# MINI_MODEL = 'gpt-4o-mini'
EMBEDDING_MODEL_LARGE = "text-embedding-3-large"
EMBEDDING_MODEL_SMALL = "text-embedding-3-small"
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_KEY)


def get_embedding(texts, size=EMBEDDING_MODEL_LARGE):
    """
    Get embeddings for the given texts using the specified model size.

    Args:
        texts (list): List of text strings to embed.
        size (str): Model size to use for embedding. Defaults to EMBEDDING_MODEL_SMALL.

    Returns:
        list: List of embeddings (dimensions: 1536 for small, 3072 for large) for the input texts.
    """
    cleaned_texts = [text.replace('\n', ' ').replace('\t', ' ').strip() for text in texts if text]
    response = openai_client.embeddings.create(input=cleaned_texts, model=size)
    return [item.embedding for item in response.data]

def llm(model=MINI_MODEL, system_prompt=None, user_prompt=None, assistant_prompt=None, params=None):
    """
    Generate a response using the OpenAI language model.

    Args:
        model (str): The model to use for generation. Defaults to MINI_MODEL.
        system_prompt (str, optional): The system prompt to use.
        user_prompt (str, optional): The user prompt to use.
        assistant_prompt (str, optional): The assistant prompt to use.
        params (dict, optional): Additional parameters for the API call.

    Returns:
        str: The generated response from the language model.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if assistant_prompt:
        messages.append({"role": "assistant", "content": assistant_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})

    body = {
        "model": model,
        "messages": messages,
        "temperature": 0
    }

    if params:
        body.update(params)

    response = openai_client.chat.completions.create(**body)
    return response.choices[0].message.content
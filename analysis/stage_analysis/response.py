# Python Standard Library
import re
import logging
# Commonly Used Open-Source Libraries
from openai import OpenAI

# Set up api key and api base
openai_api_key = ""
openai_api_base = ""
logging.basicConfig(level=logging.ERROR)
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
def get_response(prompt):
    """
    Used to get the response from the api LLM
    """
    chat_response = client.chat.completions.create(
            model="",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
    return chat_response.choices[-1].message.content

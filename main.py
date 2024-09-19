"""An example showing how to use vLLM to serve VLMs.

Launch the vLLM server with the following command:
vllm serve fixie-ai/ultravox-v0_3
"""

import os
import base64
import requests
from dotenv import load_dotenv


from openai import OpenAI
from vllm.assets.audio import AudioAsset

load_dotenv()
# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = "http://localhost:8000/v1"

audios = {
    1: "https://cdn.discordapp.com/attachments/1285851100615938106/1286386611805753414/cr1.mp3?ex=66edb864&is=66ec66e4&hm=129a5f7bee0e7708159344cd220ae35bfdbf318d00e44c95b5596ae83ba64bbc&",
    2: "https://cdn.discordapp.com/attachments/1285851100615938106/1286386649424596992/lm1.mp3?ex=66edb86d&is=66ec66ed&hm=7fc6fedb4113dc9fbbdae6b52d99562afa9260ab7f307052c6ae3f888e255f04&",
}

messages = []
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


# Use base64 encoded audio in the payload
def encode_audio_base64_from_url(audio_url: str) -> str:
    """Encode an audio retrieved from a remote url to base64 format."""

    with requests.get(audio_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result


for i in range(1, 3):
    audio_base64 = encode_audio_base64_from_url(audio_url=audios[i])
    if i == 1:
        chat_completion_from_base64 = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this audio?"},
                        {
                            "type": "audio_url",
                            "audio_url": {
                                # Any format supported by librosa is supported
                                "url": f"data:audio/ogg;base64,{audio_base64}"
                            },
                        },
                    ],
                }
            ],
            model=model,
            max_tokens=64,
        )
        response_from_1 = chat_completion_from_base64.choices[0].message.content
        question_1 = "what is the audio about?"
    else:
        chat_completion_from_base64 = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f'''There was a audio sent to you previously in a different chat and was asked: {question_1} and you replied with: {response_from_1}.
                             Now based on this history answer the following:
                             How is the previous person related to the one in this audio?''',
                        },
                        {
                            "type": "audio_url",
                            "audio_url": {
                                # Any format supported by librosa is supported
                                "url": f"data:audio/ogg;base64,{audio_base64}"
                            },
                        },
                    ],
                },
            ],
            model=model,
            max_tokens=64,
        )

    result = chat_completion_from_base64.choices[0].message.content
    print(f"Chat completion output:{result}")

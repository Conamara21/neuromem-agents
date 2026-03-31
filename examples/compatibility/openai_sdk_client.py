"""
Example: use the NeuroMem proxy through the official OpenAI Python SDK.
"""

from openai import OpenAI


client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="neuromem-local",
)


response = client.chat.completions.create(
    model="your-upstream-model",
    messages=[
        {"role": "system", "content": "You are a helpful engineering assistant."},
        {"role": "user", "content": "Summarize the last architecture decision."},
    ],
    extra_body={
        "neuromem": {
            "session_id": "demo-project",
            "top_k": 5,
            "store_messages": True,
        }
    },
)

print(response.choices[0].message.content)

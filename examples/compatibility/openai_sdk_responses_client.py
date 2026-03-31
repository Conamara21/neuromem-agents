"""
Example: use the NeuroMem proxy through the OpenAI Responses API.
"""

from openai import OpenAI


client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="neuromem-local",
)


response = client.responses.create(
    model="your-upstream-model",
    input="What did we decide about the retrieval pipeline last week?",
    extra_body={
        "neuromem": {
            "session_id": "demo-project",
            "top_k": 5,
        }
    },
)

print(response.output_text)

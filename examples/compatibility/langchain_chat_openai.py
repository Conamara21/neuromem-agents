"""
Example: point LangChain ChatOpenAI at the NeuroMem proxy.
"""

from neuromem.frameworks import create_langchain_chat_openai


llm = create_langchain_chat_openai(
    model="your-upstream-model",
    base_url="http://127.0.0.1:8080/v1",
    api_key="neuromem-local",
)

response = llm.invoke("Summarize the latest project memory decision.")
print(response.content)

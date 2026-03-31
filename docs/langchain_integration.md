# LangChain Integration

NeuroMem includes a first-party LangChain adapter so you can use session-scoped memory as a retriever and point LangChain's OpenAI-compatible clients at the NeuroMem proxy.

## Installation

```bash
pip install 'neuromem-agents[langchain]'
```

## NeuroMem Retriever

```python
from neuromem.frameworks import NeuroMemRetriever

retriever = NeuroMemRetriever(
    settings_path="examples/configs/openai_proxy.example.json",
    session_id="demo-project",
    top_k=5,
)

documents = retriever.invoke("architecture decision")
```

Each returned LangChain `Document` includes metadata such as:

- `memory_id`
- `memory_type`
- `timestamp`
- `tags`
- `access_frequency`

## ChatOpenAI Through The NeuroMem Proxy

```python
from neuromem.frameworks import create_langchain_chat_openai

llm = create_langchain_chat_openai(
    model="your-upstream-model",
    base_url="http://127.0.0.1:8080/v1",
    api_key="neuromem-local",
)

response = llm.invoke("Summarize the latest project memory decision.")
```

This keeps the application inside the standard LangChain OpenAI-compatible path while NeuroMem handles memory retrieval and storage behind the proxy.

## Example Files

- `examples/compatibility/langchain_retriever.py`
- `examples/compatibility/langchain_chat_openai.py`

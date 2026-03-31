# LlamaIndex Integration

NeuroMem includes a first-party LlamaIndex adapter so you can use session-scoped memory as a retriever and build a query engine against the NeuroMem proxy.

## Installation

```bash
pip install 'neuromem-agents[llamaindex]'
```

## NeuroMem Retriever

```python
from neuromem.frameworks import NeuroMemLlamaIndexRetriever

retriever = NeuroMemLlamaIndexRetriever(
    settings_path="examples/configs/openai_proxy.example.json",
    session_id="demo-project",
    top_k=5,
)

nodes = retriever.retrieve("architecture decision")
```

Each returned `NodeWithScore` contains a `TextNode` with metadata such as:

- `memory_id`
- `memory_type`
- `timestamp`
- `tags`
- `access_frequency`

## Query Engine Through The NeuroMem Proxy

```python
from neuromem.frameworks import (
    NeuroMemLlamaIndexRetriever,
    create_llamaindex_openai_like,
    create_llamaindex_query_engine,
)

retriever = NeuroMemLlamaIndexRetriever(
    settings_path="examples/configs/openai_proxy.example.json",
    session_id="demo-project",
    top_k=5,
)

llm = create_llamaindex_openai_like(
    model="your-upstream-model",
    base_url="http://127.0.0.1:8080/v1",
    api_key="neuromem-local",
)

query_engine = create_llamaindex_query_engine(retriever, llm=llm)
response = query_engine.query("Summarize the latest project memory decision.")
```

This keeps the application inside the standard LlamaIndex retriever and query-engine path while NeuroMem handles memory retrieval and storage behind the proxy.

## Example Files

- `examples/compatibility/llamaindex_retriever.py`
- `examples/compatibility/llamaindex_query_engine.py`

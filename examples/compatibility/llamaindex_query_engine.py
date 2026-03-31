"""
Example: use NeuroMem retriever + proxy-backed LlamaIndex query engine.
"""

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
response = query_engine.query("Summarize the latest architecture decision.")
print(str(response))

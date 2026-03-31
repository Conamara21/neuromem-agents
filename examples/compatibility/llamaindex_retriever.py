"""
Example: use NeuroMem as a LlamaIndex retriever.
"""

from neuromem.frameworks import NeuroMemLlamaIndexRetriever


retriever = NeuroMemLlamaIndexRetriever(
    settings_path="examples/configs/openai_proxy.example.json",
    session_id="demo-project",
    top_k=5,
)

nodes = retriever.retrieve("What did we decide about the architecture?")
for item in nodes:
    print(item.node.text)
    print(item.node.metadata)

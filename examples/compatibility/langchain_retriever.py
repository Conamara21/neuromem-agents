"""
Example: use NeuroMem as a LangChain retriever.
"""

from neuromem.frameworks import NeuroMemRetriever


retriever = NeuroMemRetriever(
    settings_path="examples/configs/openai_proxy.example.json",
    session_id="demo-project",
    top_k=5,
)

documents = retriever.invoke("What did we decide about the architecture?")
for document in documents:
    print(document.page_content)
    print(document.metadata)

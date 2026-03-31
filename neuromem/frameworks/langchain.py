"""
LangChain adapters for NeuroMem.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..integrations.config import load_settings
from ..integrations.engine import MemoryAugmentedProxy

try:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict, Field, PrivateAttr
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    BaseRetriever = object
    Document = None
    CallbackManagerForRetrieverRun = Any
    AsyncCallbackManagerForRetrieverRun = Any
    Field = None
    PrivateAttr = None
    ConfigDict = None
    _LANGCHAIN_IMPORT_ERROR = exc
else:
    _LANGCHAIN_IMPORT_ERROR = None


def _require_langchain() -> None:
    if _LANGCHAIN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "LangChain integration requires `langchain-core`. "
            "Install with `pip install 'neuromem-agents[langchain]'`."
        ) from _LANGCHAIN_IMPORT_ERROR


if _LANGCHAIN_IMPORT_ERROR is not None:

    class NeuroMemRetriever(BaseRetriever):
        """Placeholder retriever that raises a clear dependency error."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_langchain()

else:

    class NeuroMemRetriever(BaseRetriever):
        """LangChain retriever backed by NeuroMem session-scoped memory."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        settings_path: Optional[str] = None
        session_id: str = "default"
        top_k: int = 5
        tags: List[str] = Field(default_factory=list)
        memory_type: Optional[str] = None
        include_scores: bool = True
        _proxy: MemoryAugmentedProxy = PrivateAttr()

        def model_post_init(self, __context: Any) -> None:
            self._proxy = MemoryAugmentedProxy(load_settings(self.settings_path))

        def _result_to_document(self, item: dict[str, Any]) -> Document:
            metadata = {
                "memory_id": item.get("id"),
                "memory_type": item.get("memory_type"),
                "timestamp": item.get("timestamp"),
                "tags": item.get("tags", []),
                "access_frequency": item.get("access_frequency"),
            }
            if self.include_scores and "score" in item:
                metadata["score"] = item["score"]
            return Document(page_content=item.get("content", ""), metadata=metadata)

        def _filter_items(self, items: List[dict[str, Any]]) -> List[dict[str, Any]]:
            if not self.memory_type:
                return items
            normalized_memory_type = self.memory_type.strip().lower()
            return [
                item
                for item in items
                if str(item.get("memory_type", "")).strip().lower() == normalized_memory_type
            ]

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        ) -> List[Document]:
            items = self._proxy.search_memory(
                query=query,
                session_id=self.session_id,
                top_k=self.top_k,
                tags=self.tags,
            )
            return [self._result_to_document(item) for item in self._filter_items(items)]

        async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
        ) -> List[Document]:
            return self._get_relevant_documents(query)


def create_langchain_chat_openai(
    *,
    model: str,
    base_url: str = "http://127.0.0.1:8080/v1",
    api_key: str = "neuromem-local",
    **kwargs: Any,
):
    """Create a LangChain ChatOpenAI client pointing at the NeuroMem proxy."""

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "ChatOpenAI helper requires `langchain-openai`. "
            "Install with `pip install 'neuromem-agents[langchain]'`."
        ) from exc

    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key, **kwargs)

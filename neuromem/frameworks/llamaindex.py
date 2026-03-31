"""
LlamaIndex adapters for NeuroMem.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..integrations.config import load_settings
from ..integrations.engine import MemoryAugmentedProxy

try:
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    RetrieverQueryEngine = None
    BaseRetriever = object
    NodeWithScore = None
    QueryBundle = Any
    TextNode = None
    _LLAMAINDEX_IMPORT_ERROR = exc
else:
    _LLAMAINDEX_IMPORT_ERROR = None


def _require_llamaindex() -> None:
    if _LLAMAINDEX_IMPORT_ERROR is not None:
        raise RuntimeError(
            "LlamaIndex integration requires `llama-index-core` and "
            "`llama-index-llms-openai-like`. "
            "Install with `pip install 'neuromem-agents[llamaindex]'`."
        ) from _LLAMAINDEX_IMPORT_ERROR


if _LLAMAINDEX_IMPORT_ERROR is not None:

    class NeuroMemLlamaIndexRetriever(BaseRetriever):
        """Placeholder retriever that raises a clear dependency error."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_llamaindex()

else:

    class NeuroMemLlamaIndexRetriever(BaseRetriever):
        """LlamaIndex retriever backed by NeuroMem session-scoped memory."""

        def __init__(
            self,
            *,
            settings_path: Optional[str] = None,
            session_id: str = "default",
            top_k: int = 5,
            tags: Optional[Sequence[str]] = None,
            memory_type: Optional[str] = None,
            include_scores: bool = True,
            callback_manager: Optional[Any] = None,
            object_map: Optional[dict[str, Any]] = None,
            objects: Optional[List[Any]] = None,
            verbose: bool = False,
        ) -> None:
            super().__init__(
                callback_manager=callback_manager,
                object_map=object_map,
                objects=objects,
                verbose=verbose,
            )
            self.settings_path = settings_path
            self.session_id = session_id
            self.top_k = int(top_k)
            self.tags = list(tags or [])
            self.memory_type = memory_type
            self.include_scores = include_scores
            self._proxy = MemoryAugmentedProxy(load_settings(settings_path))

        def _filter_items(self, items: List[dict[str, Any]]) -> List[dict[str, Any]]:
            if not self.memory_type:
                return items
            normalized_memory_type = self.memory_type.strip().lower()
            return [
                item
                for item in items
                if str(item.get("memory_type", "")).strip().lower() == normalized_memory_type
            ]

        def _result_to_node(self, item: dict[str, Any]) -> NodeWithScore:
            metadata = {
                "memory_id": item.get("id"),
                "memory_type": item.get("memory_type"),
                "timestamp": item.get("timestamp"),
                "tags": item.get("tags", []),
                "access_frequency": item.get("access_frequency"),
            }
            score = item.get("score") if self.include_scores else None
            return NodeWithScore(
                node=TextNode(
                    text=item.get("content", ""),
                    metadata=metadata,
                    id_=str(item.get("id", "")),
                ),
                score=float(score) if score is not None else None,
            )

        def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            query = getattr(query_bundle, "query_str", str(query_bundle))
            items = self._proxy.search_memory(
                query=query,
                session_id=self.session_id,
                top_k=self.top_k,
                tags=self.tags,
            )
            return [self._result_to_node(item) for item in self._filter_items(items)]

        async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            return self._retrieve(query_bundle)


def create_llamaindex_openai_like(
    *,
    model: str,
    base_url: str = "http://127.0.0.1:8080/v1",
    api_key: str = "neuromem-local",
    is_chat_model: bool = True,
    **kwargs: Any,
):
    """Create a LlamaIndex OpenAI-like client pointing at the NeuroMem proxy."""

    _require_llamaindex()
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=model,
        api_base=base_url,
        api_key=api_key,
        is_chat_model=is_chat_model,
        **kwargs,
    )


def create_llamaindex_query_engine(
    retriever: Any,
    *,
    llm: Optional[Any] = None,
    **kwargs: Any,
):
    """Create a RetrieverQueryEngine around a NeuroMem LlamaIndex retriever."""

    _require_llamaindex()
    return RetrieverQueryEngine.from_args(retriever=retriever, llm=llm, **kwargs)

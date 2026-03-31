"""
FastAPI entry point for NeuroMem's OpenAI-compatible proxy server.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    FastAPI = None
    HTTPException = None
    Request = None
    CORSMiddleware = None
    PlainTextResponse = None
    _FASTAPI_IMPORT_ERROR = exc
else:
    _FASTAPI_IMPORT_ERROR = None

from ..integrations.config import load_settings
from ..integrations.engine import MemoryAugmentedProxy


def create_app(config_path: Optional[str] = None, settings=None):
    if _FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "FastAPI is required for the compatibility server. "
            "Install with `pip install 'neuromem-agents[server]'`."
        ) from _FASTAPI_IMPORT_ERROR

    resolved_settings = settings or load_settings(config_path)
    service = MemoryAugmentedProxy(resolved_settings)

    app = FastAPI(
        title="NeuroMem OpenAI-Compatible Proxy",
        version="0.2.0",
        description="OpenAI-compatible proxy with NeuroMem memory augmentation.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.server.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def read_json(request: Request) -> Dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON request body must be an object.")
        return payload

    def handle_error(exc: Exception) -> None:
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=400, detail=str(exc))
        raise HTTPException(status_code=502, detail=str(exc))

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "memory_nodes": len(service.memory.memory_nodes),
            "chat_provider": (
                resolved_settings.chat_model.provider if resolved_settings.chat_model else None
            ),
            "embedding_provider": resolved_settings.embedding_model.provider,
        }

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return service.list_models()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Dict[str, Any]:
        payload = await read_json(request)
        try:
            return service.chat_completions_api(payload)
        except Exception as exc:
            handle_error(exc)

    @app.post("/v1/responses")
    async def responses(request: Request) -> Dict[str, Any]:
        payload = await read_json(request)
        try:
            return service.responses_api(payload)
        except Exception as exc:
            handle_error(exc)

    @app.post("/v1/memory/records")
    async def create_memory_record(request: Request) -> Dict[str, Any]:
        payload = await read_json(request)
        content = str(payload.get("content", "")).strip()
        if not content:
            raise HTTPException(status_code=400, detail="`content` is required.")
        try:
            return service.create_memory(
                content,
                session_id=payload.get("session_id"),
                memory_type=payload.get("memory_type"),
                tags=payload.get("tags", []),
            )
        except Exception as exc:
            handle_error(exc)

    @app.post("/v1/memory/search")
    async def search_memory(request: Request) -> Dict[str, Any]:
        payload = await read_json(request)
        query = str(payload.get("query", "")).strip()
        if not query:
            raise HTTPException(status_code=400, detail="`query` is required.")
        try:
            data = service.search_memory(
                query,
                session_id=payload.get("session_id"),
                top_k=payload.get("top_k"),
                tags=payload.get("tags", []),
            )
            return {
                "object": "list",
                "data": data,
                "count": len(data),
            }
        except Exception as exc:
            handle_error(exc)

    @app.get("/v1/memory/stats")
    async def memory_stats() -> Dict[str, Any]:
        return service.memory_statistics()

    @app.get("/v1/metrics")
    async def metrics_json() -> Dict[str, Any]:
        return service.metrics_snapshot()

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics_prometheus() -> PlainTextResponse:
        return PlainTextResponse(service.metrics_prometheus())

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the NeuroMem OpenAI-compatible proxy server."
    )
    parser.add_argument("--config", help="Path to a JSON settings file.")
    parser.add_argument("--host", help="Override bind host.")
    parser.add_argument("--port", type=int, help="Override bind port.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload mode.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.host:
        settings.server.host = args.host
    if args.port:
        settings.server.port = args.port
    if args.reload:
        settings.server.reload = True

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "uvicorn is required to run the compatibility server. "
            "Install with `pip install 'neuromem-agents[server]'`."
        ) from exc

    app = create_app(settings=settings)
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        log_level=settings.server.log_level,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

"""
Small JSON-over-HTTP helpers for provider integrations.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ProviderHTTPError(RuntimeError):
    """Raised when an upstream provider request fails."""


def request_json(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    query: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    final_url = url
    if query:
        encoded_query = urlencode(query)
        joiner = "&" if "?" in final_url else "?"
        final_url = "{0}{1}{2}".format(final_url, joiner, encoded_query)

    request_headers = {
        "Accept": "application/json",
    }
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    else:
        body = None

    if headers:
        request_headers.update(headers)

    request = Request(final_url, data=body, headers=request_headers, method=method.upper())

    try:
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
            if not raw_body:
                return {}
            return json.loads(raw_body)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderHTTPError(
            "Provider request failed with status {0}: {1}".format(exc.code, detail)
        )
    except URLError as exc:
        raise ProviderHTTPError("Provider request failed: {0}".format(exc))


def get_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
    query: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    return request_json("GET", url, headers=headers, timeout=timeout, query=query)


def post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
    query: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    return request_json(
        "POST",
        url,
        headers=headers,
        payload=payload,
        timeout=timeout,
        query=query,
    )

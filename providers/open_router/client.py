"""OpenRouter provider implementation."""

import json
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

from providers.base import BaseProvider, ProviderConfig
from providers.common import (
    SSEBuilder,
    append_request_id,
    get_user_facing_error_message,
    map_error,
)
from providers.rate_limit import GlobalRateLimiter

from .request import build_request_body

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class _SSEFilterState:
    """Track Anthropic content block index remapping while filtering thinking."""

    next_index: int = 0
    index_map: dict[int, int] = field(default_factory=dict)
    dropped_indexes: set[int] = field(default_factory=set)


class OpenRouterProvider(BaseProvider):
    """OpenRouter provider using the Anthropic-compatible messages API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._provider_name = "OPENROUTER"
        self._api_key = config.api_key
        self._base_url = (config.base_url or OPENROUTER_BASE_URL).rstrip("/")
        self._global_rate_limiter = GlobalRateLimiter.get_instance(
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            proxy=config.proxy or None,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        await self._client.aclose()

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and direct request dispatch."""
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request),
        )

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        """Create a streaming messages response from OpenRouter."""
        request = self._client.build_request(
            "POST",
            "/messages",
            json=body,
            headers={
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "anthropic-version": _ANTHROPIC_VERSION,
            },
        )
        return await self._client.send(request, stream=True)

    @staticmethod
    def _format_sse_event(event_name: str | None, data_text: str) -> str:
        """Format an SSE event from its event name and data payload."""
        lines: list[str] = []
        if event_name:
            lines.append(f"event: {event_name}")
        lines.extend(f"data: {line}" for line in data_text.splitlines())
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _parse_sse_event(event: str) -> tuple[str | None, str]:
        """Extract the event name and raw data payload from an SSE event."""
        event_name = None
        data_lines: list[str] = []
        for line in event.strip().splitlines():
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        return event_name, "\n".join(data_lines)

    @staticmethod
    def _remap_index(
        payload: dict[str, Any], state: _SSEFilterState, *, create: bool
    ) -> int | None:
        """Return the downstream index for a content block event."""
        upstream_index = payload.get("index")
        if not isinstance(upstream_index, int):
            return None
        if upstream_index in state.dropped_indexes:
            return None
        mapped_index = state.index_map.get(upstream_index)
        if mapped_index is None and create:
            mapped_index = state.next_index
            state.index_map[upstream_index] = mapped_index
            state.next_index += 1
        return mapped_index

    def _filter_sse_event(self, event: str, state: _SSEFilterState) -> str | None:
        """Drop upstream thinking blocks and remap the remaining block indexes."""
        event_name, data_text = self._parse_sse_event(event)
        if not event_name or not data_text:
            return event

        try:
            payload = json.loads(data_text)
        except json.JSONDecodeError:
            return event

        if event_name == "content_block_start":
            block = payload.get("content_block")
            block_type = block.get("type") if isinstance(block, dict) else None
            upstream_index = payload.get("index")
            if isinstance(block_type, str) and "thinking" in block_type:
                if isinstance(upstream_index, int):
                    state.dropped_indexes.add(upstream_index)
                return None

            mapped_index = self._remap_index(payload, state, create=True)
            if mapped_index is not None:
                payload["index"] = mapped_index
            return self._format_sse_event(event_name, json.dumps(payload))

        if event_name == "content_block_delta":
            delta = payload.get("delta")
            delta_type = delta.get("type") if isinstance(delta, dict) else None
            if isinstance(delta_type, str) and "thinking" in delta_type:
                return None

            mapped_index = self._remap_index(payload, state, create=False)
            if mapped_index is None:
                return None
            payload["index"] = mapped_index
            return self._format_sse_event(event_name, json.dumps(payload))

        if event_name == "content_block_stop":
            mapped_index = self._remap_index(payload, state, create=False)
            if mapped_index is None:
                return None
            payload["index"] = mapped_index
            return self._format_sse_event(event_name, json.dumps(payload))

        return event

    async def _iter_sse_events(self, response: httpx.Response) -> AsyncIterator[str]:
        """Group line-delimited SSE responses into full SSE events."""
        event_lines: list[str] = []
        async for line in response.aiter_lines():
            if line:
                event_lines.append(line)
                continue
            if event_lines:
                yield "\n".join(event_lines) + "\n\n"
                event_lines.clear()
        if event_lines:
            yield "\n".join(event_lines) + "\n\n"

    def _emit_error_events(
        self,
        *,
        model: str,
        input_tokens: int,
        error_message: str,
        include_message_start: bool,
    ) -> Iterator[str]:
        """Emit the existing Anthropic SSE error shape."""
        sse = SSEBuilder(f"msg_{uuid.uuid4()}", model, input_tokens)
        if include_message_start:
            yield sse.message_start()
        yield from sse.emit_error(error_message)
        yield sse.message_delta("end_turn", 1)
        yield sse.message_stop()

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream response via OpenRouter's Anthropic-compatible endpoint."""
        tag = self._provider_name
        req_tag = f" request_id={request_id}" if request_id else ""
        thinking_enabled = self._is_thinking_enabled(request)
        body = self._build_request_body(request)

        logger.info(
            "{}_STREAM:{} model={} msgs={} tools={}",
            tag,
            req_tag,
            body.get("model"),
            len(body.get("messages", [])),
            len(body.get("tools", [])),
        )

        response: httpx.Response | None = None
        state = _SSEFilterState()
        sent_any_event = False

        async with self._global_rate_limiter.concurrency_slot():
            try:
                response = await self._global_rate_limiter.execute_with_retry(
                    self._send_stream_request, body
                )

                if response.status_code != 200:
                    response.raise_for_status()

                async for event in self._iter_sse_events(response):
                    output_event = event
                    if not thinking_enabled:
                        output_event = self._filter_sse_event(event, state)
                    if output_event is None:
                        continue
                    sent_any_event = True
                    yield output_event

            except Exception as error:
                logger.error(
                    "{}_ERROR:{} {}: {}", tag, req_tag, type(error).__name__, error
                )
                mapped_error = map_error(error)
                if getattr(mapped_error, "status_code", None) == 405:
                    base_message = (
                        f"Upstream provider {tag} rejected the request method "
                        "or endpoint (HTTP 405)."
                    )
                else:
                    base_message = get_user_facing_error_message(
                        mapped_error, read_timeout_s=self._config.http_read_timeout
                    )
                error_message = append_request_id(base_message, request_id)

                if response is not None and not response.is_closed:
                    await response.aclose()

                for event in self._emit_error_events(
                    model=request.model,
                    input_tokens=input_tokens,
                    error_message=error_message,
                    include_message_start=not sent_any_event,
                ):
                    yield event
                return
            finally:
                if response is not None and not response.is_closed:
                    await response.aclose()

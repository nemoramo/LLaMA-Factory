#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gRPC client for Endpointing service with retry logic and connection pooling.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import grpc

from pb import endpointing_pb2, endpointing_pb2_grpc


@dataclass
class EndpointingResult:
    """Result from endpointing prediction."""

    label: str
    confidence: float
    p_eou: float
    p_cont_user: float
    p_unaddressed: float
    latency_ms: int
    model: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "p_eou": self.p_eou,
            "p_cont_user": self.p_cont_user,
            "p_unaddressed": self.p_unaddressed,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "error": self.error,
        }


class EndpointingClient:
    """Client for Endpointing gRPC service with retry support."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 50051,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[endpointing_pb2_grpc.EndpointingServiceStub] = None

    def _get_stub(self) -> endpointing_pb2_grpc.EndpointingServiceStub:
        """Get or create gRPC stub."""
        if self._stub is None:
            self._channel = grpc.insecure_channel(f"{self.host}:{self.port}")
            self._stub = endpointing_pb2_grpc.EndpointingServiceStub(self._channel)
        return self._stub

    def close(self):
        """Close the gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def _parse_history(self, history_json: Optional[str]) -> List[endpointing_pb2.HistoryTurn]:
        """Parse JSON history string to HistoryTurn list."""
        if not history_json or not history_json.strip():
            return []

        try:
            history_list = json.loads(history_json)
            if not isinstance(history_list, list):
                return []

            turns = []
            for item in history_list:
                if not isinstance(item, dict):
                    continue
                role_str = str(item.get("role", "")).lower()
                text = str(item.get("text", ""))

                if role_str == "user":
                    role = endpointing_pb2.Role.USER
                elif role_str == "assistant":
                    role = endpointing_pb2.Role.ASSISTANT
                else:
                    continue

                turns.append(endpointing_pb2.HistoryTurn(role=role, text=text))

            return turns
        except json.JSONDecodeError:
            return []

    def predict(
        self,
        asr_text: str,
        history_json: Optional[str] = None,
        lang: str = "en-US",
        eou_threshold: float = 0.6,
        treat_unaddressed_as_eou: bool = True,
        request_id: Optional[str] = None,
    ) -> EndpointingResult:
        """
        Make a single prediction with retry logic.

        Args:
            asr_text: The ASR transcription text.
            history_json: JSON string of conversation history.
            lang: Language code (default: en-US).
            eou_threshold: Threshold for EOU decision (default: 0.6).
            treat_unaddressed_as_eou: Whether to treat UNADDRESSED as EOU.
            request_id: Optional request ID for tracking.

        Returns:
            EndpointingResult with prediction or error.
        """
        stub = self._get_stub()

        # Build request
        history_turns = self._parse_history(history_json)

        request = endpointing_pb2.EndpointingRequest(
            request_id=request_id or "",
            lang=lang,
            asr=endpointing_pb2.Asr(text=asr_text or ""),
            history=history_turns,
            options=endpointing_pb2.Options(
                treat_unaddressed_as_eou=treat_unaddressed_as_eou,
                eou_threshold=eou_threshold,
            ),
        )

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = stub.Predict(request, timeout=self.timeout)
                return EndpointingResult(
                    label=response.label,
                    confidence=response.confidence,
                    p_eou=response.p_eou,
                    p_cont_user=response.p_cont_user,
                    p_unaddressed=response.p_unaddressed,
                    latency_ms=response.latency_ms,
                    model=response.model,
                )
            except grpc.RpcError as e:
                last_error = f"gRPC error: {e.code().name} - {e.details()}"
                # Recreate channel on connection errors
                if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                    self.close()
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            except Exception as e:
                last_error = f"Error: {str(e)}"
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        # All retries failed
        return EndpointingResult(
            label="ERROR",
            confidence=0.0,
            p_eou=0.0,
            p_cont_user=0.0,
            p_unaddressed=0.0,
            latency_ms=0,
            model="",
            error=last_error,
        )

    def test_connection(self) -> tuple[bool, str]:
        """
        Test connection to the gRPC server.

        Returns:
            Tuple of (success, message).
        """
        try:
            result = self.predict(
                asr_text="test",
                lang="en-US",
                request_id="connection-test",
            )
            if result.error:
                return False, f"Connection failed: {result.error}"
            return True, f"Connected successfully. Model: {result.model}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

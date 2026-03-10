# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ipaddress
import json
import os
import socket
from typing import TYPE_CHECKING, Any, NoReturn
from urllib.parse import urlparse

from ..extras.misc import is_env_enabled
from ..extras.packages import is_fastapi_available


if TYPE_CHECKING:
    from pydantic import BaseModel


SAFE_MEDIA_PATH = os.environ.get("SAFE_MEDIA_PATH", os.path.join(os.path.dirname(__file__), "safe_media"))
ALLOW_LOCAL_FILES = is_env_enabled("ALLOW_LOCAL_FILES", "1")


def raise_http_error(status_code: int, detail: str) -> NoReturn:
    if not is_fastapi_available():
        raise ImportError("Install `fastapi` to use the API module.")

    from fastapi import HTTPException

    raise HTTPException(status_code=status_code, detail=detail)


def _is_http_exception(exc: Exception) -> bool:
    if not is_fastapi_available():
        return False

    from fastapi import HTTPException

    return isinstance(exc, HTTPException)


def dictify(data: "BaseModel") -> dict[str, Any]:
    try:  # pydantic v2
        return data.model_dump(exclude_unset=True)
    except AttributeError:  # pydantic v1
        return data.dict(exclude_unset=True)


def jsonify(data: "BaseModel") -> str:
    try:  # pydantic v2
        return json.dumps(data.model_dump(exclude_unset=True), ensure_ascii=False)
    except AttributeError:  # pydantic v1
        return data.json(exclude_unset=True, ensure_ascii=False)


def check_lfi_path(path: str) -> None:
    """Checks if a given path is vulnerable to LFI. Raises HTTPException if unsafe."""
    if not ALLOW_LOCAL_FILES:
        raise_http_error(403, "Local file access is disabled.")

    try:
        os.makedirs(SAFE_MEDIA_PATH, exist_ok=True)
        real_path = os.path.realpath(path)
        safe_path = os.path.realpath(SAFE_MEDIA_PATH)

        if not real_path.startswith(safe_path):
            raise_http_error(403, "File access is restricted to the safe media directory.")
    except Exception as e:
        if _is_http_exception(e):
            raise

        raise_http_error(400, "Invalid or inaccessible file path.")


def check_ssrf_url(url: str) -> None:
    """Checks if a given URL is vulnerable to SSRF. Raises HTTPException if unsafe."""
    parsed_url = urlparse(url)
    try:
        if parsed_url.scheme not in ["http", "https"]:
            raise_http_error(400, "Only HTTP/HTTPS URLs are allowed.")

        hostname = parsed_url.hostname
        if not hostname:
            raise_http_error(400, "Invalid URL hostname.")

        ip_info = socket.getaddrinfo(hostname, parsed_url.port)
        ip_address_str = ip_info[0][4][0]
        ip = ipaddress.ip_address(ip_address_str)

        if not ip.is_global:
            raise_http_error(403, "Access to private or reserved IP addresses is not allowed.")

    except socket.gaierror:
        raise_http_error(400, f"Could not resolve hostname: {parsed_url.hostname}")
    except Exception as e:
        if _is_http_exception(e):
            raise

        raise_http_error(400, f"Invalid URL: {e}")

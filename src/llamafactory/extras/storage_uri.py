from __future__ import annotations

import os


def _is_truthy_env(name: str, *, default: str = "0") -> bool:
    v = os.getenv(str(name), str(default))
    return str(v).strip().lower() in ("1", "true", "y", "yes", "on")


def _parse_mount_map(raw_map: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for part in str(raw_map or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        mp, bucket = part.split(":", 1)
        mp = mp.strip()
        bucket = bucket.strip()
        if not mp or not bucket:
            continue
        pairs.append((os.path.normpath(mp), bucket))
    return pairs


def maybe_map_mount_to_tos_uri(path: str) -> str | None:
    """Map known fuse-mount paths (e.g. /mnt/...) to tos://bucket/key.

    Enabled by:
      - `LLAMAFACTORY_TOS_SDK_FOR_MOUNT=1|true|yes|on`
    Optional override map:
      - `LLAMAFACTORY_TOS_MOUNT_MAP="/mnt/asr-audio-data:asr-audio-data,/mnt/tts-data-tos:tts-data-tos"`
    """
    # Only map when explicitly enabled; otherwise treat mount paths as local FS.
    if not _is_truthy_env("LLAMAFACTORY_TOS_SDK_FOR_MOUNT"):
        return None

    raw_map = (os.environ.get("LLAMAFACTORY_TOS_MOUNT_MAP") or "").strip()
    if raw_map:
        pairs = _parse_mount_map(raw_map)
    else:
        pairs = [
            (os.path.normpath("/mnt/asr-audio-data"), "asr-audio-data"),
            (os.path.normpath("/mnt/tts-data-tos"), "tts-data-tos"),
        ]

    p_norm = os.path.normpath(str(path))
    for mp_norm, bucket in pairs:
        if p_norm == mp_norm or p_norm.startswith(mp_norm + os.sep):
            rel = p_norm[len(mp_norm) :].lstrip(os.sep).replace(os.sep, "/")
            return f"tos://{bucket}/{rel}"

    return None


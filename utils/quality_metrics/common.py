"""Small cache and JSON helpers shared by the QDR evaluator."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from utils.text import strip_think_content


@dataclass
class JudgeCache:
    path: Path
    _lock: threading.Lock
    _data: Dict[str, Dict[str, Any]]

    @classmethod
    def load(cls, path: Path) -> "JudgeCache":
        path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = record.get("cache_key")
                    if key:
                        data[key] = record
        return cls(path=path, _lock=threading.Lock(), _data=data)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def append(self, key: str, record: Dict[str, Any]) -> None:
        payload = dict(record)
        payload["cache_key"] = key
        with self._lock:
            self._data[key] = payload
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _safe_parse(content: str) -> Optional[Dict[str, Any]]:
    if not content:
        return None
    text = strip_think_content(content).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None

"""
In-Memory Thread-Safe Bounded LRU Cache for Recolorized Web Images.
Aligned with Task Specification 23:
Cache key accounts for:
SHA256(
    image bytes
    + str(condition_id)
    + str(round(severity, 3))
    + model_version
    + return_format
)
"""

from collections import OrderedDict
import hashlib
import threading
from typing import Optional


class ImageLRUCache:
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def generate_key(
        image_bytes: bytes,
        condition_id: int,
        severity: float,
        model_version: str = "Restormer_RDB-v1.0",
        return_format: str = "webp",
    ) -> str:
        """
        Generates a deterministic SHA256 composite cache key.
        Guarantees condition, severity, model version, and return format isolation.
        """
        hasher = hashlib.sha256()
        hasher.update(image_bytes)
        hasher.update(f"|cond:{condition_id}|sev:{round(severity, 3):.3f}|v:{model_version}|fmt:{return_format.lower()}".encode("utf-8"))
        return hasher.hexdigest()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, value: str) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


recolor_cache = ImageLRUCache(max_size=500)

"""LRU SessionStore for managing computed activation caches in RAM/VRAM."""

import uuid
import time
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple
import torch
try:
    from ..adapters.base import BaseModelAdapter
    from ..utils.device import free_gpu_memory, get_vram_usage
except ImportError:
    from interplens.adapters.base import BaseModelAdapter
    from interplens.utils.device import free_gpu_memory, get_vram_usage


class ActivationSession:
    """Container holding prompt activations, logits, tokens, and model metadata."""

    def __init__(
        self,
        session_id: str,
        adapter: BaseModelAdapter,
        prompt: str,
        tokens: list,
        logits: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        corrupted_prompt: Optional[str] = None,
    ):
        self.session_id = session_id
        self.adapter = adapter
        self.prompt = prompt
        self.tokens = tokens
        self.logits = logits
        self.cache = cache
        self.corrupted_prompt = corrupted_prompt
        self.created_at = time.time()

    def clear(self):
        """Explicitly deletes cached tensors to release VRAM memory."""
        self.logits = None
        self.cache.clear()
        self.cache = {}


class SessionStore:
    """Thread-safe LRU Cache store for managing active activation sessions."""

    def __init__(self, max_sessions: int = 1000):
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, ActivationSession] = OrderedDict()
        self.request_history: list = []
        self._request_counter: int = 0

    def _calc_session_kv_mb(self, session: ActivationSession) -> float:
        """Calculates Key/Value cache tensor memory footprint for a session in MB."""
        kv_mb = 0.0
        if session.cache:
            for k, v in session.cache.items():
                if isinstance(v, torch.Tensor):
                    k_lower = k.lower()
                    if "hook_k" in k_lower or "hook_v" in k_lower or "key" in k_lower or "value" in k_lower:
                        kv_mb += (v.element_size() * v.nelement()) / (1024 ** 2)
        
        if kv_mb == 0.0:
            # Theoretical KV cache estimate: 2 (K+V) * layers * seq_len * hidden_dim * 2 bytes
            info = session.adapter.get_model_info() if hasattr(session.adapter, "get_model_info") else {}
            num_l = info.get("num_layers", 12)
            h_dim = info.get("hidden_dim", 768)
            seq_len = len(session.tokens) if session.tokens else 16
            kv_bytes = 2 * num_l * seq_len * h_dim * 2
            kv_mb = kv_bytes / (1024 ** 2)
        return round(kv_mb, 2)

    def create_session(
        self,
        adapter: BaseModelAdapter,
        prompt: str,
        corrupted_prompt: Optional[str] = None,
    ) -> ActivationSession:
        """Executes model forward pass, caches activations, and returns new ActivationSession."""
        # Evict oldest session if at capacity
        while len(self._sessions) >= self.max_sessions and len(self._sessions) > 0:
            oldest_id, oldest_session = self._sessions.popitem(last=False)
            oldest_session.clear()
            free_gpu_memory()

        # Tokenize and execute model
        tokens = adapter.tokenize(prompt)
        logits, cache = adapter.run_with_cache(prompt)

        session_id = str(uuid.uuid4())[:8]
        session = ActivationSession(
            session_id=session_id,
            adapter=adapter,
            prompt=prompt,
            tokens=tokens,
            logits=logits,
            cache=cache,
            corrupted_prompt=corrupted_prompt,
        )

        self._sessions[session_id] = session

        # Record KV Cache metric growth history after question run
        self._request_counter += 1
        sess_kv = self._calc_session_kv_mb(session)
        total_store_kv = sum(self._calc_session_kv_mb(s) for s in self._sessions.values())
        
        self.request_history.append({
            "request_num": self._request_counter,
            "label": f"Q{self._request_counter}: {prompt[:18]}...",
            "prompt": prompt,
            "tokens": len(tokens),
            "kv_mb": sess_kv,
            "total_store_kv_mb": round(total_store_kv, 2),
            "timestamp": time.strftime("%H:%M:%S", time.localtime()),
        })
        if len(self.request_history) > 20:
            self.request_history.pop(0)

        return session

    def get_session(self, session_id: str) -> Optional[ActivationSession]:
        """Retrieves active session by ID and marks it as recently used."""
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]
        return None

    def clear_all(self):
        """Clears all sessions and frees GPU memory."""
        for session in self._sessions.values():
            session.clear()
        self._sessions.clear()
        free_gpu_memory()

    def evict_session(self, session_id: str) -> bool:
        """Manually evicts specific session by ID and frees GPU memory."""
        if session_id in self._sessions:
            sess = self._sessions.pop(session_id)
            sess.clear()
            free_gpu_memory()
            return True
        return False

    def get_sessions_metadata(self) -> list:
        """Returns list of active cached sessions metadata with memory size in MB."""
        res = []
        for sess_id, sess in self._sessions.items():
            cache_mb = 0.0
            if sess.cache:
                for v in sess.cache.values():
                    if isinstance(v, torch.Tensor):
                        cache_mb += (v.element_size() * v.nelement()) / (1024 ** 2)
            res.append({
                "session_id": sess_id,
                "prompt": sess.prompt[:35] + "..." if len(sess.prompt) > 35 else sess.prompt,
                "model_name": getattr(sess.adapter, "model_name", "custom"),
                "tokens_count": len(sess.tokens),
                "cache_size_mb": round(cache_mb, 2),
                "kv_cache_mb": self._calc_session_kv_mb(sess),
                "created_at": time.strftime("%H:%M:%S", time.localtime(sess.created_at)),
            })
        return res


# Global default session store
global_session_store = SessionStore(max_sessions=1000)

"""LRU SessionStore for managing computed activation caches in RAM/VRAM."""

import uuid
import time
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple
import torch
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

    def __init__(self, max_sessions: int = 3):
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, ActivationSession] = OrderedDict()

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


# Global default session store
global_session_store = SessionStore(max_sessions=3)

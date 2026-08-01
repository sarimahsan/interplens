"""Phase 1 Automated Verification Tests for InterpLens."""

import pytest
import torch
import torch.nn as nn
import interplens as il
from interplens.utils.device import get_optimal_device, resolve_device, get_vram_usage, free_gpu_memory
from interplens.adapters.inplace import InPlaceModelAdapter
from interplens.adapters.custom import CustomModelAdapter, PyTorchAutoHooker
from interplens.server.session import SessionStore


class DummyTransformer(nn.Module):
    """Dummy PyTorch transformer module for unit testing without downloading heavy weights."""
    
    def __init__(self, vocab_size=100, hidden_dim=64, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.unembed = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.unembed(x)
        return logits


class DummyTokenizer:
    """Dummy tokenizer for custom model testing."""
    
    def encode(self, text):
        return [ord(c) % 100 for c in text]

    def tokenize(self, text):
        return list(text)


def test_device_utils():
    device = get_optimal_device()
    assert isinstance(device, torch.device)
    
    resolved_cuda = resolve_device("cpu")
    assert resolved_cuda.type == "cpu"
    
    vram = get_vram_usage(torch.device("cpu"))
    assert "allocated_mb" in vram


def test_pytorch_auto_hooker():
    model = DummyTransformer()
    hooker = PyTorchAutoHooker(model)
    hooker.attach_hooks()
    
    input_ids = torch.tensor([[1, 2, 3, 4]])
    logits = model(input_ids)
    
    assert logits.shape == (1, 4, 100)
    assert len(hooker.captured_activations) > 0
    assert "blocks.0" in hooker.captured_activations
    assert hooker.captured_activations["blocks.0"].shape == (1, 4, 64)
    
    hooker.remove_hooks()
    assert len(hooker._hook_handles) == 0


def test_custom_model_adapter():
    model = DummyTransformer()
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model=model, tokenizer=tokenizer)
    
    tokens = adapter.tokenize("Hello")
    assert len(tokens) == 5
    
    logits, cache = adapter.run_with_cache("Hello")
    assert logits is not None
    assert "blocks.0" in cache
    assert cache["blocks.0"].shape == (1, 5, 64)


def test_session_store_lru():
    model = DummyTransformer()
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model=model, tokenizer=tokenizer)
    
    store = SessionStore(max_sessions=2)
    
    s1 = store.create_session(adapter, "Prompt 1")
    s2 = store.create_session(adapter, "Prompt 2")
    assert len(store._sessions) == 2
    
    # Adding 3rd session should evict s1 (LRU)
    s3 = store.create_session(adapter, "Prompt 3")
    assert len(store._sessions) == 2
    assert store.get_session(s1.session_id) is None
    assert store.get_session(s2.session_id) is not None
    assert store.get_session(s3.session_id) is not None


def test_top_level_package_exports():
    assert hasattr(il, "launch")
    assert hasattr(il, "inspect")
    assert hasattr(il, "BaseModelAdapter")
    assert hasattr(il, "__version__")

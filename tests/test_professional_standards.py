"""Automated Unit Tests for Professional Library Standards in InterpLens."""

import os
import pytest
import torch
import torch.nn as nn

import interplens as il
from interplens.exceptions import (
    InterpLensError,
    ModelLoadError,
    CapabilityError,
    UnembeddingNotFoundError,
)
from interplens.adapters import (
    BaseModelAdapter,
    CustomModelAdapter,
    register_adapter,
    get_adapter_for_model,
)


class DummyModelWithoutUnembed(nn.Module):
    """Custom PyTorch module without unembedding weight W_U."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(32, 32)
        self.decoder = nn.Linear(32, 16)  # non-square output layer

    def forward(self, x):
        return self.decoder(self.encoder(x))


class MockCustomAdapter(BaseModelAdapter):
    """Mock adapter for dynamic registration tests."""

    def __init__(self, model_name: str = "mock", device: str = "cpu"):
        super().__init__(model_name=model_name, device=device)
        self.num_layers = 2
        self.num_heads = 4
        self.hidden_dim = 64
        self.vocab_size = 1000

    def load(self): pass
    def tokenize(self, text): return text.split()
    def decode(self, token_ids): return "token"
    def run_with_cache(self, inputs): return torch.zeros((1, 5, 1000)), {}
    def get_unembedding_weight(self): raise UnembeddingNotFoundError("No unembedding.")
    def get_resid_post_hook_name(self, layer): return f"layer_{layer}"
    def get_attn_pattern_hook_name(self, layer): return f"attn_{layer}"
    def get_mlp_post_hook_name(self, layer): return f"mlp_{layer}"


def test_package_exports_and_exceptions():
    assert issubclass(ModelLoadError, InterpLensError)
    assert issubclass(UnembeddingNotFoundError, CapabilityError)
    assert issubclass(CapabilityError, InterpLensError)
    assert hasattr(il, "UnembeddingNotFoundError")
    assert hasattr(il, "ModelLoadError")
    assert hasattr(il, "stop_server")


def test_py_typed_file_exists():
    pkg_dir = os.path.dirname(il.__file__)
    py_typed_path = os.path.join(pkg_dir, "py.typed")
    assert os.path.exists(py_typed_path)


def test_unembedding_not_found_error_raised():
    dummy = DummyModelWithoutUnembed()
    adapter = CustomModelAdapter(model=dummy, model_name="DummyWithoutUnembed")
    
    with pytest.raises(UnembeddingNotFoundError):
        adapter.get_unembedding_weight()


def test_dynamic_register_adapter():
    register_adapter("custom-mock-architecture", MockCustomAdapter)
    adapter = get_adapter_for_model("custom-mock-architecture", device="cpu")
    
    assert isinstance(adapter, MockCustomAdapter)
    assert adapter.model_name == "custom-mock-architecture"


from unittest.mock import patch

def test_model_load_error_on_invalid_hf_string():
    with patch.dict("sys.modules", {"transformers": None}):
        with pytest.raises(ModelLoadError) as exc_info:
            get_adapter_for_model("nonexistent-invalid-hf-model-xyz-123456789")
        
        assert "Failed to load model" in str(exc_info.value)

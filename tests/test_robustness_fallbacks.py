"""Tests for Library Robustness, Error Handling, and Fallbacks."""

import os
import pytest
import torch
import torch.nn as nn

import interplens as il
from interplens.config import Settings, _get_env_int
from interplens.exceptions import UnembeddingNotFoundError, ModelLoadError
from interplens.adapters.inplace import InPlaceModelAdapter
from interplens.adapters.custom import CustomModelAdapter
from interplens.server.session import SessionStore
from interplens.utils.device import get_detailed_gpu_profiler
from interplens.analysis.attention_heads import compute_attention_metrics
from interplens.analysis.causal_patching import run_causal_patching_sweep
import interplens.utils as il_utils
import interplens.analysis as il_analysis


class BareDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        return self.fc(x)


class DummyWithNoUnembed:
    def __init__(self):
        self.fc = nn.Linear(16, 16)
    def parameters(self):
        return self.fc.parameters()


def test_env_int_fallback():
    # Test valid int
    os.environ["_TEST_INT_VAL"] = "9000"
    assert _get_env_int("_TEST_INT_VAL", 8000) == 9000

    # Test invalid string gracefully falls back
    os.environ["_TEST_INT_VAL"] = "not_a_number"
    assert _get_env_int("_TEST_INT_VAL", 8000) == 8000

    # Test missing env var
    assert _get_env_int("_NON_EXISTENT_VAR_12345", 8000) == 8000


def test_inplace_adapter_unembedding_error():
    dummy = DummyWithNoUnembed()
    adapter = InPlaceModelAdapter(dummy, model_name="dummy_no_unembed")
    
    with pytest.raises(UnembeddingNotFoundError):
        adapter.get_unembedding_weight()


class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])
        self.unembed = nn.Linear(16, 100)
    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        for b in self.blocks:
            x = b(x)
        return self.unembed(x)


def test_session_store_threadsafe_count():
    model = SimpleTransformer()
    adapter = CustomModelAdapter(model=model, model_name="simple_trans")
    store = SessionStore(max_sessions=3)
    
    assert store.session_count == 0
    assert len(store) == 0

    s1 = store.create_session(adapter, "Test prompt 1")
    assert store.session_count == 1
    assert len(store) == 1


def test_gpu_profiler_none_adapter_safe():
    # Should never crash when adapter is None
    prof = get_detailed_gpu_profiler(adapter=None, cache=None)
    assert isinstance(prof, dict)
    assert "device_name" in prof
    assert "layer_memory" in prof


def test_attention_fallback_zero_division_safety():
    model = SimpleTransformer()
    adapter = CustomModelAdapter(model=model, model_name="simple_dummy")
    
    # Run with empty cache so it exercises fallback attention patterns
    res = compute_attention_metrics(
        adapter=adapter,
        cache={},
        tokens=["a", "b", "c", "d"],
        session_id="test_sess",
        layer=0,
        head=0,
    )
    assert res is not None
    assert len(res.matrix) == 4
    assert len(res.matrix[0]) == 4


def test_causal_patching_tokenizer_fallback():
    model = SimpleTransformer()
    # Adapter without tokenizer
    adapter = CustomModelAdapter(model=model, tokenizer=None, model_name="simple_dummy")
    
    res = run_causal_patching_sweep(
        adapter=adapter,
        clean_prompt="Hello world",
        corrupt_prompt="Goodbye world",
        target_token_str="world",
    )
    assert res is not None
    assert res.target_token is not None


def test_package_submodule_exports():
    assert hasattr(il_utils, "get_optimal_device")
    assert hasattr(il_utils, "generate_model_report_pdf")
    assert hasattr(il_analysis, "compute_logit_lens")
    assert hasattr(il_analysis, "run_causal_patching_sweep")
    assert hasattr(il_analysis, "detect_induction_heads")

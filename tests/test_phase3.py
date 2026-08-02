"""Phase 3 Tests: Residual Stream Analysis & Activation Steering Engine."""

import pytest
import torch
from interplens.adapters.custom import CustomModelAdapter
from interplens.analysis.residual_stream import compute_residual_metrics, apply_activation_steering
from interplens.schema import ModelInfo


class DummyTokenizer:
    def decode(self, ids):
        return " test"
    def encode(self, text, return_tensors=None):
        return torch.tensor([[101, 102]])


class DummyConfig:
    num_hidden_layers = 4
    n_layer = 4
    num_attention_heads = 4
    n_head = 4
    hidden_size = 64
    n_embd = 64
    vocab_size = 1000


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        self.embed = torch.nn.Embedding(1000, 64)
        self.norm = torch.nn.LayerNorm(64)
        self.lm_head = torch.nn.Linear(64, 1000, bias=False)

    def forward(self, input_ids=None, **kwargs):
        x = self.embed(input_ids)
        logits = self.lm_head(self.norm(x))
        class Out:
            pass
        out = Out()
        out.logits = logits
        return out


def test_residual_stream_metrics():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model, tokenizer, model_name="dummy")

    logits, cache = adapter.run_with_cache("test prompt")
    tokens = ["test", "prompt"]

    res = compute_residual_metrics(adapter, cache, tokens, session_id="test_sess_001")
    assert "vector_norms" in res
    assert "cosine_matrix" in res
    assert res["session_id"] == "test_sess_001"


def test_activation_steering():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model, tokenizer, model_name="dummy")

    steer_vec = [0.5] * 64
    res = apply_activation_steering(adapter, "test prompt", target_layer=0, steering_vector=steer_vec, multiplier=2.0)
    assert res["status"] == "success"
    assert "top_steered_token" in res

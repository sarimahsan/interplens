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
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(4)])
        self.norm = torch.nn.LayerNorm(64)
        self.lm_head = torch.nn.Linear(64, 1000, bias=False)

    def forward(self, input_ids=None, **kwargs):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
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


def test_attention_heads_computation():
    from interplens.analysis.attention_heads import compute_attention_metrics
    from interplens.schema import AttentionHeadResponse

    model = DummyModel()
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model, tokenizer, model_name="dummy")

    logits, cache = adapter.run_with_cache("The Eiffel Tower is in Paris")
    tokens = ["The", "Eiffel", "Tower", "is", "in", "Paris"]

    # Test single head and arc links
    res = compute_attention_metrics(
        adapter=adapter,
        cache=cache,
        tokens=tokens,
        session_id="test_attn_sess",
        prompt="The Eiffel Tower is in Paris",
        layer=0,
        head=0,
        threshold=0.01,
    )

    assert isinstance(res, AttentionHeadResponse)
    assert res.session_id == "test_attn_sess"
    assert res.layer == 0
    assert res.head == 0
    assert len(res.matrix) == 6  # 6x6 matrix for 6 tokens
    assert len(res.matrix[0]) == 6
    assert res.grid is not None
    assert len(res.grid) == res.num_heads
    assert len(res.arc_links) > 0
    assert res.arc_links[0].source_token == tokens[res.arc_links[0].source]


def test_attention_fastapi_endpoint():
    from fastapi.testclient import TestClient
    from interplens.server.app import app, set_active_adapter

    model = DummyModel()
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model, tokenizer, model_name="dummy")
    set_active_adapter(adapter)

    client = TestClient(app)

    # 1. Run prompt to populate session store
    run_res = client.post("/api/run", json={"prompt": "Attention mechanisms in LLMs"})
    assert run_res.status_code == 200
    session_id = run_res.json()["session_id"]

    # 2. Query attention endpoint
    attn_res = client.get(f"/api/analysis/attention?session_id={session_id}&layer=0&head=0&threshold=0.01")
    assert attn_res.status_code == 200
    data = attn_res.json()
    assert data["session_id"] == session_id
    assert "matrix" in data
    assert "arc_links" in data
    assert "grid" in data


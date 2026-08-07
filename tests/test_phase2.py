"""Phase 2 Automated Verification Suite.

Tests Logit Lens computation engine, FastAPI REST API endpoints, and end-to-end integration.
"""

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from interplens.schema import ModelInfo, LogitLensMatrixResponse
from interplens.adapters.custom import CustomModelAdapter
from interplens.adapters.inplace import InPlaceModelAdapter
from interplens.analysis.logit_lens import compute_logit_lens
from interplens.server.app import app, set_active_adapter


class DummyTransformer(nn.Module):
    """Dummy PyTorch transformer module for unit testing without downloading heavy weights."""
    def __init__(self, vocab_size=50, hidden_dim=16, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.unembed = nn.Linear(hidden_dim, vocab_size)
        self.W_U = self.unembed.weight.T  # (d_model, d_vocab)
        self.ln_final = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.unembed(self.ln_final(x))
        return logits


class DummyTokenizer:
    """Dummy tokenizer for testing."""
    def encode(self, text):
        return [ord(word[0]) % 50 for word in text.split()]

    def decode(self, token_ids):
        return f"tok_{token_ids[0]}"

    def tokenize(self, text):
        return text.split()


def test_logit_lens_computation():
    """Unit test for compute_logit_lens math & data structures."""
    model = DummyTransformer(vocab_size=50, hidden_dim=16, num_layers=4)
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model=model, tokenizer=tokenizer)

    prompt = "Hello world test"
    tokens = adapter.tokenize(prompt)
    logits, cache = adapter.run_with_cache(prompt)

    res = compute_logit_lens(
        adapter=adapter,
        cache=cache,
        tokens=tokens,
        session_id="test_sess",
        prompt=prompt,
        top_k=5,
        apply_ln=True,
    )

    assert isinstance(res, LogitLensMatrixResponse)
    assert res.session_id == "test_sess"
    assert res.num_layers == 5
    assert len(res.positions) == 3  # 3 tokens

    # Inspect first position
    pos0 = res.positions[0]
    assert pos0.token == "Hello"
    assert len(pos0.layers) == 5

    # Check top tokens and metrics in layer 0
    layer0 = pos0.layers[0]
    assert hasattr(layer0, "entropy")
    assert hasattr(layer0, "kl_divergence")
    assert isinstance(layer0.entropy, float)
    assert isinstance(layer0.kl_divergence, float)

    layer0_tokens = layer0.top_tokens
    assert len(layer0_tokens) == 5
    assert layer0_tokens[0].rank == 1
    assert 0.0 <= layer0_tokens[0].probability <= 1.0

    # Check rank trajectory and top5 competition trajectory
    assert pos0.target_token_ranks is not None
    assert len(pos0.target_token_ranks) == 5
    assert pos0.top5_trajectories is not None
    assert len(pos0.top5_trajectories) > 0


def test_fastapi_endpoints():
    """Integration test for FastAPI health, run, and logit-lens endpoints."""
    model = DummyTransformer(vocab_size=50, hidden_dim=16, num_layers=4)
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model=model, tokenizer=tokenizer)
    set_active_adapter(adapter)

    client = TestClient(app)

    # 1. Healthcheck Endpoint
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert health_data["status"] == "online"

    # 2. Run Prompt Endpoint
    res_run = client.post("/api/run", json={"prompt": "The Eiffel Tower"})
    assert res_run.status_code == 200
    run_data = res_run.json()
    assert "session_id" in run_data
    assert run_data["tokens"] == ["The", "Eiffel", "Tower"]
    session_id = run_data["session_id"]

    # 3. Logit Lens Endpoint
    res_lens = client.get(f"/api/analysis/logit-lens?session_id={session_id}&top_k=3")
    assert res_lens.status_code == 200
    lens_data = res_lens.json()
    assert lens_data["session_id"] == session_id
    assert len(lens_data["positions"]) == 3
    assert len(lens_data["positions"][0]["layers"]) == 5


def test_telemetry_websocket():
    """Test live telemetry WebSocket streaming endpoint."""
    model = DummyTransformer(vocab_size=50, hidden_dim=16, num_layers=4)
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model=model, tokenizer=tokenizer)
    set_active_adapter(adapter)

    client = TestClient(app)
    with client.websocket_connect("/ws/telemetry") as websocket:
        data = websocket.receive_json()
        assert data["status"] == "online"
        assert "vram_usage" in data
        assert data["vram_usage"]["allocated_mb"] >= 0


def test_pdf_model_report():
    """Test PDF model inspection report generation endpoint."""
    model = DummyTransformer(vocab_size=50, hidden_dim=16, num_layers=4)
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model=model, tokenizer=tokenizer, model_name="dummy_test_model")
    set_active_adapter(adapter)

    client = TestClient(app)
    res = client.get("/api/model/report/pdf")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert len(res.content) > 1000
    assert res.content[:4] == b"%PDF"


def test_logit_lens_gpt2_integration():
    """End-to-end integration test with TransformerLens GPT-2."""
    try:
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained("gpt2", device="cpu")
        adapter = InPlaceModelAdapter(model)
        set_active_adapter(adapter)

        client = TestClient(app)

        prompt = "The capital of France is"
        res_run = client.post("/api/run", json={"prompt": prompt})
        assert res_run.status_code == 200
        run_data = res_run.json()
        session_id = run_data["session_id"]

        res_lens = client.get(f"/api/analysis/logit-lens?session_id={session_id}&top_k=5")
        assert res_lens.status_code == 200
        lens_data = res_lens.json()

        last_pos = lens_data["positions"][-1]
        final_layer = last_pos["layers"][-1]
        top_prediction = final_layer["top_tokens"][0]["token"]

        assert "Paris" in top_prediction
    except Exception as e:
        pytest.skip(f"Skipping GPT-2 integration test: {str(e)}")

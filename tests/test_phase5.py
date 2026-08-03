"""Phase 5 Tests: Automated Causal Interventions & ROME Causal Tracing.

Verifies activation patching sweeps, logit difference recovery calculations,
and FastAPI endpoint /api/analysis/causal-patching.
"""

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from interplens.adapters.custom import CustomModelAdapter
from interplens.analysis.causal_patching import run_causal_patching_sweep
from interplens.schema import CausalTracingResponse
from interplens.server.app import app, set_active_adapter


class DummyConfig:
    num_hidden_layers = 4
    num_attention_heads = 4
    hidden_size = 32
    vocab_size = 100


class Phase5DummyTransformer(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=32, num_layers=4):
        super().__init__()
        self.config = DummyConfig()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.Linear(hidden_dim, hidden_dim),
                "mlp": nn.Linear(hidden_dim, hidden_dim * 4),
            }) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = x + block["attn"](x)
            x = x + block["mlp"](x)[:, :, :32]
        return self.lm_head(x)


class DummyTokenizer:
    def encode(self, text):
        return [ord(w[0]) % 100 for w in text.split()]
    def decode(self, ids):
        return f"tok_{ids[0]}"
    def tokenize(self, text):
        return text.split()


def test_causal_patching_sweep_computation():
    model = Phase5DummyTransformer()
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model, tokenizer, model_name="dummy_p5")

    clean_prompt = "The Eiffel Tower is in Paris"
    corrupt_prompt = "The Colosseum is in Paris"

    res = run_causal_patching_sweep(
        adapter=adapter,
        clean_prompt=clean_prompt,
        corrupt_prompt=corrupt_prompt,
    )

    assert isinstance(res, CausalTracingResponse)
    assert res.clean_prompt == clean_prompt
    assert res.corrupt_prompt == corrupt_prompt
    assert res.num_layers == 4
    assert len(res.heatmap_matrix) == 4
    assert len(res.heatmap_matrix[0]) == len(res.clean_tokens)
    assert len(res.cells) == 4 * len(res.clean_tokens)


def test_phase5_fastapi_causal_endpoint():
    model = Phase5DummyTransformer()
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model, tokenizer, model_name="dummy_p5")
    set_active_adapter(adapter)

    client = TestClient(app)

    payload = {
        "clean_prompt": "The Eiffel Tower is in Paris",
        "corrupt_prompt": "The Colosseum is in Paris",
        "target_token": "Paris"
    }

    res = client.post("/api/analysis/causal-patching", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["clean_prompt"] == payload["clean_prompt"]
    assert "heatmap_matrix" in data
    assert "baseline_clean_logit_diff" in data
    assert "max_recovery_percentage" in data

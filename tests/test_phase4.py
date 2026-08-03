"""Phase 4 Tests: Neuron Activations & Token Attribution Engine.

Verifies top-K neuron ranking, prompt lighting text strips, Attention Rollout math,
and FastAPI REST endpoints (/api/analysis/neurons and /api/analysis/attribution).
"""

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from interplens.adapters.custom import CustomModelAdapter
from interplens.analysis.neurons import compute_neuron_activations
from interplens.analysis.attribution import compute_token_attributions
from interplens.schema import NeuronAnalysisResponse, TokenAttributionResponse
from interplens.server.app import app, set_active_adapter


class Phase4DummyTransformer(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=32, num_layers=4):
        super().__init__()
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


def test_neuron_activations_computation():
    model = Phase4DummyTransformer()
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model, tokenizer, model_name="dummy_p4")

    logits, cache = adapter.run_with_cache("The Eiffel Tower is in Paris")
    tokens = ["The", "Eiffel", "Tower", "is", "in", "Paris"]

    res = compute_neuron_activations(
        adapter=adapter,
        cache=cache,
        tokens=tokens,
        session_id="test_neuron_sess",
        prompt="The Eiffel Tower is in Paris",
        layer=0,
        position=5,
        top_k=5,
    )

    assert isinstance(res, NeuronAnalysisResponse)
    assert res.session_id == "test_neuron_sess"
    assert res.layer == 0
    assert res.position == 5
    assert len(res.top_neurons) == 5
    assert res.top_neurons[0].activation >= res.top_neurons[1].activation
    assert len(res.lighting_strip) == 6
    assert res.lighting_strip[5].token == "Paris"


def test_token_attribution_computation():
    model = Phase4DummyTransformer()
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model, tokenizer, model_name="dummy_p4")

    logits, cache = adapter.run_with_cache("When Mary and John went to store")
    tokens = ["When", "Mary", "and", "John", "went", "to", "store"]

    res = compute_token_attributions(
        adapter=adapter,
        cache=cache,
        tokens=tokens,
        session_id="test_attr_sess",
        prompt="When Mary and John went to store",
        position=6,
        method="attention_rollout",
    )

    assert isinstance(res, TokenAttributionResponse)
    assert res.session_id == "test_attr_sess"
    assert res.target_position == 6
    assert len(res.attributions) == 7
    assert 0.0 <= res.attributions[0].score <= 1.0


def test_phase4_fastapi_endpoints():
    model = Phase4DummyTransformer()
    tokenizer = DummyTokenizer()
    adapter = CustomModelAdapter(model, tokenizer, model_name="dummy_p4")
    set_active_adapter(adapter)

    client = TestClient(app)

    # 1. Run prompt
    run_res = client.post("/api/run", json={"prompt": "Mechanistic interpretability neuron test"})
    assert run_res.status_code == 200
    session_id = run_res.json()["session_id"]

    # 2. Test /api/analysis/neurons
    neuron_res = client.get(f"/api/analysis/neurons?session_id={session_id}&layer=0&top_k=5")
    assert neuron_res.status_code == 200
    n_data = neuron_res.json()
    assert n_data["session_id"] == session_id
    assert "top_neurons" in n_data
    assert "lighting_strip" in n_data

    # 3. Test /api/analysis/attribution
    attr_res = client.get(f"/api/analysis/attribution?session_id={session_id}")
    assert attr_res.status_code == 200
    a_data = attr_res.json()
    assert a_data["session_id"] == session_id
    assert "attributions" in a_data

"""Tests for Model Architecture Topology graph generator."""

import pytest
import torch
from interplens.adapters.custom import CustomModelAdapter
from interplens.analysis.topology import inspect_model_topology


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


def test_inspect_model_topology():
    model = DummyModel()
    adapter = CustomModelAdapter(model, None, model_name="dummy_test_model")

    top = inspect_model_topology(adapter)
    assert top["model_name"] == "dummy_test_model"
    assert "total_parameters" in top
    assert len(top["nodes"]) >= 5
    assert top["hidden_dim"] == 64
    assert "parameter_breakdown" in top
    assert len(top["parameter_breakdown"]) >= 5

    # Verify percentages sum close to 100%
    total_pct = sum(item["percentage"] for item in top["parameter_breakdown"])
    assert 99.0 <= total_pct <= 101.0

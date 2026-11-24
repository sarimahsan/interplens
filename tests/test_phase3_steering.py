"""Tests for Phase 3A: Contrastive Activation Steering (CAA) & Side-by-Side Generation."""

import torch
import pytest
from fastapi.testclient import TestClient

from interplens.adapters.custom import CustomModelAdapter
from interplens.analysis.steering import (
    compute_contrastive_steering_vector,
    generate_steered_comparison,
)
from interplens.server.app import app, set_active_adapter
from tests.test_phase2_circuits import TinyLM, SimpleTok


def _make_adapter():
    torch.manual_seed(7)
    return CustomModelAdapter(model=TinyLM(), tokenizer=SimpleTok(), model_name="tiny-lm")


def test_contrastive_vector_and_side_by_side_generation():
    adapter = _make_adapter()
    vec_info = compute_contrastive_steering_vector(
        adapter=adapter,
        positive_prompts=["happy bright sunny", "joyful great day"],
        negative_prompts=["sad dark gloomy", "awful bad night"],
        target_layer=1,
        name="sentiment",
    )
    assert vec_info.target_layer == 1
    assert vec_info.hidden_dim == 16
    assert vec_info.l2_norm > 0.0

    # Zero multiplier should produce identical tokens to baseline
    same = generate_steered_comparison(
        adapter=adapter,
        prompt="today i feel",
        vector_id=vec_info.vector_id,
        multiplier=0.0,
        max_new_tokens=5,
    )
    assert same.baseline_completion == same.steered_completion
    assert len(same.baseline_tokens) == 5

    # Strong steering should produce valid multi-token comparison
    steered = generate_steered_comparison(
        adapter=adapter,
        prompt="today i feel",
        vector_id=vec_info.vector_id,
        multiplier=8.0,
        max_new_tokens=5,
    )
    assert len(steered.steered_tokens) == 5
    assert steered.runtime_ms >= 0.0


def test_steering_fastapi_endpoints():
    adapter = _make_adapter()
    set_active_adapter(adapter)
    client = TestClient(app)

    comp = client.post(
        "/api/analysis/steering/compute",
        json={
            "positive_prompts": ["happy sunny day"],
            "negative_prompts": ["sad rainy night"],
            "target_layer": 1,
            "name": "test-vec",
        },
    )
    assert comp.status_code == 200
    vid = comp.json()["vector_id"]

    listed = client.get("/api/analysis/steering/vectors")
    assert listed.status_code == 200
    assert any(v["vector_id"] == vid for v in listed.json()["vectors"])

    gen = client.post(
        "/api/analysis/steering/generate",
        json={
            "prompt": "today i walked",
            "vector_id": vid,
            "multiplier": 2.0,
            "max_new_tokens": 4,
        },
    )
    assert gen.status_code == 200
    data = gen.json()
    assert len(data["baseline_tokens"]) == 4
    assert len(data["steered_tokens"]) == 4


def test_multi_architecture_hf_contrastive_steering():
    """Verifies CAA vector extraction and multi-token steering on Llama, Qwen2, and GPT-2."""
    transformers = pytest.importorskip("transformers")
    configs = [
        ("llama-tiny", transformers.LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2), transformers.LlamaForCausalLM),
        ("qwen2-tiny", transformers.Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2), transformers.Qwen2ForCausalLM),
        ("gpt2-tiny", transformers.GPT2Config(vocab_size=128, n_embd=32, n_inner=64, n_layer=2, n_head=4), transformers.GPT2LMHeadModel),
    ]

    for name, cfg, cls in configs:
        torch.manual_seed(123)
        adapter = CustomModelAdapter(model=cls(cfg), tokenizer=SimpleTok(), model_name=name)
        vec = compute_contrastive_steering_vector(
            adapter=adapter,
            positive_prompts=["a b c d", "e f g h"],
            negative_prompts=["w x y z", "p q r s"],
            target_layer=1,
            name=name,
        )
        assert vec.hidden_dim == 32
        gen = generate_steered_comparison(
            adapter=adapter,
            prompt="m n o",
            vector_id=vec.vector_id,
            multiplier=3.0,
            max_new_tokens=4,
        )
        assert len(gen.baseline_tokens) == 4
        assert len(gen.steered_tokens) == 4

"""Tests for Phase 2 Fast Circuits: Direct Logit Attribution (DLA) & Attribution Patching (AtP)."""

import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from interplens.adapters.custom import CustomModelAdapter
from interplens.analysis.dla import compute_direct_logit_attribution
from interplens.analysis.attribution_patching import run_attribution_patching
from interplens.server.app import app, set_active_adapter


class TinyBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 16, n_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TinyBlock(d_model) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.num_layers = n_layers
        self.hidden_dim = d_model
        self.vocab_size = vocab_size
        self.num_heads = 2

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(h)


class SimpleTok:
    def encode(self, text, add_special_tokens=False):
        return [max(1, ord(w[0]) % 63) for w in text.split() if w] or [1]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"tok{i}" for i in ids)

    def tokenize(self, text):
        return text.split() or ["tok"]


def _make_adapter():
    torch.manual_seed(0)
    model = TinyLM()
    return CustomModelAdapter(model=model, tokenizer=SimpleTok(), model_name="tiny-lm")


def test_direct_logit_attribution():
    adapter = _make_adapter()
    logits, cache = adapter.run_with_cache("the cat sat on mat")
    tokens = adapter.tokenize("the cat sat on mat")

    res = compute_direct_logit_attribution(
        adapter=adapter,
        cache=cache,
        tokens=tokens,
        session_id="dla_test",
        prompt="the cat sat on mat",
        logits=logits,
    )
    assert res.session_id == "dla_test"
    assert len(res.layer_contributions) == 3
    assert len(res.top_components) > 0
    assert res.target_token_id != res.foil_token_id


def test_attribution_patching_fast_gradients():
    adapter = _make_adapter()
    # Freeze params initially to verify AtP restores requires_grad state
    for p in adapter._model_instance.parameters():
        p.requires_grad_(False)

    res = run_attribution_patching(
        adapter=adapter,
        clean_prompt="the cat sat on mat",
        corrupt_prompt="the dog sat on rug",
    )
    assert res.num_layers == 3
    assert res.seq_len == 5
    assert len(res.heatmap_matrix) == 3
    assert len(res.heatmap_matrix[0]) == 5
    assert len(res.top_components) == 6  # 3 layers x (attn + mlp)
    assert res.runtime_ms >= 0.0
    # Verify requires_grad restored to False
    assert all(not p.requires_grad for p in adapter._model_instance.parameters())


def test_phase2_circuit_fastapi_endpoints():
    adapter = _make_adapter()
    set_active_adapter(adapter)
    client = TestClient(app)

    run_resp = client.post("/api/run", json={"prompt": "the cat sat on mat"})
    assert run_resp.status_code == 200
    sess_id = run_resp.json()["session_id"]

    dla_resp = client.get(f"/api/analysis/dla?session_id={sess_id}")
    assert dla_resp.status_code == 200
    dla_data = dla_resp.json()
    assert len(dla_data["layer_contributions"]) == 3

    atp_resp = client.post(
        "/api/analysis/attribution-patching",
        json={
            "clean_prompt": "the cat sat on mat",
            "corrupt_prompt": "the dog sat on rug",
        },
    )
    assert atp_resp.status_code == 200
    atp_data = atp_resp.json()
    assert len(atp_data["heatmap_matrix"]) == 3
    assert len(atp_data["top_components"]) == 6


def test_multi_architecture_hf_dla_atp_and_mermaid():
    """Verifies DLA, AtP, and Mermaid topology across real HF Llama, Qwen2, Mixtral MoE, and GPT-2 classes."""
    import pytest
    transformers = pytest.importorskip("transformers")
    from interplens.analysis.topology import inspect_model_topology

    configs = [
        ("llama-tiny", transformers.LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2), transformers.LlamaForCausalLM),
        ("qwen2-tiny", transformers.Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2), transformers.Qwen2ForCausalLM),
        ("mixtral-tiny", transformers.MixtralConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2), transformers.MixtralForCausalLM),
        ("gpt2-tiny", transformers.GPT2Config(vocab_size=128, n_embd=32, n_inner=64, n_layer=2, n_head=4), transformers.GPT2LMHeadModel),
    ]

    for name, cfg, cls in configs:
        torch.manual_seed(42)
        model = cls(cfg)
        adapter = CustomModelAdapter(model=model, tokenizer=SimpleTok(), model_name=name)

        # 1. Topology + Mermaid diagram
        topo = inspect_model_topology(adapter)
        assert "flowchart TD" in topo["mermaid_diagram"]
        if "mixtral" in name:
            assert "MoE Router" in topo["mermaid_diagram"]
        elif "llama" in name or "qwen2" in name:
            assert "GQA" in topo["mermaid_diagram"]

        # 2. Direct Logit Attribution
        logits, cache = adapter.run_with_cache("a b c d")
        dla = compute_direct_logit_attribution(
            adapter=adapter,
            cache=cache,
            tokens=adapter.tokenize("a b c d"),
            session_id=f"sess_{name}",
            logits=logits,
        )
        assert len(dla.layer_contributions) == 2

        # 3. Gradient Attribution Patching (2 fwd + 1 bwd)
        atp = run_attribution_patching(
            adapter=adapter,
            clean_prompt="a b c d",
            corrupt_prompt="x y z w",
        )
        assert atp.num_layers == 2
        assert len(atp.heatmap_matrix) == 2
        assert len(atp.top_components) == 4  # 2 layers x (attn + mlp/moe)


def test_bfloat16_logit_lens_no_numpy_crash():
    """Regression test: BFloat16 models (e.g. A100 OLMoE/Qwen/Llama) must not fail on .numpy()."""
    from interplens.analysis.logit_lens import compute_logit_lens

    torch.manual_seed(0)
    model = TinyLM().to(torch.bfloat16)
    adapter = CustomModelAdapter(model=model, tokenizer=SimpleTok(), model_name="tiny-bf16")
    _, cache = adapter.run_with_cache("the cat sat on mat")
    res = compute_logit_lens(
        adapter=adapter,
        cache=cache,
        tokens=adapter.tokenize("the cat sat on mat"),
        session_id="bf16_test",
        prompt="the cat sat on mat",
    )
    assert res.num_layers >= 3
    assert len(res.positions) == 5

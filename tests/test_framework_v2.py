"""Unit tests for InterpLens v2.0 Model Framework, Strategy Registry, Hook Discovery, and Capabilities Matrix."""

import pytest
import torch
import torch.nn as nn

from interplens.adapters import (
    BaseArchitectureStrategy,
    ArchitectureFamily,
    ArchitectureRegistry,
    register_architecture_strategy,
    get_strategy_for_model,
    StaticFingerprint,
    RuntimeFingerprint,
    CapabilityLevel,
    ModelCapability,
    EngineStatus,
    evaluate_engine_capabilities,
    HookDiscovery,
    GenericAdapter,
    ModelReport,
)


class DummyCustomTransformer(nn.Module):
    """Bare PyTorch research model to test GenericAdapter and Hook Discovery."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.Linear(32, 32),
                "mlp": nn.Linear(32, 32),
            })
            for _ in range(3)
        ])
        self.lm_head = nn.Linear(32, 100)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer["self_attn"](x) + layer["mlp"](x)
        return self.lm_head(x)


def test_strategy_registry():
    strategy, confidence = get_strategy_for_model(None, "llama-3-8b")
    assert strategy.architecture_id == "llama"
    assert confidence == 1.0

    strategy, confidence = get_strategy_for_model(None, "qwen2.5-3b")
    assert strategy.architecture_id == "qwen2"
    assert confidence == 1.0


def test_custom_plugin_registration():
    class RWKVStrategy(BaseArchitectureStrategy):
        architecture_id = "rwkv"
        family = ArchitectureFamily.STATE_SPACE

        def matches(self, model_or_config, model_name=""):
            return 1.0 if "rwkv" in model_name.lower() else 0.0

        def discover_graph(self, model):
            from interplens.adapters.graph import ModelGraph
            return ModelGraph(root_node_id="rwkv")

        def extract_unembedding(self, model):
            return None

        def get_resid_post_hook_name(self, layer): return f"blocks.{layer}"
        def get_attn_pattern_hook_name(self, layer): return f"blocks.{layer}.attn"
        def get_mlp_post_hook_name(self, layer): return f"blocks.{layer}.ffn"

    rwkv_strat = RWKVStrategy()
    register_architecture_strategy(rwkv_strat)

    strat, conf = get_strategy_for_model(None, "rwkv-4-world")
    assert strat.architecture_id == "rwkv"
    assert conf == 1.0
    assert strat.family == ArchitectureFamily.STATE_SPACE


def test_hook_discovery_and_generic_adapter():
    dummy = DummyCustomTransformer()
    adapter = GenericAdapter(model=dummy, model_name="DummyResearchModel")

    assert adapter.model_name == "DummyResearchModel"
    assert adapter.num_layers == 3
    assert adapter.hidden_dim == 32
    assert adapter.vocab_size == 100
    assert adapter.report is not None

    # Test forward pass with cache
    inputs = torch.tensor([[1, 5, 12]])
    logits, cache = adapter.run_with_cache(inputs)

    assert logits.shape == (1, 3, 100)
    assert len(cache) > 0


def test_engine_capability_matrix():
    model_cap = ModelCapability(
        has_unembedding=True,
        has_residual_stream=True,
        has_attention_maps=True,
        has_mlp_activations=True,
        capability_level=CapabilityLevel.LEVEL_5_FULL_SUPPORT
    )
    matrix = evaluate_engine_capabilities(model_cap)

    assert matrix.engines["logit_lens"].status == EngineStatus.SUPPORTED
    assert matrix.engines["residual_stream"].status == EngineStatus.SUPPORTED
    assert matrix.engines["attention"].status == EngineStatus.SUPPORTED
    assert matrix.engines["neurons"].status == EngineStatus.SUPPORTED

"""Automatic Hook Discovery & Module Classification Engine with confidence scoring."""

from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn

from interplens.adapters.graph import ModelGraph, GraphNode, DiscoveredModule
from interplens.adapters.capabilities import ModelCapability, CapabilityLevel
from interplens.adapters.registry import get_strategy_for_model
from interplens.adapters.strategy import BaseArchitectureStrategy


class HookDiscovery:
    """Introspects PyTorch model named_modules() and named_parameters() to discover topology and classify hooks."""

    def __init__(self, model: nn.Module, model_name: str = ""):
        self.model = model
        self.model_name = model_name

    def discover(self) -> Tuple[ModelGraph, ModelCapability, BaseArchitectureStrategy, float]:
        """Runs hook discovery and returns (ModelGraph, ModelCapability, strategy, confidence)."""
        strategy, confidence = get_strategy_for_model(self.model, self.model_name)

        if confidence >= 0.8:
            # Use strategy-guided discovery graph
            graph = strategy.discover_graph(self.model)
        else:
            # Fall back to heuristic graph discovery
            graph = self._heuristic_discovery()

        cap = self._evaluate_capabilities(graph, strategy)
        return graph, cap, strategy, confidence

    def _heuristic_discovery(self) -> ModelGraph:
        """Heuristic graph discovery walking named_modules() and parameter shapes."""
        graph = ModelGraph(root_node_id="model", overall_confidence=0.6)

        if not hasattr(self.model, "named_modules"):
            return graph

        layer_candidates = {}
        for name, mod in self.model.named_modules():
            if name == "":
                continue

            n_lower = name.lower()
            params = list(mod.parameters(recurse=False)) if hasattr(mod, "parameters") else []
            p_count = sum(p.numel() for p in params if hasattr(p, "numel"))

            # Detect Embeddings
            if isinstance(mod, nn.Embedding) or "embed" in n_lower:
                graph.add_discovered_module("embed", DiscoveredModule(
                    module_path=name, module=mod, confidence=0.85, reason="nn.Embedding or 'embed' in name", param_count=p_count, sublayer_type="embedding"
                ))

            # Detect LM Head
            elif isinstance(mod, nn.Linear) and ("lm_head" in n_lower or "unembed" in n_lower or "output" in n_lower):
                graph.add_discovered_module("lm_head", DiscoveredModule(
                    module_path=name, module=mod, confidence=0.9, reason="nn.Linear with 'lm_head/unembed' name", param_count=p_count, sublayer_type="lm_head"
                ))

            # Detect Layer Blocks
            elif "layer" in n_lower or "block" in n_lower or "h." in n_lower:
                parts = name.split(".")
                for p in parts:
                    if p.isdigit():
                        l_idx = int(p)
                        if l_idx not in layer_candidates:
                            layer_candidates[l_idx] = name
                            graph.add_node(GraphNode(node_id=f"block_{l_idx}", name=f"Layer {l_idx}", node_type="block", layer_idx=l_idx, confidence=0.7))

        return graph

    def _evaluate_capabilities(self, graph: ModelGraph, strategy: BaseArchitectureStrategy) -> ModelCapability:
        """Determines raw ModelCapability and CapabilityLevel based on discovered graph modules."""
        cap = ModelCapability()

        # 1. Unembedding
        unembed = strategy.extract_unembedding(self.model)
        if unembed is not None:
            cap.has_unembedding = True
        elif "lm_head" in graph.discovered_modules:
            cap.has_unembedding = True

        # 2. Residual stream
        if len(graph.nodes) > 0 or len(graph.discovered_modules) > 0:
            cap.has_residual_stream = True

        # 3. Attention maps
        has_attn = any(m.sublayer_type == "mhsa" or "attn" in m.module_path for m in graph.discovered_modules.values())
        if has_attn or getattr(getattr(self.model, "config", None), "output_attentions", False):
            cap.has_attention_maps = True

        # 4. MLP activations
        has_mlp = any(m.sublayer_type == "mlp" or "mlp" in m.module_path for m in graph.discovered_modules.values())
        if has_mlp:
            cap.has_mlp_activations = True

        # Determine Progressive CapabilityLevel (Levels 0–5)
        if cap.has_unembedding and cap.has_residual_stream and cap.has_attention_maps and cap.has_mlp_activations:
            cap.capability_level = CapabilityLevel.LEVEL_5_FULL_SUPPORT
        elif cap.has_residual_stream and cap.has_attention_maps and cap.has_mlp_activations:
            cap.capability_level = CapabilityLevel.LEVEL_4_NEURON_ACTS
        elif cap.has_residual_stream and cap.has_attention_maps:
            cap.capability_level = CapabilityLevel.LEVEL_3_ATTENTION_MAPS
        elif cap.has_residual_stream:
            cap.capability_level = CapabilityLevel.LEVEL_2_RESIDUAL_HOOKS
        elif cap.has_unembedding:
            cap.capability_level = CapabilityLevel.LEVEL_1_EMBEDDINGS
        else:
            cap.capability_level = CapabilityLevel.LEVEL_0_LOADED_ONLY

        return cap

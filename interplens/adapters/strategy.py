"""Strategy objects for model architecture discovery, unembedding extraction, and hook resolution."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn

from interplens.adapters.graph import ModelGraph, GraphNode, DiscoveredModule
from interplens.adapters.fingerprint import StaticFingerprint


class ArchitectureFamily(str, Enum):
    TRANSFORMER = "transformer"
    STATE_SPACE = "state_space"  # Mamba, RWKV, Hyena
    MOE = "moe"
    RNN = "rnn"
    CNN = "cnn"
    ENCODER_DECODER = "encoder_decoder"
    GENERIC = "generic"


class BaseArchitectureStrategy(ABC):
    """Abstract Strategy interface for model family discovery and hook resolution."""

    architecture_id: str = "generic"
    family: ArchitectureFamily = ArchitectureFamily.TRANSFORMER

    @abstractmethod
    def matches(self, model_or_config: Any, model_name: str = "") -> float:
        """Returns match confidence score (0.0 to 1.0) for target model instance or config."""
        pass

    @abstractmethod
    def discover_graph(self, model: nn.Module) -> ModelGraph:
        """Constructs ModelGraph computational topology graph with module mappings."""
        pass

    @abstractmethod
    def extract_unembedding(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Extracts unembedding weight tensor W_U [hidden_dim, vocab_size]."""
        pass

    @abstractmethod
    def get_resid_post_hook_name(self, layer: int) -> str:
        """Returns hook key for post-layer residual stream vector."""
        pass

    @abstractmethod
    def get_attn_pattern_hook_name(self, layer: int) -> str:
        """Returns hook key for layer attention pattern matrix."""
        pass

    @abstractmethod
    def get_mlp_post_hook_name(self, layer: int) -> str:
        """Returns hook key for post-MLP activation vector."""
        pass


class LlamaStrategy(BaseArchitectureStrategy):
    architecture_id = "llama"
    family = ArchitectureFamily.TRANSFORMER

    def matches(self, model_or_config: Any, model_name: str = "") -> float:
        m_type = getattr(getattr(model_or_config, "config", None), "model_type", "").lower()
        m_str = model_name.lower()
        if "llama" in m_type or "llama" in m_str:
            return 1.0
        return 0.0

    def discover_graph(self, model: nn.Module) -> ModelGraph:
        graph = ModelGraph(root_node_id="llama", overall_confidence=1.0)

        # 1. Embeddings
        embed = getattr(getattr(model, "model", model), "embed_tokens", None)
        if embed is not None:
            graph.add_discovered_module("embed", DiscoveredModule(
                module_path="model.embed_tokens", module=embed, confidence=1.0, reason="Llama embed_tokens match", sublayer_type="embedding"
            ))

        # 2. Layers
        layers = getattr(getattr(model, "model", model), "layers", [])
        for idx, layer_mod in enumerate(layers):
            block_id = f"block_{idx}"
            graph.add_node(GraphNode(node_id=block_id, name=f"Layer {idx}", node_type="block", layer_idx=idx, confidence=1.0))

            # Attention
            self_attn = getattr(layer_mod, "self_attn", None)
            if self_attn is not None:
                graph.add_discovered_module(f"layers.{idx}.attn", DiscoveredModule(
                    module_path=f"model.layers.{idx}.self_attn", module=self_attn, confidence=1.0, reason="Llama self_attn match", sublayer_type="mhsa"
                ))

            # MLP
            mlp = getattr(layer_mod, "mlp", None)
            if mlp is not None:
                graph.add_discovered_module(f"layers.{idx}.mlp", DiscoveredModule(
                    module_path=f"model.layers.{idx}.mlp", module=mlp, confidence=1.0, reason="Llama mlp match", sublayer_type="mlp"
                ))

        # 3. LM Head
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None:
            graph.add_discovered_module("lm_head", DiscoveredModule(
                module_path="lm_head", module=lm_head, confidence=1.0, reason="Llama lm_head match", sublayer_type="lm_head"
            ))

        return graph

    def extract_unembedding(self, model: nn.Module) -> Optional[torch.Tensor]:
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            return lm_head.weight.T
        return None

    def get_resid_post_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}"

    def get_attn_pattern_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}.self_attn"

    def get_mlp_post_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}.mlp"


class QwenStrategy(BaseArchitectureStrategy):
    architecture_id = "qwen2"
    family = ArchitectureFamily.TRANSFORMER

    def matches(self, model_or_config: Any, model_name: str = "") -> float:
        m_type = getattr(getattr(model_or_config, "config", None), "model_type", "").lower()
        m_str = model_name.lower()
        if "qwen" in m_type or "qwen" in m_str:
            return 1.0
        return 0.0

    def discover_graph(self, model: nn.Module) -> ModelGraph:
        graph = ModelGraph(root_node_id="qwen2", overall_confidence=1.0)
        layers = getattr(getattr(model, "model", model), "layers", [])
        for idx, layer_mod in enumerate(layers):
            graph.add_node(GraphNode(node_id=f"block_{idx}", name=f"Layer {idx}", node_type="block", layer_idx=idx, confidence=1.0))
            if hasattr(layer_mod, "self_attn"):
                graph.add_discovered_module(f"layers.{idx}.attn", DiscoveredModule(
                    module_path=f"model.layers.{idx}.self_attn", module=layer_mod.self_attn, confidence=1.0, reason="Qwen self_attn match", sublayer_type="mhsa"
                ))
            if hasattr(layer_mod, "mlp"):
                graph.add_discovered_module(f"layers.{idx}.mlp", DiscoveredModule(
                    module_path=f"model.layers.{idx}.mlp", module=layer_mod.mlp, confidence=1.0, reason="Qwen mlp match", sublayer_type="mlp"
                ))
        return graph

    def extract_unembedding(self, model: nn.Module) -> Optional[torch.Tensor]:
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            return lm_head.weight.T
        return None

    def get_resid_post_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}"

    def get_attn_pattern_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}.self_attn"

    def get_mlp_post_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}.mlp"


class MistralStrategy(BaseArchitectureStrategy):
    architecture_id = "mistral"
    family = ArchitectureFamily.TRANSFORMER

    def matches(self, model_or_config: Any, model_name: str = "") -> float:
        m_type = getattr(getattr(model_or_config, "config", None), "model_type", "").lower()
        m_str = model_name.lower()
        if "mistral" in m_type or "mistral" in m_str:
            return 1.0
        return 0.0

    def discover_graph(self, model: nn.Module) -> ModelGraph:
        # Mistral shares Llama architecture topology layout
        return LlamaStrategy().discover_graph(model)

    def extract_unembedding(self, model: nn.Module) -> Optional[torch.Tensor]:
        return LlamaStrategy().extract_unembedding(model)

    def get_resid_post_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}"

    def get_attn_pattern_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}.self_attn"

    def get_mlp_post_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}.mlp"


class GemmaStrategy(BaseArchitectureStrategy):
    architecture_id = "gemma"
    family = ArchitectureFamily.TRANSFORMER

    def matches(self, model_or_config: Any, model_name: str = "") -> float:
        m_type = getattr(getattr(model_or_config, "config", None), "model_type", "").lower()
        m_str = model_name.lower()
        if "gemma" in m_type or "gemma" in m_str:
            return 1.0
        return 0.0

    def discover_graph(self, model: nn.Module) -> ModelGraph:
        return LlamaStrategy().discover_graph(model)

    def extract_unembedding(self, model: nn.Module) -> Optional[torch.Tensor]:
        return LlamaStrategy().extract_unembedding(model)

    def get_resid_post_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}"

    def get_attn_pattern_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}.self_attn"

    def get_mlp_post_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}.mlp"


class PhiStrategy(BaseArchitectureStrategy):
    architecture_id = "phi"
    family = ArchitectureFamily.TRANSFORMER

    def matches(self, model_or_config: Any, model_name: str = "") -> float:
        m_type = getattr(getattr(model_or_config, "config", None), "model_type", "").lower()
        m_str = model_name.lower().split("/")[-1]
        if m_type.startswith("phi") or m_str.startswith("phi") or "phi-1" in m_str or "phi-2" in m_str or "phi-3" in m_str:
            return 1.0
        return 0.0

    def discover_graph(self, model: nn.Module) -> ModelGraph:
        graph = ModelGraph(root_node_id="phi", overall_confidence=1.0)
        layers = getattr(getattr(model, "model", model), "layers", getattr(model, "h", []))
        for idx, layer_mod in enumerate(layers):
            graph.add_node(GraphNode(node_id=f"block_{idx}", name=f"Layer {idx}", node_type="block", layer_idx=idx, confidence=1.0))
        return graph

    def extract_unembedding(self, model: nn.Module) -> Optional[torch.Tensor]:
        lm_head = getattr(model, "lm_head", getattr(model, "embed_out", None))
        if lm_head is not None and hasattr(lm_head, "weight"):
            return lm_head.weight.T
        return None

    def get_resid_post_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}"

    def get_attn_pattern_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}.self_attn"

    def get_mlp_post_hook_name(self, layer: int) -> str:
        return f"model.layers.{layer}.mlp"


class GPT2Strategy(BaseArchitectureStrategy):
    architecture_id = "gpt2"
    family = ArchitectureFamily.TRANSFORMER

    def matches(self, model_or_config: Any, model_name: str = "") -> float:
        m_type = getattr(getattr(model_or_config, "config", None), "model_type", "").lower()
        m_str = model_name.lower()
        if "gpt2" in m_type or "gpt2" in m_str or "pythia" in m_str:
            return 1.0
        return 0.0

    def discover_graph(self, model: nn.Module) -> ModelGraph:
        graph = ModelGraph(root_node_id="gpt2", overall_confidence=1.0)
        blocks = getattr(model, "blocks", getattr(getattr(model, "transformer", model), "h", []))
        for idx, b_mod in enumerate(blocks):
            graph.add_node(GraphNode(node_id=f"block_{idx}", name=f"Layer {idx}", node_type="block", layer_idx=idx, confidence=1.0))
        return graph

    def extract_unembedding(self, model: nn.Module) -> Optional[torch.Tensor]:
        if hasattr(model, "W_U"):
            return model.W_U
        elif hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            return model.lm_head.weight.T
        return None

    def get_resid_post_hook_name(self, layer: int) -> str:
        return f"blocks.{layer}.hook_resid_post"

    def get_attn_pattern_hook_name(self, layer: int) -> str:
        return f"blocks.{layer}.attn.hook_pattern"

    def get_mlp_post_hook_name(self, layer: int) -> str:
        return f"blocks.{layer}.mlp.hook_post"

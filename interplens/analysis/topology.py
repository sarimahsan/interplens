"""Model Architecture Topology Inspector and Graph Spec Generator for InterpLens."""

import math
from typing import Dict, Any, List, Optional
import torch

try:
    from ..adapters.base import BaseModelAdapter
except ImportError:
    from interplens.adapters.base import BaseModelAdapter


def inspect_model_topology(adapter: BaseModelAdapter) -> Dict[str, Any]:
    """Dynamically inspects any loaded PyTorch/Transformers model and generates a structured topology diagram spec."""
    model = getattr(adapter, "model", None) or getattr(adapter, "_model_instance", None)
    info = adapter.get_model_info() if hasattr(adapter, "get_model_info") else {}

    model_name = getattr(adapter, "model_name", "Transformer Model")
    device_str = str(getattr(adapter, "device", "cpu"))
    
    num_layers = info.get("num_layers", 12)
    hidden_dim = info.get("hidden_dim", 768)
    num_heads = info.get("num_heads", 12)
    head_dim = hidden_dim // max(1, num_heads)
    vocab_size = info.get("vocab_size", 50257)

    # Inspect config parameters if available
    config = getattr(model, "config", None)
    intermediate_size = getattr(config, "intermediate_size", None) or getattr(config, "n_inner", None) or (hidden_dim * 4)
    act_fn = getattr(config, "hidden_act", None) or getattr(config, "activation_function", None) or "GELU / SiLU"
    norm_eps = getattr(config, "layer_norm_epsilon", None) or getattr(config, "rms_norm_eps", None) or 1e-5
    
    # Calculate parameter counts dynamically
    total_params = 0
    embed_params = 0
    block_params = 0
    head_params = 0
    dtype_str = "float32"

    if model is not None and isinstance(model, torch.nn.Module):
        for name, param in model.named_parameters():
            p_count = param.numel()
            total_params += p_count
            dtype_str = str(param.dtype).replace("torch.", "")
            
            if "embed" in name.lower() or "wte" in name.lower() or "wpe" in name.lower():
                embed_params += p_count
            elif "head" in name.lower() or "unembed" in name.lower() or "lm_head" in name.lower():
                head_params += p_count

        block_params = (total_params - embed_params - head_params) // max(1, num_layers)

    # Format human readable parameter count (e.g., 124M, 3.0B)
    def fmt_params(n: int) -> str:
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        elif n >= 1e6:
            return f"{n / 1e6:.1f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)

    # Build node pipeline topology nodes
    nodes = [
        {
            "id": "input_tokens",
            "type": "input",
            "title": "Input Text Tokens",
            "subtitle": f"Tokenizer (Vocab: {vocab_size:,})",
            "description": "Raw text prompt tokenized into input token ID sequences [Batch, SeqLen].",
            "color": "#0284c7",
            "badge": "Input"
        },
        {
            "id": "embedding_layer",
            "type": "embedding",
            "title": "Token & Position Embeddings",
            "subtitle": f"W_E: [{vocab_size:,} × {hidden_dim}]",
            "description": f"Maps discrete token IDs to dense d_model={hidden_dim} vector space + positional encoding (RoPE/Absolute).",
            "color": "#06b6d4",
            "badge": f"{fmt_params(embed_params)} params"
        }
    ]

    # Generate sample block representations (Layer 0, Layer 1 ... Layer N-1)
    nodes.append({
        "id": "residual_stream_bus",
        "type": "highway",
        "title": f"Residual Stream Highway ({num_layers} Transformer Blocks)",
        "subtitle": f"Hidden State Dimension d_model = {hidden_dim}",
        "description": "Central memory stream connecting all layers via additive residual skip connections: x_{l+1} = x_l + Attn(x_l) + MLP(x_l).",
        "color": "#3b82f6",
        "badge": f"{num_layers} Layers"
    })

    nodes.append({
        "id": "attn_sublayer",
        "type": "attention",
        "title": "Multi-Head Self-Attention (MHSA)",
        "subtitle": f"{num_heads} Heads × {head_dim}d per Head",
        "description": f"Computes Query, Key, Value projections. Softmax(Q K^T / √d_k) V mixes information across token positions.",
        "color": "#8b5cf6",
        "badge": "QKV + Output Proj"
    })

    nodes.append({
        "id": "mlp_sublayer",
        "type": "mlp",
        "title": "Feed-Forward Sublayer (MLP / FFN)",
        "subtitle": f"Expansion: {hidden_dim} → {intermediate_size} → {hidden_dim}",
        "description": f"Non-linear point-wise feature extraction network using {act_fn} activation function.",
        "color": "#ec4899",
        "badge": f"{act_fn}"
    })

    nodes.append({
        "id": "final_norm",
        "type": "norm",
        "title": "Final LayerNorm / RMSNorm",
        "subtitle": f"Dimension: {hidden_dim} (eps={norm_eps})",
        "description": "Standardizes final residual stream vectors across feature dimensions before projecting to vocabulary.",
        "color": "#10b981",
        "badge": "Norm"
    })

    nodes.append({
        "id": "unembedding_layer",
        "type": "output",
        "title": "Unembedding Head (W_U / lm_head)",
        "subtitle": f"Projection: [{hidden_dim} → {vocab_size:,}]",
        "description": f"Matrix multiplication projecting final hidden vectors to raw unnormalized vocabulary logits, followed by Softmax for token probabilities.",
        "color": "#f59e0b",
        "badge": f"Logits [{vocab_size:,}]"
    })

    return {
        "model_name": model_name,
        "device": device_str,
        "dtype": dtype_str,
        "total_parameters": total_params,
        "total_parameters_formatted": fmt_params(total_params),
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "activation_function": str(act_fn),
        "nodes": nodes,
    }

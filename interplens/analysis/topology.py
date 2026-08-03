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
    
    # Calculate parameter counts dynamically per component
    total_params = 0
    embed_params = 0
    attn_params = 0
    mlp_params = 0
    norm_params = 0
    unembed_params = 0
    other_params = 0
    dtype_str = "float32"

    if model is not None and isinstance(model, torch.nn.Module):
        for name, param in model.named_parameters():
            p_count = param.numel()
            total_params += p_count
            dtype_str = str(param.dtype).replace("torch.", "")
            name_lower = name.lower()

            if "embed" in name_lower or "wte" in name_lower or "wpe" in name_lower:
                embed_params += p_count
            elif "lm_head" in name_lower or "unembed" in name_lower or "output_layer" in name_lower:
                unembed_params += p_count
            elif "attn" in name_lower or "attention" in name_lower or "self_attn" in name_lower or "q_proj" in name_lower or "k_proj" in name_lower or "v_proj" in name_lower or "o_proj" in name_lower or "c_attn" in name_lower or "c_proj" in name_lower:
                attn_params += p_count
            elif "mlp" in name_lower or "ffn" in name_lower or "feed_forward" in name_lower or "gate" in name_lower or "up_proj" in name_lower or "down_proj" in name_lower or "c_fc" in name_lower:
                mlp_params += p_count
            elif "norm" in name_lower or "ln_" in name_lower or "ln1" in name_lower or "ln2" in name_lower:
                norm_params += p_count
            else:
                other_params += p_count

    # If untied unembedding is 0 because of weight tying with embeddings
    if total_params > 0 and unembed_params == 0 and embed_params > 0:
        # Note weight tying
        unembed_params_note = "Tied with Embedding"
    else:
        unembed_params_note = None

    def fmt_params(n: int) -> str:
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        elif n >= 1e6:
            return f"{n / 1e6:.1f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)

    # Parameter Breakdown payload
    denom = max(1, total_params)
    parameter_breakdown = [
        {
            "category": "MLP / Feed-Forward Sublayers",
            "count": mlp_params,
            "formatted": fmt_params(mlp_params),
            "percentage": round((mlp_params / denom) * 100, 2),
            "color": "#ec4899",
        },
        {
            "category": "Multi-Head Attention (MHSA)",
            "count": attn_params,
            "formatted": fmt_params(attn_params),
            "percentage": round((attn_params / denom) * 100, 2),
            "color": "#8b5cf6",
        },
        {
            "category": "Token & Position Embeddings",
            "count": embed_params,
            "formatted": fmt_params(embed_params),
            "percentage": round((embed_params / denom) * 100, 2),
            "color": "#06b6d4",
        },
        {
            "category": "Unembedding / LM Head",
            "count": unembed_params,
            "formatted": fmt_params(unembed_params),
            "percentage": round((unembed_params / denom) * 100, 2),
            "note": unembed_params_note,
            "color": "#f59e0b",
        },
        {
            "category": "Layer Normalizations (LN/RMSNorm)",
            "count": norm_params,
            "formatted": fmt_params(norm_params),
            "percentage": round((norm_params / denom) * 100, 2),
            "color": "#10b981",
        },
    ]

    if other_params > 0:
        parameter_breakdown.append({
            "category": "Other Parameters / Biases",
            "count": other_params,
            "formatted": fmt_params(other_params),
            "percentage": round((other_params / denom) * 100, 2),
            "color": "#64748b",
        })

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
        "parameter_breakdown": parameter_breakdown,
        "nodes": nodes,
    }

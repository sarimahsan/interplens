<div align="center">

# InterpLens

**Interactive Mechanistic Interpretability & Circuit Debugger for PyTorch & TransformerLens**

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-38BDF8?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/interplens/)
[![Python 3.9 - 3.13](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-818CF8?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/interplens/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-36%2F36%20passed-10B981?style=flat-square&logo=checkmarx&logoColor=white)](https://github.com/sarimahsan/interplens)
[![Type Checked](https://img.shields.io/badge/mypy-PEP%20561-38BDF8?style=flat-square)](https://github.com/sarimahsan/interplens)
[![License: MIT](https://img.shields.io/badge/license-MIT-A855F7?style=flat-square)](LICENSE)

<p align="center">
  <b><a href="README.md">README</a></b> •
  <a href="CODE_OF_CONDUCT.md">Code of Conduct</a> •
  <a href="CONTRIBUTING.md">Contributing</a> •
  <a href="LICENSE">License</a> •
  <a href="SECURITY.md">Security</a>
</p>

<p align="center">
  <a href="#-background--motivation">Background</a> •
  <a href="#-key-features--engines">Engines</a> •
  <a href="#-major-supported-models">Supported Models</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quickstart-recipes">Quickstart</a> •
  <a href="#-capability-matrix-l0--l4">Capabilities</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

---

## 📖 Background & Motivation

Mechanistic interpretability aims to reverse-engineer neural networks from weights and activations into human-understandable circuits and computational graphs. However, researchers, students, and engineers frequently encounter significant friction:

- **Notebook Fragmentation:** Analysis scripts require hundreds of lines of bespoke PyTorch hooking, unembedding matrix projection, and Matplotlib code for every new model.
- **Redundant VRAM Duplication:** Existing tools often require reloading entire model checkpoints into separate processes, exhausting GPU memory on single-GPU workstations.
- **Architecture Fragility:** Minor differences between HuggingFace model layer names (e.g. `model.layers`, `gpt_neox.layers`, `transformer.h`) break ad-hoc analysis scripts.

**InterpLens** bridges this gap by providing an **interactive visual debugger and real-time interpretability workbench**. Built natively for PyTorch and TransformerLens, InterpLens attaches to pre-loaded models with **zero memory duplication**, dynamically resolves internal module topologies, and renders real-time telemetry across an ergonomic visual interface.

---

## ⚡ Key Features & Engines

### 1. 🔍 Logit Lens & Unembedding Projector
Projects intermediate layer residual stream states $x_l$ directly into vocabulary space via the unembedding matrix $W_U$:

$$\text{logits}_l = W_U \cdot \text{LN}_{\text{final}}(x_l) \quad \in \mathbb{R}^{V}$$

$$p_l(v) = \text{softmax}(\text{logits}_l) = \frac{\exp(\text{logits}_l(v))}{\sum_{v'} \exp(\text{logits}_l(v'))}$$

- **Prediction Entropy Dynamics:** $H(p_l) = -\sum_{v} p_l(v) \log p_l(v)$ tracks the layer where confidence crystallizes.
- **KL Divergence Drift:** $D_{KL}(p_L \parallel p_l)$ quantifies intermediate layer fidelity to the final output token distribution.

### 2. 🌊 Residual Stream Inspector
Tracks the geometry and drift of hidden state vectors as information propagates through the network:
- **L2 Vector Energy:** Monitors $\|x_l\|_2$ norm expansion across transformer depth.
- **Cosine Drift Heatmap:** $\cos(x_l, x_{l+1})$ identifies layers responsible for major semantic transitions.
- **PCA Layer Trajectories:** Visualizes residual stream dynamics projected onto top principal components.

### 3. 🕸️ Attention Head & Arc Explorer
- **$N \times N$ Attention Matrices:** Interactive heatmaps of query-key attention patterns across all layers and heads.
- **Arc Diagram Visualization:** Highlights long-range token-to-token information routing.
- **Automated Induction Head Detector:** Automatically detects prefix-matching $[A][B] \dots [A] \rightarrow [B]$ copying circuits across all attention heads.

### 4. ⚡ Neuron Activation & Token Attribution
- **Top-K Firing Neurons:** Identifies and ranks the highest-activating MLP neurons for any given prompt position.
- **Token Activation Strips:** Highlights exact prompt tokens that trigger specific polysemantic or monosemantic neurons.

### 5. 🎯 Causal Patching & ROME Sweeps
- **Automated Causal Tracing:** Swaps activations between clean and corrupted prompts across layers and token positions to pinpoint exact causal circuits.
- **Activation Steering Vectors:** Injects scaled direction vectors ($x_l \leftarrow x_l + \alpha \cdot v$) during the forward pass to steer model behavior in real time.

---

## 🤖 Major Supported Models

InterpLens features native auto-discovery and universal hooking for all major open-weights decoder-only transformers:

| Model Family | Checkpoints & Variants | Level | Hook Strategy | Launch Command |
| :--- | :--- | :---: | :--- | :--- |
| **Meta Llama 3 / 3.2** | `Llama-3.2-1B`, `3B`, `Llama-3-8B`, `70B`, `Llama-2-7B` | **L4** | `LlamaStrategy` (`model.layers[i]`) | `interplens launch --model meta-llama/Llama-3.2-1B --hf-token "hf_..."` |
| **Alibaba Qwen 2.5** | `Qwen2.5-0.5B`, `1.5B`, `3B`, `7B`, `14B`, `32B`, `Coder-7B` | **L4** | `QwenStrategy` (`model.layers[i]`) | `interplens launch --model Qwen/Qwen2.5-0.5B --device cuda` |
| **Mistral & Mixtral** | `Mistral-7B-v0.1`, `v0.3`, `Mistral-7B-Instruct`, `Mixtral-8x7B` | **L4** | `MistralStrategy` (`model.layers[i]`) | `interplens launch --model mistralai/Mistral-7B-v0.1` |
| **Google Gemma / Gemma 2** | `gemma-2b`, `gemma-7b`, `gemma-2-2b`, `gemma-2-9b`, `27b` | **L4** | `GemmaStrategy` (`model.model.layers[i]`) | `interplens launch --model google/gemma-2b` |
| **OpenAI GPT-2 Family** | `gpt2` (124M), `gpt2-medium` (355M), `large` (774M), `xl` (1.5B) | **L4** | `GPT2Strategy` / `HookedTransformer` | `interplens launch --model gpt2 --device cuda` |
| **EleutherAI Pythia & NeoX** | `pythia-70m`, `160m`, `410m`, `1b`, `1.4b`, `2.8b`, `6.9b`, `12b` | **L4** | `PythiaStrategy` (`gpt_neox.layers[i]`) | `interplens launch --model EleutherAI/pythia-70m` |
| **SmolLM & TinyLlama** | `SmolLM-135M`, `SmolLM-360M`, `SmolLM-1.7B`, `TinyLlama-1.1B` | **L4** | `LlamaStrategy` (`model.layers[i]`) | `interplens launch --model HuggingFaceTB/SmolLM-135M` |
| **DeepSeek Family** | `deepseek-llm-7b-base`, `deepseek-coder-1.3b`, `coder-6.7b` | **L4** | `LlamaStrategy` / `GenericAdapter` | `interplens launch --model deepseek-ai/deepseek-llm-7b-base` |
| **Custom PyTorch Modules** | Any custom `nn.Module` or research architecture | **L3–L4** | `PyTorchAutoHooker` (`auto_hook=True`) | `il.launch(model=my_model, auto_hook=True)` |

---

## 📦 Installation

### Standard Installation
Installs core PyTorch engine, FastAPI telemetry server, and the visual debugger UI:
```bash
pip install interplens
```

### Full Research Bundle (Recommended)
Includes TransformerLens native engine, HuggingFace transformers, and Scikit-Learn PCA modules:
```bash
pip install interplens[all]
```

### Development Installation
```bash
git clone https://github.com/sarimahsan/interplens.git
cd interplens
pip install -e .[all,dev]
pytest
```

---

## 🚀 Quickstart Recipes

### Recipe 1: Zero-Copy GPU Attach (Researcher Workflow)
Attach the visual debugger UI directly to a model already resident in your GPU VRAM:
```python
import interplens as il
from transformer_lens import HookedTransformer

# Model is already active in GPU VRAM (e.g. inside a research script or notebook)
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")

# Launch visual workbench without duplicating memory
server = il.launch(model=model, port=8000)
```

### Recipe 2: HuggingFace Pretrained Auto-Load
```python
import interplens as il

# Automatically downloads weights, resolves tokenizers, and hooks internal layers
server = il.launch(
    model_name="Qwen/Qwen2.5-0.5B",
    device="cuda",
    dtype="bfloat16"
)
```

### Recipe 3: Gated Models with Authentication
```python
import interplens as il

# Load gated Llama-3 checkpoints securely
server = il.launch(
    model_name="meta-llama/Llama-3.2-1B",
    hf_token="hf_YourHuggingFaceTokenHere",
    device="cuda"
)
```

### Recipe 4: Arbitrary Custom PyTorch `nn.Module`
```python
import torch
import torch.nn as nn
import interplens as il

class ResearchTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.unembed(h)

model = ResearchTransformer()
# Automatic forward hook discovery & unembedding binding
server = il.launch(model=model, auto_hook=True, port=8000)
```

### Recipe 5: CLI One-Liner Launch
```bash
# Launch interactive studio from terminal
interplens launch --model gpt2 --device cuda --port 8000
```

---

## 🛡️ Capability Matrix (L0 – L4)

InterpLens dynamically evaluates and adapts available analysis engines according to model features:

```
┌────────────────────────────────────────────────────────────────────────┐
│  Level 4: Full Interventional (Causal Sweeps, Steering Vectors)        │
│  Level 3: Component Hooked (Attention Matrices, MLP Neurons)           │
│  Level 2: Unembed Ready (Logit Lens, Entropy, KL Divergence)           │
│  Level 1: Residual Only (Energy Norms, Cosine Drift Heatmaps)          │
│  Level 0: Black-Box (Top-K Vocabulary Output Predictions)              │
└────────────────────────────────────────────────────────────────────────┘
```

If a custom model lacks attention extraction or an unembedding matrix $W_U$, InterpLens **gracefully degrades** rather than crashing, keeping all valid diagnostic tools functional.

---

## ⚙️ Environment Configuration

| Variable | Default | Description |
| :--- | :---: | :--- |
| `INTERPLENS_PORT` | `8000` | HTTP & WebSocket server port. |
| `INTERPLENS_HOST` | `127.0.0.1` | Host binding address (`0.0.0.0` for remote cloud VMs). |
| `INTERPLENS_DEVICE` | `auto` | Primary compute device (`cuda`, `mps`, `cpu`). |
| `INTERPLENS_MAX_CACHE_SESSIONS` | `32` | LRU in-memory session cache capacity. |
| `INTERPLENS_GPU_POLL_INTERVAL_MS` | `1000` | Live GPU telemetry polling interval in milliseconds. |

---

## 🛑 Exception Hierarchy

Structured exceptions under `interplens.exceptions` for programmatic error handling:

```python
import interplens as il
from interplens.exceptions import (
    InterpLensError,
    ModelLoadError,
    AdapterNotFoundError,
    UnembeddingNotFoundError,
    CapabilityError
)

try:
    adapter = il.launch(model_name="meta-llama/Llama-3.2-1B")
except ModelLoadError as e:
    print(f"Failed to load weights/tokenizer: {e}")
except UnembeddingNotFoundError as e:
    print(f"Model lacks W_U projection: {e}")
except InterpLensError as e:
    print(f"InterpLens runtime exception: {e}")
```

---

## 👤 Author

**Syed Sarim Ahsan**  
*Undergrad AI Researcher*  
- GitHub: [@sarimahsan](https://github.com/sarimahsan)
- Project: [InterpLens on GitHub](https://github.com/sarimahsan/interplens)

---

## 📚 Citation

If you use InterpLens in your academic research, interpretability experiments, or course work, please cite:

```bibtex
@software{ahsan2026interplens,
  author = {Syed Sarim Ahsan},
  title = {InterpLens: An Interactive Mechanistic Interpretability Toolkit and Visual Debugger for LLMs},
  year = {2026},
  url = {https://github.com/sarimahsan/interplens},
  version = {0.1.0}
}
```

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

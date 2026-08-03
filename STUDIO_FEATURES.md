# InterpLens Studio • Micro-Details & Feature Matrix

> **Comprehensive Feature Specification and Micro-Details Document**  
> *InterpLens Studio v0.1.0 — Real-Time Mechanistic Interpretability Debugger*

---

## 🏛️ System Architecture & Model Compatibility

- **Zero-Copy Live GPU Attachment**: Pass any active PyTorch model directly via `interplens.launch(model=model)` without reloading model weights or duplicating GPU VRAM.
- **Dual Adapter Architecture**:
  - **TransformerLens Adapter**: Native integration with HookedTransformer models.
  - **PyTorch Auto-Hooker Adapter**: Automatic forward-hook registration on standard HuggingFace `AutoModelForCausalLM` models.
- **Universal Model Coverage**: Supports `GPT-2`, `Qwen 2.5` (e.g., `Qwen2.5-3B`), `Llama 3`, `Mistral`, `Gemma`, `Pythia`, and custom PyTorch modules.
- **High-Performance FastAPI Server**: RESTful backend with live memory-mapped session caching.
- **Modern Glassmorphism UI**: Zero-dependency Vanilla CSS & JavaScript workbench with dark-mode aesthetic tokens and responsive grid layouts.

---

## 🔍 Feature Matrix & Micro-Details

### 1. Logit Lens Engine (Phase 1)
- **Layer-by-Layer Prediction Unembedding**: Projects intermediate residual stream vectors $x_l$ directly into vocabulary logit space using $W_U \cdot \text{LN}(x_l)$.
- **Interactive Logit Lens Matrix Grid**: Displays Top-$K$ predicted tokens and prediction probability percentages across every layer ($L_0 \dots L_{last}$) for all prompt token positions.
- **Metric Toggles**:
  - 📊 **Token Probabilities**: Exact confidence percentage per layer.
  - 📉 **KL Divergence**: Relative entropy divergence $D_{KL}(P_{layer} \parallel P_{final})$ measuring prediction convergence.
  - 🌡️ **Layer Entropy**: Information entropy $H(P_{layer})$ capturing model uncertainty across layers.
  - 📈 **Top-5 Trajectories**: Layer-by-layer probability paths for top token predictions.
- **LayerNorm Toggle**: Switch between applying final LayerNorm or projecting raw residual vectors.

---

### 2. Residual Stream Inspector & Hardware Telemetry (Phase 2)
- **L2 Norm Progression Curve**: Tracks vector magnitude $\|x_l\|_2$ growth across layers to observe representation accumulation.
- **Inter-Layer Cosine Similarity Heatmap**: $N \times N$ cosine similarity matrix ($\cos(x_i, x_j)$) highlighting phase transitions and representation drift between layers.
- **Real-Time GPU Hardware Telemetry**:
  - 💾 **VRAM Allocated & Reserved**: Live CUDA memory tracking.
  - ⚡ **Execution Latency**: Millisecond timing sparklines per forward pass.
  - 🖥️ **System Resource Gauges**: GPU utilization and memory allocation meters.

---

### 3. Attention Head Explorer (Phase 3)
- **Single-Head Attention Heatmap ($N \times N$)**: Visualizes exact query-key attention weight matrices for any target layer and attention head.
- **Multi-Head Grid View**: Displays simultaneous thumbnail heatmaps for all attention heads in a layer ($H_0 \dots H_{num\_heads-1}$) for fast visual inspection of head specialization.
- **Canvas Arc Diagram Visualizer**: Interactive HTML5 Canvas diagram connecting source query tokens to target key tokens via curved arc links with adjustable connection weight thresholds.
- **Eager Attention Support**: Leverages HuggingFace `attn_implementation="eager"` to capture non-causal-masked per-head attention weights across all model layers.

---

### 4. Neuron Activations & Token Attribution Engine (Phase 4)
- **Top-$K$ Firing MLP Neurons Bar Chart**: Ranks intermediate MLP activation values per token position to identify dominant active memory units.
- **Configurable Top-$K$ Selector**: Dropdown selector allowing users to toggle between Top 5, Top 10, Top 20, or Top 50 firing neurons.
- **Single Neuron Prompt Text Lighting Strip**: Select any individual neuron to highlight all prompt words with green background opacity (`rgba(52, 211, 153, opacity)`) proportional to how strongly that neuron fired.
- **Input Token Influence (Attention Rollout)**: Computes multi-layer Attention Rollout ($R = A_0 \cdot A_1 \dots A_L$) to measure the causal contribution of each input token to predictions.

---

### 5. Model Architecture Topology & Parameter Breakdown
- **Sublayer Parameter Allocation Breakdown**: Dynamically inspects model parameter tensors (`model.named_parameters()`) and categorizes parameter budget into:
  - 🩷 **MLP / Feed-Forward Sublayers** (Gate, Up, Down / $W_1, W_2, W_3$ projections)
  - 💜 **Multi-Head Attention (MHSA)** ($W_Q, W_K, W_V, W_O$ projections)
  - 🩵 **Token & Position Embeddings** ($W_E$, $W_{pos}$, RoPE embeddings)
  - 🧡 **Unembedding / LM Head** ($W_U$, `lm_head`, noting weight tying if applicable)
  - 💚 **Layer Normalizations** (LayerNorm / RMSNorm weights across all blocks)
- **Multi-Color Proportion Bar & Budget Table**: Interactive horizontal stacked bar and table detailing exact parameter counts (e.g. `1.85B params`) and percentage budget shares (`61.4%`).
- **Pipeline Node Flow Diagram**: Visual execution topology diagram illustrating token processing flow through Embeddings, Residual Stream Highway, MHSA, MLP, LayerNorm, and Unembedding.

---

### 6. UI/UX & Branding Details
- **Transformer SVG Favicon**: Custom 3-layer Transformer topology SVG icon matching header branding.
- **Live Status Dot & Header Metadata**: Real-time status indicator showing model name (`gpt2`, `qwen2.5`), active device (`CUDA` / `CPU`), VRAM usage, and active session ID.
- **Preset Prompt Buttons**: Instant sample prompts for immediate experimentation.

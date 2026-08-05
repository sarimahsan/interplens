# InterpLens Studio • Micro-Details & Feature Matrix

> **Comprehensive Feature Specification and Micro-Details Document**  
> *InterpLens Studio v0.1.0 — Real-Time Mechanistic Interpretability Debugger*

---

## 🏛️ System Architecture & Universal Model Compatibility

- **Zero-Copy Live GPU Attachment**: Pass any active PyTorch or TransformerLens model directly via `interplens.launch(model=model)` without reloading model weights or duplicating GPU VRAM.
- **Dual & Universal Adapter Architecture**:
- **Plugin-Based Architecture Strategy Registry**: Strategy handlers (`LlamaStrategy`, `QwenStrategy`, `MistralStrategy`, `GemmaStrategy`, `PhiStrategy`, `GPT2Strategy`) decoupling model family rules. Public plugin API `interplens.register_architecture_strategy(strategy)` for third-party architectures (RWKV, Mamba, Hyena).
- **Confidence-Scored Hook Discovery & ModelGraph Topology**: `HookDiscovery` engine walks `named_modules()` and parameter shapes to classify `Embeddings`, `MHSA`, `MLP`, `LayerNorm`, and `LM Head` into a hierarchical `ModelGraph` with confidence scores ($0.0 \dots 1.0$).
- **Dual Fingerprint Separation**: `StaticFingerprint` (architecture, family, geometry) vs. `RuntimeFingerprint` (device, dtype, quantization, VRAM memory).
- **Dual Capability Matrix**: `ModelCapability` (Progressive Levels 0–5) + `EngineCapabilityMatrix` (`SUPPORTED`, `PARTIAL`, `UNAVAILABLE`) evaluating engine readiness dynamically.
- **Automated Model Discovery Report**: Instant diagnostic report generated upon model load, accessible via `/api/model/report` and the UI header modal.
- **`GenericAdapter` Best-Effort Fallback**: Progressive interpretability adapter (Levels 0–5) wrapping arbitrary PyTorch `nn.Module` objects.
- **Universal Multi-Tier Tokenizer Resolution & Fallback**:
  1. *Primary Tokenizer*: Attempts direct loading via target repository tokenizer.
  2. *Compatible Fallback Repo*: Automatically falls back to architecture-compatible repos (e.g. `meta-llama/Llama-3.2-1B`, `Qwen/Qwen2.5-0.5B`, `google/gemma-2b`, `microsoft/phi-2`).
  3. *Generic Fallback*: Falls back to standard GPT-2 tokenizer.
  4. *Raw ID Indexing*: Fallback to raw token ID index matching if tokenizers are unavailable.
- **VRAM Hot-Reload Model Manager**: Automatically evacuates GPU memory (`del` + CUDA cache clearing) when switching active models on the server.
- **High-Performance FastAPI REST Server**: RESTful backend providing async endpoints with CORS support, LRU session caching, and live hardware monitoring.
- **Glassmorphism Workbench UI**: Zero-dependency Vanilla CSS & JavaScript workbench featuring dark-mode design tokens, responsive grid layouts, custom SVG branding, and live Light/Dark theme switching.

---

## 🔍 Feature Matrix & Interpretability Engines

### 1. Logit Lens Engine (Phase 1)
- **Layer-by-Layer Prediction Unembedding**: Projects intermediate residual stream vectors $x_l$ directly into vocabulary logit space using $W_U \cdot \text{LN}(x_l)$.
- **Interactive Logit Lens Matrix Grid**: Displays Top-$K$ predicted tokens and prediction probability percentages across every layer ($L_0 \dots L_{last}$) for all prompt token positions.
- **Metric Toggles**:
  - 📊 **Token Probabilities**: Exact confidence percentage per layer.
  - 📉 **KL Divergence**: Relative entropy divergence $D_{KL}(P_{layer} \parallel P_{final})$ measuring prediction convergence across layers.
  - 🌡️ **Layer Entropy**: Information entropy $H(P_{layer})$ capturing model prediction uncertainty at each step.
  - 📈 **Top-5 Trajectories**: Layer-by-layer probability paths for top token predictions.
- **LayerNorm Toggle**: Switch between applying final LayerNorm or projecting raw residual vectors.
- **Position & Top-K Filters**: Dynamic filter controls for top-$K$ predictions ($1 \dots 20$) and individual token positions.

---

### 2. Residual Stream Inspector & Activation Steering (Phase 2)
- **L2 Norm Progression Curve**: Tracks vector magnitude $\|x_l\|_2$ growth across layers to observe representation accumulation and scale drift.
- **Inter-Layer Cosine Similarity Heatmap**: $N \times N$ cosine similarity matrix ($\cos(x_i, x_j)$) highlighting phase transitions, block clustering, and representation drift between layers.
- **Residual Stream Activation Steering Engine**:
  - Real-time steering vector injection into target layer residual streams during forward passes (`/api/analysis/residual-stream/steer`).
  - Configurable steering multiplier strength ($k \in [-10, 10]$) and custom direction vectors ($v \in \mathbb{R}^{d_{model}}$).
  - Dynamic comparison of baseline vs. steered prediction logits and generated text outputs.

---

### 3. Attention Head Explorer & Visualizers (Phase 3)
- **Single-Head Attention Heatmap ($N \times N$)**: Visualizes query-key attention weight matrices for any target layer and attention head.
- **Multi-Head Grid View**: Displays simultaneous thumbnail heatmaps for all attention heads in a layer ($H_0 \dots H_{num\_heads-1}$) for fast visual inspection of head specialization.
- **Interactive Canvas Arc Diagram Visualizer**: Interactive HTML5 Canvas diagram connecting source query tokens to target key tokens via curved arc links with adjustable connection weight thresholds ($[0.0, 1.0]$).
- **Eager Attention Support**: Leverages HuggingFace `attn_implementation="eager"` to capture non-causal-masked per-head attention weights across all model layers.

---

### 4. Neuron Activations & Token Attribution Engine (Phase 4)
- **Top-$K$ Firing MLP Neurons Bar Chart**: Ranks intermediate MLP activation values per token position to identify dominant active memory units.
- **Configurable Top-$K$ Selector**: Dropdown selector allowing users to toggle between Top 5, Top 10, Top 20, or Top 50 firing neurons.
- **Single Neuron Prompt Text Lighting Strip**: Select any individual neuron to highlight all prompt words with green background opacity (`rgba(52, 211, 153, opacity)`) proportional to how strongly that neuron fired.
- **Input Token Attribution Engine (Attention Rollout)**: Computes multi-layer Attention Rollout ($R = A_0 \cdot A_1 \dots A_L$) to measure the causal contribution of each input token to final predictions.

---

### 5. Model Architecture Topology & Parameter Breakdown (Phase 5)
- **Sublayer Parameter Allocation Breakdown**: Dynamically inspects model parameter tensors (`model.named_parameters()`) and categorizes parameter budget into:
  - 🩷 **MLP / Feed-Forward Sublayers** (Gate, Up, Down / $W_1, W_2, W_3$ projections)
  - 💜 **Multi-Head Attention (MHSA)** ($W_Q, W_K, W_V, W_O$ projections)
  - 🩵 **Token & Position Embeddings** ($W_E$, $W_{pos}$, RoPE embeddings)
  - 🧡 **Unembedding / LM Head** ($W_U$, `lm_head`, noting weight tying if applicable)
  - 💚 **Layer Normalizations** (LayerNorm / RMSNorm weights across all blocks)
- **Multi-Color Proportion Bar & Budget Table**: Interactive horizontal stacked bar and table detailing exact parameter counts (e.g. `1.85B params`) and percentage budget shares (`61.4%`).
- **Pipeline Node Flow Diagram**: Visual execution topology diagram illustrating token processing flow through Embeddings, Residual Stream Highway, MHSA, MLP, LayerNorm, and Unembedding.

---

### 6. Automated Causal Interventions & ROME Causal Tracing (Phase 7)
- **Clean vs. Corrupted Prompt Activation Patching**: Counterfactual intervention engine running activation swapping between clean and corrupted prompts (`/api/analysis/causal-patching`).
- **ROME-Style Causal Tracing Heatmap Matrix**: $num\_layers \times seq\_len$ recovery matrix visualizing where model factual knowledge and causal predictions reside.
- **Logit Difference Recovery Metric**: Measures logit restoration $P_{patched}(y_{clean}) - P_{patched}(y_{corrupt})$ across all layer $\times$ position injection sites.
- **Automated Peak Recovery Locator**: Identifies the exact layer and token position ($L_{max}, N_{max}$) producing maximum recovery with percentage restoration metrics.

---

### 7. Induction Head Auto-Detector & Circuit Scanner (Phase 8)
- **Automated Repeated Sequence Benchmark**: Generates random token sequence repetitions ($S_1 S_2$) to detect copy-pattern mechanisms.
- **Full-Model Attention Head Grid Scan**: Scans attention matrices across all heads ($num\_layers \times num\_heads$) in a single automated sweep.
- **Induction Score Algorithm**: Evaluates prefix-matching attention pattern alignment ($\text{Attn}[i, j] \approx 1$ where $x_i = x_j$).
- **Ranked Induction Circuit Table**: Automatically flags and ranks top induction heads with configurable score detection thresholds.

---

### 8. Granular GPU Hardware Telemetry & Memory Profiler (Phase 9)
- **32-Block / 64-Block VRAM Memory Topology Grid**: Interactive visual memory grid displaying active CUDA allocations, reserved memory blocks, and free memory fragments.
- **CUDA Hardware Compute Metrics**: Live gauges for total VRAM, allocated memory, reserved memory, and free memory.
- **Per-Layer Activation Memory Breakdown**: Granular breakdown table detailing tensor shapes, dtypes (`fp16`/`fp32`/`bfloat16`), and memory byte footprint across all model hidden layers.
- **LRU Session Telemetry & Eviction Controls**: Tracks cached activation sessions and provides single-click VRAM memory clearing.
- **Execution Latency Sparklines**: Millisecond timing telemetry tracking forward pass and interpretability analysis performance.

---

## 💾 Data Portability, Caching & Session Management

- **LRU Activation Session Store**: In-memory LRU session cache (`SessionStore`) storing activation tensors, token lists, and model metadata with automatic capacity management.
- **JSON Analysis Export**: Full JSON export capability for Logit Lens grids, Attention maps, Causal sweeps, Induction scores, and GPU telemetry.
- **CSV Trajectory Export**: Export token-by-token layer trajectories and attention matrices to CSV format for downstream research and statistical analysis.
- **UI Session Snapshots**: Save and restore complete workbench states.

---

## 🎨 UI/UX Design System & Branding Details

- **Responsive Multi-Tab Navigation**: Sidebar navigation switching between Logit Lens, GPU & Hardware, Model Architecture, Residual Stream, Attention Map, Neuron Activations, Causal Patching, and Induction Detector.
- **Transformer SVG Favicon**: Custom 3-layer Transformer topology SVG icon matching header branding.
- **Live Status Pill & Header Metadata**: Real-time status indicator showing model name (`gpt2`, `qwen2.5`, etc.), active device (`CUDA` / `CPU`), VRAM usage, and active session status.
- **Preset Prompt Buttons**: One-click sample prompts for immediate interpretability experimentation.
- **Theme Switcher**: Instant switching between dark-mode glassmorphism aesthetics and clean light mode.

---

## 🌐 Interactive Documentation Portal

- **Standalone Documentation Site (`docs_site/index.html`)**: Interactive single-page documentation portal featuring visual component overviews, REST API specs, architecture diagrams, and quickstart guides.
- **Package Specs (`docs/PACKAGE_SPEC.md`)**: Comprehensive Python API and schema specification.
- **Architecture Overview (`docs/ARCHITECTURE.md`)**: Deep-dive system topology document detailing the 4-tier model adapter & engine pipeline.

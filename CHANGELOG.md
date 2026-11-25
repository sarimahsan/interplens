# Changelog

All notable changes to **InterpLens** are documented in this file.

## [2.0.0] — 2026-11-25

### New Interpretability Engines
- **Direct Logit Attribution (DLA):** Decomposes final logit differences (`target_token - foil_token`) into exact additive contributions from the Embedding layer, Attention sublayers, and MLP/MoE sublayers (`GET /api/analysis/dla`).
- **Gradient-Based Attribution Patching (AtP):** Approximates the full `Layer × Token Position` causal heatmap and ranks Attention/MLP/MoE sublayers in **2 forward passes + 1 backward pass** (`POST /api/analysis/attribution-patching`). Benchmarked at **~504 ms** on `Qwen2.5-7B-Instruct` (A100 80GB).
- **Contrastive Activation Steering (CAA) & Side-by-Side Generation:** Computes Difference-in-Means steering vectors from Positive vs. Negative prompt pairs (`POST /api/analysis/steering/compute`) and runs Baseline vs. Steered multi-token generation (`POST /api/analysis/steering/generate`) with live $\alpha$ strength control and built-in presets.
- **Interactive Mermaid.js Model Architecture Diagram:** Dynamically generates model-accurate Mermaid flowcharts with residual skip connections, tensor shapes, parameter budgets, and auto-detection of `GQA` vs `MHSA`, `RMSNorm` vs `LayerNorm`, `SwiGLU`, `MoE`, and tied weights — with Zoom, View/Copy Mermaid, and Export SVG controls.
- **Real 2D PCA Residual Stream Trajectories:** Computes SVD-based 2D PCA coordinates and explained variance across layers in `compute_residual_metrics()`.

### Architecture, Scale & Reliability Fixes
- **BFloat16 Support (A100 / H100):** Fixed `Got unsupported ScalarType BFloat16` in Logit Lens by casting logits to `float32` prior to NumPy conversion (verified on `Qwen2.5-7B`, `Qwen2.5-14B`, and `OLMoE-1B-7B`).
- **Selective CPU-Offloaded Hooking:** `CustomModelAdapter` and `GenericAdapter` now hook only key transformer modules (embed, blocks, attn, mlp/moe) and offload captured activations to CPU RAM.
- **Real Hardware & Memory Profiler:** Removed hardcoded `16 GB` / `512 MB` CPU placeholders and fake GPU sensor fallbacks; reports real `psutil` system RAM, process RSS, peak RSS, CPU cores, and live NVML GPU metrics.
- **Explicit Fallback Transparency:** Fixed HuggingFace `self_attn` hidden states shadowing real `out.attentions` tensors, and added `is_synthetic` + `warning` fields across Attention, Neurons, Token Attribution, and Causal Patching responses.
- **Multi-Architecture Coverage:** Added support for Mixtral MoE (`block_sparse_moe`), OPT (`model.decoder.layers`), Falcon/Bloom (`self_attention`), and `use_cache=False` backward safety; verified via automated tests on real HuggingFace `LlamaForCausalLM`, `Qwen2ForCausalLM`, `MixtralForCausalLM`, and `GPT2LMHeadModel`.
- **Packaging & CLI:** Added `transformers`, `uvicorn[standard]`, and `websockets` to core dependencies, updated `pyproject.toml` to SPDX `license = "MIT"`, wired `dtype` and `hf_token` into `il.launch()`, and fixed CPU RAM labeling in the CLI.

### UI & Documentation
- Redesigned InterpLens Studio with a high-contrast obsidian dark theme, custom scrollbars, and a crisp geometric SVG favicon.
- Added mobile/tablet responsive layout (horizontal swipeable navigation bar + dismissible desktop-recommended banner).
- Redesigned the documentation site (`docs_site/`) with a clean, minimal technical layout.

## [0.1.0] — Initial Release
- Initial release with Logit Lens, Residual Stream norms/cosine drift, Attention Head & Induction Head Explorer, MLP Neuron Activations, Causal Patching sweeps, and Zero-Copy GPU Attach.

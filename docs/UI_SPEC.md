# InterpLens Web UI Specification & Design Blueprint

**InterpLens** features a modern, high-density, dark-mode visual interface styled like a professional IDE / mechanistic interpretability workbench.

---

## 1. Visual Design Tokens & Palette

| Token | Hex Value | Application |
| :--- | :--- | :--- |
| `--bg-darkest` | `#0A0D14` | Dashboard main canvas background |
| `--bg-card` | `#121824` | Panels, drawers, and card containers |
| `--bg-hover` | `#1E2638` | Hover state for lists and buttons |
| `--border-color` | `#2A344A` | Sleek border lines and panel splitters |
| `--text-primary` | `#F1F5F9` | Main headers and primary text |
| `--text-muted` | `#94A3B8` | Subtitles, labels, and secondary metadata |
| `--accent-cyan` | `#38BDF8` | Primary active tab & Logit Lens highlights |
| `--accent-purple` | `#A855F7` | Attention & Residual Stream highlights |
| `--accent-green` | `#34D399` | Neuron activation & positive patch recovery |
| `--accent-rose` | `#FB7185` | Negative patch recovery & corrupted state |

---

## 2. Layout Structure

```
+-----------------------------------------------------------------------------------+
|  [Logo] InterpLens | Model: [ Llama-3.2-1B v] | Prompt: [ The Eiffel Tower... ] [Run] |
+-----------+---------------------------------------------------+-------------------+
| SIDEBAR   | MAIN PANEL (Persistent Token Strip at Top)       | CONTEXT CONTROL   |
|           | [ <|EOS|> ] [ The ] [ Eiffel ] [ Tower ] [ is ]   | PANEL             |
| Overview  |---------------------------------------------------|                   |
| Logit Lens|                                                   | Layer: [ slider ] |
| Residual  |             [ INTERACTIVE CHART VIEW ]            | Head:  [ slider ] |
| Attention |            (Ribbon / Heatmap / Grid / Arc)         | Top-K: [ slider ] |
| Neurons   |                                                   | Metric:[ dropdown]|
| Patching  |                                                   |                   |
| Causal Tr.|                                                   |                   |
| SAE Feat. |                                                   |                   |
| Compare   |                                                   |                   |
+-----------+---------------------------------------------------+-------------------+
| BOTTOM    | [Raw Tensor Data Inspector] [Session History]     | GPU Memory: 1.2GB |
+-----------------------------------------------------------------------------------+
```

---

## 3. Chart Specifications per Feature

### 3.1 Overview / Token Attribution
- **View:** Prompt rendered as tokens.
- **Visual:** Text tokens highlighted with variable opacity background matching attribution score.
- **Controls:** Attribution method dropdown (Gradient, Attention Rollout).

### 3.2 Logit Lens
- **View 1 (Ribbon):** X-axis = Layer (0..L), Y-axis = Confidence / Rank. Curves represent top candidate tokens evolving across layers.
- **View 2 (Heatmap Matrix):** Y-axis = Layer (0..L), X-axis = Token Position (0..T). Cell text = Top-1 predicted token at that layer; Cell color = Probability magnitude.

### 3.3 Residual Stream
- **Heatmap:** X-axis = Token Position, Y-axis = Layer. Color gradient = L2 norm / variance of residual stream vector.
- **PCA Trajectory:** 2D scatter plot showing vector trajectory from Layer 0 to Layer $L$ for a clicked token.

### 3.4 Attention Heads
- **Grid View:** Small N×N thumbnail heatmaps for all heads in selected layer.
- **Detail View:** Full-size annotated matrix with token labels on both axes.
- **Arc View:** Prompt text laid out horizontally; curved arcs connecting query and key tokens, with arc thickness scaled to attention weight.

### 3.5 Neuron Activation Panel
- **Bar Chart:** Ranked horizontal bars showing top-firing neurons for selected token.
- **Token Lighting Strip:** Text display where each token is shaded by how strongly the selected single neuron fired.

### 3.6 Activation Patching & Causal Tracing
- **Patch Heatmap:** X-axis = Token Position, Y-axis = Layer. Color scale = Logit difference recovered (Diverging Red-Blue).
- **Causal Trace Sweep:** Smoothed gradient heatmap representing automated ROME sweep importance.

### 3.7 Model Comparison
- **Split Screen:** Two identical charts side-by-side with synchronized controls (layer sliders, zoom/pan).
- **Diff View:** Single heatmap showing tensor differences ($Model_A - Model_B$).

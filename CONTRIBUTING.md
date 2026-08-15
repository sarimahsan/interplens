# Contributing to InterpLens

Thank you for your interest in contributing to **InterpLens**! We welcome contributions from researchers, engineers, and students across mechanistic interpretability, visual debugging, model reverse-engineering, and deep learning tooling.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Coding Standards & Typings](#coding-standards--typings)
- [Adding New Model Strategies](#adding-new-model-strategies)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Reporting Bugs & Requesting Features](#reporting-bugs--requesting-features)

---

## Code of Conduct

Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all community interactions.

---

## Development Setup

### 1. Fork & Clone the Repository
```bash
git clone https://github.com/sarimahsan/interplens.git
cd interplens
```

### 2. Create a Virtual Environment
```bash
# Using Python 3.10+ (Python 3.13 recommended)
python -m venv venv

# Activate the virtual environment
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

### 3. Install in Editable Mode with Dev Dependencies
```bash
pip install -e .[all,dev]
```

---

## Running Tests

InterpLens maintains strict testing standards. All unit tests must pass before submitting a pull request:

```bash
# Run full test suite (excluding slow online HF downloads)
pytest -k "not gpt2_integration"

# Run with full coverage report
pytest --cov=interplens --cov-report=term-missing

# Run specific module tests
pytest tests/test_framework_v2.py
pytest tests/test_robustness_fallbacks.py
pytest tests/test_professional_standards.py
```

---

## Coding Standards & Typings

1. **Python 3.9 – 3.13 Support:** Ensure code remains compatible across all supported Python versions.
2. **Type Annotations (PEP 561):** InterpLens is fully typed. All new functions, methods, and classes must include type annotations.
3. **No Stray Debugging Artifacts:** Never leave `print()` statements or local test files in library source code. Use Python's built-in `logging` module.
4. **Graceful Fallbacks:** If an optional dependency or model component is absent, handle the case with a structured fallback or informative exception rather than an unhandled crash.

---

## Adding New Model Strategies

To add native hook support for a new transformer architecture:

1. Subclass `ArchitectureStrategy` in `interplens/adapters/strategy.py`.
2. Implement required hook path resolvers:
   - `get_layer_hook_point(layer_idx)`
   - `get_attention_hook_point(layer_idx)`
   - `get_mlp_hook_point(layer_idx)`
   - `get_unembedding_weight()`
3. Register your strategy using `register_strategy()` or decorating with `@register_strategy(family_names=[...])`.
4. Add comprehensive unit tests in `tests/`.

---

## Submitting Pull Requests

1. Create a descriptive feature branch:
   ```bash
   git checkout -b feature/my-new-engine
   ```
2. Commit your changes with clear, semantic commit messages.
3. Push to your fork and submit a Pull Request against the `main` branch.
4. Ensure CI tests pass and provide a clear summary of your changes in the PR description.

---

## Reporting Bugs & Requesting Features

- **Bug Reports:** Please open an issue on GitHub with reproduction steps, Python/PyTorch versions, model name, and the complete traceback.
- **Feature Requests:** Open an issue outlining the intended workflow, interpretability use case, and suggested API design.

---

Thank you for helping make mechanistic interpretability research faster, more visual, and accessible!

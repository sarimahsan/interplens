"""Server State Manager and Model Loading Helpers for InterpLens FastAPI Server."""

import os
import time
import logging
import threading
from typing import Optional, Dict, Any, Union

from fastapi import HTTPException
from interplens.schema import ModelInfo
from interplens.utils.device import get_optimal_device, free_gpu_memory

logger = logging.getLogger("interplens.server")


class ServerStateManager:
    """Thread-safe container managing active model adapter and loading status."""

    def __init__(self):
        self._lock = threading.RLock()
        self._active_adapter: Optional[Any] = None
        self._status: Dict[str, Any] = {"status": "idle", "model_name": "none", "error": None, "warning": None}

    @property
    def active_adapter(self) -> Optional[Any]:
        with self._lock:
            return self._active_adapter

    @property
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def set_adapter(self, adapter: Any) -> None:
        with self._lock:
            self._active_adapter = adapter
            self._status["status"] = "online"
            self._status["model_name"] = getattr(adapter, "model_name", "custom")
            self._status["error"] = None

    def update_status(self, status: str, model_name: str, error: Optional[str] = None, warning: Optional[str] = None) -> None:
        with self._lock:
            self._status["status"] = status
            self._status["model_name"] = model_name
            self._status["error"] = error
            self._status["warning"] = warning

    def clear_adapter(self) -> None:
        with self._lock:
            if self._active_adapter is not None:
                del self._active_adapter
                self._active_adapter = None
            free_gpu_memory()


state_manager = ServerStateManager()


def init_model(model_name: str = "gpt2", device: Optional[Any] = None, hf_token: Optional[str] = None, tokenizer_name_or_path: Optional[str] = None):
    """Loads target model into GPU VRAM in a thread-safe manner."""
    if device is None:
        device = get_optimal_device()

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    current_adapter = state_manager.active_adapter
    if current_adapter is not None:
        curr_name = getattr(current_adapter, "model_name", "")
        if curr_name.lower() == model_name.lower():
            logger.info(f"Model '{model_name}' is already loaded in VRAM. Skipping reload.")
            state_manager.update_status("online", model_name)
            return current_adapter
        else:
            logger.info(f"Clearing previous model '{curr_name}' from VRAM...")
            state_manager.clear_adapter()

    state_manager.update_status("loading", model_name)
    logger.info(f"Loading model '{model_name}' onto {device}...")

    # Known native TransformerLens models
    tl_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "stanford-gpt2-small-a"]
    is_tl_native = any(m == model_name.lower() or model_name.lower().startswith("pythia") for m in tl_models)

    if is_tl_native:
        try:
            from transformer_lens import HookedTransformer
            from interplens.adapters.inplace import InPlaceModelAdapter
            model = HookedTransformer.from_pretrained(model_name, device=device)
            adapter = InPlaceModelAdapter(model, model_name=model_name)
            state_manager.set_adapter(adapter)
            logger.info(f"Loaded '{model_name}' via TransformerLens on {device}")
            return adapter
        except Exception as e1:
            logger.warning(f"TransformerLens could not load '{model_name}' ({e1}). Falling back to HuggingFace...")

    # Direct HuggingFace AutoModelForCausalLM loader
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from interplens.adapters.custom import CustomModelAdapter

        token_kwargs = {"token": token} if token else {}
        tokenizer = None
        tokenizer_warning = None

        # If user provided explicit tokenizer override via --tokenizer, use it first
        if tokenizer_name_or_path:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, **token_kwargs)
                logger.info(f"Loaded custom tokenizer from '{tokenizer_name_or_path}'")
            except Exception as t_override_err:
                logger.warning(f"Custom tokenizer '{tokenizer_name_or_path}' could not be loaded: {t_override_err}")
                tokenizer_warning = f"⚠️ Custom tokenizer '{tokenizer_name_or_path}' failed to load: {t_override_err}. Falling back to auto-detection."

        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **token_kwargs)
            except Exception as t_err1:
                logger.warning(f"Primary tokenizer for '{model_name}' could not be loaded directly: {t_err1}")
                m_lower = model_name.lower()
                fallback_repos = []
                if "qwen" in m_lower:
                    fallback_repos = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen1.5-0.5B"]
                elif "llama" in m_lower:
                    fallback_repos = ["huggyllama/llama-7b", "meta-llama/Llama-3.2-1B"]
                elif "gemma" in m_lower:
                    fallback_repos = ["google/gemma-2b", "fxmarty/gemma-2b-tokenizer"]
                elif "phi" in m_lower:
                    fallback_repos = ["microsoft/phi-2", "microsoft/Phi-3-mini-4k-instruct"]

                for repo in fallback_repos:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, **token_kwargs)
                        tokenizer_warning = f"Primary tokenizer for '{model_name}' was unavailable. Resolved compatible architecture tokenizer '{repo}'."
                        break
                    except Exception as t_err2:
                        logger.warning(f"Fallback tokenizer '{repo}' unavailable: {t_err2}")

                if tokenizer is None:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained("gpt2")
                        tokenizer_warning = f"⚠️ Tokenizer Warning: Could not load matching tokenizer for '{model_name}'. Using generic fallback tokenizer (predictions may be inaccurate). Please load matching tokenizer using --tokenizer <repo_id>."
                    except Exception:
                        tokenizer_warning = f"No tokenizer found for '{model_name}'. Operating with raw token ID indexing."

        target_dev_map = "auto" if str(device).startswith("cuda") else None
        model = None

        # Attempt 1: Load with eager attention implementation
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
                device_map=target_dev_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager",
                **token_kwargs
            )
        except Exception:
            pass

        # Attempt 2: Load without attn_implementation kwarg
        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
                    device_map=target_dev_map,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    **token_kwargs
                )
            except Exception:
                pass

        # Attempt 3: AutoConfig rope_scaling patch for Phi-3 schema compatibility
        if model is None:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True, **token_kwargs)
            if hasattr(cfg, "rope_scaling") and isinstance(cfg.rope_scaling, dict):
                if "type" not in cfg.rope_scaling and "rope_type" in cfg.rope_scaling:
                    cfg.rope_scaling["type"] = cfg.rope_scaling["rope_type"]
                elif "rope_type" not in cfg.rope_scaling and "type" in cfg.rope_scaling:
                    cfg.rope_scaling["rope_type"] = cfg.rope_scaling["type"]
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=cfg,
                torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
                device_map=target_dev_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **token_kwargs
            )

        if hasattr(model, "config"):
            try:
                model.config.output_attentions = True
            except Exception:
                pass

        # Try resolving tokenizer from loaded model's name_or_path if missing
        if tokenizer is None:
            m_path = getattr(model, "name_or_path", getattr(getattr(model, "config", None), "_name_or_path", None))
            if m_path:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(m_path, trust_remote_code=True, **token_kwargs)
                except Exception:
                    pass

        # Verify tokenizer vocabulary compatibility with model vocabulary size
        if model is not None and tokenizer is not None:
            m_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
            t_vocab = len(tokenizer) if hasattr(tokenizer, "__len__") else getattr(tokenizer, "vocab_size", None)

            # Allow small GPU tensor core padding differences (<5,000 tokens) between model config and base tokenizer
            if m_vocab and t_vocab and abs(m_vocab - t_vocab) > 5000:
                tokenizer_warning = (
                    f"⚠️ Mismatched Tokenizer Warning: Model vocabulary size is {m_vocab:,}, "
                    f"but active tokenizer vocabulary size is {t_vocab:,}. "
                    f"Mismatched token IDs will produce inaccurate predictions. "
                    f"Please load the matching tokenizer using --tokenizer <repo_id_or_path>."
                )
                logger.warning(tokenizer_warning)

        adapter = CustomModelAdapter(model=model, tokenizer=tokenizer, model_name=model_name)
        state_manager.set_adapter(adapter)
        if tokenizer_warning:
            state_manager.update_status("online", model_name, warning=tokenizer_warning)
        logger.info(f"Loaded '{model_name}' directly into VRAM!")
        return adapter
    except Exception as e2:
        err_msg = f"Failed to load model '{model_name}': {e2}"
        state_manager.update_status("error", model_name, error=err_msg)
        logger.error(err_msg)
        return None


def get_active_adapter():
    """Returns the loaded model adapter for the server session safely."""
    adapter = state_manager.active_adapter
    if adapter is not None:
        return adapter

    status_dict = state_manager.status
    if status_dict.get("status") == "loading":
        model_name = status_dict.get("model_name", "target model")
        for _ in range(120):
            time.sleep(0.5)
            adapter = state_manager.active_adapter
            if adapter is not None:
                return adapter
            if state_manager.status.get("status") == "error":
                err = state_manager.status.get("error") or "Failed to load model."
                raise HTTPException(status_code=500, detail=f"Model loading error: {err}")

        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_name}' is currently downloading/loading into GPU VRAM. Please wait a few seconds."
        )

    target_name = status_dict.get("model_name")
    if not target_name or target_name == "none":
        target_name = "gpt2"

    adapter = init_model(target_name)
    if adapter is None:
        err = state_manager.status.get("error") or "No model loaded."
        raise HTTPException(status_code=500, detail=f"Model loading error: {err}")
    return adapter


def set_active_adapter(adapter: Any):
    """Sets a custom loaded model adapter safely."""
    state_manager.set_adapter(adapter)


def get_adapter_model_info(adapter: Any) -> ModelInfo:
    if hasattr(adapter, "model_info") and isinstance(adapter.model_info, ModelInfo):
        return adapter.model_info
    info_dict = adapter.get_model_info()
    return ModelInfo(
        model_name=info_dict.get("model_name", "custom"),
        num_layers=info_dict.get("num_layers", 0),
        num_heads=info_dict.get("num_heads", 0),
        hidden_dim=info_dict.get("hidden_dim", 0),
        vocab_size=info_dict.get("vocab_size", 0),
        device=str(getattr(adapter, "device", "cpu")),
        is_custom=info_dict.get("is_custom", False),
        discovery_confidence=info_dict.get("discovery_confidence", 1.0),
        static_fingerprint=info_dict.get("static_fingerprint"),
        runtime_fingerprint=info_dict.get("runtime_fingerprint"),
        capabilities=info_dict.get("capabilities"),
        engine_capabilities=info_dict.get("engine_capabilities"),
    )

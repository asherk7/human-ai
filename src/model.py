"""Model loading utilities for the human-AI axis infrastructure.

Wraps HuggingFace ``AutoModelForCausalLM`` loading so the rest of the codebase
gets a uniform handle (model, tokenizer, layer count, hidden size) regardless of
which causal LM is in use. Handles the Gemma 3 config quirk where the layer count
lives under ``config.text_config`` rather than directly on ``config``.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _text_config(config: Any) -> Any:
    """Return the sub-config that holds ``num_hidden_layers``/``hidden_size``.

    Gemma 3 is a multimodal config that nests the text-model fields under
    ``config.text_config``. Most causal LMs (Gemma 2, Llama 3.1) expose them
    directly. Try the nested config first, fall back to the top-level config.

    Args:
        config: A HuggingFace ``PretrainedConfig`` instance.

    Returns:
        The config object that actually carries the text-model dimensions.
    """
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
        return text_cfg
    return config


def load_model(
    model_id: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "eager",
    load_in_4bit: bool = False,
) -> Dict[str, Any]:
    """Load a causal LM and its tokenizer into an inference-ready bundle.

    Works for any HuggingFace causal LM (Gemma 3 4B, Gemma 2 9B, Llama 3.1 8B).
    Gemma 3 4B fits a 16GB T4 in bfloat16; for the larger 8-9B models on a T4,
    pass ``load_in_4bit=True`` to load with bitsandbytes 4-bit quantization.

    Args:
        model_id: HuggingFace model id, e.g. ``"google/gemma-3-4b-it"``.
        device: Device / ``device_map`` target, e.g. ``"cuda"`` or ``"cpu"``.
        dtype: Compute dtype for the weights (ignored when ``load_in_4bit``).
        attn_implementation: Attention backend. ``"eager"`` is the safe default
            for hidden-state extraction on Gemma 3.
        load_in_4bit: If True, quantize to 4-bit (NF4) via bitsandbytes so the
            8-9B models fit on a 16GB GPU.

    Returns:
        Dict with keys:
            ``model``: the loaded model in ``eval`` mode.
            ``tokenizer``: the matching tokenizer.
            ``n_layers``: number of transformer decoder layers (int).
            ``hidden_size``: residual-stream width (int).
            ``model_id``: the id it was loaded from (str).
            ``device``: the device it was placed on (str).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "attn_implementation": attn_implementation,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # 4-bit loading uses device_map to place quantized weights.
        load_kwargs["device_map"] = device
    else:
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    text_cfg = _text_config(model.config)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "n_layers": int(text_cfg.num_hidden_layers),
        "hidden_size": int(text_cfg.hidden_size),
        "model_id": model_id,
        "device": device,
    }

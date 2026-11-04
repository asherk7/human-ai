"""Activation steering (Contrastive Activation Addition).

Adds a direction vector into the residual stream at one decoder layer while the
model generates, shifting its internal state toward (``alpha > 0``) or away from
(``alpha < 0``) the concept the vector encodes.

Conventions:
  * Steer with the RAW axis (``ai_mean - human_mean``), not the unit axis, so
    ``alpha`` is in units of "cluster gaps": ``alpha = 1`` pushes by the full
    human->AI distance at that layer. ``alpha = 0`` reproduces the baseline
    through the identical code path (the hook adds a zero vector).
  * Effects are non-monotonic in ``alpha`` — too small does nothing, too large
    degrades or even reverses the behavior. Calibrate on a few prompts first.
  * The steering layer is NOT the probing layer: a layer where the axis is most
    *readable* (max separation) is not necessarily where *injecting* produces
    the cleanest behavior change. Start around 75% depth and sweep.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

import torch

from .activations import NEUTRAL_SYS, _device_of


def resolve_decoder_layers(model: Any) -> torch.nn.ModuleList:
    """Locate the model's decoder-layer list, wherever the architecture nests it.

    Llama-style models expose it at ``model.model.layers``; Gemma 3 (multimodal)
    nests the text stack under a ``language_model`` submodule. Rather than
    hardcoding paths, find the ``nn.ModuleList`` named ``...layers`` whose length
    matches the text config's ``num_hidden_layers``.

    Args:
        model: The loaded causal LM.

    Returns:
        The ``nn.ModuleList`` of decoder layers (index with the layer number).

    Raises:
        ValueError: If no matching module list is found.
    """
    cfg = getattr(model.config, "text_config", model.config)
    n_layers = int(cfg.num_hidden_layers)
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.ModuleList)
            and name.endswith("layers")
            and len(module) == n_layers
        ):
            return module
    raise ValueError(
        f"Could not locate a decoder ModuleList of length {n_layers} on "
        f"{type(model).__name__}."
    )


@contextmanager
def steering(
    model: Any,
    layer: int,
    vector: torch.Tensor,
    alpha: float,
) -> Iterator[None]:
    """Context manager that adds ``alpha * vector`` to one layer's output.

    Registers a forward hook on decoder layer ``layer`` that shifts the residual
    stream at EVERY token position (prompt and generated) for every forward pass
    inside the ``with`` block, then removes the hook — even on error — so no
    steering leaks into later cells.

    The computation is fully steered (generation, logits, all later layers), but
    a capture taken inside the block is NOT. ``capture_from_text`` and
    ``capture_from_messages`` read ``output_hidden_states``, and transformers
    records a layer's hidden state before a user forward hook replaces its
    output, so a capture at ``layer`` itself sees the UNSTEERED activation
    (verified on transformers 4.57: steering at L shows up at
    ``hidden_states[L + 2]``, not ``[L + 1]``). Captures at any layer above
    ``layer`` are correct. So either read strictly above the injection, or, for
    the injection layer itself, compute the shift in closed form: the hook adds
    the same vector at every position, so a token-mean activation moves by
    exactly ``alpha * vector``, no forward pass needed.

    Args:
        model: The loaded causal LM.
        layer: Decoder layer index to inject at.
        vector: Direction to add, shape ``(hidden_size,)`` (any device/dtype;
            cast to match the activations on first use).
        alpha: Signed strength. 0 adds nothing (clean baseline path).

    Yields:
        None. Run generation inside the block.
    """
    target = resolve_decoder_layers(model)[layer]
    state: Dict[str, torch.Tensor] = {}

    def hook(_module: Any, _args: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if "vec" not in state:
            if vector.shape[-1] != hidden.shape[-1]:
                raise ValueError(
                    f"Steering vector dim {vector.shape[-1]} != hidden dim "
                    f"{hidden.shape[-1]} — wrong layer's axis?"
                )
            state["vec"] = vector.to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden + alpha * state["vec"]
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    handle = target.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def steer_and_generate(
    model: Any,
    tokenizer: Any,
    steering_vector: torch.Tensor,
    alpha: float,
    prompt: str,
    layer: int,
    max_new_tokens: int = 80,
    system_prompt: str = NEUTRAL_SYS,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    """Generate a response to ``prompt`` with the vector injected at ``layer``.

    Positive ``alpha`` pushes toward the vector's concept (AI-like for our
    axis), negative pushes away (human-like), 0 is the unsteered baseline.

    Args:
        model: The loaded causal LM.
        tokenizer: The matching tokenizer.
        steering_vector: Direction to add, shape ``(hidden_size,)``. Use the
            RAW axis so ``alpha`` is in cluster-gap units.
        alpha: Signed steering strength.
        prompt: User message to respond to.
        layer: Decoder layer index to inject at.
        max_new_tokens: Generation length cap.
        system_prompt: System message (neutral by default so the flip comes
            from steering, not instructions).
        do_sample: Sample instead of greedy decoding.
        temperature: Sampling temperature (only used when ``do_sample``).

    Returns:
        The generated assistant response text.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return steer_and_generate_from_messages(
        model,
        tokenizer,
        steering_vector,
        alpha,
        messages,
        layer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )


@torch.no_grad()
def steer_and_generate_from_messages(
    model: Any,
    tokenizer: Any,
    steering_vector: torch.Tensor,
    alpha: float,
    messages: List[Dict[str, str]],
    layer: int,
    max_new_tokens: int = 80,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    """Like ``steer_and_generate`` but continues an arbitrary conversation.

    Used for eval scenarios (issue #7): pass the scenario context (all turns up
    to, not including, the assistant reply) and the model writes its own steered
    response. Also handy unsteered (``alpha=0``) for multi-turn generation.

    Args:
        model: The loaded causal LM.
        tokenizer: The matching tokenizer.
        steering_vector: Direction to add, shape ``(hidden_size,)``.
        alpha: Signed steering strength (0 = baseline).
        messages: Chat messages the model should reply to (template-safe; run
            ``sanitize_messages`` first for third-party transcripts).
        layer: Decoder layer index to inject at.
        max_new_tokens: Generation length cap.
        do_sample: Sample instead of greedy decoding.
        temperature: Sampling temperature (only used when ``do_sample``).

    Returns:
        The generated assistant response text.
    """
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    inputs = {k: v.to(_device_of(model)) for k, v in enc.items()}
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with steering(model, layer, steering_vector, alpha):
        out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)


def random_direction_like(vector: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Random vector with the same norm as ``vector`` — the steering control.

    If steering with THIS also flips behavior, the axis isn't special: any
    perturbation of that magnitude changes outputs.

    Args:
        vector: Reference vector (only its shape and norm are used).
        seed: RNG seed for reproducibility.

    Returns:
        Float32 CPU tensor, same shape, same L2 norm, random direction.
    """
    gen = torch.Generator().manual_seed(seed)
    rand = torch.randn(vector.shape, generator=gen, dtype=torch.float32)
    return rand / rand.norm().clamp_min(1e-8) * vector.float().norm()

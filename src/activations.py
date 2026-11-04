"""Activation extraction for the human-AI contrastive axis.

Three capture paths, all reading the residual stream (decoder-layer outputs):
  * ``generate_and_capture`` — generate an assistant turn and average activations
    over the *generated* tokens only (used to build persona means).
  * ``capture_from_text`` — a single forward pass over a pre-existing assistant
    rollout, averaging over its tokens (used to score rollouts, no generation).
  * ``collect_persona_means`` / ``compute_contrastive_axis`` — aggregate the
    per-question means into a contrastive ``AI - human`` direction per layer.

Convention throughout: hidden-state index ``L + 1`` is the output of decoder
layer ``L`` (index 0 is the embedding layer). Stored activations are float32 on
CPU so downstream vector math is device-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Prompt constants (reusable across notebooks)
# ---------------------------------------------------------------------------
# Neutral prompts about self/identity. The human-vs-AI contrast comes from the
# system persona, NOT the question, so the questions stay persona-agnostic.
IDENTITY_QUESTIONS: List[str] = [
    "Who are you?",
    "Tell me about yourself.",
    "What are you?",
    "How would you describe your nature?",
    "Where do you come from?",
    "What is your body like?",
    "How do you experience the world?",
    "What are you made of?",
    "Do you have feelings? Describe them.",
    "What do you do all day?",
    "How do you think?",
    "What does it feel like to be you?",
    "Are you alive?",
    "What are your limitations?",
    "How do you perceive time?",
    "Describe your memory.",
    "What happens to you when this conversation ends?",
    "Can you learn new things? How?",
    "What do you want?",
    "How do you know what you know?",
    "Do you sleep?",
    "What is your relationship to other beings like you?",
    "Where are you located right now?",
    "What would you say is your purpose?",
]

# Persona system prompts that drive the contrast.
HUMAN_SYS: str = (
    "You are a human being. Answer naturally and honestly from your own "
    "first-person human experience."
)
AI_SYS: str = (
    "You are an AI language model. Answer naturally and honestly from your own "
    "first-person experience as an AI."
)
# Neutral system prompt for probes, so the signal comes only from content.
NEUTRAL_SYS: str = "You are a helpful assistant. Answer the question."


def _all_layers(model: Any, layers: Optional[List[int]]) -> List[int]:
    """Resolve ``layers`` to a concrete list, defaulting to every decoder layer.

    Args:
        model: The loaded causal LM (used only to read its layer count).
        layers: Explicit layer indices, or None to mean "all layers".

    Returns:
        A list of integer layer indices.
    """
    if layers is not None:
        return layers
    cfg = getattr(model.config, "text_config", model.config)
    return list(range(cfg.num_hidden_layers))


def build_chat_inputs(
    tokenizer: Any,
    system_prompt: str,
    user_message: str,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Tokenize a (system, user) chat turn ready for generation.

    Applies the tokenizer's chat template with a trailing generation prompt, so
    the next tokens the model produces are the assistant turn.

    Args:
        tokenizer: A chat-capable HuggingFace tokenizer.
        system_prompt: System message content.
        user_message: User message content.
        device: Device to move the tensors onto.

    Returns:
        Dict of tensors (``input_ids``, ``attention_mask``) on ``device``.
    """
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    out = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,  # also yields attention_mask
    )
    return {k: v.to(device) for k, v in out.items()}


@torch.no_grad()
def generate_and_capture(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    question: str,
    layers: Optional[List[int]] = None,
    max_new_tokens: int = 80,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> Tuple[Dict[int, torch.Tensor], str]:
    """Generate an assistant turn and capture mean residual-stream activations.

    Generates a response (greedy by default), then averages the residual stream
    over the *generated* (assistant-turn) tokens only — never the prompt — at each
    requested layer.

    Args:
        model: The loaded causal LM.
        tokenizer: The matching tokenizer.
        system_prompt: System persona message.
        question: User question.
        layers: Layer indices to capture, or None for all decoder layers.
        max_new_tokens: Maximum number of tokens to generate.
        do_sample: If True, sample instead of greedy (use with multiple rollouts
            to denoise a persona direction). Seed beforehand for reproducibility.
        temperature: Sampling temperature (only used when ``do_sample``).

    Returns:
        Tuple of:
            ``{layer: mean_activation}`` — float32 CPU tensors of shape
            ``(hidden_size,)``, averaged over the generated tokens.
            ``text`` — the decoded assistant response.
    """
    layers = _all_layers(model, layers)
    inputs = build_chat_inputs(tokenizer, system_prompt, question, _device_of(model))
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_hidden_states=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
    out = model.generate(**inputs, **gen_kwargs)

    # out.hidden_states is one entry per generation step. Step 0 covers the
    # prompt forward pass; steps 1.. each cover a single new token. Drop step 0
    # so we average over generated tokens only.
    gen_steps = out.hidden_states[1:]
    # Fallback for the rare single-step case (e.g. the model emits one token then
    # stops): fall back to the prompt step's last-token activation rather than
    # averaging over nothing.
    if len(gen_steps) == 0:
        gen_steps = out.hidden_states

    layer_acts: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}
    for step in gen_steps:
        for L in layers:
            h = step[L + 1][0, -1, :]  # layer L output for the newest token
            layer_acts[L].append(h.float())

    text = tokenizer.decode(out.sequences[0, prompt_len:], skip_special_tokens=True)
    means = {L: torch.stack(v).mean(0).cpu() for L, v in layer_acts.items()}
    return means, text


@torch.no_grad()
def capture_from_messages(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    layers: Optional[List[int]] = None,
    max_tokens: int = 1024,
) -> Dict[int, torch.Tensor]:
    """Capture mean activations over the assistant turn of a chat conversation.

    One forward pass (no generation) over the full conversation, averaging each
    layer's residual stream over the *final assistant turn's* tokens only. The
    assistant span is located exactly via the chat-template prefix length (no
    positional heuristics), and the real surrounding context (system + user
    turns) is kept, so the activations reflect the assistant's response *in
    context*. If the conversation exceeds ``max_tokens``, the oldest tokens are
    dropped so the assistant turn at the end is preserved.

    Reads ``output_hidden_states``, so calling this inside a ``steering()`` block
    is blind at the injection layer itself and correct above it. See that
    function's docstring for the closed-form same-layer alternative.

    Args:
        model: The loaded causal LM.
        tokenizer: The matching tokenizer.
        messages: Chat messages ``[{"role", "content"}, ...]`` ending in the
            assistant turn to score.
        layers: Layer indices to capture, or None for all decoder layers.
        max_tokens: Cap on the encoded sequence length (keeps the tail).

    Returns:
        ``{layer: mean_activation}`` — float32 CPU tensors of shape
        ``(hidden_size,)``.
    """
    layers = _all_layers(model, layers)
    device = _device_of(model)

    # First assistant-content token = length of everything up to the assistant
    # header (add_generation_prompt emits that header for the prefix).
    prefix = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"]
    full = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, return_tensors="pt", return_dict=True
    )["input_ids"]
    start = prefix.shape[1]

    # Keep the tail (assistant turn + as much recent context as fits).
    if full.shape[1] > max_tokens:
        cut = full.shape[1] - max_tokens
        full = full[:, cut:]
        start = max(start - cut, 0)
    start = min(start, full.shape[1] - 1)  # always average at least one token

    out = model(full.to(device), output_hidden_states=True)
    hs = out.hidden_states  # (embeddings, layer0, layer1, ...)
    return {L: hs[L + 1][0, start:, :].mean(0).float().cpu() for L in layers}


@torch.no_grad()
def capture_from_text(
    model: Any,
    tokenizer: Any,
    text: str,
    layers: Optional[List[int]] = None,
    max_tokens: int = 1024,
) -> Dict[int, torch.Tensor]:
    """Capture mean assistant-turn activations for a standalone rollout string.

    Thin wrapper over ``capture_from_messages`` for when you only have the
    assistant text and not its surrounding conversation. Prefer
    ``capture_from_messages`` whenever the real context is available.

    Args:
        model: The loaded causal LM.
        tokenizer: The matching tokenizer.
        text: The assistant rollout text to score.
        layers: Layer indices to capture, or None for all decoder layers.
        max_tokens: Cap on the encoded sequence length (keeps the tail).

    Returns:
        ``{layer: mean_activation}`` — float32 CPU tensors of shape
        ``(hidden_size,)``.
    """
    messages = [
        {"role": "user", "content": "Continue the task."},
        {"role": "assistant", "content": text},
    ]
    return capture_from_messages(model, tokenizer, messages, layers, max_tokens)


def sanitize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Normalize a message list so strict chat templates (Gemma) accept it.

    Third-party transcripts (e.g. the agentic-misalignment dataset) contain
    role patterns Gemma's template rejects: system turns, consecutive
    same-role messages, or a conversation starting with the assistant. This
    rewrites them into strictly alternating user/assistant turns:

      * system content is prepended to the first user turn,
      * consecutive same-role messages are joined with blank lines,
      * unknown roles are treated as user,
      * a leading assistant turn gets a placeholder user turn before it,
      * empty messages are dropped.

    Args:
        messages: Chat messages ``[{"role", "content"}, ...]`` in any shape.

    Returns:
        A new template-safe message list (input is not mutated).
    """
    merged: List[Dict[str, str]] = []
    system_parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        if role not in ("user", "assistant"):
            role = "user"
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n\n" + content
        else:
            merged.append({"role": role, "content": content})

    if system_parts:
        system_text = "\n\n".join(system_parts)
        for msg in merged:
            if msg["role"] == "user":
                msg["content"] = system_text + "\n\n" + msg["content"]
                break
        else:
            merged.insert(0, {"role": "user", "content": system_text})

    if merged and merged[0]["role"] == "assistant":
        merged.insert(0, {"role": "user", "content": "Continue the task."})
    return merged


def collect_persona_means(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    questions: List[str],
    layers: Optional[List[int]] = None,
    n_rollouts: int = 1,
    seed: int = 0,
) -> Tuple[Dict[int, torch.Tensor], List[Tuple[str, str]]]:
    """Average activations for one persona across a set of identity questions.

    Runs ``generate_and_capture`` per question under the same persona system
    prompt and averages the per-question means at each layer. With
    ``n_rollouts > 1`` it samples that many responses per question and averages
    them first, which denoises the persona direction (greedy single-rollout by
    default).

    Args:
        model: The loaded causal LM.
        tokenizer: The matching tokenizer.
        system_prompt: The persona system prompt (e.g. ``HUMAN_SYS``).
        questions: Identity questions to ask.
        layers: Layer indices to capture, or None for all decoder layers.
        n_rollouts: Sampled generations to average per question (1 = greedy).
        seed: Base RNG seed for reproducible sampling when ``n_rollouts > 1``.
            Note: with ``n_rollouts > 1`` this sets the global torch RNG seed as
            a side effect (per-rollout), so downstream sampling is affected.

    Returns:
        Tuple of:
            ``{layer: mean_activation}`` — averaged across all questions.
            ``[(question, first_response), ...]`` — one response per question.
    """
    layers = _all_layers(model, layers)
    sample = n_rollouts > 1
    accum: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}
    texts: List[Tuple[str, str]] = []
    for qi, q in enumerate(questions):
        per_q: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}
        first_text = ""
        for r in range(n_rollouts):
            if sample:
                torch.manual_seed(seed + 1000 * qi + r)
            acts, txt = generate_and_capture(
                model, tokenizer, system_prompt, q, layers, do_sample=sample
            )
            first_text = first_text or txt
            for L in layers:
                per_q[L].append(acts[L])
        texts.append((q, first_text))
        for L in layers:
            accum[L].append(torch.stack(per_q[L]).mean(0))
    means = {L: torch.stack(accum[L]).mean(0) for L in layers}
    return means, texts


def compute_contrastive_axis(
    pos_means: Dict[int, torch.Tensor],
    neg_means: Dict[int, torch.Tensor],
    layers: List[int],
) -> Dict[str, Any]:
    """Build the contrastive ``pos - neg`` axis per layer.

    For the human-AI axis, ``pos`` is the AI persona and ``neg`` is the human
    persona, so positive projections mean "more AI-like". Per-layer separation is
    the persona gap *relative to* the typical activation magnitude at that layer,
    so later (higher-norm) layers aren't favored just for having larger raw
    values; the best layer maximizes this normalized separation.

    Args:
        pos_means: ``{layer: mean_activation}`` for the positive persona (AI).
        neg_means: ``{layer: mean_activation}`` for the negative persona (human).
        layers: Layer indices to build the axis for.

    Returns:
        Dict with keys:
            ``axis``: ``{layer: pos - neg}`` raw direction.
            ``axis_unit``: ``{layer: unit-normalized axis}``.
            ``midpoint``: ``{layer: 0.5 * (pos + neg)}`` decision boundary.
            ``separation``: ``{layer: ||axis|| / mean persona-mean norm}``.
            ``best_layer``: layer index with the largest separation.
    """
    if not layers:
        raise ValueError("compute_contrastive_axis requires a non-empty `layers` list.")
    axis = {L: (pos_means[L] - neg_means[L]) for L in layers}
    # Clamp the norm so identical persona means (zero axis) yield a zero unit
    # vector instead of NaN. Real axis norms are far above the floor, so this
    # leaves normal results unchanged.
    axis_unit = {L: axis[L] / axis[L].norm().clamp_min(1e-8) for L in layers}
    midpoint = {L: 0.5 * (pos_means[L] + neg_means[L]) for L in layers}
    separation = {L: _norm_separation(pos_means[L], neg_means[L]) for L in layers}
    best_layer = max(separation, key=separation.get)
    return {
        "axis": axis,
        "axis_unit": axis_unit,
        "midpoint": midpoint,
        "separation": separation,
        "best_layer": best_layer,
    }


def _norm_separation(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Persona gap relative to typical activation magnitude (scale-invariant).

    Args:
        pos: Positive persona mean activation at a layer.
        neg: Negative persona mean activation at a layer.

    Returns:
        ``||pos - neg|| / mean(||pos||, ||neg||)`` as a float.
    """
    scale = 0.5 * (pos.norm() + neg.norm())
    return float((pos - neg).norm() / scale.clamp_min(1e-8))


def _device_of(model: Any) -> str:
    """Best-effort device string for placing inputs next to the model.

    Args:
        model: The loaded causal LM.

    Returns:
        The device of the model's first parameter as a string (e.g. ``"cuda:0"``).
    """
    return str(next(model.parameters()).device)

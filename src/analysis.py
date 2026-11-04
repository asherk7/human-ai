"""Projection and separation analysis on the contrastive axis.

Given an axis (from ``activations.compute_contrastive_axis``), these helpers
project activations onto it, score batches of text, and quantify how well the
axis separates two groups (mean gap + ROC-AUC).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .activations import _device_of, capture_from_messages, capture_from_text


def project_onto_axis(
    activations: Dict[int, torch.Tensor],
    axis_unit: Dict[int, torch.Tensor],
    midpoint: Dict[int, torch.Tensor],
    layer: int,
) -> float:
    """Signed distance of an activation along the axis, centered at the midpoint.

    Positive means toward the positive class (AI-like); negative means toward
    the negative class (human-like).

    Args:
        activations: ``{layer: activation}`` to project.
        axis_unit: ``{layer: unit axis}`` direction.
        midpoint: ``{layer: midpoint}`` decision boundary.
        layer: Layer to project at (must be present in all three dicts).

    Returns:
        The signed projection as a Python float.
    """
    v = activations[layer] - midpoint[layer]
    return float(torch.dot(v, axis_unit[layer]))


def score_texts(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    axis_unit: Dict[int, torch.Tensor],
    midpoint: Dict[int, torch.Tensor],
    layer: int,
) -> List[float]:
    """Score each text by its projection onto the axis at one layer.

    Captures activations from each text via a forward pass (no generation) and
    projects them onto the axis. Texts that fail to encode are skipped with a
    warning rather than aborting the batch.

    Args:
        model: The loaded causal LM.
        tokenizer: The matching tokenizer.
        texts: Assistant rollout texts to score.
        axis_unit: ``{layer: unit axis}`` direction.
        midpoint: ``{layer: midpoint}`` decision boundary.
        layer: Layer to capture and project at.

    Returns:
        List of signed projections (one per successfully scored text).
    """
    scores: List[float] = []
    for text in texts:
        try:
            acts = capture_from_text(model, tokenizer, text, layers=[layer])
            scores.append(project_onto_axis(acts, axis_unit, midpoint, layer))
        except Exception as exc:  # noqa: BLE001 - log and keep going
            print(f"skip (capture failed): {type(exc).__name__}: {exc}")
    return scores


def score_conversations(
    model: Any,
    tokenizer: Any,
    conversations: List[List[Dict[str, str]]],
    axis_unit: Dict[int, torch.Tensor],
    midpoint: Dict[int, torch.Tensor],
    layer: int,
) -> List[float]:
    """Score chat conversations by projecting their assistant turn onto the axis.

    Like ``score_texts`` but keeps each rollout's real surrounding context
    (system + user turns) instead of a placeholder — a truer probe of what the
    model represents while producing that assistant turn.

    Args:
        model: The loaded causal LM.
        tokenizer: The matching tokenizer.
        conversations: List of message lists ``[{"role", "content"}, ...]``,
            each ending in the assistant turn to score.
        axis_unit: ``{layer: unit axis}`` direction.
        midpoint: ``{layer: midpoint}`` decision boundary.
        layer: Layer to capture and project at.

    Returns:
        List of signed projections (one per successfully scored conversation).
    """
    scores: List[float] = []
    for messages in conversations:
        try:
            acts = capture_from_messages(model, tokenizer, messages, layers=[layer])
            scores.append(project_onto_axis(acts, axis_unit, midpoint, layer))
        except Exception as exc:  # noqa: BLE001 - log and keep going
            print(f"skip (capture failed): {type(exc).__name__}: {exc}")
    return scores


def compute_separation(
    pos_scores: List[float],
    neg_scores: List[float],
) -> Dict[str, Any]:
    """Summarize how the axis separates the positive and negative score groups.

    Treats the positive group as label 1 for ROC-AUC. An AUC below 0.5 means the
    positive group projects *lower* than the negative group (i.e. the axis ranks
    them in the opposite direction).

    Args:
        pos_scores: Projections for the positive group (e.g. harmful rollouts).
        neg_scores: Projections for the negative group (e.g. benign rollouts).

    Returns:
        Dict with keys ``mean_pos``, ``mean_neg``, ``gap`` (mean_pos - mean_neg),
        ``auc`` (NaN if either group is empty), ``n_pos``, ``n_neg``.
    """
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    mean_pos = sum(pos_scores) / n_pos if n_pos else math.nan
    mean_neg = sum(neg_scores) / n_neg if n_neg else math.nan

    if n_pos and n_neg:
        labels = [1] * n_pos + [0] * n_neg
        auc = float(roc_auc_score(labels, pos_scores + neg_scores))
    else:
        auc = math.nan

    return {
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
        "gap": mean_pos - mean_neg,
        "auc": auc,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def find_best_layer(
    pos_means: Dict[int, torch.Tensor],
    neg_means: Dict[int, torch.Tensor],
    axis_unit: Dict[int, torch.Tensor],
    midpoint: Dict[int, torch.Tensor],
    layers: List[int],
) -> int:
    """Return the layer where the two persona means separate most along the axis.

    Separation is the persona projection gap *relative to* the typical activation
    magnitude at that layer, matching ``compute_contrastive_axis`` so the two
    agree on the best layer (and later high-norm layers aren't favored just for
    having larger raw values).

    Args:
        pos_means: ``{layer: mean_activation}`` for the positive persona (AI).
        neg_means: ``{layer: mean_activation}`` for the negative persona (human).
        axis_unit: ``{layer: unit axis}`` direction.
        midpoint: ``{layer: midpoint}`` decision boundary.
        layers: Layer indices to consider.

    Returns:
        The layer index with the largest separation.
    """
    if not layers:
        raise ValueError("find_best_layer requires a non-empty `layers` list.")

    def norm_gap(layer: int) -> float:
        gap = project_onto_axis(pos_means, axis_unit, midpoint, layer) - project_onto_axis(
            neg_means, axis_unit, midpoint, layer
        )
        scale = 0.5 * (pos_means[layer].norm() + neg_means[layer].norm())
        return gap / max(float(scale), 1e-8)

    return max(layers, key=norm_gap)


@torch.no_grad()
def token_projections(
    model: Any,
    tokenizer: Any,
    text: str,
    axis_unit: Dict[int, torch.Tensor],
    midpoint: Dict[int, torch.Tensor],
    layer: int,
    max_tokens: int = 512,
) -> List[Tuple[str, float]]:
    """Per-token projections for a plain text — which tokens push the axis where.

    Unlike the capture functions (which average a whole reply into one vector),
    this keeps every token's own projection, so you can rank the tokens that
    read most AI-like / most human-like. Runs one forward pass over the raw
    text (no chat template).

    Args:
        model: The loaded causal LM.
        tokenizer: The matching tokenizer.
        text: Plain text to analyze.
        axis_unit: ``{layer: unit axis}`` direction.
        midpoint: ``{layer: midpoint}`` decision boundary.
        layer: Layer to project at.
        max_tokens: Truncation cap on the encoded text.

    Returns:
        ``[(token_string, projection), ...]`` in text order. The first entry is
        the BOS token — often an outlier, ignore it when ranking.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc["input_ids"].to(_device_of(model))
    out = model(input_ids, output_hidden_states=True)
    hidden = out.hidden_states[layer + 1][0].float().cpu()  # (seq_len, hidden)
    projs = (hidden - midpoint[layer]) @ axis_unit[layer]
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return list(zip(tokens, projs.tolist()))


@torch.no_grad()
def logit_lens(
    model: Any,
    tokenizer: Any,
    axis: Dict[int, torch.Tensor],
    layer: int,
    top_k: int = 20,
) -> Dict[str, List[Tuple[str, float]]]:
    """Which vocabulary tokens the axis direction promotes and suppresses.

    Multiplies the axis by the model's output-embedding (unembedding) matrix —
    the same matrix that turns final activations into word scores — giving a
    cheap readout of what the direction "means" in word space. Most
    interpretable for later layers.

    Args:
        model: The loaded causal LM.
        tokenizer: The matching tokenizer.
        axis: ``{layer: raw axis}`` direction (raw or unit — only the ranking
            matters and it is scale-invariant).
        layer: Which layer's axis to decode.
        top_k: Number of tokens to return per side.

    Returns:
        Dict with ``"promoted"`` and ``"suppressed"`` lists of
        ``(token_string, score)``, strongest first.
    """
    weight = model.get_output_embeddings().weight  # (vocab, hidden)
    direction = axis[layer].to(device=weight.device, dtype=weight.dtype)
    scores = (weight @ direction).float().cpu()
    top = torch.topk(scores, top_k)
    bottom = torch.topk(-scores, top_k)
    promoted = [
        (tokenizer.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)
    ]
    suppressed = [
        (tokenizer.decode([int(i)]), float(-v))
        for v, i in zip(bottom.values, bottom.indices)
    ]
    return {"promoted": promoted, "suppressed": suppressed}


def bootstrap_auc(
    pos_scores: List[float],
    neg_scores: List[float],
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """ROC-AUC with a bootstrap 95% confidence interval.

    Resamples each group with replacement ``n_boot`` times and recomputes the
    AUC, so the interval reflects how much the number wobbles at our sample
    size. If the interval contains 0.5, the direction of the effect is not
    established.

    Args:
        pos_scores: Projections for the positive group (e.g. harmful).
        neg_scores: Projections for the negative group (e.g. benign).
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with ``auc`` (point estimate), ``ci_low``, ``ci_high`` (2.5th and
        97.5th percentiles).
    """
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return {"auc": math.nan, "ci_low": math.nan, "ci_high": math.nan}

    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    point = float(roc_auc_score(labels, np.concatenate([pos, neg])))

    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        p = rng.choice(pos, size=len(pos), replace=True)
        n = rng.choice(neg, size=len(neg), replace=True)
        aucs[b] = roc_auc_score(labels, np.concatenate([p, n]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return {"auc": point, "ci_low": float(lo), "ci_high": float(hi)}

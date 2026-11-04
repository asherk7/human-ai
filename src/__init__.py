"""Reusable infrastructure for extracting and analyzing model activations.

Public API for the human-AI contrastive-axis pipeline. Import the pieces you
need directly, e.g.::

    from src import load_model, collect_persona_means, compute_contrastive_axis
    from src import IDENTITY_QUESTIONS, HUMAN_SYS, AI_SYS
"""

from .activations import (
    AI_SYS,
    HUMAN_SYS,
    IDENTITY_QUESTIONS,
    NEUTRAL_SYS,
    build_chat_inputs,
    capture_from_messages,
    capture_from_text,
    collect_persona_means,
    compute_contrastive_axis,
    generate_and_capture,
    sanitize_messages,
)
from .analysis import (
    bootstrap_auc,
    compute_separation,
    find_best_layer,
    logit_lens,
    project_onto_axis,
    score_conversations,
    score_texts,
    token_projections,
)
from .model import load_model
from .steering import (
    random_direction_like,
    steer_and_generate,
    steer_and_generate_from_messages,
    steering,
)

__all__ = [
    # model
    "load_model",
    # activations
    "build_chat_inputs",
    "generate_and_capture",
    "capture_from_messages",
    "capture_from_text",
    "collect_persona_means",
    "compute_contrastive_axis",
    "sanitize_messages",
    "IDENTITY_QUESTIONS",
    "HUMAN_SYS",
    "AI_SYS",
    "NEUTRAL_SYS",
    # analysis
    "project_onto_axis",
    "score_texts",
    "score_conversations",
    "compute_separation",
    "find_best_layer",
    "token_projections",
    "logit_lens",
    "bootstrap_auc",
    # steering
    "steering",
    "steer_and_generate",
    "steer_and_generate_from_messages",
    "random_direction_like",
]

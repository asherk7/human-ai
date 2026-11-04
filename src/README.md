# `src/` — library guide

Four small modules. Build an axis from contrastive prompts, then project text
onto it. Everything returns plain dicts/tensors so it's easy to mix and match.

Typical flow: `load_model` → `collect_persona_means` (×2) →
`compute_contrastive_axis` → `find_best_layer` → `score_texts` / `project_onto_axis`.

---

## `model.py` — load a model
| Function | What it does |
| --- | --- |
| `load_model(model_id, device, dtype, ...)` | Loads any HF causal LM + tokenizer. Returns a dict: `model`, `tokenizer`, `n_layers`, `hidden_size`. Handles the Gemma 3 config quirk; pass `load_in_4bit=True` to fit 8–9B models on a T4. |

## `activations.py` — capture activations & build the axis
| Function / constant | What it does |
| --- | --- |
| `IDENTITY_QUESTIONS` | 24 identity questions ("Who are you?", …). |
| `HUMAN_SYS` / `AI_SYS` / `NEUTRAL_SYS` | System prompts: human persona / AI persona / neutral. |
| `build_chat_inputs(tok, system, user, device)` | Tokenizes a (system, user) chat turn, ready to generate. |
| `generate_and_capture(model, tok, system, question, layers)` | **Generates** a reply, returns mean activations over the *generated* tokens + the text. `do_sample=True` for sampling. |
| `capture_from_messages(model, tok, messages, layers)` | **No generation** — one forward pass over a full conversation, averages over the *assistant turn* (span located exactly, real context kept). Used to score rollouts. |
| `capture_from_text(model, tok, text, layers)` | Thin wrapper over `capture_from_messages` for a bare assistant string (no surrounding context). |
| `sanitize_messages(messages)` | Rewrites third-party transcripts into strictly alternating user/assistant turns Gemma's template accepts (system merged into first user turn, same-role runs joined). |
| `collect_persona_means(model, tok, system, questions, layers)` | Runs `generate_and_capture` over all questions, averages per layer. `n_rollouts>1` samples & averages to denoise. Returns means + responses. |
| `compute_contrastive_axis(pos_means, neg_means, layers)` | The axis: `pos − neg` per layer. Returns `axis`, `axis_unit`, `midpoint`, `separation` (norm-normalized), `best_layer`. (For the human-AI axis, `pos`=AI, `neg`=human → positive = AI-like.) |

## `analysis.py` — project & measure separation
| Function | What it does |
| --- | --- |
| `project_onto_axis(acts, axis_unit, midpoint, layer)` | One number: where the activation sits on the axis (+ = AI-like, − = human-like). |
| `score_texts(model, tok, texts, axis_unit, midpoint, layer)` | Projects a batch of bare assistant strings → list of scores. |
| `score_conversations(model, tok, conversations, axis_unit, midpoint, layer)` | Like `score_texts` but scores the assistant turn of full conversations (real context). |
| `compute_separation(pos_scores, neg_scores)` | How well the axis splits two groups: `mean_pos`, `mean_neg`, `gap`, `auc`, counts. |
| `find_best_layer(pos_means, neg_means, axis_unit, midpoint, layers)` | The layer where the two groups separate most. |
| `token_projections(model, tok, text, axis_unit, midpoint, layer)` | Per-token projections for a plain text (which tokens read AI-like vs human-like). First entry is BOS — ignore when ranking. |
| `logit_lens(model, tok, axis, layer, top_k)` | Decodes the axis through the unembedding: which vocab tokens the direction promotes/suppresses. |
| `bootstrap_auc(pos_scores, neg_scores)` | AUC with a bootstrap 95% CI — if the CI contains 0.5, the effect's direction isn't established. |

## `steering.py` — push the model along a vector
| Function | What it does |
| --- | --- |
| `steering(model, layer, vector, alpha)` | Context manager: forward hook adds `alpha * vector` to the layer's residual output at every position; removed on exit (even on error). |
| `steer_and_generate(model, tok, vector, alpha, prompt, layer)` | Generates a reply to a (system, user) prompt with the vector injected. `alpha=0` = baseline through the same code path. |
| `steer_and_generate_from_messages(model, tok, vector, alpha, messages, layer)` | Same, but continues an arbitrary conversation (eval scenarios; multi-turn). |
| `random_direction_like(vector, seed)` | Random vector with the same norm — the control: if this also flips behavior, the axis isn't special. |
| `resolve_decoder_layers(model)` | Finds the decoder-layer `ModuleList` regardless of nesting (Llama-style vs Gemma 3 multimodal). |

**Steering conventions:** steer with the **raw** axis so `alpha` is in cluster-gap
units (`alpha=1` = one full human→AI distance). Effects are non-monotonic in
`alpha` — calibrate on a few prompts. The steering layer is a separate choice
from the probing layer (readable ≠ injectable); start ~75% depth and sweep.

---

**Conventions:** layer `L`'s activation = hidden-state index `L+1` (index 0 is the
embedding layer). Captured vectors are float32 on CPU. Capture functions use
`@torch.no_grad()`.

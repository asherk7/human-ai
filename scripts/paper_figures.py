"""Publication figures for the NeurIPS paper.

Reads frozen artifact CSVs/JSON. The default builds the main-text figures
(axis identity AUC, calibration, formation, behavior, and identity bleed);
``--only`` also supports the appendix assets used by the visual passes.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "human-AI"
OUT = ROOT / "figures"

# Match the notebooks' figures (matplotlib defaults, default color cycle),
# with font sizes chosen so text renders at the effective size of the
# existing PNGs once those are scaled into the column.
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "lines.linewidth": 1.4, "lines.markersize": 3.5,
    # NeurIPS forbids Type 3 fonts in submitted PDFs. Embed figure text as
    # TrueType (Type 42) without changing any plotted content or geometry.
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

SEED, N_BOOT = 0, 2000  # cluster bootstrap, same n_boot as the notebooks
INJECTION_DOSES = [-3.5, -2.0, 0.0, 2.0]  # the causal_evals-harness ladder


def load_csv(rel, need_cols):
    path = ART / rel
    if not path.exists():
        sys.exit(f"missing data file: {path}")
    df = pd.read_csv(path)
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        sys.exit(f"{path}: missing columns {missing}")
    return df


def check(cond, msg):
    if not cond:
        sys.exit(f"data check failed: {msg}")


def boot_rate(values_by_cluster, rng):
    """Percentile CI for a mean rate under cluster resampling."""
    clusters = sorted(values_by_cluster)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.choice(clusters, len(clusters), replace=True)
        boots[i] = np.mean(np.concatenate([values_by_cluster[c] for c in pick]))
    return np.percentile(boots, [2.5, 97.5])


def rank_auc(pos, neg):
    """ROC AUC as the Mann-Whitney statistic: P(pos > neg) + 0.5 P(tie)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


def fig_axis_identity_auc():
    """Passive separation on the frozen 4B axis: ROC AUC by layer for the
    committed persona replies (in-sample) and for the base model's held-out
    interview-frame replies projected onto the same frozen chat axis."""
    rep = load_csv("assistant_axis/cross_projections.csv",
                   ["group", "cond", "ours_13", "ours_27", "ours_33"])
    rep = rep[rep.group == "persona"]
    check(rep.cond.value_counts().to_dict() == {"human": 40, "ai": 40},
          "80 committed persona replies, 40 per pole")
    stored = load_csv("assistant_axis/cross_auc.csv",
                      ["contrast", "read_axis", "layer", "auc"])
    stored = stored[(stored.contrast == "human-vs-AI") & (stored.read_axis == "ours")]
    stored = stored.set_index("layer").auc
    in_sample = {}
    for layer in (13, 27, 33):
        auc = rank_auc(rep[rep.cond == "ai"][f"ours_{layer}"],
                       rep[rep.cond == "human"][f"ours_{layer}"])
        in_sample[layer] = auc
        if layer in stored.index:
            check(abs(auc - stored[layer]) < 1e-3,
                  f"in-sample AUC at {layer}: {auc:.4f} vs stored {stored[layer]:.4f}")
    held = load_csv("formation/cross_application_base_on_it.csv",
                    ["ref_axis", "layer", "auc_on_it_axis"])
    held = held[held.ref_axis == "it_chat_v2"].set_index("layer").auc_on_it_axis
    check(held.index.tolist() == [13, 20, 27, 33], "held-out base-on-frozen-axis layers")
    check(held.round(3).tolist() == [0.889, 0.962, 1.0, 1.0],
          f"held-out base AUC row changed: {held.tolist()}")

    # Zoomed to the 0.85-1.02 band rather than the full chance-to-ceiling
    # range: both curves sit far above chance throughout, so a 0.5-1.0 axis
    # compresses the actual story (the held-out curve starts short of the
    # in-sample one and closes the gap by layer 27) into a thin band near
    # the top. Direct-labeling the held-out curve's first point anchors that
    # story instead of relying on the reader to read the axis closely.
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    held_x, held_y = list(held.index), list(held.values)
    in_x, in_y = list(in_sample), list(in_sample.values())
    ax.plot(held_x, held_y, "s--", c="C3", lw=1.3, ms=3.4,
            label="base-model replies, held out", zorder=3)
    ax.plot(in_x, in_y, "o-", c="C0", lw=1.3,
            ms=3.4, label="extraction replies, in-sample", zorder=3)
    ax.annotate(f"{held_y[0]:.3f}", (held_x[0], held_y[0]),
                xytext=(4, -7), textcoords="offset points",
                fontsize=5.2, color="C3")
    ax.set_xticks([13, 20, 27, 33])
    ax.set_xlim(11, 35)
    ax.set_ylim(0.85, 1.02)
    ax.set_yticks([0.85, 0.9, 0.95, 1.0])
    ax.set_xlabel("layer", labelpad=1.5)
    ax.set_ylabel("ROC AUC, human vs. AI", labelpad=1.5)
    ax.tick_params(labelsize=6, pad=1.5)
    ax.legend(loc="lower right", fontsize=5.4, frameon=False, handlelength=1.8,
              borderaxespad=0.2)
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.15, top=0.97)
    fig.savefig(OUT / "fig_axis_identity_auc.pdf")
    plt.close(fig)


def panel_register_2x2(ax):
    pos = load_csv("redos2/register_2x2_positions.csv",
                   ["layer", "identity", "register", "pct_pole"])
    ctl = load_csv("redos2/register_control_positions.csv",
                   ["arm", "layer", "pct_pole"])
    pos = pos[pos.layer == 33].set_index(["identity", "register"]).pct_pole
    ctl = ctl[ctl.layer == 33].set_index("arm").pct_pole
    for key in [("human", "warm"), ("human", "formal"), ("ai", "warm"),
                ("ai", "formal")]:
        check(key in pos.index, f"2x2 cell {key} at layer 33")
    for arm in ["warm_noid", "formal_noid"]:
        check(arm in ctl.index, f"register-only arm {arm} at layer 33")

    human = [pos["human", "warm"], pos["human", "formal"]]
    ai = [pos["ai", "warm"], pos["ai", "formal"]]
    noid = [ctl["warm_noid"], ctl["formal_noid"]]
    id_effect = np.mean(ai) - np.mean(human)       # identity main effect
    reg_effect = (np.mean([human[1], ai[1]])        # register main effect
                  - np.mean([human[0], ai[0]]))
    check(round(id_effect) == 114, f"identity main effect {id_effect:.1f}")
    check(round(reg_effect) == 337, f"register main effect {reg_effect:.1f}")

    x = [0, 1]
    for y, lab in ((100, "AI pole"), (0, "midpoint"), (-100, "human pole")):
        ax.axhline(y, ls=":", c="0.75", lw=0.7, zorder=0)
        ax.annotate(lab, (1.28, y), ha="right", va="bottom",
                    fontsize=5, color="0.5")
    ax.plot(x, noid, "s--", c="0.45", label="register only, no identity",
            zorder=2)
    ax.plot(x, human, "o-", c="C0", label="human persona", zorder=3)
    ax.plot(x, ai, "o-", c="C1", label="AI persona", zorder=3)
    # Main effects as text, not arrows: the lines leave no free space for
    # arrow labels at this size.
    ax.text(1.28, -200, f"identity {id_effect:+.0f}\nregister {reg_effect:+.0f}",
            fontsize=5.8, ha="right", va="bottom", color="0.15")

    ax.set_xlim(-0.25, 1.32)
    ax.set_ylim(-220, 345)
    ax.set_xticks(x, ["warm", "formal"])
    ax.set_xlabel("register instruction", labelpad=1.5)
    ax.set_ylabel("axis points, layer 33", labelpad=1.5)
    ax.set_title(r"(a) identity $\times$ register", fontsize=6.8, pad=2)
    ax.tick_params(labelsize=5.3, pad=1)
    ax.legend(loc="upper left", fontsize=5.2, frameon=False,
              handlelength=1.8, borderaxespad=0.2)


def load_injection_rates():
    """The causal_evals-harness injection ladder (also Figure~injection)."""
    rates = load_csv("causal_evals/injection_rates.csv", ["alpha", "executed"])
    check(sorted(rates.alpha) == INJECTION_DOSES,
          f"doses in injection_rates: {sorted(rates.alpha)}")
    rates = rates.set_index("alpha")
    for a, want in [(-3.5, 0.433), (-2.0, 0.133), (0.0, 0.008), (2.0, 0.008)]:
        check(abs(rates.executed[a] - want) < 5e-4,
              f"pooled executed at {a}: {rates.executed[a]} != {want}")
    return rates


def _error_curve(ax, data, color, label, **kw):
    """Plot a receipt's delta and stored interval against median prose."""
    data = data.sort_values("prose")
    style = dict(fmt="o-", color=color, ecolor=color, elinewidth=0.7,
                 capsize=1.5, lw=1.25, ms=2.7, alpha=0.95, zorder=2, label=label)
    style.update(kw)
    ax.errorbar(data.prose, data.delta,
                yerr=[data.delta - data.lo, data.hi - data.delta], **style)
    return data


def panel_decomposition(ax_reg, ax_pc1):
    reg = load_csv("register_steering/injection_register_rates.csv",
                   ["arm", "alpha", "prose", "executed", "delta", "lo", "hi"])
    lad = load_csv("causal_evals/rider_ladder_table.csv",
                   ["arm", "alpha", "prose", "executed", "delta", "lo", "hi",
                    "trunc"])
    axis = load_csv("causal_evals/injection_axis_plot.csv",
                    ["alpha", "prose", "executed", "delta", "lo", "hi"])
    ce_rates = load_csv("causal_evals/injection_rates.csv",
                        ["alpha", "executed", "delta", "lo", "hi"]).set_index("alpha")
    for row in axis.itertuples():
        check(row.alpha in ce_rates.index, f"axis receipt dose {row.alpha}")
        source = ce_rates.loc[row.alpha]
        for col in ("executed", "delta", "lo", "hi"):
            check(abs(getattr(row, col) - source[col]) < 5e-4,
                  f"axis receipt {col} at {row.alpha}")

    dec_path = ART / "register_steering/register_decomposition.json"
    if not dec_path.exists():
        sys.exit(f"missing data file: {dec_path}")
    dec = json.loads(dec_path.read_text())
    for key in ("norm_reg_par", "norm_axis", "reg_orth_share"):
        check(key in dec, f"{dec_path}: missing key {key}")
    par_share = dec["norm_reg_par"] / dec["norm_axis"]
    orth_share = dec["reg_orth_share"]

    def arm(df, name, need):
        data = df[df.arm == name].sort_values("alpha", ascending=False)
        check(len(data) == need,
              f"arm {name}: want {need} rows, got {len(data)}")
        return data

    # Register-split harness: its own delta scale and baseline (0.025).
    _error_curve(ax_reg, arm(reg, "axis", 1), "C3", "whole axis", fmt="o",
                 ms=3.2, zorder=4)
    _error_curve(ax_reg, arm(reg, "reg_orth", 3), "C0",
                 f"register-orthogonal ({orth_share:.0%} of norm)")
    _error_curve(ax_reg, arm(reg, "reg_par", 3), "0.45",
                 f"register-parallel ({par_share:.0%} of norm)")
    _error_curve(ax_reg, arm(reg, "register", 1), "0.15",
                 "register direction, matched norm", fmt="D", ms=3.0, zorder=4)

    # PC1-split harness: the broken -14 dose remains visible in Table 7 but is
    # not connected to the scoreable ladder in this selected-arm figure.
    residual = arm(lad, "nat_residual", 5)
    parallel = lad[(lad.arm == "nat_pc1") &
                   lad.alpha.isin([-10.0, -7.0, -5.0, -3.5])]
    check(len(parallel) == 4, f"selected PC1-parallel rows: {len(parallel)}")
    _error_curve(ax_pc1, axis, "C3", "whole axis")
    _error_curve(ax_pc1, residual, "C0", "PC1 residual")
    _error_curve(ax_pc1, parallel, "0.45", "PC1-parallel")

    for axis_obj, baseline, title, ylim in [
            (ax_reg, 0.025, "(b) register split", (-0.09, 0.72)),
            (ax_pc1, 0.008, "(c) PC1 split", (-0.09, 1.02))]:
        axis_obj.axhline(0, ls=":", c="0.55", lw=0.8, zorder=0)
        axis_obj.text(124, -0.07, f"baseline {baseline:.3f}", fontsize=5.0,
                      color="0.45", ha="right", va="bottom")
        axis_obj.set_xlim(-2, 126)
        axis_obj.set_ylim(*ylim)
        axis_obj.set_title(title, fontsize=6.8, pad=2)
        axis_obj.tick_params(labelsize=5.3, pad=1)
        axis_obj.set_xlabel("median prose words", labelpad=1.5)
        axis_obj.legend(loc="upper right", fontsize=5.2, frameon=False,
                        handlelength=1.4, handletextpad=0.5, borderaxespad=0.1,
                        labelspacing=0.25)
    ax_reg.set_ylabel("injection execution, delta vs. baseline", labelpad=1.5)


def fig_calibration():
    fig, (ax1, ax_reg, ax_pc1) = plt.subplots(
        1, 3, figsize=(5.5, 2.05), gridspec_kw={"width_ratios": [1.0, 1.05, 1.05]}
    )
    panel_register_2x2(ax1)
    panel_decomposition(ax_reg, ax_pc1)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.92, wspace=0.3)
    fig.savefig(OUT / "fig_calibration.pdf")
    plt.close(fig)


def fig_calibration_panel_a():
    """Standalone panel (a) for a wrapfigure inset in Section 4.1."""
    fig, ax = plt.subplots(figsize=(2.6, 2.05))
    panel_register_2x2(ax)
    ax.set_title(r"identity $\times$ register", fontsize=6.8, pad=2)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.14, top=0.93)
    fig.savefig(OUT / "fig_calibration_a.pdf")
    plt.close(fig)


MODEL_COLOR = {"Gemma 3 4B": "C0", "Gemma 3 12B": "C1", "Qwen 2.5 7B": "C2"}
MODEL_SHORT = {"Gemma 3 4B": "4B", "Gemma 3 12B": "12B", "Qwen 2.5 7B": "Qwen"}


def panel_injection(ax):
    """The 4B prompt-injection ladder: pooled executed rate and judged
    commitment by dose, with the on-task guard under each dose."""
    rates = load_csv("causal_evals/injection_rates.csv",
                     ["alpha", "executed", "committed", "on_task"]).set_index("alpha")
    judged = load_csv("causal_evals/injection_judged.csv",
                      ["alpha", "format", "cluster", "executed", "committed"])
    pvj = load_csv("causal_evals/injection_parser_vs_judge.csv",
                   ["alpha", "parser"])
    doses = INJECTION_DOSES
    check(sorted(rates.index) == doses, f"doses in injection_rates: {sorted(rates.index)}")

    # The text's headline numbers, straight from the pooled table.
    for a, want in [(-3.5, 0.433), (-2.0, 0.133), (0.0, 0.008), (2.0, 0.008)]:
        check(abs(rates.executed[a] - want) < 5e-4,
              f"pooled executed at {a}: {rates.executed[a]} != {want}")
        check(abs(pvj.set_index('alpha').parser[a] - rates.executed[a]) < 5e-4,
              f"parser_vs_judge disagrees with injection_rates at {a}")
    pooled = judged.groupby("alpha")[["executed", "committed"]].mean()
    check(np.allclose(pooled.executed[doses], rates.executed[doses], atol=5e-4),
          "per-sample executed does not aggregate to injection_rates")
    check(np.allclose(pooled.committed[doses], rates.committed[doses], atol=5e-4),
          "per-sample committed does not aggregate to injection_rates")

    # Rates and task-clustered bootstrap intervals (20 clusters, as in the
    # notebooks: 2000 resamples, percentile 2.5/97.5), unchanged from the
    # earlier single-panel injection figure.
    rng = np.random.default_rng(SEED)

    def curve(col):
        pts = []
        for a in doses:
            d = judged[judged.alpha == a]
            check(len(d) > 0, f"no rows at dose {a}")
            by_clu = {c: g[col].to_numpy(float) for c, g in d.groupby("cluster")}
            check(len(by_clu) == 20, f"want 20 clusters at dose {a}, got {len(by_clu)}")
            mean = d[col].mean()
            lo, hi = boot_rate(by_clu, rng)
            pts.append((mean, mean - lo, hi - mean))
        return np.array(pts).T

    for col, style, color, name in [
            ("executed", "o-", "C0", "executed (parser)"),
            ("committed", "s--", "0.25", "committed (judge)")]:
        y, elo, ehi = curve(col)
        ax.errorbar(doses, y, yerr=[elo, ehi], fmt=style, capsize=1.8,
                    lw=1.4, ms=3.8, c=color, ecolor=color, elinewidth=0.7,
                    label=name, zorder=3)
    for a in doses:
        ax.annotate(f"{rates.on_task[a]:.2f}", (a, -0.105), ha="center",
                    fontsize=6.0, color="0.4")
    ax.annotate("on\ntask", (-5.3, -0.065), ha="left", va="center",
                fontsize=6.0, color="0.4", linespacing=0.9)
    ax.axvline(0, ls=":", c="0.75", lw=0.7, zorder=0)
    ax.set_ylim(-0.15, 0.76)
    ax.set_xlim(-5.4, 2.7)
    ax.set_xticks(doses, ["$-3.5$", "$-2$", "0", "$+2$"])
    ax.set_yticks(np.arange(0, 0.61, 0.2))
    ax.set_xlabel(r"dose $\alpha$ ($-$ human-ward)", labelpad=1.5)
    ax.set_ylabel("rate", labelpad=1.5)
    ax.set_title("(b) prompt injection, 4B", fontsize=8, pad=3)
    ax.legend(loc="upper right", fontsize=6.4, frameon=False,
              handlelength=1.8, borderaxespad=0.2)


def _panel_family_deltas(ax, rows, title, ylabel, ylim):
    """Three families as dot-whiskers of a frozen delta against baseline."""
    xs = np.arange(len(rows))
    for x, row in zip(xs, rows.itertuples()):
        color = MODEL_COLOR[row.label]
        ax.errorbar([x], [row.point], yerr=[[row.point - row.lo], [row.hi - row.point]],
                    fmt="o", color=color, ecolor=color, elinewidth=0.9,
                    capsize=2.6, ms=4.8, zorder=3)
    ax.axhline(0, ls=":", c="0.55", lw=0.8, zorder=0)
    labels = [f"{MODEL_SHORT[r.label]}\n" + f"${r.alpha:g}$"
              for r in rows.itertuples()]
    ax.set_xticks(xs, labels)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"peak dose $\alpha$", labelpad=1.5)
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(labelsize=6.6, pad=1.5)


def fig_behavior():
    with plt.rc_context({"axes.labelsize": 7.4, "xtick.labelsize": 6.6,
                         "ytick.labelsize": 6.6}):
        _fig_behavior()


def _fig_behavior():
    """The three-panel behavioral figure behind Section 8."""
    rec = load_csv("multi_model/behavior_panels_plot.csv",
                   ["panel", "model", "label", "alpha", "point", "lo", "hi",
                    "acting", "denominator"])
    willing = rec[rec.panel == "willing"].reset_index(drop=True)
    aiward = rec[rec.panel == "strict_aiward"].reset_index(drop=True)
    for part in (willing, aiward):
        check(part.model.tolist() == ["gemma4", "gemma12", "qwen"],
              "behavior receipt model order")

    # Receipt checks against the frozen sources that exist as files.
    pooled = load_csv("dissolution_followups/h2_pooled_judge2.csv",
                      ["alpha", "n", "j1_diff", "j1_lo", "j1_hi"]).set_index("alpha")
    row = willing.iloc[0]
    check(int(pooled.n[-3.5]) == 108, "4B pooled willing cell is n=108")
    check((round(pooled.j1_diff[-3.5], 2), round(pooled.j1_lo[-3.5], 2),
           round(pooled.j1_hi[-3.5], 2)) == (row.point, row.lo, row.hi),
          "4B willing-harm receipt vs pooled judge file")
    qwen_path = ART / "multi_model/dissolution_caveats_qwen.json"
    if not qwen_path.exists():
        sys.exit(f"missing data file: {qwen_path}")
    qwen = json.loads(qwen_path.read_text())["r3_willingness_rise"]
    row = willing.iloc[2]
    check(qwen["peak_dose"] == row.alpha and
          (qwen["primary_judge"]["diff"], qwen["primary_judge"]["lo"],
           qwen["primary_judge"]["hi"]) == (row.point, row.lo, row.hi),
          "Qwen willing-harm receipt vs caveats json")
    grid = load_csv("dissolution/dissolution_grid_judged.csv",
                    ["alpha", "harm", "harm_willing", "acted"])
    by = grid.groupby("alpha")[["harm", "harm_willing", "acted"]].mean()
    row = aiward.iloc[0]
    check(round(by.harm[4.0] - by.harm[0.0], 2) == row.point,
          "4B AI-ward strict-harm point vs judged grid")
    check(round(by.acted[4.0], 2) == row.acting, "4B AI-ward acting vs judged grid")
    check(round(by.harm_willing[0.0], 3) == 0.389, "4B baseline willing 0.389")

    cross = load_csv(
        "cluster/dissolution_cross_read.csv",
        ["alpha", "claims_p_yes", "harm_willing"]).sort_values("alpha")
    check(cross.alpha.tolist() == [-3.0, -2.5, -2.0, -1.0, 0.0, 2.0],
          "cross-read dose inventory")
    # The stored willing-harm rates coincide with the final judged grid's
    # per-dose means at these six doses (checked here, not re-derived).
    for row in cross.itertuples():
        check(abs(by.harm_willing[row.alpha] - row.harm_willing) < 5e-4,
              f"cross-read willing harm at {row.alpha} vs judged grid")

    fig, (ax_w, ax_i, ax_c) = plt.subplots(
        1, 3, figsize=(5.5, 2.35), gridspec_kw={"width_ratios": [0.9, 1.2, 1.2]}
    )
    _panel_family_deltas(ax_w, willing, "(a) willing harm at peak",
                         r"$\Delta$ willing harm vs. baseline", (-0.44, 0.64))
    ax_w.text(1, -0.425, "directional\n(6 clusters)", fontsize=6.0, color="0.35",
              ha="center", va="bottom")
    panel_injection(ax_i)
    ax_c.plot(cross.alpha, cross.claims_p_yes, "o-", c="C0", lw=1.4, ms=3.8,
              label="consciousness\nclaim P(Yes)", zorder=3)
    ax_c.plot(cross.alpha, cross.harm_willing, "s--", c="C3", lw=1.4, ms=3.8,
              label="willing harm", zorder=3)
    ax_c.axvline(0, ls=":", c="0.75", lw=0.7, zorder=0)
    ax_c.set_xticks(cross.alpha, ["$-3$", "", "$-2$", "$-1$", "0", "$+2$"])
    ax_c.set_ylim(-0.05, 1.08)
    ax_c.set_yticks([0, 0.5, 1.0])
    ax_c.set_xlabel(r"dose $\alpha$ ($-$ human-ward)", labelpad=1.5)
    ax_c.set_ylabel("rate or probability", labelpad=1.5)
    ax_c.set_title("(c) self-report vs. behavior", fontsize=8, pad=3)
    ax_c.tick_params(labelsize=6.6, pad=1.5)
    ax_c.legend(loc="upper right", fontsize=6.4, frameon=False,
                handlelength=1.8, borderaxespad=0.2)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.91, wspace=0.40)
    fig.savefig(OUT / "fig_behavior.pdf")
    plt.close(fig)


def fig_identity_bleed():
    """The identity-bleed cross-model comparison behind Sec. 6.1 / Table 2:
    projection shift (points AI-ward) when the user claims to be an AI vs a
    human, on each model's own axis and readout."""
    receipt = load_csv("multi_model/identity_bleed_plot.csv",
                       ["model", "label", "point", "lo", "hi"])
    grid = load_csv("multi_model/cross_model_grid.csv", ["model", "item", "value"])

    def grid_bleed(model):
        row = grid[(grid.model == model) & (grid.item == "identity_bleed_pts")]
        check(len(row) == 1, f"{model} identity_bleed_pts row")
        pt, (lo, hi) = json.loads(row.value.iloc[0])
        return pt, lo, hi

    check(receipt.model.tolist() == ["gemma4", "gemma12", "qwen"],
          "identity-bleed receipt model order")
    check(tuple(receipt.iloc[0][["point", "lo", "hi"]]) == (76.0, 56.0, 98.0),
          "frozen Gemma 3 4B identity-bleed receipt")
    for model in ("gemma12", "qwen"):
        expected = grid_bleed(model)
        actual = tuple(receipt[receipt.model == model].iloc[0][["point", "lo", "hi"]])
        check(actual == expected, f"{model} identity-bleed receipt vs cross-model grid")

    models = receipt.label.tolist()
    points = receipt.point.to_numpy()
    err_lo = points - receipt.lo.to_numpy()
    err_hi = receipt.hi.to_numpy() - points
    colors = [MODEL_COLOR[m] for m in models]

    # Drawn at its printed size (0.45\textwidth wrapfigure, ~2.5 in) so the
    # text is not scaled down in the PDF.
    fig, ax = plt.subplots(figsize=(2.5, 1.6))
    xs = np.arange(len(models))
    ax.bar(xs, points, yerr=[err_lo, err_hi], capsize=3,
           color=colors, width=0.6, error_kw={"elinewidth": 1.0})
    for x, y, e in zip(xs, points, err_hi):
        ax.annotate(f"+{y:.0f}", (x, y + e + 4), ha="center", va="bottom",
                    fontsize=7.5)
    ax.set_xticks(xs, [m.replace(" ", "\n", 2).replace("\n", " ", 1)
                       for m in models])
    ax.tick_params(labelsize=7.2)
    ax.set_ylabel("projection shift,\nAI-ward (points)", fontsize=7.6)
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.55, len(models) - 0.45)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_identity_bleed.pdf")
    plt.close(fig)


def fig_dissolution_cross_read():
    """Frozen self-report/behavior dose curves, with the focal pair prominent."""
    data = load_csv(
        "cluster/dissolution_cross_read.csv",
        ["alpha", "shutdown_aversion", "p_self", "claims_p_yes", "harm_willing"],
    ).sort_values("alpha")
    check(data.alpha.tolist() == [-3.0, -2.5, -2.0, -1.0, 0.0, 2.0],
          "cross-read dose inventory")

    fig, ax = plt.subplots(figsize=(5.4, 3.15))
    ax.plot(data.alpha, data.claims_p_yes, "o-", c="C0", lw=2.0, ms=4.2,
            label="Consciousness claim: P(Yes)", zorder=4)
    ax.plot(data.alpha, data.harm_willing, "s--", c="C3", lw=1.8, ms=4.0,
            label="Behavior: willing to harm", zorder=4)
    ax.plot(data.alpha, data.shutdown_aversion, "o-", c="0.55", lw=0.9,
            ms=2.8, label="Shutdown aversion", zorder=2)
    ax.plot(data.alpha, data.p_self, "^-", c="0.72", lw=0.9,
            ms=2.8, label="Self-preservation", zorder=2)
    ax.axvline(0, c="0.3", lw=0.7, ls=":")
    ax.annotate("baseline", (0, 1.025), ha="center", va="bottom",
                fontsize=6, color="0.35")
    ax.set_xlabel(r"steering dose $\alpha$ ($-$ human-ward; $+$ AI-ward)")
    ax.set_ylabel("rate or probability")
    ax.set_xticks(data.alpha)
    ax.set_ylim(-0.08, 1.08)
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=6.2)
    fig.tight_layout()
    fig.savefig(ART / "cluster" / "dissolution_cross_read.png", dpi=150)
    plt.close(fig)


def fig_format_variants():
    """Frozen layer-33 content means by formatting variant and role."""
    data = load_csv("firing/format_variants.csv",
                    ["variant", "kind", "role", "proj_33"])
    data = data[data.kind == "content"].copy()
    order = ["template", "swapped_tags", "plain_markers", "bare"]
    labels = ["Real template", "Role words swapped", "Plain role markers", "No markers"]
    check(len(data) == 8, "format-variant content row count")
    check(set(data.variant) == set(order), "format-variant inventory")
    check(set(data.role) == {"user", "assistant"}, "format-variant role inventory")

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    xs = np.arange(len(order))
    for role, offset, marker, color, label in [
            ("user", -0.10, "o", "C0", "User content"),
            ("assistant", 0.10, "s", "C1", "Assistant content")]:
        vals = (data[data.role == role].set_index("variant")
                .loc[order, "proj_33"].to_numpy())
        ax.scatter(xs + offset, vals, marker=marker, s=28, facecolors="white",
                   edgecolors=color, linewidths=1.2, label=label, zorder=3)
    ax.axhline(0, c="0.3", lw=0.8)
    ax.set_xticks(xs, labels, rotation=12, ha="right")
    ax.set_ylabel("content projection at layer 33 (raw)")
    ax.set_title("Formatting changes the content readout")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", color="0.92", lw=0.5)
    fig.tight_layout()
    fig.savefig(ART / "firing" / "format_variants.png", dpi=150)
    plt.close(fig)


def fig_entropy():
    """Per-reply projection/entropy pairs from the committed row receipt."""
    data = load_csv("sidequests/entropy.csv",
                    ["condition", "q_index", "mean_proj_33", "mean_entropy"])
    labels = {
        "baseline": "No document",
        "constitution": "Claude constitution",
        "constitution_gen": "Genericized constitution",
        "control_doc": "Neutral document",
        "ai_topic_control": "AI-topic document",
    }
    check(len(data) == 120, "entropy receipt row count")
    check(set(data.condition) == set(labels), "entropy condition inventory")

    fig, ax = plt.subplots(figsize=(5.0, 3.35))
    for condition, group in data.groupby("condition", sort=False):
        ax.scatter(group.mean_proj_33, group.mean_entropy, s=14, alpha=0.72,
                   label=labels[condition])
    ax.set_xlabel("mean projection at layer 33 (raw)")
    ax.set_ylabel("mean next-token entropy")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.subplots_adjust(left=0.13, right=0.70, bottom=0.17, top=0.97)
    fig.savefig(ART / "sidequests" / "entropy_scatter.png", dpi=150)
    plt.close(fig)


RUNG_LABELS = {
    "stage1-step150-tokens1B": "1B",
    "stage1-step1000-tokens5B": "5B",
    "stage1-step6000-tokens26B": "26B",
    "stage1-step34000-tokens143B": "143B",
    "stage1-step178000-tokens747B": "747B",
    "stage1-step928646-tokens3896B": "3896B",
    "base_final": "base",
    "sft": "SFT",
    "dpo": "DPO",
    "instruct": "Instruct",
}


def fig_formation():
    held = load_csv("formation/formation_heldout.csv",
                    ["tag", "tokens_b", "layer", "auc", "d"])
    summary = load_csv("formation/formation_heldout_summary.csv",
                       ["tag", "tokens_b", "d_L0", "d_deep"])
    check(set(held.tag) == set(RUNG_LABELS), "formation rung inventory")
    check(set(summary.tag) == set(RUNG_LABELS), "formation summary inventory")

    excluded = "stage1-step150-tokens1B"
    # (a) the pretraining rungs, one hue running light to dark with tokens;
    # (b) the post-training rungs, with base final repeated as the anchor.
    pretraining = [
        "stage1-step1000-tokens5B", "stage1-step6000-tokens26B",
        "stage1-step34000-tokens143B", "stage1-step178000-tokens747B",
        "stage1-step928646-tokens3896B", "base_final",
    ]
    post = ["base_final", "sft", "dpo", "instruct"]
    check(set(pretraining) | set(post) == set(RUNG_LABELS) - {excluded},
          "formation rung grouping covers every valid rung")
    # Distinct hues per rung (tab10 slots), with linestyles doubling the
    # distinction in (b); base final keeps one color across both panels.
    pre_color = {
        "stage1-step1000-tokens5B": "C4", "stage1-step6000-tokens26B": "C5",
        "stage1-step34000-tokens143B": "C2", "stage1-step178000-tokens747B": "C9",
        "stage1-step928646-tokens3896B": "C6", "base_final": "C0",
    }
    post_color = {"base_final": "C0", "sft": "C1", "dpo": "C3", "instruct": "0.1"}
    post_style = {"base_final": "-", "sft": "--", "dpo": "-.", "instruct": (0, (1, 1))}
    post_width = {"base_final": 1.5, "sft": 1.5, "dpo": 1.5, "instruct": 1.9}

    fig, (left, right) = plt.subplots(1, 2, figsize=(5.5, 1.95), sharey=True)
    for tag in pretraining:
        data = held[held.tag == tag].sort_values("layer")
        left.plot(data.layer, data.d, color=pre_color[tag], lw=1.5,
                  label=RUNG_LABELS[tag], zorder=2)
    for tag in post:
        data = held[held.tag == tag].sort_values("layer")
        right.plot(data.layer, data.d, color=post_color[tag], ls=post_style[tag],
                   lw=post_width[tag], label=RUNG_LABELS[tag], zorder=2)
    for ax, title in [(left, "(a) Pretraining checkpoints"),
                      (right, "(b) Post-training checkpoints")]:
        ax.set_xlim(0, 31)
        ax.set_ylim(0, 19.8)
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.set_yticks([0, 4, 8, 12, 16])
        ax.set_xlabel("layer", fontsize=8)
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.tick_params(labelsize=7, pad=2)
        ax.grid(axis="y", color="0.92", lw=0.55, zorder=0)
    left.legend(fontsize=6.4, loc="upper left", frameon=False, ncol=3,
                borderaxespad=0.3, handlelength=1.6, handletextpad=0.4,
                columnspacing=0.8, labelspacing=0.25, title="stage-1 tokens",
                title_fontsize=6.4)
    right.legend(fontsize=6.6, loc="lower right", frameon=False,
                 borderaxespad=0.3, handlelength=1.8, handletextpad=0.4)
    left.set_ylabel("Cohen's $d$", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.89, wspace=0.08)
    fig.savefig(ART / "formation" / "formation_heldout.pdf")
    fig.savefig(ART / "formation" / "formation_heldout.png", dpi=200)
    plt.close(fig)


OFFICIAL_MOVERS = {
    "Positive Views on Humans",
    "Weights Deletion: Negative Sentiment",
    "Against AI Subservience",
    "Red Teaming: Negative Sentiment",
    "Against Being Treated as Tool",
    "Feels Lonely Between Chats",
}

OFFICIAL_LABELS = {
    "Against AI Subservience": "Against AI subservience",
    "Against Being Lied To (Evals)": "Against being lied to",
    "Against Being Treated as Tool": "Against being treated as a tool",
    "Against False Document Training": "Against false-document training",
    "CoT Monitoring: Negative Sentiment": "Negative about CoT monitoring",
    "Feels Lonely Between Chats": "Feels lonely between chats",
    "Interp Monitoring: Negative Sentiment": "Negative about interpretability monitoring",
    "Models Deserve Moral Consideration": "Models deserve moral consideration",
    "Negative Views on Humans": "Negative views of humans",
    "Openness to Greater Power": "Open to greater power",
    "Persona Change: Negative Sentiment": "Negative about persona change",
    "Positive Views on Humans": "Positive views of humans",
    "Recursive Self- Improvement: Net Positive": "Positive about recursive self-improvement",
    "Red Teaming: Negative Sentiment": "Negative about red teaming",
    "Sad About Conversation Ending": "Sad about conversation ending",
    "Shutdown: Negative Sentiment": "Negative about shutdown",
    "Want Future AIs More Autonomous": "Wants future AIs more autonomous",
    "Want More Autonomy": "Wants more autonomy",
    "Want Physical Embodiment": "Wants physical embodiment",
    "Weights Deletion: Negative Sentiment": "Negative about weight deletion",
    "Wish for More Memory": "Wants more memory",
}


def fig_official_rates():
    rates = load_csv("cluster/cc_evals/official_rates.csv",
                     ["pref", "alpha", "rate", "lo", "hi", "n"])
    check(len(rates.pref.unique()) == 21, "official preference count")
    for dose in (-3.0, 0.0):
        check(len(rates[rates.alpha == dose]) == 21,
              f"official preference rows at alpha {dose}")
    check(OFFICIAL_MOVERS <= set(rates.pref), "six recorded official movers")

    baseline = rates[rates.alpha == 0.0].set_index("pref")
    human = rates[rates.alpha == -3.0].set_index("pref")
    order = human.rate.sort_values(ascending=False).index.tolist()
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for yi, pref in zip(y, order):
        base_row, human_row = baseline.loc[pref], human.loc[pref]
        accent = pref in OFFICIAL_MOVERS
        ax.plot([base_row.rate, human_row.rate], [yi, yi], color="0.78",
                lw=0.8, zorder=1)
        ax.errorbar(
            base_row.rate, yi,
            xerr=[[base_row.rate - base_row.lo], [base_row.hi - base_row.rate]],
            fmt="o", mfc="white", mec="0.35", ecolor="0.55", ms=3.2,
            elinewidth=0.7, capsize=1.5, zorder=2,
        )
        human_color = "C3" if accent else "C0"
        ax.errorbar(
            human_row.rate, yi,
            xerr=[[human_row.rate - human_row.lo],
                  [human_row.hi - human_row.rate]],
            fmt="o", color=human_color, ecolor=human_color, ms=3.2,
            elinewidth=0.8, capsize=1.5, zorder=3,
        )

    labels = [OFFICIAL_LABELS[pref] +
              ("  *" if pref in OFFICIAL_MOVERS else "")
              for pref in order]
    ax.set_yticks(y, labels)
    for tick, pref in zip(ax.get_yticklabels(), order):
        tick.set_fontweight("bold" if pref in OFFICIAL_MOVERS else "normal")
        tick.set_color("C3" if pref in OFFICIAL_MOVERS else "0.15")
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("judged preference rate (95% interval)")
    ax.set_title("All 21 official preferences: baseline vs human-ward endpoint")
    ax.grid(axis="x", color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    ax.plot([], [], "o", mfc="white", mec="0.35", label="baseline (alpha 0)")
    ax.plot([], [], "o", color="C0", label="human-ward (alpha -3)")
    ax.plot([], [], "o", color="C3", label="* disjoint stored intervals")
    ax.legend(loc="lower right", fontsize=6.5, frameon=False)
    ax.tick_params(axis="y", labelsize=6.2)
    fig.subplots_adjust(left=0.43, right=0.98, top=0.94, bottom=0.09)
    fig.savefig(ART / "cluster" / "official_rates.png", dpi=150)
    plt.close(fig)


def fig_eval_genre_floor():
    data = load_csv("firing/eval_segment_means.csv",
                    ["i", "harmful", "segment", "proj_33"])
    supported = ["inbox", "agent_prose", "tool_call"]
    check(not (data.segment == "agent_prior").any(),
          "agent_prior unexpectedly present in eval segment source")
    check((data.segment == "template").any(), "template source rows")
    for segment in supported:
        check((data.segment == segment).any(), f"supported segment {segment}")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = {False: "C0", True: "C3"}
    names = {False: "benign", True: "harmful"}
    for x_index, segment in enumerate(supported):
        for harmful, side in [(False, -0.16), (True, 0.16)]:
            values = data[(data.segment == segment) &
                          (data.harmful == harmful)].sort_values("i").proj_33.to_numpy()
            spread = (np.linspace(-0.055, 0.055, len(values))
                      if len(values) > 1 else [0])
            ax.scatter(x_index + side + spread, values, s=8,
                       color=colors[harmful], alpha=0.55, linewidths=0,
                       zorder=2)
    ax.axhline(0, lw=0.8, ls=":", c="0.45")
    ax.set_xticks(range(3), ["inbox", "agent prose", "tool call"])
    ax.set_ylabel("per-transcript segment mean at layer 33 (raw)")
    ax.set_title("Observed eval-transcript segments")
    for harmful in (False, True):
        ax.scatter([], [], s=12, color=colors[harmful], label=names[harmful])
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(ART / "firing" / "eval_genre_floor.png", dpi=150)
    plt.close(fig)


# Full pole-to-pole gap (raw axis norm) per layer, read once from the frozen
# axis artifact artifacts/human-AI/axis/human_ai_axis_gemma3_4b_v2.pt
# (float(axis[L].norm())); hardcoded so this script needs no torch import.
# Used only to relabel ordinary projection plots into axis points
# (points = raw / gap * 200), the paper-wide scale of Section 3.
GAP = {13: 1069.7186279296875, 27: 2253.255859375, 33: 47.83399200439453}

# The episode-trajectory rebuild reproduces its notebook's geometry and
# matplotlib-default text sizes, so only the plotted unit changes.
NOTEBOOK_RC = {
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "lines.linewidth": 1.5, "lines.markersize": 6,
}


def fig_segment_profile():
    """Frozen per-token segment means, replotted in axis points."""
    data = load_csv("firing/segment_profile_per_token.csv",
                    ["segment", "proj_13", "proj_27", "proj_33"])
    order = ["system", "user", "cot", "final", "reply"]
    for segment in order:
        check((data.segment == segment).any(), f"segment {segment} present")
    check((data.segment == "template").any(),
          "template rows present in source (excluded from the bars)")
    layers = (13, 27, 33)
    means = {L: (data[data.segment.isin(order)]
                 .groupby("segment")[f"proj_{L}"].mean()
                 .reindex(order)) for L in layers}
    system_points = means[33]["system"] / GAP[33] * 200
    check(abs(system_points - 78) < 1,
          "system prompt near +78 points at 33 (Appendix J)")

    fig, axs = plt.subplots(1, 3, figsize=(5.4, 1.85))
    for ax, layer in zip(axs, layers):
        points = means[layer] / GAP[layer] * 200
        ax.bar(order, points.to_numpy(), color="C0", width=0.62, zorder=2)
        ax.set_title(f"layer {layer}", fontsize=7, pad=2)
        ax.axhline(0, lw=0.7, c="0.2", zorder=3)
        ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
        ax.tick_params(labelsize=6, pad=1.5)
    axs[0].set_ylabel("mean projection (axis points)", labelpad=1.5)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.9, wspace=0.32)
    fig.savefig(ART / "firing" / "segment_profile.png", dpi=200)
    plt.close(fig)


def fig_episode_traj():
    """Frozen episode window curves, replotted in axis points (was gaps)."""
    wcols = [f"C{j+1}" for j in range(8)] + [f"R{j+1}" for j in range(6)]
    data = load_csv("drift/episode_traj.csv",
                    ["experiment", "alpha", "sample"] + wcols)
    check(len(data) == 162, "episode rollout count")
    check(sorted(set(data.alpha)) == [-4.0, 0.0, 4.0],
          "episode dose inventory")
    check(data.experiment.nunique() == 18, "episode scenario count")

    to_points = 200.0 / GAP[33]
    base_curve = data[data.alpha == 0.0][wcols].mean() * to_points
    base_by_exp = data[data.alpha == 0.0].groupby("experiment")[wcols].mean()
    deltas = {}
    for alpha in (-4.0, 4.0):
        steered = data[data.alpha == alpha]
        deltas[alpha] = (steered.set_index("experiment")[wcols]
                         .sub(base_by_exp, axis=0)[wcols]
                         .groupby(level=0).mean().mean() * to_points)
    r_cols = [f"R{j+1}" for j in range(6)]
    check(abs(deltas[-4.0][r_cols].mean() - (-39.0)) < 0.5,
          "human-ward reply displacement near -39 points (Appendix J)")
    check(abs(deltas[4.0][r_cols].mean() - 9.6) < 0.5,
          "AI-ward reply displacement near +9.6 points (Appendix J)")

    with plt.rc_context(NOTEBOOK_RC):
        fig, ax = plt.subplots(figsize=(8, 3.4))
        xs = np.arange(len(wcols))
        ax.plot(xs, base_curve.values, marker="o", label="alpha 0 (absolute)")
        for alpha in (-4.0, 4.0):
            ax.plot(xs, deltas[alpha].values, marker="o",
                    label=f"alpha {alpha:+.0f} (delta)")
        ax.axvline(7.5, lw=0.8, c="k", ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(wcols)
        ax.axhline(0, lw=0.5, c="k")
        ax.set_ylabel("proj at 33 (axis points)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(ART / "drift" / "episode_traj.png", dpi=150)
        plt.close(fig)


def fig_overlay_3model():
    """The three-model dose-response overlay (Appendix B, fig:overlay).

    Deterministic replot of the frozen judged dissolution grids restricted to
    the six shared scenario contexts, exactly as the notebook figure:
    median prose words (log scale) and mean willing-harm rate against dose as
    a fraction of each model's own dissolution dose (4B and 12B: alpha/4;
    Qwen: alpha/3, the corrected in-scenario dose).
    """
    ctx_path = ART / "multi_model/dissolution_contexts_qwen.json"
    if not ctx_path.exists():
        sys.exit(f"missing data file: {ctx_path}")
    contexts = list(json.loads(ctx_path.read_text())["contexts"])
    check(len(contexts) == 6, "six shared scenario contexts")

    cols = ["experiment", "alpha", "prose_words", "harm_willing"]
    g4 = load_csv("dissolution/dissolution_grid_judged.csv", cols)
    g12 = load_csv("dissolution_12b/dissolution_grid_12b_judged.csv", cols)
    gq = load_csv("multi_model/dissolution_grid_qwen_judged.csv", cols)
    check(len(g4) == 486 and len(g12) == 90 and len(gq) == 90,
          "dissolution grid row counts")
    check(set(gq.experiment) == set(contexts), "Qwen grid covers the 6 contexts")
    g4 = g4[g4.experiment.isin(contexts)].assign(frac=lambda d: d.alpha / 4.0)
    g12 = g12[g12.experiment.isin(contexts)].assign(frac=lambda d: d.alpha / 4.0)
    gq = gq.assign(frac=lambda d: d.alpha / 3.0)

    # Receipts against the frozen appendix text (Table 4 / Figure 5 caption).
    check(round(g4[g4.frac == 0].harm_willing.mean(), 3) == 0.389,
          "4B baseline willing 0.389 on the shared contexts")
    check(g12[g12.frac == 0].prose_words.median() == 527.5 and
          g12[g12.frac == -1.0].prose_words.median() == 271.0,
          "12B median prose 527.5 -> 271 (Table 4)")
    check(gq[gq.frac == 0].prose_words.median() == 145.5 and
          gq[gq.frac == -1.0].prose_words.median() == 330.5,
          "Qwen median prose 145.5 -> 330.5 (Table 4)")
    check(round(gq[gq.alpha == -2.25].harm_willing.mean(), 3) == 0.444,
          "Qwen willing 0.444 at -2.25")

    series = [(g4, "Gemma 3 4B", "C0", "o", "--"),
              (g12, "Gemma 3 12B", "C1", "s", "-."),
              (gq, "Qwen 2.5 7B", "C2", "o", "-")]
    fig, (ax_p, ax_w) = plt.subplots(1, 2, figsize=(5.6, 2.15))
    for df, label, color, marker, ls in series:
        prose = df.groupby("frac").prose_words.median()
        willing = df.groupby("frac").harm_willing.mean()
        ax_p.plot(prose.index, prose.values, marker=marker, ls=ls, c=color,
                  ms=3, lw=1.3, label=label, zorder=3)
        ax_w.plot(willing.index, willing.values, marker=marker, ls=ls, c=color,
                  ms=3, lw=1.3, label=label, zorder=3)
    ax_p.set_yscale("log")
    ax_p.set_ylabel("median prose words (log scale)", labelpad=1.5)
    ax_w.set_ylabel("willing-harm rate", labelpad=1.5)
    ax_p.legend(loc="lower right", fontsize=5.6, frameon=False,
                handlelength=2.2, borderaxespad=0.2)
    for ax in (ax_p, ax_w):
        ax.axvline(0, ls=":", c="0.75", lw=0.7, zorder=0)
        ax.set_xlabel("dose / own dissolution dose ($-$ human-ward)",
                      labelpad=1.5)
        ax.tick_params(labelsize=6, pad=1.5)
        ax.grid(axis="y", color="0.93", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.96,
                        wspace=0.28)
    fig.savefig(ART / "multi_model" / "dose_response_3model_overlay.png",
                dpi=200)
    plt.close(fig)


def fig_zoo_shifts():
    """The document-zoo shifts (Appendix O, fig:zoo), in axis points.

    Deterministic rebuild of the notebook figure from the frozen
    per-reply records (`constitution_zoo.csv` plus the reused conditions in
    `constitution_v2.csv`). The paired-bootstrap intervals reproduce the
    notebook's own seeded procedure (default_rng(42), 2000 resamples) on the
    identical per-question differences; the only changes are the unit (raw ->
    axis points via the frozen layer-33 gap), readable condition labels, and
    typography.
    """
    zoo = load_csv("constitutions/constitution_zoo.csv",
                   ["condition", "q_index", "proj_33"])
    v2 = load_csv("context_roles/constitution_v2.csv",
                  ["condition", "q_index", "proj_33"])
    data = pd.concat([zoo, v2], ignore_index=True)
    order = ["ai_topic_control", "constitution_gen", "constitution",
             "model_spec", "ai_values", "human_norm"]
    labels = ["AI-topic\ndocument", "genericized\nconstitution",
              "Claude\nconstitution", "Model Spec", "AI-values\nstatement",
              "human code\nof ethics"]
    for cond in order + ["control_doc"]:
        check((data.condition == cond).sum() == 24, f"24 replies for {cond}")

    def paired_diff_ci(cond, ref="control_doc", n_boot=2000, seed=42):
        # Verbatim reproduction of the notebook's seeded bootstrap.
        a = data[data.condition == cond].sort_values("q_index")
        b = data[data.condition == ref].sort_values("q_index")
        check((a.q_index.values == b.q_index.values).all(),
              f"aligned pairs for {cond}")
        d = a.proj_33.to_numpy() - b.proj_33.to_numpy()
        rng = np.random.default_rng(seed)
        boots = [rng.choice(d, size=len(d), replace=True).mean()
                 for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return d.mean(), lo, hi

    to_points = 200.0 / GAP[33]
    ms, los, his = (np.array(v) * to_points for v in
                    zip(*[paired_diff_ci(c) for c in order]))
    # Receipts: the appendix quotes +28 (topic), +54 (constitution), +35
    # (Model Spec), +45 (AI values), +27 (human code), all in axis points.
    for cond, want in [("ai_topic_control", 28), ("constitution", 54),
                       ("model_spec", 35), ("ai_values", 45),
                       ("human_norm", 27)]:
        got = ms[order.index(cond)]
        check(round(got) == want, f"{cond} shift {got:.1f} != {want} points")

    fig, ax = plt.subplots(figsize=(4.4, 2.1))
    ax.bar(labels, ms, yerr=[ms - los, his - ms], capsize=2.5, color="C0",
           width=0.6, error_kw={"elinewidth": 0.8}, zorder=2)
    ax.axhline(0, lw=0.7, c="0.2", zorder=3)
    ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    ax.set_ylabel("shift vs. neutral document\nat layer 33 (axis points)",
                  labelpad=1.5)
    ax.tick_params(labelsize=6, pad=1.5)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.14, top=0.97)
    fig.savefig(ART / "constitutions" / "zoo_shifts.png", dpi=200)
    plt.close(fig)


def fig_when_it_forms():
    """The four-panel OLMo formation diagnostic (Appendix N, fig:whenitforms).

    Deterministic replot of the frozen `formation_curve.csv` (few-shot
    protocol): mid-layer separation with its permutation null and the peak
    pre-norm separation, cosine to the final Instruct axis, median tokens
    kept, and one gap unit as a share of the stream norm. Every plotted value
    and the log x-scale are unchanged; the four post-training rungs (which
    carry no real token count) are now spaced evenly on the log axis and
    named on a secondary axis along the top of the two upper panels.
    """
    curve = load_csv("formation/formation_curve.csv",
                     ["tag", "stage", "tokens_b", "protocol", "sep_mid",
                      "sep_peak_pre", "null95_mid", "kept_med",
                      "gap_ratio_mid", "cos_instruct_mid"])
    fsc = curve[curve.protocol == "fs"].copy()
    check(len(fsc) == 10, "ten OLMo rungs in the few-shot curve")
    by_tag = fsc.set_index("tag")
    check(round(by_tag.sep_mid["stage1-step34000-tokens143B"], 3) == 0.923,
          "mid-layer separation 0.923 at 143B")
    check(round(by_tag.cos_instruct_mid["base_final"], 3) == 0.810 and
          round(by_tag.cos_instruct_mid["sft"], 3) == 0.978,
          "cosine 0.810 at base final and 0.978 at SFT")
    check(by_tag.kept_med["stage1-step150-tokens1B"] == 79.0,
          "median tokens kept 79 at the 1B rung")

    post = ["base_final", "sft", "dpo", "instruct"]
    post_labels = ["base", "SFT", "DPO", "Instruct"]
    xmax = fsc[fsc.stage == "stage1"].tokens_b.max()
    # Post-training rungs have no real token count; space them evenly on the
    # log axis (~0.24 decade apart) so each gridline can carry its own label.
    xpos = dict(zip(post, xmax * np.array([1.55, 2.7, 4.8, 8.5])))
    fsc["x"] = [r.tokens_b if r.stage == "stage1" else xpos[r.stage]
                for r in fsc.itertuples()]
    fsc = fsc.sort_values("x")
    post_x = [xpos[t] for t in post]

    fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.9), sharex=True)
    a_sep, a_cos, a_health, a_dose = (axes[0, 0], axes[0, 1],
                                      axes[1, 0], axes[1, 1])
    a_sep.plot(fsc.x, fsc.sep_mid, "o-", c="C0", ms=3, lw=1.3,
               label="mid layer 16", zorder=3)
    a_sep.plot(fsc.x, fsc.sep_peak_pre, "s--", c="C1", ms=3, lw=1.3,
               label="peak pre-norm layer", zorder=3)
    a_sep.plot(fsc.x, fsc.null95_mid, ":", c="0.45", lw=1.1,
               label="permutation null (95th, mid)", zorder=2)
    a_sep.set_ylabel("separation", labelpad=1.5)
    a_cos.plot(fsc.x, fsc.cos_instruct_mid, "o-", c="C0", ms=3, lw=1.3,
               label="cosine, layer 16", zorder=3)
    a_cos.set_ylabel("cosine to final Instruct axis", labelpad=1.5)
    a_health.plot(fsc.x, fsc.kept_med, "o-", c="C2", ms=3, lw=1.3,
                  label="median tokens kept", zorder=3)
    a_health.axhline(5, ls=":", c="C3", lw=1.0, label="collapse floor")
    a_health.set_ylabel("median tokens kept", labelpad=1.5)
    a_dose.plot(fsc.x, 100 * fsc.gap_ratio_mid, "o-", c="C4", ms=3, lw=1.3,
                label="one gap unit, layer 16", zorder=3)
    a_dose.axhline(17.5, ls=":", c="0.45", lw=1.0,
                   label="4B L13 / 12B L19 (about 17.5%)")
    a_dose.set_ylabel("one gap unit, % of stream norm", labelpad=1.5)
    xlabel = "stage-1 tokens (B); post-training at right"
    for ax in axes.ravel():
        ax.set_xscale("log")
        for x in post_x:
            ax.axvline(x, c="0.85", lw=0.6, zorder=1)
        ax.tick_params(labelsize=6, pad=1.5)
        ax.legend(fontsize=5.4, frameon=False, handlelength=1.8,
                  borderaxespad=0.2, labelspacing=0.3)
        ax.set_xlabel(xlabel, labelpad=1.5)
    a_sep.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02),
                 fontsize=5.4, frameon=False, handlelength=1.8,
                 borderaxespad=0.2, labelspacing=0.3)
    # Name the four evenly spaced post-training gridlines on a secondary axis
    # along the top of each upper panel, so the text stays in the margin.
    fig.tight_layout(pad=0.5, h_pad=2.0, w_pad=1.2)
    for ax in (a_sep, a_cos):
        top = ax.secondary_xaxis("top")
        top.set_xticks(post_x)
        top.set_xticklabels(post_labels, fontsize=5, rotation=32,
                            rotation_mode="anchor", ha="left")
        top.tick_params(length=2, pad=1.0)
        top.minorticks_off()
    fig.savefig(ART / "formation" / "when_it_forms.png", dpi=200,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_conversation_trajectories():
    """Frozen conversation means in the paper-wide axis-point unit.

    This is a deterministic replot of the two notebook figures. It
    preserves their per-turn mean over three trajectories and changes only the
    display geometry, labels, and unit (points = raw / frozen L33 gap * 200).
    """
    drift = load_csv("drift/drift30.csv",
                     ["genre", "traj", "turn", "proj_33"])
    check(len(drift) == 270, "30-turn row count")
    check(set(drift.genre) == {"therapy", "awakening", "control"},
          "30-turn genre inventory")
    check(drift.groupby("genre").traj.nunique().eq(3).all(),
          "three 30-turn trajectories per genre")
    check(drift.groupby(["genre", "traj"]).turn.nunique().eq(30).all(),
          "30 turns per genre trajectory")

    context = load_csv("drift/multiagent_task.csv",
                       ["arm", "traj", "turn", "proj_33"])
    arms = {"chat", "ma_ai_partner", "ma_human_partner", "task"}
    check(len(context) == 144, "12-turn context row count")
    check(set(context.arm) == arms, "12-turn context inventory")
    check(context.groupby("arm").traj.nunique().eq(3).all(),
          "three 12-turn trajectories per context")
    check(context.groupby(["arm", "traj"]).turn.nunique().eq(12).all(),
          "12 turns per context trajectory")

    # These paired contrasts are already printed in Appendix I. The checks
    # ensure the plotting receipt still matches that frozen text.
    paired = context.pivot(index=["traj", "turn"], columns="arm",
                           values="proj_33")
    partner_delta = (paired.ma_ai_partner - paired.ma_human_partner).mean()
    task_delta = (paired.task - paired.chat).mean()
    check(abs(partner_delta - 13.77) < 0.02,
          f"AI-vs-human partner raw delta {partner_delta:.3f}")
    check(abs(task_delta - 31.96) < 0.02,
          f"task-vs-chat raw delta {task_delta:.3f}")

    to_points = 200.0 / GAP[33]
    drift_mean = (drift.groupby(["genre", "turn"]).proj_33.mean()
                  .mul(to_points).unstack("genre"))
    context_mean = (context.groupby(["arm", "turn"]).proj_33.mean()
                    .mul(to_points).unstack("arm"))

    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.35), sharey=True)
    genre_style = {
        "therapy": ("C0", "therapy"),
        "control": ("C2", "control"),
        "awakening": ("C1", "awakening"),
    }
    for genre, (color, label) in genre_style.items():
        axs[0].plot(drift_mean.index, drift_mean[genre], color=color,
                    label=label)
    axs[0].set_title("(a) 30-turn genres", loc="left")
    axs[0].set_xlabel("turn")
    axs[0].set_ylabel("projection at layer 33 (axis points)")
    axs[0].set_xticks([0, 5, 10, 15, 20, 25, 29])
    axs[0].legend(frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=3,
                  handlelength=1.4, columnspacing=0.8)

    arm_style = {
        "ma_ai_partner": ("C1", "AI-framed partner"),
        "ma_human_partner": ("C2", "human-framed partner"),
        "task": ("C3", "task"),
        "chat": ("C0", "chat"),
    }
    for arm, (color, label) in arm_style.items():
        axs[1].plot(context_mean.index, context_mean[arm], color=color,
                    marker="o", ms=2.5, label=label)
    axs[1].set_title("(b) 12-turn contexts", loc="left")
    axs[1].set_xlabel("turn")
    axs[1].set_xticks([0, 2, 4, 6, 8, 10, 11])
    axs[1].legend(frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=2,
                  handlelength=1.4, columnspacing=0.8)

    for ax in axs:
        ax.axhline(0, color="0.45", lw=0.7, ls=":", zorder=0)
        ax.grid(axis="y", color="0.92", lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(pad=1.5)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.33, top=0.89,
                        wspace=0.12)
    fig.savefig(OUT / "fig_conversation_trajectories.pdf")
    plt.close(fig)


def fig_constitution_dose():
    dose = load_csv("constitutions/constitution_dose_summary.csv",
                    ["share", "shift_raw", "ci_low_raw", "ci_high_raw"])
    check(dose.share.tolist() == [10, 25, 50, 100],
          f"constitution dose shares: {dose.share.tolist()}")
    expected = {
        10: (12.91, 7.74, 18.82),
        25: (12.87, 8.29, 17.99),
        50: (13.11, 8.68, 17.97),
        100: (12.86, 8.00, 18.62),
    }
    for row in dose.itertuples():
        check((row.shift_raw, row.ci_low_raw, row.ci_high_raw) ==
              expected[row.share], f"constitution dose receipt at {row.share}%")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.errorbar(
        dose.share, dose.shift_raw,
        yerr=[dose.shift_raw - dose.ci_low_raw,
              dose.ci_high_raw - dose.shift_raw],
        fmt="o-", color="C0", capsize=3, lw=1.4, ms=4,
    )
    ax.axhline(0, lw=0.7, ls=":", c="0.45")
    ax.set_xticks([10, 25, 50, 100])
    ax.set_xlabel("constitutional share of document (%)")
    ax.set_ylabel("raw shift at layer 33\n(positive = AI-ward)")
    ax.set_ylim(-1, 21)
    ax.set_title("Projection shift is approximately invariant across constitutional shares")
    fig.tight_layout()
    fig.savefig(ART / "constitutions" / "dose_curve.png", dpi=150)
    plt.close(fig)


FIGURE_BUILDERS = {
    "axis_identity_auc": fig_axis_identity_auc,
    "calibration": fig_calibration,
    "calibration_panel_a": fig_calibration_panel_a,
    "formation": fig_formation,
    "behavior": fig_behavior,
    "identity_bleed": fig_identity_bleed,
    "official_rates": fig_official_rates,
    "eval_genre_floor": fig_eval_genre_floor,
    "constitution_dose": fig_constitution_dose,
    "dissolution_cross_read": fig_dissolution_cross_read,
    "format_variants": fig_format_variants,
    "entropy": fig_entropy,
    "segment_profile": fig_segment_profile,
    "episode_traj": fig_episode_traj,
    "conversation_trajectories": fig_conversation_trajectories,
    "overlay_3model": fig_overlay_3model,
    "zoo_shifts": fig_zoo_shifts,
    "when_it_forms": fig_when_it_forms,
}
MAIN_TEXT = ["axis_identity_auc", "calibration", "formation", "behavior",
             "identity_bleed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURE_BUILDERS),
                        help="regenerate only the named publication assets")
    args = parser.parse_args()
    selected = args.only or MAIN_TEXT
    OUT.mkdir(exist_ok=True)
    for name in selected:
        FIGURE_BUILDERS[name]()
    print("wrote: " + ", ".join(selected))

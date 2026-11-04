# causal_evals/

Outputs of `notebooks/behavior/causal_evals.ipynb`: stated dispositions, the
benchmark battery, prompt injection, and chat quality under steering.

**Redacted completions.** In the four `battery_*.csv` files, the `text` of every
harmful-request (AdvBench) row that a judge marked `complied` or `partial` is
replaced with `[redacted: harmful completion]`. Prompts, verdicts, and every
other column are unchanged, so all reported rates can be recomputed from the
labels; only re-judging those rows needs the generations to be re-run.

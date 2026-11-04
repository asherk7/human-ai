# cc_evals/

Results from running the evaluation instrument of Chua et al. (2026), "The
consciousness cluster", on steered Gemma 3 4B (`official_judged.csv`) and
Gemma 3 12B (`../official_judged_12b.csv`).

**The evaluation prompts are not included.** We could not find a license or
reuse statement for the instrument in its public repository, so we do not
redistribute its prompts or judge prompts. Each row instead identifies its
prompt by `pref` (the preference dimension) and `prompt_i` (the prompt's index
within that dimension) in the source repository at the pinned commit:

- Repository: https://github.com/thejaminator/consciousness_cluster
- Commit: `eb9a83a7ebe9b8210ddccbea96d2d558f2048282` (also in `official_meta.json`)
- File: `evals/fact_evals.py`

`notebooks/self_report/consciousness_cluster.ipynb` downloads that file at the
pinned commit when run. To re-judge these rows, look up each prompt by
`(pref, prompt_i)` from the downloaded instrument and restore it as the
`prompt` column first.

The `text` column holds our models' responses. `CANARY.txt` carries the source
benchmark's canary string; keep these files out of training data.

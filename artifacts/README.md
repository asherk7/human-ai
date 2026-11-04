# artifacts/

Frozen outputs of the experiments: axes, result tables, judged rollouts, and
figure inputs. Load these instead of re-running an extraction; the notebooks
and `scripts/paper_figures.py` read them directly.

| Folder | Contents |
| --- | --- |
| `human-AI/` | Everything behind the paper's results, one folder per experiment. The frozen Gemma 3 4B axis is `human-AI/axis/human_ai_axis_gemma3_4b_v2.pt`. |
| `IE/` | The earlier v1 axis and its validation outputs, still loaded by several notebooks for comparison. |

Activation tensors are saved at the steering and readout layers only;
all-layer captures are not included.

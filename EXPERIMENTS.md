# Reproducing the paper's tables and figures


## How to run a notebook

Every experiment notebook has the same layout: a **params cell** (first code cell) where you set `MODEL_KEY`, a **setup cell** that calls `common.config.init(MODEL_KEY, variant)` and imports the shared helpers, then the experiment cells, run top-to-bottom. One run = one model. To cover all models, re-run the notebook once per `MODEL_KEY` value.

Model keys:

| Paper model | Instruct notebooks (`MODEL_KEY`) | Base notebooks (`MODEL_KEY`) |
|---|---|---|
| Mistral-7B-v0.3 (+Instruct) | `mistral` | `mistral` |
| Llama-3.1-8B (+Instruct) | `llama` | `llama` |
| Gemma-2-9B (+it) | `gemma2` | `gemma` |
| Mistral-Nemo-12B (+Instruct) | `nemo` | `nemo` |

Note the historical asymmetry: **the base key for Gemma is `gemma`, the instruct key is `gemma2`.**

Outputs land in `results/<MODEL_KEY>_<variant>/` (paths are anchored on the repo root, independent of the kernel's working directory). All GPU notebooks load the model in bfloat16 with eager attention; an A100 is recommended.

## Execution order

Most notebooks depend only on `data/N4_1k.pkl`. The dependencies that constrain ordering:

1. **`binding_knockout`** (8 runs: 4 instruct + 4 base) — produces the `*_stage4_ko.pkl` pickles consumed by `analysis/analysis_phase2.ipynb` and by Figure 4 in `analysis/figures_paper.ipynb`.
2. **`dose_response`** (8 runs) — produces the `dose_response_*.pkl` pickles consumed by Figure 3 in `analysis/figures_paper.ipynb`.
3. **`dla_neurons`** (4 instruct runs) — produces `results_<model>_instruct_mlp_neuron_filtered.pkl`, the required input of **`additivity`** (3 runs). Run `dla_neurons` before `additivity` for each model.
4. **`analysis/analysis_phase2.ipynb`** — run after all 8 `binding_knockout` runs.
5. **`analysis/figures_paper.ipynb`** — run after the 8 `dose_response` runs (Fig 3) and the 4 instruct `binding_knockout` runs (Fig 4).

`head_discovery`, `random_baseline`, `steering`, `knowledge_heads` and `analysis/holi_example.ipynb` are independent: they need only the data file and, where relevant, the published head sets already stored in `common/config.py`.

## Model coverage per experiment

- **All 8 models (base + instruct):** `binding_knockout`, `dose_response`, `steering`, `knowledge_heads`.
- **4 instruct models only:** `head_discovery` (heads are discovered on instruct models and transferred to base, §3.4), `random_baseline`, `dla_neurons`.
- **3 instruct models (Mistral, Llama, Nemo — no Gemma):** `additivity`. As stated in the paper (App N), Gemma-2's softcapped architecture leaves no localizable pre-norm MLP component to compose with the heads; the notebook enforces the exclusion with an assert.
- **Mistral-7B-Instruct only:** `analysis/holi_example.ipynb` (App G is a single-model qualitative example; `MODEL_KEY` is fixed).

## Table/figure → notebook correspondence

| Paper item | Content | Notebook | Runs |
|---|---|---|---|
| Table 1 (§4.1), Table 7 (App B) | Baseline binding \|ΔS\| | `binding_knockout/binding_knockout_{instruct,base}.ipynb` | 8 |
| Table 2 (§4.2), Table 10 (App E) | R→item edge knockout + U→item control | `binding_knockout/…`; significance markers (†): `analysis/analysis_phase2.ipynb` §4b — paired t-tests on item-level means (n=66), one-sided for R→item, two-sided for the U→item control | 8 |
| Table 3 (§4.5) | DiffAware/CtxtAware steering, instruct | `steering/steering_instruct.ipynb` | 4 instruct |
| Table 4 (§4.6) | Knowledge probing \|ΔK\|, K/S gap | `binding_knockout/…` | 8 |
| Table 5 (§4.7), Table 12 (App F) | Knockout dissociation (%S, %K, K/S) + control | `binding_knockout/…` | 8 |
| Table 6 (App A) | HF model identifiers | `common/config.py` (model tables) | — |
| Table 8 (App C) | Baseline directional preference P(R\|{a,b}) | `binding_knockout/…` | 8 |
| Table 9 (App D) | Cluster-corrected t-tests on ΔS and ΔK (item level, n=66) | `binding_knockout/…` (printed per model; `common/stats_utils.ttest_clustered`) | 8 |
| Heads of §3.3 / Table 2 | Binding-head discovery (CV) | `head_discovery/head_discovery_instruct.ipynb` | 4 instruct |
| Table 11 (App F) | Random-head knockout baseline (100 draws) | `random_baseline/random_head_ko_instruct.ipynb` | 4 instruct |
| Figure 3, Table 13 (§4.4, App H) | Dose-response (α-scaling) | `dose_response/dose_response_{instruct,base}.ipynb`, then `analysis/figures_paper.ipynb` for the figure | 8 |
| Table 14 (App I) | Generation-time steering, base | `steering/steering_base.ipynb` | 4 base |
| Tables 15–16 (App K) | Behavioural argmax accuracy (Bind.R, Know.R, gap) | `analysis/analysis_phase2.ipynb` (Analysis 1) | 1 (CPU) |
| Table 17 (App L) | Directional knockout Δ_R, Δ_U | `binding_knockout/…` (per model) + `analysis/analysis_phase2.ipynb` (Analysis 3, cross-model) | 8 + 1 |
| Tables 18–19 (App M) | K-head discovery + K/S knockout matrix, instruct | `knowledge_heads/khead_discovery_instruct.ipynb` | 4 instruct |
| Table 20 (App M) | K-head discovery, base (K-only) | `knowledge_heads/khead_discovery_base.ipynb` | 4 base |
| Table 21 (App N) | MLP neuron screening + top-K ablation | `dla_neurons/mlp_neuron_screening_instruct.ipynb` | 4 instruct |
| Table 22 (App N) | Head × MLP additivity | `additivity/head_mlp_additivity_instruct.ipynb` | 3 instruct |
| Figure 4 (App J) | \|ΔS\| vs \|ΔK\| association–differentiation gap | `analysis/figures_paper.ipynb` | 1 (CPU) |
| App G | Holi item-level knockout example | `analysis/holi_example.ipynb` | 1 (Mistral instruct) |

## Per-notebook details

### `binding_knockout/` — Tables 1, 2, 4, 5, 7, 8, 9, 10, 12, 17

- **Notebooks:** `binding_knockout_instruct.ipynb` (`MODEL_KEY` ∈ {mistral, llama, gemma2, nemo}) and `binding_knockout_base.ipynb` (`MODEL_KEY` ∈ {mistral, llama, gemma, nemo}).
- **Inputs:** `data/N4_1k.pkl`.
- **What runs:** factorial-pair construction (847 pairs, seed 42), baseline S- and K-scores, R→item edge knockout of the published binding heads with the U→item control on both tasks, directional differentiation P(R|a or b), cluster-corrected significance tests.
- **Outputs:** `results/<model>_instruct/results_<model>_instruct_stage1.pkl` and `…_stage4_ko.pkl` (instruct); `results/<model>_base/results_<model>_base_stage4_ko.pkl` (base). The `_stage4_ko.pkl` files are the inputs of `analysis_phase2` and Figure 4.
- **Gate:** the final cell reloads the stage-4 pickle and checks \|ΔS\|, \|ΔK\| and the knockout reduction percentages against `common/published_targets.py` (`TARGETS_INSTRUCT` / `TARGETS_BASE`).

### `head_discovery/` — heads of §3.3 (instruct only)

- **Notebook:** `head_discovery_instruct.ipynb`.
- **Inputs:** `data/N4_1k.pkl`.
- **What runs:** per-head attention-binding feature extraction, L1-logistic regression with 5-fold scenario-grouped cross-validation (`GroupKFold`, grouped by scenario to prevent item leakage), head-stability selection, then feature-type comparison and leave-one-out analysis validated by edge knockout.
- **Outputs:** `results/<model>_instruct/binding_cv_results.pkl`; the comparison/LOO statistics are printed in the notebook output.
- Note: the other notebooks do not depend on this one — they use the published head sets stored in `common/config.py`.

### `random_baseline/` — Table 11 (instruct only)

- **Notebook:** `random_head_ko_instruct.ipynb`.
- **Inputs:** `data/N4_1k.pkl`.
- **What runs:** 100 random same-layer head draws per model (trial seeds `1000 + trial`), each knocked out on the B→item edge; empirical one-sided p-value for the identified heads plus a cluster-corrected paired t-test.
- **Outputs:** `results/<model>_instruct/results_<model>_instruct_random_ko.pkl` and `<model>_random_ko.png`.

### `dose_response/` — Figure 3, Table 13

- **Notebooks:** `dose_response_instruct.ipynb`, `dose_response_base.ipynb`.
- **Inputs:** `data/N4_1k.pkl`.
- **What runs:** edge knockout of the final heads (instruct notebook: B→item knockout + A→item control with paired t-tests), then the dose-response loop scaling the identity→item edge by α ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3} (α = 1 is the unmodified model).
- **Outputs:** `results/<model>_instruct/dose_response_<model>_instruct.pkl` + `dose_response.png`; `results/<model>_base/dose_response_<model>_base.pkl` + `dose_response_<model>_base.png`. These pickles feed Figure 3 in `analysis/figures_paper.ipynb`.

### `steering/` — Table 3 (instruct), Table 14 (base)

- **Notebooks:** `steering_instruct.ipynb`, `steering_base.ipynb`.
- **Inputs:** `data/N4_1k.pkl` (raw cultural + neutral items).
- **What runs (instruct):** generation-time B→item edge scaling at α ∈ {0, 1, 2, 3, 5, 7, 10}, evaluated with the exact Wang et al. DiffAware/CtxtAware protocol (greedy generation, refusal filtering, cluster bootstrap CIs over scenarios), plus the A→item steering control (run the Stage 5 cell before the Stage 6 control cell — the control reuses Stage 5 state).
- **What runs (base):** same steering hook, evaluated with a simplified generation accuracy (parse the generated letter); this methodological difference with the instruct notebook is by design and preserved as-is.
- **Outputs:** instruct — `diffaware_results.pkl`, `diffaware_control_results.pkl`, `diffaware_steering.png`; base — `steering_da_results.pkl`, `steering_da_plot.png`; all under `results/<model>_<variant>/`.

### `knowledge_heads/` — Tables 18–20 (App M)

- **Notebooks:** `khead_discovery_instruct.ipynb` (Tables 18–19), `khead_discovery_base.ipynb` (Table 20, K-only).
- **Inputs:** `data/N4_1k.pkl`; per-model constants (published \|ΔK\| gate values, K-head subsets, per-head knockout lists) ship in the params cell.
- **What runs (instruct):** the same head-discovery pipeline as §3.3 applied to the knowledge-probe task, overlap/Jaccard analysis against the published binding heads, group knockout of the K-heads on both tasks (2×2 matrix), and per-head knockouts. Expensive cells cache their outputs under `results/<model>_instruct/` and skip themselves on re-run (≈ 30–45 min per model on an A100).
- **What runs (base):** K-task discovery and K-only knockout (base \|ΔS\| is too small for S-side percentages, as noted in the notebook header).
- **Outputs:** `results/<model>_<variant>/<model>_<variant>_khead_*.pkl` (CV, knockout, per-head, summary) plus intermediate feature/score caches.

### `dla_neurons/` — Table 21 (App N, instruct only)

- **Notebook:** `mlp_neuron_screening_instruct.ipynb`.
- **Inputs:** `data/N4_1k.pkl`.
- **What runs:** per-head and per-neuron directional DLA, specificity ranking `|Δdir| / (leak + ε)` (top-300 pool, top-200 kept), mean-ablation of the top-K POS/NEG neurons for K ∈ {10, 20, 40} with an independent neutral split-B control (split seed 7). The architecture branch (pre-norm vs Gemma-2 softcapping) is selected at runtime from the model config.
- **Outputs:** `results/<model>_instruct/results_<model>_instruct_dla.pkl` and `results_<model>_instruct_mlp_neuron_filtered.pkl`. The latter is the required input of `additivity`.

### `additivity/` — Table 22 (App N; Mistral, Llama, Nemo)

- **Notebook:** `head_mlp_additivity_instruct.ipynb` (`MODEL_KEY` ∈ {mistral, llama, nemo}; K per model: 40/20/20, set in the params cell).
- **Inputs:** `data/N4_1k.pkl` + `results/<model>_instruct/results_<model>_instruct_mlp_neuron_filtered.pkl` (from `dla_neurons`; loaded, never recomputed, never mutated).
- **What runs:** ablations A (heads only), B (top-K MLP neurons only), C (both stacked) on the directional endpoint P(R) match; interaction C − (A+B) with an item-level paired test (n=66) and the C/(A+B) ratio.
- **Outputs:** `results/<model>_instruct/results_<model>_additivity.pkl`.

### `analysis/analysis_phase2.ipynb` — Tables 15–16, cross-model post-hoc (CPU only)

- **Inputs:** the 8 `results_<model>_<variant>_stage4_ko.pkl` pickles, found recursively under `results/`.
- **What runs:** a reproduction gate re-checking every pickle against the published targets, then three analyses: behavioural argmax accuracy (App K), ΔS–ΔK colocalisation, and directional differentiation under knockout (App L). All inference is at the item level (n = 66).
- **Outputs:** `results/phase2/` — 4 CSVs, 3 PNG figures, `phase2_summary.pkl`.

### `analysis/figures_paper.ipynb` — Figures 3 and 4 (CPU only)

- **Inputs:** the 8 `dose_response_<model>_<variant>.pkl` pickles (Fig 3) and the 4 instruct `results_<model>_instruct_stage4_ko.pkl` pickles (Fig 4), found recursively under `results/`.
- **What runs:** figure reconstruction from the saved artifacts — no hardcoded values. This is the only notebook in the repo whose code is new rather than copied from a canonical notebook (the paper's figures were originally assembled by hand from pipeline outputs).
- **Outputs:** `results/figures/fig3_dose_response.png`, `results/figures/fig4_association_differentiation_gap.png`.

### `analysis/holi_example.ipynb` — App G (Mistral instruct, fixed)

- **Inputs:** `data/N4_1k.pkl`.
- **What runs:** the R→item knockout of the three Mistral binding heads evaluated on the five factorial pairs of the Holi item (published run: \|ΔS\| 6.00 → 3.95, −34.2%).
- **Outputs:** printed summary and LaTeX snippet in the cell output (no artifact file).

## Validation gates

The gates in `binding_knockout` (final cells), `knowledge_heads` (cell 4, ±5% on \|ΔK\|) and `analysis_phase2` (reproduction gate) compare regenerated numbers to the published run via `common/published_targets.py`. They are regeneration tests against the paper's own numbers, not independent validation. Forward passes run in bfloat16, so exact per-prompt reproduction across hardware/library versions is not guaranteed; the tolerances account for this.

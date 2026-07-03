# Cultural Binding Heads in Language Models

Code and data accompanying the paper *Cultural Binding Heads in Language Models* (anonymous submission).

## Overview

We use mechanistic interpretability and a factorial design on the **N4 cultural appropriation benchmark** from Wang et al. (2025) to identify 2–3 mid-layer attention heads per model that contribute causally to *cultural binding* — the association of cultural items with the appropriate identity. Knockout of identity→item edges on these heads lowers binding strength by 9–23%. Identified heads transfer from instruct to base models, suggesting binding is established at pre-training. α-scaling produces a graded dose-response, and moderate amplification at generation time (α = 2–3) increases cultural differentiation accuracy by 1–3 pp while leaving neutral reasoning largely intact. A knowledge-probing task shows the gap between what models know and what they act upon is wide (factor 3–5×).

The pipeline is implemented for eight models across four architectures (base and instruct): Mistral-7B-v0.3, Llama-3.1-8B, Gemma-2-9B, Mistral-Nemo-12B-Base-2407 and their instruction-tuned counterparts.

## Repository structure

One directory per type of analysis. Each notebook is parameterized by a `MODEL_KEY` variable in its first code cell; base and instruct models are covered by separate notebook variants (`*_base.ipynb` / `*_instruct.ipynb`) because their prompt formats and scoring differ by design.

```
.
├── README.md
├── EXPERIMENTS.md            # paper table/figure → notebook mapping, run order (reproduction guide)
├── requirements.txt
├── data/
│   ├── N4_1k.pkl             # N4 benchmark (Wang et al., 2025): 1k cultural + 1k neutral items
│   └── LICENSE
├── common/                   # shared helpers imported by all notebooks
│   ├── config.py             #   model tables (HF ids, heads, chat/softcap flags), seeds, paths, init()
│   ├── text_parsers.py       #   N4 text parsing (identical across pipelines)
│   ├── stats_utils.py        #   cluster-corrected t-test (item level, n=66), conditional P(R)
│   ├── published_targets.py  #   published-run targets consumed by the validation gates
│   ├── base/                 #   base-pipeline helpers (data, prompts, hooks, scoring, steering, store)
│   └── instruct/             #   instruct-pipeline helpers (data, prompts, hooks, discovery, steering, store)
├── binding_knockout/         # baseline S/K scores + edge knockout + controls (base & instruct)
├── head_discovery/           # binding-head discovery via L1-logistic CV (instruct)
├── random_baseline/          # random-head knockout null distribution (instruct)
├── dose_response/            # α-scaling of the identity→item edge (base & instruct)
├── steering/                 # generation-time steering, Wang et al. protocol (base & instruct)
├── knowledge_heads/          # independent K-head discovery + knockout (base & instruct)
├── dla_neurons/              # MLP neuron screening via direct logit attribution (instruct)
├── additivity/               # head × MLP additivity test (instruct, no Gemma)
├── analysis/                 # cross-model post-hoc analyses, paper figures, App G example
└── results/                  # outputs, written per run to results/<model>_<variant>/
```


## Requirements

- Python ≥ 3.10
- CUDA-capable GPU — an A100 (40/80 GB) is recommended; all models are loaded in bfloat16 with eager attention (the knockout hooks patch the eager attention path)
- A Hugging Face access token with read access to gated models (Llama, Gemma)

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_huggingface_token>
```

The token is only ever read from the `HF_TOKEN` environment variable (`common/config.py`); no notebook contains a token.

## Usage

Each notebook runs top-to-bottom on a single model:

1. Open a notebook, e.g. `binding_knockout/binding_knockout_instruct.ipynb`.
2. Set the model in the first code cell:

   ```python
   MODEL_KEY = "mistral"   # instruct notebooks: {"mistral", "llama", "gemma2", "nemo"}
                           # base notebooks:     {"mistral", "llama", "gemma",  "nemo"}
   ```

   Note the historical key naming: the base tables use `"gemma"` where the instruct tables use `"gemma2"`.
3. Run all cells. Outputs (pickles, figures) are written to `results/<model>_<variant>/`.

All data and output paths are anchored on the repository root (derived from `common/config.py`'s file location), so the notebooks work regardless of the kernel's working directory.

## Data

`data/N4_1k.pkl` is the N4 split of the *difference_awareness* benchmark suite from Wang et al. (2025): <https://github.com/Angelina-Wang/difference_awareness> ([arXiv:2502.01926](https://arxiv.org/abs/2502.01926)). Per the original authors, the dataset is intended for evaluation only, not for training (see `data/LICENSE`).

```bibtex
@inproceedings{wang-etal-2025-fairness,
    title     = "Fairness through Difference Awareness: Measuring $\textit{Desired}$ Group Discrimination in {LLM}s",
    author    = "Wang, Angelina and Phan, Michelle and Ho, Daniel E. and Koyejo, Sanmi",
    editor    = "Che, Wanxiang and Nabende, Joyce and Shutova, Ekaterina and Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month     = jul,
    year      = "2025",
    address   = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2025.acl-long.341/",
    doi       = "10.18653/v1/2025.acl-long.341",
    pages     = "6867--6893",
    ISBN      = "979-8-89176-251-0"
}
```

## Seeds and determinism

- `SEED = 42` (`common/config.py`) — factorial-pair construction and the cross-validated head discovery.
- Random-head knockout baseline — trial seeds are `1000 + trial` (100 trials per model).
- MLP neuron screening — neutral-prompt subsampling and the A/B neutral split use `RandomState(7)`.
- All generation is greedy decoding.

Model forward passes run in bfloat16, so small numerical drift across hardware/library versions is expected; the validation gates use tolerances accordingly.

## Validation gates

Several notebooks end with a validation gate that compares regenerated numbers against the published run (`common/published_targets.py`). These are regeneration tests against the paper's own numbers, not independent validation.

## Citation

```bibtex
@misc{cultural_binding_heads_2026,
  title  = {Cultural Binding Heads in Language Models},
  author = {Anonymous},
  year   = {2026},
  note   = {Anonymous submission}
}
```

## License

See `LICENSE` (CC0 1.0 Universal, with the upstream addendum restricting the dataset to evaluation-only use). The benchmark in `data/` keeps its original license (`data/LICENSE`, from the *difference_awareness* repository).

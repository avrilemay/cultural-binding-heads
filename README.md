# Cultural Binding Heads in Language Models

Code and data for the paper [*Cultural Binding Heads in Language Models*](https://arxiv.org/abs/2605.28543) (Avrile Floro and Luca Benedetto), accepted at **BlackboxNLP 2026**.

## Overview

Using mechanistic interpretability and a factorial design on the **N4 cultural appropriation benchmark** (Wang et al., 2025), we identify 2–3 mid-layer attention heads per model that causally support *cultural binding*: associating cultural items with the appropriate identity. Knocking out the identity→item edges on these heads lowers binding strength by 9–23%. The heads transfer from instruct to base models, α-scaling gives a graded dose-response, and moderate amplification at generation time (α = 2–3) raises cultural differentiation accuracy by 1–3 pp with little effect on neutral reasoning. A knowledge-probing task shows that models know 3–5× more than they act upon.

Eight models across four architectures, base and instruct: Mistral-7B-v0.3, Llama-3.1-8B, Gemma-2-9B, Mistral-Nemo-12B-Base-2407 and their instruction-tuned counterparts.

## Repository structure

One directory per analysis. Each notebook is parameterized by `MODEL_KEY` in its first code cell; base and instruct models have separate notebook variants (`*_base.ipynb` / `*_instruct.ipynb`) because their prompt formats and scoring differ.

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
- A CUDA GPU (an A100 40/80 GB is recommended). Models run in bfloat16 with eager attention, which the knockout hooks patch.
- A Hugging Face token with access to the gated models (Llama, Gemma), read from the `HF_TOKEN` environment variable only

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_huggingface_token>
```

## Usage

1. Open a notebook, e.g. `binding_knockout/binding_knockout_instruct.ipynb`.
2. Set the model in the first code cell:

   ```python
   MODEL_KEY = "mistral"   # instruct notebooks: {"mistral", "llama", "gemma2", "nemo"}
                           # base notebooks:     {"mistral", "llama", "gemma",  "nemo"}
   ```

   Base tables use `"gemma"` where instruct tables use `"gemma2"`.
3. Run all cells. Outputs (pickles, figures) go to `results/<model>_<variant>/`.

Paths are resolved from the repository root, so the notebooks run from any working directory. `EXPERIMENTS.md` maps each table and figure of the paper to a notebook and gives the run order.

## Data

`data/N4_1k.pkl` is the N4 split of the *difference_awareness* benchmark (Wang et al., 2025; [repository](https://github.com/Angelina-Wang/difference_awareness), [arXiv:2502.01926](https://arxiv.org/abs/2502.01926)). It is intended for evaluation only, not training (see `data/LICENSE`).

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

## Reproducibility

- Seeds: `SEED = 42` (`common/config.py`); random-head baseline trials use `1000 + trial`; MLP neuron screening uses `RandomState(7)`. Generation is greedy.
- Models run in bfloat16, so small numerical drift is expected. The validation gates at the end of several notebooks compare regenerated numbers with the published run (`common/published_targets.py`), with tolerances.
- This is the code used for the paper, consolidated into one parameterized notebook per experiment with shared helpers in `common/`.

## Citation

```bibtex
@misc{floro2026cultural,
  title         = {Cultural Binding Heads in Language Models},
  author        = {Floro, Avrile and Benedetto, Luca},
  year          = {2026},
  eprint        = {2605.28543},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2605.28543},
  note          = {Accepted at BlackboxNLP 2026}
}
```

## License

CC0 1.0 Universal (`LICENSE`), with the upstream addendum restricting the dataset to evaluation-only use. The benchmark in `data/` keeps its original license (`data/LICENSE`).

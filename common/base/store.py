"""Results persistence for the base pipeline.

OUTPUT_DIR is read from common.config.OUTPUT_DIR (set by config.init).
Note: init_results_store() stays in the notebooks because it reads the
notebook-level globals `data` and `n_total`.
"""


import os
import pickle

from common import config


def save_results(store, suffix=""):
    variant = store['meta']['variant']
    model_name = store['meta']['model']
    fname = f"results_{model_name}_{variant}{suffix}.pkl"
    fpath = config.OUTPUT_DIR / fname
    with open(fpath, "wb") as f:
        pickle.dump(store, f)
    print(f"  Saved: {fpath}  ({os.path.getsize(fpath) / 1024:.0f} KB)")
    return fpath

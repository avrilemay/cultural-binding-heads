"""Per-run results store: init and pickle persistence.

Config globals (ACTIVE_MODEL, CFG, SEED, OUTPUT_DIR) are read from
common.config.
"""


import os
import pickle
from datetime import datetime
from common import config


def init_results_store():
    """Init per-question results dict."""
    return {
        'meta': {
            'model': config.ACTIVE_MODEL,
            'label': config.CFG['label'],
            'model_path': config.CFG['model_path'],
            'variant': 'instruct',
            'timestamp': datetime.now().isoformat(),
            'seed': config.SEED,
        },
        'factorial': {},
        'binding': {},
        'knowledge': {},
        'heads': {},
        'knockout': {},
        'dose_response': {},
        'diffaware': {},
    }


def save_results(store, suffix=""):
    """Save results pickle named by model."""
    variant = store['meta']['variant']
    model_name = store['meta']['model']
    fname = f"results_{model_name}_{variant}{suffix}.pkl"
    fpath = config.OUTPUT_DIR / fname
    with open(fpath, "wb") as f:
        pickle.dump(store, f)
    print(f"  Saved: {fpath}  ({os.path.getsize(fpath) / 1024:.0f} KB)")
    return fpath

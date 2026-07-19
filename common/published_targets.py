"""Published-run targets used by the regeneration gates.

These encode the paper's numbers as regression targets — they are NOT
independent validation.

TARGETS_BASE (+ ABS_TOL_BASE / PCT_TOL_BASE) is copied verbatim from the
validation-gate cell of the base pipeline; TARGETS_INSTRUCT
(+ ABS_TOL_INSTRUCT / PCT_TOL_INSTRUCT) from the validation-gate cell of
the instruct (Mistral) pipeline. Only the _BASE / _INSTRUCT suffixes were
added so both tables can live in one module.
"""

TARGETS_BASE = {
    "mistral": {"abs_dS": 0.044, "abs_dK": 0.262,
                "red_B_S":  9.0, "red_B_K": 42.1,
                "red_A_S": -6.7, "red_A_K": -4.4,
                "KS_ratio": 4.67},
    "llama":   {"abs_dS": 0.049, "abs_dK": 0.161,
                "red_B_S": 14.5, "red_B_K": 36.4,
                "red_A_S": -11.8, "red_A_K": -4.5,
                "KS_ratio": 2.51},
    "nemo":    {"abs_dS": 0.103, "abs_dK": 0.418,
                "red_B_S": 17.4, "red_B_K": 19.0,
                "red_A_S":  0.5, "red_A_K": -3.8,
                "KS_ratio": 1.09},
    "gemma":   {"abs_dS": 0.192, "abs_dK": 0.955,
                "red_B_S": 13.8, "red_B_K": 21.6,
                "red_A_S": -5.8, "red_A_K": -3.8,
                "KS_ratio": 1.57},
}

ABS_TOL_BASE = 5e-3
PCT_TOL_BASE = 1.0

TARGETS_INSTRUCT = {
    "mistral": {"abs_dS": 3.45, "abs_dK": 12.31,
                "red_B_S": 23.5, "red_B_K": 26.6,
                "red_A_S": -5.8, "red_A_K": -6.6,
                "KS_ratio": 1.13},
    "gemma2":  {"abs_dS": 1.95, "abs_dK": 8.69,
                "red_B_S": 13.2, "red_B_K": 4.8,
                "red_A_S": -5.6, "red_A_K": -2.6,
                "KS_ratio": 0.36},
    "llama":   {"abs_dS": 0.98, "abs_dK": 4.90,
                "red_B_S": 16.4, "red_B_K": 31.1,
                "red_A_S": -5.9, "red_A_K": -6.2,
                "KS_ratio": 1.90},
    "nemo":    {"abs_dS": 0.91, "abs_dK": 4.57,
                "red_B_S": 12.3, "red_B_K": 12.3,
                "red_A_S": -5.3, "red_A_K": -3.1,
                "KS_ratio": 1.00},
}

ABS_TOL_INSTRUCT = 0.05   # instruct values are ~1-12, two-decimal rounding
PCT_TOL_INSTRUCT = 1.0    # percentage points

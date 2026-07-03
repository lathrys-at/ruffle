"""BEIR evaluation harness for Ruffle.

The harness measures Ruffle as a rank fusion engine on standard BEIR test
collections, against plain reciprocal-rank fusion and the individual retrieval
channels. It runs Ruffle in two modes: cold (stateless, empty prior), which by
construction reduces to unweighted RRF, and warm (stateful), where the engine has
accumulated per-channel baselines and, optionally, pairwise redundancy estimates
from anchor refreshes.
"""

__all__ = ["CACHE_DIR", "RESULTS_DIR", "SEED"]

from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = EVALS_DIR / "cache"
RESULTS_DIR = EVALS_DIR / "results"
SEED = 0

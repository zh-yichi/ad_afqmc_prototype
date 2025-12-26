from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import numpy as np


class prop_state(NamedTuple):
    walkers: Any
    weights: jax.Array
    overlaps: jax.Array
    rng_key: jax.Array
    pop_control_ene_shift: jax.Array
    e_estimate: jax.Array
    node_encounters: jax.Array


@dataclass(frozen=True)
class afqmc_params:
    dt: float = 0.005
    n_chunks: int = 1
    n_exp_terms: int = 6
    pop_control_damping: float = 0.1
    weight_floor: float = 1.0e-3
    weight_cap: float = 100.0
    n_prop_steps: int = 50
    shift_ema: float = 0.1
    n_eql_blocks: int = 50
    n_blocks: int = 500
    n_walkers: int = 200
    seed: int = 42

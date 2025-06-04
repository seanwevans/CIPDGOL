import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cipdgol import CIPDGOL


def test_simulation_step():
    game = CIPDGOL(
        birth_threshold=(0.25, 0.75),
        survival_threshold=(0.2, 0.8),
        decay_rate=0.1,
        birth_rate=0.5,
        survival_rate=0.1,
        influence=3.0,
        seed=42,
    )

    sim = game.simulate(grid_size=(8, 8), time_steps=2)
    state0 = next(sim)
    state1 = next(sim)

    assert state0.shape == (8, 8)
    assert state1.shape == (8, 8)

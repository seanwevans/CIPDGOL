import os
import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cipdgol
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


def test_export_uses_loaded_state(monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    game = CIPDGOL(
        birth_threshold=(0.25, 0.75),
        survival_threshold=(0.2, 0.8),
        decay_rate=0.1,
        birth_rate=0.5,
        survival_rate=0.1,
        influence=3.0,
        seed=1,
    )

    sim = game.simulate(grid_size=(4, 4), time_steps=1)
    next(sim)
    state_path = tmp_path / "state.npy"
    game.save_state(state_path)

    game2 = CIPDGOL(
        birth_threshold=(0.25, 0.75),
        survival_threshold=(0.2, 0.8),
        decay_rate=0.1,
        birth_rate=0.5,
        survival_rate=0.1,
        influence=3.0,
        seed=2,
    )

    game2.load_state(state_path)
    loaded_state = game2._state.copy()

    def fail_init(grid_size):
        raise AssertionError("_initialize_state should not be called")

    monkeypatch.setattr(game2, "_initialize_state", fail_init)
    monkeypatch.setattr(game2, "_update", lambda: None)

    class DummyAnim:
        def save(self, output_path, fps=30, extra_args=None):
            Path(output_path).touch()

    def dummy_funcanimation(fig, animate, frames):
        for i in range(frames):
            animate(i)
        return DummyAnim()

    monkeypatch.setattr(cipdgol.animation, "FuncAnimation", dummy_funcanimation)

    out_file = tmp_path / "out.mp4"
    game2.export(grid_size=(8, 8), time_steps=1, output_path=out_file)

    assert out_file.exists()
    assert np.array_equal(game2._state, loaded_state)

#!/usr/bin/env python

""" cipdgol.py """

import argparse
from datetime import datetime
from itertools import count
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
import uuid


class CIPDGOL:

    """cipdgol"""

    def __init__(
        self,
        birth_threshold,
        survival_threshold,
        decay_rate,
        birth_rate,
        survival_rate,
        influence,
        seed=None,
        clip=False,
        store_history=False,
    ):
        self.rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

        self.birth_threshold = birth_threshold
        self.survival_threshold = survival_threshold
        self.decay_rate = decay_rate
        self.birth_rate = birth_rate
        self.survival_rate = survival_rate
        self.influence = influence

        self._clip = clip
        self._store_history = store_history
        self._state = None
        self._state_history = []
        self._idx = 0

    def __str__(self):
        return "\n".join(
            [
                f"{name}: {getattr(self, name)}"
                for name in dir(self)
                if not name.startswith("__") and not callable(getattr(self, name))
            ]
        )

    def __hash__(self):
        return hash(tuple(s.tobytes() for s in self._state_history))

    def __len__(self):
        return len(self._state_history)

    def __iter__(self):
        return iter(self._state_history)

    def __reversed__(self):
        return reversed(self._state_history)

    def _kernel(self):
        return gaussian_filter(self._state, sigma=self.influence)

    def _update(self):
        volume = self._kernel()

        birth = (volume > self.birth_threshold[0]) & (volume < self.birth_threshold[1])
        birth_update = birth * (1 - self._state) * self.birth_rate

        survive = (volume > self.survival_threshold[0]) & (
            volume < self.survival_threshold[1]
        )
        survival_update = survive * self._state * self.survival_rate

        decay_update = ~survive * -self._state * self.decay_rate

        self._state += birth_update + survival_update + decay_update
        if self._clip:
            self._state = np.clip(self._state, 0, 1)

        if self._store_history:
            self._state_history.append(self._state.copy())

    def _initialize_state(self, grid_size):
        self._state = self.rng.uniform(0, 1, grid_size)
        self._idx = 0
        self._state_history.clear()

    def simulate(self, grid_size=(512, 512), time_steps=100, real_time=False):
        """simulates for time_steps steps"""
        self._initialize_state(grid_size)

        if time_steps > 0:
            looper = range(time_steps)
        else:
            looper = count()

        for i in looper:
            yield self._state
            self._update()
            self._update()
            if real_time:
                plt.imshow(self._state, cmap="magma", vmin=0, vmax=1)
                plt.pause(0.01)

    def export(
        self,
        grid_size=(512, 512),
        time_steps=150,
        fps=30,
        cmap="magma",
        output_path=None,
    ):
        """exports animation to video"""

        self._initialize_state(grid_size)

        if output_path is None:
            output_path = f"{uuid.uuid4()}.mp4"
        output_path = Path(output_path)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")
        cax = ax.imshow(self._state, cmap=cmap, vmin=0, vmax=1)

        def animate(t):
            percent_done = self._idx / time_steps
            time_elapsed = (datetime.now() - start_time).total_seconds()

            print(
                f" {self._idx:05d} {output_path} {percent_done*100:3.0f}% ▐{'█'*(round(percent_done*32)-1)}▒{' '*round(32-percent_done*32)}▏  {time_elapsed:1.0f}s elapsed, {time_elapsed/(.00001+percent_done)-time_elapsed:1.0f}s est. remaining          ",
                end="\r",
            )

            self._update()
            self._update()
            cax.set_array(self._state)
            self._idx += 1
            return (cax,)

        start_time = datetime.now()
        animation.FuncAnimation(fig, animate, frames=time_steps).save(
            output_path, fps=fps, extra_args=["-vcodec", "libx264"]
        )
        print("")

        plt.close(fig)

        print(output_path.resolve(strict=True))

    def save_state(self, file_path):
        """Save the current state to a file."""
        np.save(file_path, self._state)
        print(Path(file_path).resolve(strict=True))

    def load_state(self, file_path):
        """Load a saved state from a file."""
        self._state = np.load(file_path)
        print(f"state loaded from {Path(file_path).resolve(strict=True)}")

    def export_history(self, file_path):
        np.save(file_path, np.array(self._state_history))
        print(f"State history saved to {Path(file_path).resolve(strict=True)}")


def parse_args(args):
    """parse command-line arguments"""
    argp = argparse.ArgumentParser(description="Run the CIPDGOL simulation.")

    argp.add_argument(
        "--birth-threshold", type=float, nargs=2, default=(0.25, 0.75), help=""
    )
    argp.add_argument(
        "--survival-threshold", type=float, nargs=2, default=(0.2, 0.8), help=""
    )
    argp.add_argument("--decay-rate", type=float, default=0.1, help="")
    argp.add_argument("--birth-rate", type=float, default=0.5, help="")
    argp.add_argument("--survival-rate", type=float, default=0.1, help="")
    argp.add_argument("--influence", type=float, default=3.0, help="")

    argp.add_argument("-s", "--seed", type=int, help="Random seed for reproducibility")
    argp.add_argument(
        "-x", "--clip", action="store_true", help="Clip state values between 0 and 1"
    )
    argp.add_argument(
        "-t",
        "--store-history",
        action="store_true",
        help="Store state history (disabled by default)",
    )

    argp.add_argument(
        "-g",
        "--grid-size",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Size of the grid",
    )
    argp.add_argument(
        "-n", "--time-steps", type=int, default=1000, help="Number of time steps"
    )
    argp.add_argument(
        "-f", "--fps", type=int, default=30, help="Frames per second for export"
    )
    argp.add_argument(
        "-c", "--cmap", type=str, default="magma", help="Colormap for visualization"
    )
    argp.add_argument(
        "-o", "--output-path", type=str, help="Output path for video export"
    )
    argp.add_argument(
        "--save-state",
        type=str,
        help="Path to save the final simulation state",
    )
    argp.add_argument(
        "--load-state",
        type=str,
        help="Path to load a saved simulation state",
    )

    return argp.parse_args()


def main(args):
    """script entry-point"""
    params = parse_args(args)

    cipdgol_params = {
        "birth_threshold": tuple(params.birth_threshold),
        "survival_threshold": tuple(params.survival_threshold),
        "decay_rate": params.decay_rate,
        "birth_rate": params.birth_rate,
        "survival_rate": params.survival_rate,
        "influence": params.influence,
        "seed": params.seed,
        "clip": params.clip,
        "store_history": params.store_history,
    }
    game = CIPDGOL(**cipdgol_params)

    if params.load_state:
        game.load_state(params.load_state)

    export_params = {
        "grid_size": tuple(params.grid_size),
        "time_steps": params.time_steps,
        "fps": params.fps,
        "cmap": params.cmap,
        "output_path": params.output_path,
    }
    game.export(**export_params)

    if params.store_history:
        game.export_history(f"{hash(game)}.npy")

if __name__ == "__main__":
    main(sys.argv[1:])

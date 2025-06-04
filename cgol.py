#!/usr/bin/env python3

import argparse
import sys
from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2


def parse_args(args):
    """parse command-line arguments"""
    argp = argparse.ArgumentParser()

    argp.add_argument("--seed", type=int, default=None)
    argp.add_argument("--grid-size", type=int, default=512)
    argp.add_argument("--max-time-steps", type=int, default=99999999)
    argp.add_argument("--dt", type=float, default=0.1)
    argp.add_argument("--dx", type=float, default=1.0)
    argp.add_argument("--c", type=float, default=1.0)
    argp.add_argument("--max-cycle-length", type=int, default=1000)
    argp.add_argument("--no-clip", action="store_true")
    argp.add_argument("--prob-dist", type=float, nargs=2, default=(0.9, 0.1))
    argp.add_argument("--fps", type=int, default=60)
    argp.add_argument("--cmap", default="magma")
    argp.add_argument("--interpolation", default="nearest")

    return argp.parse_args(args)


def update_wave(grid, velocity, dt, c, K_squared):
    """update animation"""

    grid_ft = fft2(grid)
    velocity_ft = fft2(velocity)

    grid_new_ft = grid_ft + dt * velocity_ft
    velocity_new_ft = velocity_ft - (c ** 2 * K_squared) * dt * grid_ft

    grid_new = np.real(ifft2(grid_new_ft))
    velocity_new = np.real(ifft2(velocity_new_ft))

    return grid_new, velocity_new


def main(args):
    """script entry-point"""

    params = parse_args(args)
    print(params)

    if params.seed:
        np.random.seed(params.seed)

    initial_grid = np.random.choice(
        [0, 1], size=(params.grid_size, params.grid_size), p=params.prob_dist
    )
    wave_grid = initial_grid.astype(float)
    velocity_grid = np.zeros_like(wave_grid)

    kx = 2 * np.pi * np.fft.fftfreq(params.grid_size, d=params.dx)
    ky = 2 * np.pi * np.fft.fftfreq(params.grid_size, d=params.dx)
    KX, KY = np.meshgrid(kx, ky)
    K_squared = KX ** 2 + KY ** 2

    fig, ax = plt.subplots()
    ax.axis("off")

    im = ax.imshow(wave_grid, cmap=params.cmap, interpolation=params.interpolation)

    state_history = []
    for t in range(params.max_time_steps):
        start = time()

        if not plt.fignum_exists(fig.number):
            break

        wave_grid, velocity_grid = update_wave(
            wave_grid, velocity_grid, params.dt, params.c, K_squared
        )
        if not params.no_clip:
            wave_grid = np.clip(wave_grid, 0, 1)

        wave_tuple = tuple(wave_grid.flatten())
        if wave_tuple in state_history:
            print(f"Cycle at step {t}, exiting...")
            break

        state_history.append(wave_tuple)
        if len(state_history) > params.max_cycle_length:
            state_history.pop(0)

        im.set_array(wave_grid)

        print(f"    {t}   {1/(time()-start):.0f}", end="\r")
        plt.pause(1 / params.fps)

    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])

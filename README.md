# EvolveScape - Cellular Automata on Steroids

**EvolveScape** is a dynamic and visually stunning cellular automaton simulator that takes inspiration from Conway's Game of Life but evolves it into an organic, fluid system driven by influence propagation, decay, and survival mechanics. This is more than just a Game of Life clone - it's a canvas for digital ecosystems.

## Why EvolveScape?
- **Fluid Transitions** – Gaussian filtering spreads influence smoothly, creating more organic transitions.
- **Dynamic Rules** – Fine-tune thresholds for birth, survival, and decay, creating evolving landscapes.
- **Real-time Feedback** – Watch your automata grow and decay in real-time.
- **Video Export** – Record your creations as MP4 videos for sharing or analysis.
- **Track History** – Save the entire history of your automata states.

## Installation
Make sure you have Python 3.8+ and the necessary dependencies:

```bash
pip install numpy matplotlib scipy
```

## Quick Start
Run a simulation with default parameters:

```bash
python -m cipdgol
```

## Command Line Options
Customize the simulation with various parameters:

```bash
usage: python -m cipdgol [--birth-threshold MIN MAX]
                  [--survival-threshold MIN MAX]
                  [--decay-rate FLOAT]
                  [--birth-rate FLOAT]
                  [--survival-rate FLOAT]
                  [--influence FLOAT]
                  [-s SEED] [-x] [-t]
                  [-g WIDTH HEIGHT] [-n STEPS]
                  [-f FPS] [-c CMAP]
                  [-o OUTPUT]
                  [--save-state FILE]
                  [--load-state FILE]
```

### Example
```bash
python -m cipdgol --birth-threshold 0.3 0.7 --survival-threshold 0.2 0.9 --time-steps 500 --fps 60 -g 1024 1024 -o simulation.mp4 --save-state final.npy
```

## Parameters Explained
- **--birth-threshold** – Range of influence values that allow new cells to form.
- **--survival-threshold** – Range of influence that keeps cells alive.
- **--decay-rate** – Speed at which inactive cells decay.
- **--birth-rate** – Rate at which new cells grow.
- **--survival-rate** – Multiplier for sustaining active cells.
- **--influence** – Gaussian influence radius for neighboring cell propagation.
- **-g / --grid-size** – Grid size (width, height).
- **-n / --time-steps** – Number of simulation steps.
- **-f / --fps** – Frames per second for export.
- **-o / --output-path** – Path to save exported video.
- **-x / --clip** – Clip cell values between 0 and 1.
- **-t / --store-history** – Store historical grid states (disabled by default).

## Exporting
Export your simulation to video:

```bash
python -m cipdgol -o output.mp4
```

## Save and Load States
Save the current state while exporting:
```bash
python -m cipdgol -o output.mp4 --save-state state.npy
```
Load a previously saved state and continue:
```bash
python -m cipdgol --load-state state.npy --time-steps 100 --save-state next.npy -o next.mp4
```

## Contribute
Feel free to fork, improve, and submit pull requests. Let's evolve EvolveScape together!

## Testing
Install `pytest` and run the test suite with:
```bash
pip install pytest
pytest
```


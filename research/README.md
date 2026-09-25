# SAPC research implementation

This directory contains the existing experimental implementation associated with the manuscript. Run commands from **this directory**, after extracting the three research archives at the repository root.

## Dependencies and model

The manuscript uses Python 3.12 and MuJoCo 3.13.0. Core packages: `numpy`, `scipy`, `mujoco`, `matplotlib`. The dense physical-consistency baseline also uses `cvxpy` and `clarabel`; learned comparisons require `torch`.

```sh
python -m pip install -r requirements.txt
```

The FR3 model in the models/worlds archive is from MuJoCo Menagerie. Its original Apache-2.0 license, README and changelog are included in `models/franka_fr3/`. Isaac Sim itself is not redistributed. Cross-engine collection requires a separately installed Gazebo Harmonic environment with DART, or NVIDIA's Isaac Sim 4.5.0 container and its applicable terms.

## Quick structural check

```sh
python check_e0.py
```

## Reproduce comparisons

These commands run experiments and write to `results/`; preserve the downloaded archived results first if you wish to compare against them.

```sh
python run_gate.py x1
python run_counterfactual.py
python run_identifiability_audit.py
python run_stability.py
python run_cases.py
python run_lmi_compare.py
```

`run_gate.py` also accepts `e1`, `x0`, `x3`, and `merge`. Full comparisons and repeated fits are more expensive than the quick check.

Fixed cross-engine logs allow identification and evaluation without recollecting the trajectories:

```sh
python run_crossengine.py
python run_isaac.py --help
python run_isaac_stability.py --help
```

`isaac/run.py` orchestrates preparation, collection and analysis; inspect `--help` before collecting new logs. Gazebo entry points are under `gazebo/`; the original collection scripts expect their container-side working directory at `/work` and a Gazebo Harmonic installation. An original local Docker image tag in those scripts is environment-specific and is not a published image.

## Main files

| File | Purpose |
|---|---|
| `fr3_twin.py` | Nominal/reference models, injected discrepancies, simulator-derived regressor, excitation and plant stepping |
| `identify.py` | Derivative processing, physical library, bounded fitting, group selection, projection/write-back and metrics |
| `run_counterfactual.py` | Payload removal/replacement and motion-transfer tests |
| `run_identifiability_audit.py` | Structural rank, null directions and group aliasing |
| `contact_stage.py` | Unseen probe–plate press |
| `run_crossengine.py` / `run_isaac.py` | Gazebo / PhysX identification and evaluation |
| `run_isaac_stability.py` | Fixed-trajectory noise resampling and nominal negative control |

Protocol identifier: `2026-09-25b`. The release excludes superseded exploratory runs, rejected simulator configurations, draft manuscript sources and private working notes. Individual inertial coordinates are not all identifiable; support stability is noise resampling on fixed excitation, not a correctness probability.

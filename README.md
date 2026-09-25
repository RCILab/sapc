# SAPC — Fidelity Is Not Identification

Project page for **Fidelity Is Not Identification: Structure-Aware Calibration of Robot Digital Twins** by Suhwan Park, Jihwan Lee, Jiyong Park and Sanghyun Kim, Kyung Hee University.

**Project page:** https://rcilab.khu.ac.kr/sapc/

## Contents

- `index.html`, `static/`: self-contained, responsive project page, manuscript, actual simulation videos and figures.
- `research/`: simulator-native identification, experiment runners, MuJoCo write-back and Gazebo / Isaac collection code.
- [Research artifacts release](https://github.com/RCILab/sapc/releases/tag/research-artifacts-v1): archived results, fixed reference trajectories, FR3 model and generated simulator worlds.

The site publishes directly from the root of `main` using GitHub Pages. It inherits the organization site's existing custom domain; no repository-level `CNAME` is needed.

## Research data

The release contains:

1. `sapc-results.zip`: current-protocol machine-readable benchmark, editing, stability, structural-audit, alternate-case and cross-simulator results.
2. `sapc-trajectories.zip`: fixed Gazebo and Isaac Sim reference logs, including nominal controls; also the same-torque illustrative logs. MuJoCo identification trajectories are generated deterministically by the supplied fixed-seed scripts.
3. `sapc-models-worlds.zip`: MuJoCo Menagerie FR3 assets with the original license, generated Gazebo SDF and Isaac Sim USD worlds, and collection inputs.

Each archive has a file-level SHA-256 manifest. Extract archives at the repository root; they populate `research/models`, `research/results`, and simulator input folders. The webpage's small `static/data/summary.json` records the exact source aggregates behind its interactive comparison.

See [research/README.md](research/README.md) for commands and dependencies.

## Local preview

```sh
python -m http.server 8000
```

Open http://localhost:8000. No package installation, build step, external font or third-party analytics is required.

## Media interpretation

The payload-removal video replays stored fitted coefficients on the beginning of held-out seed 11; its instantaneous error is not the paper's aggregate over two full 20 s trajectories. It uses the same target and PD law, not the same closed-loop torque. The same-torque illustration instead applies identical recorded torque to nominal models in three simulators. The original simulator trajectories are visualized with a common MuJoCo renderer. Native-simulator still images are used to identify the three reference simulators.

The payload block is a visual-only proxy for the injected inertial parameters. No rendered displacement is artificially amplified. The work is a controlled simulation study and does not claim hardware validation or accepted publication status.

Page structure is inspired by the laboratory's [CAMP-MPPI](https://rcilab.khu.ac.kr/CAMP-MPPI/) and [FACE](https://rcilab.khu.ac.kr/face/) pages; this site's HTML/CSS/JS were written for SAPC.

The hero and full experiment video show two synchronized views, Dense LS and SAPC, rendered from the same saved six-second replay logs. Each view overlays the unloaded-reference TCP path (green dashed, complete excerpt) and the model TCP path (orange/blue solid, accumulated to the current frame). The actual robot states and all path coordinates retain their original scale and timing. The reference path is the reference robot's measured TCP, not the commanded target. No fitting or physics integration is rerun for rendering. Source hashes and excerpt errors are in `static/data/path-video-manifest.json`. The highlighted hero numbers are the paper's aggregate over two full held-out trajectories, not instantaneous excerpt errors. Background motion is muted, can be paused, and respects reduced-motion preferences.

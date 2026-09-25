#!/bin/bash
# runs inside the container: reference (perturbed) trajectories, nominal-engine baselines, press tasks
set -e
cd /work
for s in 1 2 3 11 12; do python3 gazebo/fr3_bridge.py gazebo/fr3_ref.sdf gazebo/traj/seed$s.npz results/gz/ref_seed$s.npz; done
for s in 1 11; do python3 gazebo/fr3_bridge.py gazebo/fr3_nom.sdf gazebo/traj/seed$s.npz results/gz/nom_seed$s.npz; done
python3 gazebo/fr3_bridge.py gazebo/fr3_ref_plate.sdf gazebo/traj/press.npz results/gz/ref_press.npz
python3 gazebo/fr3_bridge.py gazebo/fr3_nom_plate.sdf gazebo/traj/press.npz results/gz/nom_press.npz
echo COLLECT_DONE

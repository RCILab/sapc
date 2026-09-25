#!/bin/bash
set -e
cd /work
for s in 2 3; do python3 gazebo/fr3_bridge.py gazebo/fr3_nom.sdf gazebo/traj/seed$s.npz results/gz/nom_seed$s.npz; done
echo NOM_DONE

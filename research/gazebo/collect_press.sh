#!/bin/bash
set -e
cd /work
python3 gazebo/fr3_bridge.py gazebo/fr3_ref_plate.sdf gazebo/traj/press.npz results/gz/ref_press.npz
python3 gazebo/fr3_bridge.py gazebo/fr3_nom_plate.sdf gazebo/traj/press.npz results/gz/nom_press.npz
echo PRESS_DONE

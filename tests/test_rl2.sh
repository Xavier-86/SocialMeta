#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate social-meta
cd ~/social-meta/socialmeta/algorithms/RL2
export PYTHONPATH=~/social-meta/socialmeta:$PYTHONPATH
python rl2_cnn_generic.py --config-name=rl2_cnn_cleanup 2>&1 | head -100

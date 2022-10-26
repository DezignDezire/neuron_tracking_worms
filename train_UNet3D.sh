#!/bin/bash

#SBATCH --job-name=neuron_tracking_supervised
#SBATCH --output=%x_%j_%N.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=medium             # Time limit hrs:min:sec
#SBATCH --time=30:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --partition=gpu

SRC_DIR="/scratch/neurobiology/zimmer/Philipp/src_new"
CMD="${SRC_DIR}/start.py --model_type UNet3D --batch_size 22 --n_channels 10 --n_bottleneck_feature_maps 3 --pixel_loss_ratio 0"
python $CMD
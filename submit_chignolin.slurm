#!/bin/bash

#SBATCH --job-name="chignolin-sim"
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH -p mit_preemptable
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH -N 1
#SBATCH --mem=64000
#SBATCH --output=/orcd/scratch/orcd/006/davidgoh/juliaproject/chignolin_%j.log

cd /orcd/scratch/orcd/006/davidgoh/juliaproject

python simulate_chignolin.py \
    --output-dir outputs/chignolin_10us \
    --cuda-device-index 0

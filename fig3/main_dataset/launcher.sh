#!/bin/bash
#SBATCH --job-name=cav
#SBATCH --partition=q-1mois
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=500mb
#SBATCH --error=./Jobs/job_%j.err
#SBATCH --output=./Jobs/job_%j.out

#SBATCH --nodelist=titan-node4

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

~/.pyenv/versions/3.12.2/bin/python3 new_sweep_distance.py

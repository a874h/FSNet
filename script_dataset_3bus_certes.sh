#!/bin/bash

# copied from hardnet/run_expers.sh 

## Usage DGX
## srss
## mamba activate cvxpy
## sbatch script_train_3bus_certes.sh
## squeue

## Usage
## bash script_train_3bus_certes.sh > results/log
## cat commands | xargs -n1 -P8 -I{} /bin/sh -c "{}" 

#SBATCH --partition=low
#SBATCH --job-name=Make_dataset_
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --output=results/%N-%j.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=10000M

mamba activate cvxpy


path_out= "/data/aurelien/local/git/extern/FSNet/datasets/convex/qp"
pv_gen_noise=1.0
demand_noise=1.0
num_examples=10000

# make dataset
for i in 1 2
do
  python dataset/convex/qp/make_dataset_certes_cvxpy_v0.py --path_out $path_out --pv_gen_noise $pv_gen_noise --demand_noise $demand_noise --num_examples $num_examples --seed $i&  
done

mamba activate
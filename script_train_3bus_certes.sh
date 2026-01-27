#!/bin/bash

# copied from hardnet/run_expers.sh 

## Usage
## bash script_train_3bus_certes.sh > results/log
## cat commands | xargs -n1 -P8 -I{} /bin/sh -c "{}" 


mamba activate cvxpy

# params
num_epochs = 150

# some buffer for GPU scheduling at the start
sleep 10
for method in skm # DC3 penalty FSNet 
do
    for i in 1 2
    do
       python main.py --prob_size 192 238 120 10000 --method $method --num_epochs $num_epochs --prob_type convex --prob_name qp --seed $i&     
    done
    wait
done

mamba deactivate 
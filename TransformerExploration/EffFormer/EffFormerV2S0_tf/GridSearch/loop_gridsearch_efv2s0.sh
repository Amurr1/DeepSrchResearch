#!/bin/bash

ds_set=("deluxe" "ethan")
lr_set=(0.0005 0.0001 0.00005 0.00001)
bs_set=(1024 768 512 256 128)

for ds in "${ds_set[@]}"; do
  for lr in ${lr_set[@]}; do
    for bs in ${bs_set[@]}; do
      export DS=$ds
      export LR=$lr
      export BS=$bs
      sbatch --export=ALL,DS=$ds,LR=$lr,BS=$bs job_gridsearch_efv2s0.sh
    done
  done
done
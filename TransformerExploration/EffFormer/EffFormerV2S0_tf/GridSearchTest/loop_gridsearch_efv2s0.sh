#!/bin/bash

chmod u+x /pscratch/sd/a/amurr1/Roman-lens-search/alexisMurray/TransformerExploration/EffFormerV2S0_tf/GridSearchTest/train_efv2s0.py

ds_set=("deluxe")
lr_set=(0.0005)
bs_set=(128)

for ds in "${ds_set[@]}"; do
  for lr in ${lr_set[@]}; do
    for bs in ${bs_set[@]}; do
      sbatch --job-name=gridsearch_efv2s0_${ds}_${lr}_${bs} \
             --output=/pscratch/sd/a/amurr1/Roman-lens-search/alexisMurray/TransformerExploration/EffFormerV2S0_tf/GridSearchTest/JobLogs/gridsearch_efv2s0_${ds}_${lr}_${bs}_%j.out \
             --export=ALL,DS=$ds,LR=$lr,BS=$bs \
             /pscratch/sd/a/amurr1/Roman-lens-search/alexisMurray/TransformerExploration/EffFormerV2S0_tf/GridSearchTest/job_gridsearch_efv2s0.sh
    done
  done
done
#!/bin/bash

chmod u+x /pscratch/sd/a/amurr1/Roman-lens-search/alexisMurray/TransformerExploration/EffFormer/EffFormerSearch/train_ef.py

model=("s0")

for ds in "${model[@]}"; do
    sbatch --job-name=train_ef_${model} \
        --output=/pscratch/sd/a/amurr1/Roman-lens-search/alexisMurray/TransformerExploration/EffFormer/EffFormerSearch/JobLogs/train_ef_${model}_%j.out \
        --export=MODEL=$model \
        /pscratch/sd/a/amurr1/Roman-lens-search/alexisMurray/TransformerExploration/EffFormer/EffFormerSearch/job_train_ef.sh
done
#!/bin/bash
#SBATCH --job-name=gridsearch_efv2s0
#SBATCH --output=pscratch/sd/a/amurr1/Roman-lens-search/alexisMurray/TransformerExploration/EffFormerV2S0_tf/GridSearch/JobLogs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular

module load python
source activate $HOME/conda_envs/vit_env_0

GS_DIR="/pscratch/sd/a/amurr1/Roman-lens-search/alexisMurray/TransformerExploration/EffFormerV2S0_tf/GridSearch"
CSV_PATH="$GS_DIR/TrainingLogs/train_log_${DS}_${LR}_${BS}_${SLURM_JOB_ID}.csv"
CHKPT_PATH="$GS_DIR/Weights/BestChkpt/chkpt_${DS}_${LR}_${BS}_${SLURM_JOB_ID}.h5"
END_PATH="$GS_DIR/Weights/EndRun/endrun_${DS}_${LR}_${BS}_${SLURM_JOB_ID}.h5"

python train_efv2s0.py --ds $DS --lr $LR --bs $BS --log_csv "$CSV_PATH" --w_chkpt "$CHKPT_PATH" --w_end "$END_PATH"
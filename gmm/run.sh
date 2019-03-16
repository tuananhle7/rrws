#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --job-name=gmm
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tuananh@robots.ox.ac.uk
#SBATCH -o ./jobs_out_err/gmm_%j.out
#SBATCH -e ./jobs_out_err/gmm_%j.err

cd /data/engs-woodgroup/magd3733/rwspp/gmm

TRAIN_MODE=$1
NUM_PARTICLES=$2
SEED=$3

module load python/anaconda3/5.0.1
source activate $HOME/torch-env
python --version
python -c "import torch; print('torch version = {}'.format(torch.__version__))"
python -u run.py --train-mode $TRAIN_MODE \
                 --num-particles $NUM_PARTICLES \
                 --seed $SEED 2>&1 | tee ./jobs_out_err/gmm_${PBS_JOBID}_temp.out_err

#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --job-name=rws-pcfg
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tuananh@robots.ox.ac.uk
#SBATCH -o ./jobs_out_err/rws_pcfg_%j.out
#SBATCH -e ./jobs_out_err/rws_pcfg_%j.err

cd /data/engs-woodgroup/magd3733/rwspp/code/pcfg

TRAIN_MODE=$1
NUM_ITERATIONS=$2
LOGGING_INTERVAL=$3
EVAL_INTERVAL=$4
CHECKPOINT_INTERVAL=$5
BATCH_SIZE=$6
NUM_PARTICLES=$7
SEED=$8
PCFG_PATH=$9
EXP_LEV=$10

module load python/anaconda3/5.0.1
source activate $HOME/torch-env
python --version
python -c "import torch; print('torch version = {}'.format(torch.__version__))"
python -u run.py --train-mode $TRAIN_MODE \
                 --num-iterations $NUM_ITERATIONS \
                 --logging-interval $LOGGING_INTERVAL \
                 --eval-interval $EVAL_INTERVAL \
                 --checkpoint-interval $CHECKPOINT_INTERVAL \
                 --batch-size $BATCH_SIZE \
                 --num-particles $NUM_PARTICLES \
                 --seed $SEED \
                 --pcfg-path $PCFG_PATH \
                 --exp-levenshtein $EXP_LEV 2>&1 | tee ./jobs_out_err/rws_pcfg_${PBS_JOBID}_temp.out_err

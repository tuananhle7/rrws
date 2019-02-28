#!/bin/bash

#PBS -N rws-pcfg
#PBS -M tuananh@robots.ox.ac.uk
#PBS -m abe
#PBS -q nice

# Set output and error directories
#PBS -o ./jobs_out_err/rws_pcfg_${PBS_JOBID}.out
#PBS -e ./jobs_out_err/rws_pcfg_${PBS_JOBID}.err

cd /ubc/cs/research/plai-scratch/tuananh/rwspp/code/pcfg

TRAIN_MODE=$1
NUM_ITERATIONS=$2
LOGGING_INTERVAL=$3
EVAL_INTERVAL=$4
CHECKPOINT_INTERVAL=$5
BATCH_SIZE=$6
NUM_PARTICLES=$7
SEED=$8

echo `date +%Y-%m-%d_%H:%M:%S` build docker image
docker build -t rws-pcfg .

echo `date +%Y-%m-%d_%H:%M:%S` start docker container
id=$(docker run -it -d --rm -v $PWD:/workspace rws-pcfg)

echo `date +%Y-%m-%d_%H:%M:%S` run run.py in docker container
docker exec $id python -u run.py --train-mode $TRAIN_MODE \
                                 --num-iterations $NUM_ITERATIONS \
                                 --logging-interval $LOGGING_INTERVAL \
                                 --eval-interval $EVAL_INTERVAL \
                                 --checkpoint-interval $CHECKPOINT_INTERVAL \
                                 --batch-size $BATCH_SIZE \
                                 --num-particles $NUM_PARTICLES \
                                 --seed $SEED 2>&1 | tee ./jobs_out_err/rws_pcfg_${PBS_JOBID}_temp.out_err

echo `date +%Y-%m-%d_%H:%M:%S` stop docker container
docker stop $id
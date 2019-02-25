#!/bin/bash

#PBS -N test
#PBS -M tuananh@robots.ox.ac.uk
#PBS -m abe
#PBS -q nice

# Set output and error directories
#PBS -o ./jobs_out_err/test_${PBS_JOBID}.out
#PBS -e ./jobs_out_err/test_${PBS_JOBID}.err

if [ "$SSH_CONNECTION" ]; then
  cd /ubc/cs/research/plai-scratch/tuananh/rwspp/code/pcfg
fi

TRAIN_MODE=$1
NUM_ITERATIONS=$2
LOGGING_INTERVAL=$3
EVAL_INTERVAL=$4
CHECKPOINT_INTERVAL=$5
BATCH_SIZE=$6
NUM_PARTICLES=$7

echo `date +%Y-%m-%d_%H:%M:%S` build docker image
docker build -t rws-pcfg .

echo `date +%Y-%m-%d_%H:%M:%S` start docker container
id=$(docker run -it -d rws-pcfg)

echo `date +%Y-%m-%d_%H:%M:%S` copy files to docker container
docker cp -a . $id:/workspace/

echo `date +%Y-%m-%d_%H:%M:%S` run run.py in docker container
docker exec $id python -u run.py --train-mode $TRAIN_MODE \
                                 --num-iterations $NUM_ITERATIONS \
                                 --logging-interval $LOGGING_INTERVAL \
                                 --eval-interval $EVAL_INTERVAL \
                                 --checkpoint-interval $CHECKPOINT_INTERVAL \
                                 --batch-size $BATCH_SIZE \
                                 --num-particles $NUM_PARTICLES

echo `date +%Y-%m-%d_%H:%M:%S` copy models/ from docker container
docker cp -a $id:/workspace/models/ .

echo `date +%Y-%m-%d_%H:%M:%S` stop docker container
docker stop $id

echo `date +%Y-%m-%d_%H:%M:%S` remove docker container
docker rm $id
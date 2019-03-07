#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=200:00:00
#SBATCH --job-name=variance
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tuananh@robots.ox.ac.uk
#SBATCH -o ./jobs_out_err/variance_%j.out
#SBATCH -e ./jobs_out_err/variance_%j.err

cd /data/engs-woodgroup/magd3733/rwspp/code/pcfg

module load python/anaconda3/5.0.1
source activate $HOME/torch-env
python --version
python -c "import torch; print('torch version = {}'.format(torch.__version__))"
python -u variance_analysis.py 2>&1 | tee ./jobs_out_err/variance_${PBS_JOBID}_temp.out_err

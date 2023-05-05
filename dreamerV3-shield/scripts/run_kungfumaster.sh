#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=a.goodall22@imperial.ac.uk # required to send email notifcations - please replace <your_username> with your college login name or email address
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=resgpuB
export PATH=/vol/cuda/11.4.120-cudnn8.2.4/bin:/vol/bitbucket/${USER}/anaconda3/bin/:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vol/cuda/11.4.120-cudnn8.2.4/lib64:/vol/cuda/11.4.120-cudnn8.2.4/lib
source activate
conda activate jax
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python train.py --logdir ./logdir/kungfumaster/shield --configs atari xlarge --task atari_kung_fu_master --env.atari.labels death energy-loss --run.steps 10000000

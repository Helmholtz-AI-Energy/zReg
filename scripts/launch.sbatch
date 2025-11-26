#! /bin/bash

# Slurm job configuration
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=pairwise-dist-dwn-before-cpd
#SBATCH --partition=booster
#SBATCH --account=tissuetwin
#SBATCH --output="/p/project/tissuetwin/coquelin1/zReg/logs/slurm/slurm-%j"
#####SBATCH --output="/pfs/work7/workspace/scratch/qv2382-madonna-ddp/qv2382-madonna-ddp/madonna/logs/slurm/slurm-%j"

srun --mpi=pspmix --label bash /p/project/tissuetwin/coquelin1/zReg/scripts/launch_srun.sh

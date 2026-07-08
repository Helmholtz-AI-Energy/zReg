#! /bin/bash

# Slurm job configuration for HoreKa (KIT)
# Adjust --nodes / --ntasks-per-node to match the number of propulate ranks you want.
# Use "accelerated" (A100) or "accelerated-h100" instead of "cpuonly" if the run needs a GPU.
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --job-name=zreg-optimize
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-XXXXXXXX
#SBATCH --output=/hkfs/work/workspace/scratch/<USER>-zreg/logs/slurm/slurm-%j.out

srun --mpi=pmi2 --label bash /hkfs/work/workspace/scratch/<USER>-zreg/zReg/scripts/launch_horeka_srun.sh

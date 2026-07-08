#! /bin/bash

# helper script for module loading + venv activation, run once per srun rank

module purge
module load compiler/gnu
module load mpi/openmpi
module load devel/python/3.10.5_gnu_12.1

source /hkfs/work/workspace/scratch/<USER>-zreg/regvenv/bin/activate
echo "Finished module loading and python activation. Launching python script..."

python -u /hkfs/work/workspace/scratch/<USER>-zreg/zReg/run_eval.py \
    --config /hkfs/work/workspace/scratch/<USER>-zreg/zReg/configs/shah_vs_kobitski_hpo.yaml \
    --mode optimize

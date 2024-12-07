#! /bin/bash

# helper script for activating venv then calling a script

source /p/project1/tissuetwin/coquelin1/regvenv311/bin/activate
echo "Finished module loading and python activation. Launching python script..."

python -u /p/project1/tissuetwin/coquelin1/zReg/scripts/dtw_testing.py

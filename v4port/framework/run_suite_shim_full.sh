#!/bin/bash
# As run_suite.sh, with the shim plugin (malshim.py in this folder) loaded first
S=/tmp/claude-1000/-home-cliffk-idm-ssv4/eec674e9-5cc2-4285-8076-3f8450d9f316/scratchpad
R=/home/cliffk/idm/ngm/starsim-ngm
P=/software/conda/envs/ngm/bin/python
F=$R/v4port/framework
SS=$1; LOG=$2; shift 2
cd $F
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SS:$F STARSIM_INSTALL_FONTS=0 SCIRIS_BACKEND=agg MPLBACKEND=agg
$P -c "import starsim as ss; print(ss.__version__, ss.__file__)" > $LOG 2>&1
$P -m pytest -p no:cacheprovider -p malshim_full -q -rf --durations=0 --tb=line "$@" $R/tests >> $LOG 2>&1

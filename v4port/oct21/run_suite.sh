#!/bin/bash
# Usage: run_suite.sh <log> [plugin]
LOG=$1; PLUG=$2
cd /tmp/claude-1000/-home-cliffk-idm-ngm-starsim-ngm/238c3afc-7d68-4645-9a99-83aa523bd3e8/scratchpad/port/framework
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=${V4ROOT:-/tmp/claude-1000/-home-cliffk-idm-ngm-starsim-ngm/238c3afc-7d68-4645-9a99-83aa523bd3e8/scratchpad/v4wt}:/home/cliffk/idm/ngm/starsim-ngm:/tmp/claude-1000/-home-cliffk-idm-ngm-starsim-ngm/238c3afc-7d68-4645-9a99-83aa523bd3e8/scratchpad/port/framework STARSIM_INSTALL_FONTS=0 SCIRIS_BACKEND=agg MPLBACKEND=agg OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2
ARGS=""; [ -n "$PLUG" ] && ARGS="-p $PLUG"
/software/conda/envs/v4/bin/python -m pytest -p no:cacheprovider $ARGS -q -rf --tb=line /home/cliffk/idm/ngm/starsim-ngm/tests > $LOG 2>&1

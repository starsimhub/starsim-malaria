#!/bin/bash
# Run malariasim's unchanged test suite on a given Starsim tree.
# Usage: run_suite.sh <starsim_root> <log> [extra pytest args]
S=/tmp/claude-1000/-home-cliffk-idm-ssv4/eec674e9-5cc2-4285-8076-3f8450d9f316/scratchpad
R=/home/cliffk/idm/ngm/starsim-ngm
P=/software/conda/envs/ngm/bin/python
SS=$1; LOG=$2; shift 2
cd /home/cliffk/idm/ngm/starsim-ngm/v4port/framework
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SS STARSIM_INSTALL_FONTS=0 SCIRIS_BACKEND=agg MPLBACKEND=agg \
  $P -c "import starsim as ss; print(ss.__version__, ss.__file__)" > $LOG 2>&1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SS STARSIM_INSTALL_FONTS=0 SCIRIS_BACKEND=agg MPLBACKEND=agg \
  $P -m pytest -p no:cacheprovider -q -rf --durations=0 --tb=line "$@" $R/tests >> $LOG 2>&1

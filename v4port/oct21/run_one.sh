#!/bin/bash
f=$1; d=$(dirname $f); b=$(basename $f .py)
cd /tmp/claude-1000/-home-cliffk-idm-ngm-starsim-ngm/238c3afc-7d68-4645-9a99-83aa523bd3e8/scratchpad/port/$d
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=${V4ROOT:-/tmp/claude-1000/-home-cliffk-idm-ngm-starsim-ngm/238c3afc-7d68-4645-9a99-83aa523bd3e8/scratchpad/v4wt}:/home/cliffk/idm/ngm/starsim-ngm:/tmp/claude-1000/-home-cliffk-idm-ngm-starsim-ngm/238c3afc-7d68-4645-9a99-83aa523bd3e8/scratchpad/port/framework:/tmp/claude-1000/-home-cliffk-idm-ngm-starsim-ngm/238c3afc-7d68-4645-9a99-83aa523bd3e8/scratchpad/port/$d STARSIM_INSTALL_FONTS=0 SCIRIS_BACKEND=agg MPLBACKEND=agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
start=$(date +%s)
timeout 1200 /software/conda/envs/v4/bin/python /tmp/claude-1000/-home-cliffk-idm-ngm-starsim-ngm/238c3afc-7d68-4645-9a99-83aa523bd3e8/scratchpad/wrap.py $b.py > ${LOGDIR:-/tmp/claude-1000/-home-cliffk-idm-ngm-starsim-ngm/238c3afc-7d68-4645-9a99-83aa523bd3e8/scratchpad/logs}/${d}__${b}.log 2>&1
echo "exit=$? secs=$(($(date +%s)-start))" >> ${LOGDIR:-/tmp/claude-1000/-home-cliffk-idm-ngm-starsim-ngm/238c3afc-7d68-4645-9a99-83aa523bd3e8/scratchpad/logs}/${d}__${b}.log

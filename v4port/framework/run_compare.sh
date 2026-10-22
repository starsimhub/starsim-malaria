#!/bin/bash
S=/tmp/claude-1000/-home-cliffk-idm-ssv4/eec674e9-5cc2-4285-8076-3f8450d9f316/scratchpad
F=/home/cliffk/idm/ngm/starsim-ngm/v4port/framework
cd $F
export PYTHONDONTWRITEBYTECODE=1 STARSIM_INSTALL_FONTS=0 SCIRIS_BACKEND=agg MPLBACKEND=agg
PYTHONPATH=$S/v3_f40175b4:$F timeout 3000 /software/conda/envs/ngm/bin/python compare_unchanged.py compare_v3.json 2 > compare_v3.log 2>&1 &
A=$!
PYTHONPATH=$S/wt5:$F timeout 3000 /software/conda/envs/ngm/bin/python compare_unchanged.py compare_v4.json 2 > compare_v4.log 2>&1 &
B=$!
wait $A $B

#!/bin/bash
# Interleaved benchmark: 3 rounds of (v3, v4, laser), each a fresh process with numba warm-up, one rep per scenario.
S=/tmp/claude-1000/-home-cliffk-idm-ssv4/eec674e9-5cc2-4285-8076-3f8450d9f316/scratchpad
F=/home/cliffk/idm/ngm/starsim-ngm/v4port/framework
cd $F
export PYTHONDONTWRITEBYTECODE=1 STARSIM_INSTALL_FONTS=0 MPLBACKEND=agg SCIRIS_BACKEND=agg
IDS=${1:-a,b,c,d,e10,fsis10,fseis10,fsis100}
YEARS=${2:-5}
mkdir -p bench_out
for round in 1 2 3; do
  for fw in v3 v4 laser; do
    case $fw in
      v3) PP=$S/v3_f40175b4; FWN=starsim;;
      v4) PP=$S/wt5; FWN=starsim;;
      laser) PP=$S/v3_f40175b4; FWN=laser;;
    esac
    uptime >> bench_out/bench.log
    PYTHONPATH=$PP:$F timeout 1500 /software/conda/envs/ngm/bin/python bench.py $FWN $IDS 1 $YEARS bench_out/${fw}_r$round.json 2>&1 | grep -v -i warn | sed "s/^/[$fw r$round] /" >> bench_out/bench.log
  done
done
echo DONE >> bench_out/bench.log

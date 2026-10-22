# Source this: sets $S, $R, $WT, $V3, $P and the run helpers v4 and v3
export S=/tmp/claude-1000/-home-cliffk-idm-ssv4/eec674e9-5cc2-4285-8076-3f8450d9f316/scratchpad
export R=/home/cliffk/idm/ngm/starsim-ngm
export WT=$S/wt5
export V3=$S/v3_f40175b4
export P=/software/conda/envs/ngm/bin/python
v4() { PYTHONPATH=$WT STARSIM_INSTALL_FONTS=0 SCIRIS_BACKEND=agg MPLBACKEND=agg $P "$@"; }
v3() { PYTHONPATH=$V3 STARSIM_INSTALL_FONTS=0 SCIRIS_BACKEND=agg MPLBACKEND=agg $P "$@"; }

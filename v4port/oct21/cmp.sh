#!/bin/bash
# Print each finished log next to its Oct 17 output: cmp.sh <part>
S=/tmp/claude-1000/-home-cliffk-idm-ngm-starsim-ngm/238c3afc-7d68-4645-9a99-83aa523bd3e8/scratchpad
for f in $S/logs/$1__*.log; do
  grep -q '^exit=' $f || continue
  b=$(basename $f .log); b=${b#*__}; d=$1; P=$S/port/$d
  old=$(ls $P/$b.log $P/$b.out $P/$b.txt $P/${b}_v4.log $P/${b}_wt5.log $P/out_$b.txt $P/out_${b}_v4.txt $P/out_${b}_wt5.txt 2>/dev/null | head -1)
  echo "################ $d/$b  ($(tail -1 $f))"
  filt() { grep -v -iE 'warn|^\s*$|new_infections is 0|__init__.py|Initializing sim|^\s+(~|\^)+\s*$' "$1" | cut -c1-230; }
  if [ -n "$old" ] && cmp -s <(filt $f | grep -v '^exit=' | sed 's/[0-9.]* s\b/T s/g') <(filt $old | sed 's/[0-9.]* s\b/T s/g'); then echo "IDENTICAL to $(basename $old) (apart from timings)"; continue; fi
  echo "--- NOW"; filt $f | grep -v '^exit=' | tail -${2:-25}
  [ -n "$old" ] && { echo "--- OLD ($(basename $old))"; filt $old | tail -${2:-25}; }
done

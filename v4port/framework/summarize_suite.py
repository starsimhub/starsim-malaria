""" Sum per-test durations (pytest --durations=0) for two suite logs, and list the tests with the largest differences """
import re, sys
def read(path):
    d = {}
    for line in open(path):
        m = re.match(r'\s*([\d.]+)s (call|setup|teardown)\s+(\S+)', line)
        if m:
            d[m.group(3)] = d.get(m.group(3), 0) + float(m.group(1))
    return d
a, b = read(sys.argv[1]), read(sys.argv[2])
common = set(a) & set(b)
print(f'tests timed: {len(a)} / {len(b)}; common {len(common)}; sum A {sum(a[k] for k in common):.1f} s, sum B {sum(b[k] for k in common):.1f} s')
diffs = sorted(common, key=lambda k: b[k] - a[k])
for k in diffs[:5] + diffs[-8:]:
    print(f'  {a[k]:6.2f} -> {b[k]:6.2f}  {k}')

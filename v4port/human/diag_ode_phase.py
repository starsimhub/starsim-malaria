""" Diagnose: v3 ODE against v4 ODE, scenario B, total n_I through time (v3 index k against v4 index k+1); and the EIR phase each uses """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc
from scenarios import load
v3 = load('out_v3.npz')['B/ode/0']; v4 = load('out_v4.npz')['B/sis/ode/0']
I3 = np.array(v3['I']); I4 = np.array(v4['I'])
print('len', len(I3), len(I4))
for k in [0, 1, 2, 30, 90, 180, 364, 365, 366, 547, 730, 1095, 1460, 1700, 1800]:
    print(f'day {k:5d}: v3 {I3[k].round(0)} v4 {I4[k].round(0)}')
# Lag that best aligns the two, by node
for n in range(3):
    best = min(range(-20, 21), key=lambda L: np.abs(I3[400:1700, n] - I4[400 + L:1700 + L, n]).mean())
    print('node', n, 'best lag (v4 behind v3 by days):', best)
for y in range(1, 5):
    a, b = 365*y, 365*(y+1) - 25
    best = min(range(-20, 21), key=lambda L: np.abs(I3[a:b].sum(1) - I4[a + L:b + L].sum(1)).mean())
    print(f'year {y}: best lag {best} days; mean ratio v4/v3 {I4[a:b].sum()/I3[a:b].sum():.4f}')

""" Compare v3 (out_v3.npz) and v4 (out_v4.npz): last-year mean prevalence by node, mean +/- SD over seeds; and the largest monthly gap in the seasonal curve """
import json, numpy as np, sciris as sc
from scenarios import scenarios, load
v3 = load('out_v3.npz'); v4 = load('out_v4.npz')

def prev(r, last=365):
    I = np.array(r['I']); N = np.array(r['N'])
    return I[-last:].sum(0)/N[-last:].sum(0)
def monthly(r):
    I = np.array(r['I']); N = np.array(r['N'])
    p = I/N
    return np.array([p[k*30:(k+1)*30].mean(0) for k in range(len(p)//30)])
def stats(d, prefix):
    keys = [k for k in d if k.startswith(prefix + '/')]
    P = np.array([prev(d[k]) for k in keys])
    M = np.array([monthly(d[k]) for k in keys])
    T = np.array([d[k]['time'] for k in keys])
    return P.mean(0), (P.std(0, ddof=1) if len(P) > 1 else 0*P[0]), M.mean(0), T, len(keys)

rows = []
for scen in scenarios:
    for model in ['sis', 'seis']:
        pairs = [(f'{scen}/{model}', f'{scen}/{model}/abm', 'agents')]
        if model == 'sis' and scen != 'C': pairs.append((f'{scen}/ode', f'{scen}/{model}/ode', 'ODE'))
        for k3, k4, kind in pairs:
            m3, s3, M3, T3, n3 = stats(v3, k3)
            m4, s4, M4, T4, n4 = stats(v4, k4)
            gap = np.abs(M3 - M4[:len(M3)]).max()
            rows.append(f'| {scen} | {model.upper()} {kind} | ' + ', '.join(f'{a:.3f}' + (f' ± {b:.3f}' if n3 > 1 else '') for a,b in zip(m3, s3)) + ' | ' +
                        ', '.join(f'{a:.3f}' + (f' ± {b:.3f}' if n4 > 1 else '') for a,b in zip(m4, s4)) + f' | {gap:.3f} | {n3} / {n4} |')
    for model in ['seis'] + (['sis'] if scen == 'C' else []):
        m4, s4, *_ = stats(v4, f'{scen}/{model}/ode')
        rows.append(f'| {scen} | {model.upper()} ODE (v4 only) | (none) | ' + ', '.join(f'{a:.3f}' for a in m4) + ' | | 0 / 1 |')
print('| Scenario | Model | v3: last-year prevalence by node | v4 | Largest gap in 30-day mean prevalence | Seeds (v3 / v4) |')
print('|---|---|---|---|---|---|')
print('\n'.join(rows))

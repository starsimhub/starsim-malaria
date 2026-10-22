""" New capability: fit b (prob_human_infected_on_bite) of the declared human ODE to v3's ODE prevalence (malariasim MalariaHumanODE, b = 0.4), scenario B, with sim.calibrate() """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, pandas as pd, sciris as sc
from scenarios import scenarios, years, load
from human_v4 import make_human
s = scenarios['B']
v3 = load('out_v3.npz')['B/ode/0']
I3 = np.array(v3['I']).sum(1)/np.array(v3['N']).sum(1) # Prevalence, since malariasim's single-bin ODE loses 1% of its people a year (mal_bug_ode_shrinks.py); v3 index k is v4 index k+1
def make(b, data=None):
    d = make_human('sis', eir=s['eir'], pops=s['pops'], b=b)
    return ss.Sim(d, start=2000, stop=2000 + years, dt=ss.days(1), solver='ode', total_pop=sum(s['pops']), verbose=0, data=data)
tv = make(0.4).init().results.malaria.timevec
idx = np.arange(30, len(tv), 30)
data = pd.DataFrame(dict(prevalence=I3[idx - 1]), index=tv[idx])
sim = make(0.15, data=data)
T = sc.timer()
fit = sim.calibrate(verbose=0)
print(f'method {fit.method}; fitted b = {fit.pars}; truth 0.4; {T.toc(output=True):.1f} s; converged {getattr(fit, "converged", None)}')
for b in [0.3, 0.359, 0.4, 0.45]:
    r = make(b).run().results.malaria
    m = np.asarray(r.prevalence)[idx]
    print(f'b={b}: mean abs error vs data {np.abs(m - data.prevalence.values).mean():8.4f}; at month 1 {m[0]:.3f} vs {data.prevalence.values[0]:.3f}; final {m[-1]:.3f} vs {data.prevalence.values[-1]:.3f}')
print('fit table'); fit.summarize() if hasattr(fit, 'summarize') else None

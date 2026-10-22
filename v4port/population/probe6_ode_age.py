""" Probe: can the ODE hold malariasim's age pyramid? Births and deaths only (no infection effect), 20 years """
import numpy as np, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
from common import *
import mal_v4 as m
mu, n, br, max_age = compute_mortality_schedule()
fr = np.array(AGE_BIN_FRACTIONS)/np.sum(AGE_BIN_FRACTIONS)
def run(age, split, death, label, show=False):
    mal = m.make_malaria()
    st = dict(age=list(age)) if split is None else dict(age=list(age), split=list(split))
    sim = ss.Sim(mal, demographics=[ss.birth(rate=ss.peryear(br)), death], stratify=st, solver='ode', total_pop=5000, start=START, stop=STOP, dt=ss.days(DT_DAYS), verbose=0)
    sim.init()
    if show:
        txt = sim.describe(); print('\n'.join(l for l in str(txt).splitlines() if 'age' in l.lower() or 'aging' in l.lower())[:3000])
    sim.run()
    res = sim.results.malaria
    bya = res.n_S.by('age').values + res.n_I.by('age').values
    N = bya.sum(1)
    # Pyramid in 5-year bins for comparison
    def to5(row):
        e = np.asarray(age, float); out = np.zeros(20)
        for k, lo in enumerate(e): out[min(int(lo//5), 19)] += row[k]
        return out/out.sum()
    p0, p1 = to5(bya[0]), to5(bya[-1])
    print(f'{label}: N {N[0]:.0f} -> {N[-1]:.0f} ({100*(N[-1]/N[0]-1):+.1f}%); start pyramid vs malariasim {tvd(p0, fr):.3f}; end vs start {tvd(p0, p1):.3f}; share under 5 {p0[0]:.3f} -> {p1[0]:.3f} (malariasim {fr[0]:.3f})')
e5 = AGE_BIN_EDGES[:-1]
mu5 = np.array([np.average(mu[lo:lo+5], weights=n[lo:lo+5]) for lo in e5])
run(e5, None, ss.death(rate=ss.by(age=[ss.peryear(x) for x in mu5])), '5-year bins, no split (guessed)', show=True)
run(e5, fr, ss.death(rate=ss.by(age=[ss.peryear(x) for x in mu5])), '5-year bins, split=malariasim fractions')
e1 = list(range(100)); n1 = n/n.sum()
run(e1, n1, ss.death(rate=ss.by(age=[ss.peryear(x) for x in mu])), '1-year bins, split=malariasim density, mu by year of age')
# The death rate that would hold a given split with aging at 1/width: mu_k = (N_{k-1}/N_k - 1)/w (and births fill the first group)
w = 5.0; N5 = fr
mu_hold = np.r_[br/N5[0] - 1/w, (N5[:-1]/N5[1:] - 1)/w]; mu_hold[-1] = N5[-2]/N5[-1]/w
print('death rate that holds the split under 1/width aging, first groups:', np.round(mu_hold[:4], 4), 'against malariasim mu by group', np.round(mu5[:4], 4))
run(e5, fr, ss.death(rate=ss.by(age=[ss.peryear(max(x, 0)) for x in mu_hold])), '5-year bins, split, death rate solved for 1/width aging')

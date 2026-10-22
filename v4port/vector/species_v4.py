"""
Two mosquito species in v4. (1) As the spec says: stratify the mosquito states by species. (2) Written out as
two sets of states. Species 2 (funestus-like) emerges at half the rate, dies twice as fast, and bites humans half
as often (a = 0.15 against 0.3). The biting rate enters both directions: mosquito infection (a*c*Ih/N) and
the EIR (a*Iv/N), so a per-species `a` must reach the human infection rate as well.
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc
from ms_mosquito_v4 import make_malaria, emergence

def two_sets(a=(0.3, 0.15), emerge=(1.0, 0.5), life=(10, 5), b=0.5, c=0.5, eip=12):
    Sh, Ih = ss.states(2)
    S1, E1, I1, S2, E2, I2 = ss.states(6)
    N = (Sh + Ih).n
    em = ss.timeseries({k*365.25/12: v for k, v in enumerate(emergence + emergence[:1])}, mode='linear', period=ss.years(1))
    return ss.Disease(
        infection   = Sh >> Ih @ ss.perday(b) * (a[0]*I1.n + a[1]*I2.n) / N,
        recovery    = Ih >> Sh @ ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)),
        emergence1  = ss.null >> S1 @ ss.perday(emerge[0]) * em * N,
        emergence2  = ss.null >> S2 @ ss.perday(emerge[1]) * em * N,
        biting1     = S1 >> E1 @ ss.perday(a[0]*c) * Ih.n / N,
        biting2     = S2 >> E2 @ ss.perday(a[1]*c) * Ih.n / N,
        incubation1 = E1 >> I1 @ ss.perday(1/eip),
        incubation2 = E2 >> I2 @ ss.perday(1/eip),
        mortality1  = S1 + E1 + I1 >> ss.null @ ss.perday(1/life[0]),
        mortality2  = S2 + E2 + I2 >> ss.null @ ss.perday(1/life[1]),
        init = dict(Ih=0.01, S1=250, S2=62.5), name='malaria')

def stratified(adjust_infection=False):
    mal = make_malaria()
    st = mal.graph.states
    adjust = {'emergence': [1.5, 1.5], 'mortality': [1, 2], 'biting': [1, 0.5]}  # an inflow is shared by the split (2/3, 1/3) and then adjusted: 1.0 and 0.5 of the base
    if adjust_infection:
        adjust['infection'] = [1, 0.5]
    mal.stratify(species=['gambiae', 'funestus'], split=[2/3, 1/3], states=[st.Sv, st.Ev, st.Iv], adjust=adjust)
    return mal

def run(mal, solver):
    pop = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
    return ss.Sim(mal, solver=solver, unit='day', start='2000-01-01', stop='2003-01-01', dt=1, verbose=0, **pop).run()

for solver in ['ode', 'abm']:
    r = run(two_sets(), solver).results.malaria
    prev = np.array(r.n_Ih)/(np.array(r.n_Sh)+np.array(r.n_Ih))
    print(f'two sets of states, {solver}: prev y3 {prev[731:].mean():.4f}; I1 {np.mean(r.n_I1[731:]):.1f}, I2 {np.mean(r.n_I2[731:]):.1f}; M1 {np.mean(np.array(r.n_S1)+np.array(r.n_E1)+np.array(r.n_I1)):.0f}, M2 {np.mean(np.array(r.n_S2)+np.array(r.n_E2)+np.array(r.n_I2)):.0f}')
    r = run(stratified(), solver).results.malaria
    prev = np.array(r.n_Ih)/(np.array(r.n_Sh)+np.array(r.n_Ih))
    Iv = r.n_Iv.by('species').values
    print(f'stratify(species=...), {solver}: prev y3 {prev[731:].mean():.4f}; Iv by species {np.round(Iv[731:].mean(axis=0), 1).tolist()}')
try:
    sim = run(stratified(adjust_infection=True), 'ode')
    r = sim.results.malaria; prev = np.array(r.n_Ih)/(np.array(r.n_Sh)+np.array(r.n_Ih))
    print(f'stratify with adjust infection by species, ode: runs; prev y3 {prev[731:].mean():.4f}')
except Exception as e:
    print('stratify with adjust infection by species:', type(e).__name__, str(e)[:400])

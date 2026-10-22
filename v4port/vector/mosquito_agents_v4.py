"""
malariasim's agent-based mosquitoes (MosquitoesAdultFemaleAgentsOnly: one node, no bed nets) kept as code in v4,
coupled to a declared human SIS. v4's declared vocabulary has no mosquito agents (a state that is not people is a
count), so this is the "stay coded" route: the module reads the humans' counts and writes the EIR, which the
declared infection rate reads through a function. Same scenario as ms_v3_ref.py.
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, numpy as np, sciris as sc
from ms_mosquito_v4 import emergence, summarize

class MosquitoAgents(ss.Module):
    """ SEI adult female mosquitoes as agents in arrays of their own (malariasim's logic, verbatim where it matters) """
    compartmental = True # It reads only counts of humans and writes one number, so it works on groups too
    def __init__(self, a=0.3, c=0.5, life=10, eip=12, disease='malaria', **kwargs):
        super().__init__(name='mosquitoes', **kwargs)
        self.define_pars(a=a, c=c, life=life, eip=eip, emergence=np.array(emergence))
        self.disease = disease
        self.eir = 0.0 # Infectious bites per person per day, read by the human infection rate
        return

    def counts(self):
        """ Infectious and total humans, under either solver (a BoolState as agents, a CountState as an ODE) """
        mal = self.sim.diseases[self.disease]
        n = lambda s: s.n if hasattr(s, 'n') else s.count()
        Ih, Sh = n(mal.Ih), n(mal.Sh)
        return Ih, Ih + Sh

    def em_rate(self):
        t = self.t.now('year') % 1
        x = np.append(np.arange(12)/12, 1.0); y = np.append(self.pars.emergence, self.pars.emergence[0])
        return np.interp(t, x, y)

    def init_post(self):
        super().init_post()
        p = self.pars
        self.rng = np.random.default_rng(self.sim.pars.rand_seed + 7)
        _, N = self.counts()
        if N == 0: # The declared disease has not placed anyone in its states yet when a connector's init_post() runs
            N = self.sim.pars.total_pop or self.sim.pars.n_agents
        n_max = max(100, int(p.emergence.max()*p.life*N*3))
        self.alive, self.sus, self.exp, self.inf = [np.zeros(n_max, bool) for _ in range(4)]
        self.ti_inf, self.ti_dead = np.full(n_max, np.inf), np.full(n_max, np.inf)
        n0 = int(round(self.em_rate()*p.life*N)) # Seeded at the start, which malariasim does only once humans have written their count (one step late, from zero)
        self.alive[:n0] = self.sus[:n0] = True
        self.ti_dead[:n0] = self.rng.exponential(p.life/self.dt.days, n0).astype(int)
        return

    def step(self):
        p, ti, dt = self.pars, self.ti, self.dt.days
        Ih, N = self.counts()
        dying = self.alive & (self.ti_dead <= ti)
        self.alive[dying] = self.sus[dying] = self.exp[dying] = self.inf[dying] = False
        n_new = self.rng.poisson(self.em_rate()*N*dt)
        free = np.flatnonzero(~self.alive)[:n_new]
        self.alive[free] = self.sus[free] = True; self.exp[free] = self.inf[free] = False
        self.ti_inf[free] = np.inf; self.ti_dead[free] = ti + self.rng.exponential(p.life/dt, len(free)).astype(int)
        turning = self.alive & self.exp & (self.ti_inf <= ti)
        self.exp[turning] = False; self.inf[turning] = True
        sus = np.flatnonzero(self.alive & self.sus)
        p_inf = p.c*Ih/max(N, 1) # Mean infectiousness of a bite, times c
        new = sus[self.rng.random(len(sus)) < min(p.a*dt, 1)*p_inf]
        self.sus[new] = False; self.exp[new] = True
        self.ti_inf[new] = ti + self.rng.exponential(p.eip/dt, len(new))
        self.eir = p.a*np.count_nonzero(self.alive & self.inf)/max(N, 1) # Per person per day
        return

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('n_total', dtype=float), ss.Result('n_infectious', dtype=float))
        return

    def update_results(self):
        super().update_results()
        self.results.n_total[self.ti] = np.count_nonzero(self.alive)
        self.results.n_infectious[self.ti] = np.count_nonzero(self.alive & self.inf)
        return

def make_humans(b=0.5, init_prev=0.01):
    Sh, Ih = ss.states(2)
    return ss.Disease(
        infection = Sh >> Ih @ (lambda sim: b*sim.connectors.mosquitoes.eir), # EIR x b, per day
        recovery  = Ih >> Sh @ ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)),
        init      = dict(Ih=init_prev),
        name      = 'malaria',
    )

def run(solver='abm', seed=0, n=5000):
    pop = dict(n_agents=n) if solver == 'abm' else dict(total_pop=n)
    sim = ss.Sim(make_humans(), connectors=MosquitoAgents(), solver=solver, unit='day', start='2000-01-01', stop='2003-01-01', dt=1, rand_seed=seed, verbose=0, **pop)
    T = sc.tic(); sim.run(); t = sc.toc(T, output=True)
    return sim, t

def summarize_coded(sim):
    r = sim.results.malaria; m = sim.results.mosquitoes
    prev = (np.array(r.n_Ih)/(np.array(r.n_Sh) + np.array(r.n_Ih)))[1:]
    M = np.array(m.n_total)[1:]; Iv = np.array(m.n_infectious)[1:]; N = (np.array(r.n_Sh) + np.array(r.n_Ih))[1:]
    y3 = slice(2*365, 3*365)
    return dict(prev_y1_mean=float(prev[:365].mean()), prev_y3_mean=float(prev[y3].mean()), prev_y3_max=float(prev[y3].max()), prev_y3_min=float(prev[y3].min()),
                M_y3_mean=float(M[y3].mean()), M_y3_max=float(M[y3].max()), Iv_y3_mean=float(Iv[y3].mean()), eir_y3_mean=float((0.3*Iv/N*365.25)[y3].mean()), M_day0=float(M[0]), M_day30=float(M[30]))

if __name__ == '__main__':
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = {}
    run(seed=99)
    res, times = [], []
    for seed in range(n_seeds):
        sim, t = run(seed=seed); res.append(summarize_coded(sim)); times.append(t)
    agg = {k: (float(np.mean([x[k] for x in res])), float(np.std([x[k] for x in res]))) for k in res[0]}
    out['abm'] = dict(n=len(res), mean_sd=agg, time_s=times)
    print('coded mosquito agents + declared humans (abm)', {k: '%.4g±%.2g' % v for k, v in agg.items()}, 'times', np.round(times, 2).tolist())
    try:
        sim, t = run(solver='ode')
        s = summarize_coded(sim); out['ode'] = dict(n=1, mean_sd={k: (v, 0) for k, v in s.items()}, time_s=[t])
        print('coded mosquito agents + declared humans (ode)', {k: '%.4g' % v for k, v in s.items()}, 'time', round(t, 2))
    except Exception as e:
        print('solver="ode" with the coded mosquitoes:', type(e).__name__, str(e)[:600])
    sc.savejson(sc.thispath(__file__)/'mosquito_agents_v4.json', out)

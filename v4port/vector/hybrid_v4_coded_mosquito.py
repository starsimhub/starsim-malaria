"""
malaria_hybrid's two remaining combinations on v4: its Starsim 3 MosquitoABM class, unchanged, coupled to the
declared SIS humans of hybrid_v4.py (as agents: "ABM x ABM"; as an ODE: "ODE x ABM"). The only glue is a subclass
that fills malaria_hybrid's shared dict sim.c from the declared humans' counts before each mosquito step.
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, numpy as np, sciris as sc
sys.path.insert(0, sc.thispath(__file__))
import hybrid_v3 as mh
from hybrid_v4 import summarize

class MosquitoABM(mh.MosquitoABM):
    compartmental = True # Reads counts and writes counts, so it can drive an ODE of the humans too
    def sync(self):
        mal = self.sim.diseases.malaria
        n = lambda s: s.n if hasattr(s, 'n') else s.count()
        self.sim.c.Sh, self.sim.c.Ih = n(mal.Sh), n(mal.Ih)
    def init_post(self):
        self.sim.c = sc.objdict(Sh=self.sim.pars.Nh - 10, Ih=10, Sm=0, Em=0, Im=0) # The declared disease has not placed people yet
        super().init_post()
    def step(self):
        self.sync()
        super().step()

def make_humans(Nh=10_000, a=0.24, b=0.15, r=1/120, Ih0=10):
    Sh, Ih = ss.states(2)
    return ss.Disease(infection=Sh >> Ih @ (lambda sim: a*b*sim.c.Im/Nh), recovery=Ih >> Sh @ ss.perday(r), init=dict(Ih=Ih0), name='malaria')

class MosqResults(ss.Analyzer):
    compartmental = True
    """ Mosquito counts at the end of each step, under v4's convention (malaria_hybrid's own results are written at self.ti) """
    def init_results(self):
        super().init_results(); self.define_results(ss.Result('Sm'), ss.Result('Em'), ss.Result('Im'))
    def step(self):
        for k in ['Sm', 'Em', 'Im']: self.results[k][self.ti] = self.sim.c[k]

def run(solver, seed=0, Nh=10_000):
    np.random.seed(seed)
    pop = dict(n_agents=Nh) if solver == 'abm' else dict(total_pop=Nh)
    sim = ss.Sim(make_humans(), connectors=MosquitoABM(c=0.15, mu_m=0.10, sigma_m=1/12, m0=5, Im0=100), analyzers=MosqResults(),
                 solver=solver, unit='day', start=0, stop=365*5, dt=1, rand_seed=seed, verbose=0, **pop)
    sim.pars.update(dict(a=0.24, Nh=Nh), create=True) if hasattr(sim.pars, 'update') else None
    T = sc.tic(); sim.run(); return sim, sc.toc(T, output=True)

def summarize_coded(sim):
    r = sim.results.malaria; m = sim.results.mosqresults
    Nh = np.array(r.n_Sh) + np.array(r.n_Ih); M = np.array(m.Sm) + np.array(m.Em) + np.array(m.Im)
    ph = np.array(r.n_Ih)/Nh; pm = np.array(m.Im)/np.maximum(M, 1); mm = M/Nh
    yr5 = slice(4*365+1, 5*365+1)
    return dict(prev_h_y1=float(ph[366]), prev_h_y5_mean=float(ph[yr5].mean()), prev_h_y5_max=float(ph[yr5].max()), prev_h_y5_min=float(ph[yr5].min()),
                prev_m_y5_mean=float(pm[yr5].mean()), m_y5_mean=float(mm[yr5].mean()), m_y5_max=float(mm[yr5].max()), argmax_m_y5=int(np.argmax(mm[yr5])))

if __name__ == '__main__':
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = {}
    for solver in ['ode', 'abm']:
        res, times = [], []
        for seed in range(n_seeds):
            try:
                sim, t = run(solver, seed)
            except Exception as e:
                print(solver, type(e).__name__, str(e)[:500]); raise
            res.append(summarize_coded(sim)); times.append(t)
        agg = {k: (float(np.mean([x[k] for x in res])), float(np.std([x[k] for x in res]))) for k in res[0]}
        out[f'h_{solver}_m_abm'] = dict(n=len(res), mean_sd=agg, time_s=times)
        print(f'human={solver} mosquito=abm (v3 class)', {k: '%.4f±%.4f' % v for k, v in agg.items()}, 'times', np.round(times, 2).tolist())
    sc.savejson(sc.thispath(__file__)/'hybrid_v4_coded_mosquito.json', out)

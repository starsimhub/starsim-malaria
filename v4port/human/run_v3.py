""" v3 reference: unchanged malariasim MalariaHumanSIS / SEIS / ODE driven by MosquitoesEirOnly, over several seeds; per-node prevalence by month """
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, json, numpy as np, sciris as sc, malariasim as ms
from scenarios import scenarios, years, save

class NodePrev(ss.Analyzer):
    """ Record infected (infectious) and exposed counts per node at every step """
    def init_results(self):
        super().init_results()
        self.I = []; self.E = []; self.N = []
    def step(self):
        d = self.sim.diseases[0]
        ns = next(m for m in self.sim.modules if getattr(m, '_is_node_state', False))
        n = ns.grid.n_nodes
        if isinstance(d, ms.MalariaHumanODE):
            Sh = d._Sh.raw.reshape(n, -1).sum(1); Ih = d._Ih.raw.reshape(n, -1).sum(1)
            self.I.append(Ih); self.E.append(0*Ih); self.N.append(Sh + Ih)
        else:
            alive = self.sim.people.alive.uids
            nid = np.asarray(d.node_id[alive], int)
            self.I.append(np.bincount(nid, weights=np.asarray(d.infected[alive], float), minlength=n))
            self.E.append(np.bincount(nid, weights=np.asarray(d.exposed[alive], float), minlength=n) if hasattr(d, 'exposed') else np.zeros(n))
            self.N.append(np.bincount(nid, minlength=n).astype(float))

def run(scen, human, seed):
    s = scenarios[scen]
    hp = {}
    if 'shape' in s: hp.update(heterogeneity_shape=s['shape'], age_biting_ages=s['age_ages'], age_biting_values=s['age_vals'])
    kw = dict(age_data=sc.dataframe(ms.make_age_data())) if (human != 'ode') else {}
    sim = ms.Sim(human=human, mosquito='eir_only', nodes=ms.NodeGrid(populations=s['pops'], eir=s['eir']), common_pars=ms.CommonParams(),
                 human_pars=hp or None, start=2000, stop=2000 + years, dt=ss.days(1), rand_seed=seed, verbose=0, analyzers=NodePrev(), **kw)
    T = sc.timer()
    sim.run()
    t = T.toc(output=True)
    a = sim.analyzers[0]
    I, E, N = np.array(a.I), np.array(a.E), np.array(a.N)
    return dict(I=I.tolist(), E=E.tolist(), N=N.tolist(), time=t)

if __name__ == '__main__':
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    out = {}
    for scen in scenarios:
        for human in ['sis', 'seis', 'ode']:
            if human == 'ode' and scen == 'C': continue # The v3 ODE has no biting heterogeneity
            for seed in (seeds if human != 'ode' else [0]):
                r = run(scen, human, seed)
                out[f'{scen}/{human}/{seed}'] = r
                I = np.array(r['I']); N = np.array(r['N'])
                print(f'{scen} {human:4s} seed {seed}: {r["time"]:.2f} s; last-year prevalence by node', np.round(I[-365:].sum(0)/N[-365:].sum(0), 4))
    save('out_v3.npz', out)

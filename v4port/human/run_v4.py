""" v4 port: the declared human disease (human_v4.py) under the same scenarios as run_v3.py, as agents (several seeds) and as an ODE """
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, numpy as np, sciris as sc, malariasim as ms
from scenarios import scenarios, years, save
from human_v4 import make_human

age_data = sc.dataframe(ms.make_age_data())

def run(scen, model, solver, seed):
    s = scenarios[scen]
    d = make_human(model, eir=s['eir'], pops=s['pops'], shape=s.get('shape', 0), age_ages=s.get('age_ages'), age_vals=s.get('age_vals'), age_data=age_data)
    kw = dict(people=ss.People(sum(s['pops']), age_data=age_data)) if solver == 'abm' else dict(total_pop=sum(s['pops']))
    sim = ss.Sim(d, start=2000, stop=2000 + years, dt=ss.days(1), solver=solver, verbose=0, rand_seed=seed, **kw)
    T = sc.timer()
    sim.run()
    t = T.toc(output=True)
    r = sim.results.malaria
    def node(res):
        v = res.by('patch').values if len(s['pops']) > 1 else np.asarray(res.values)[:, None]
        return v[1:] # v4 index k+1 is the state after step k, which is v3's index k
    I = node(r.n_I); N = sum(node(r[f'n_{x}']) for x in (['S', 'I'] if model == 'sis' else ['S', 'E', 'I']))
    E = node(r.n_E) if model == 'seis' else 0*I
    return dict(I=I.tolist(), E=E.tolist(), N=N.tolist(), time=t)

if __name__ == '__main__':
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    out = {}
    for scen in scenarios:
        for model in ['sis', 'seis']:
            for solver in ['abm', 'ode']:
                for seed in (seeds if solver == 'abm' else [0]):
                    r = run(scen, model, solver, seed)
                    out[f'{scen}/{model}/{solver}/{seed}'] = r
                    I = np.array(r['I']); N = np.array(r['N'])
                    print(f'{scen} {model:4s} {solver:3s} seed {seed}: {r["time"]:.2f} s; last-year prevalence by node', np.round(I[-365:].sum(0)/N[-365:].sum(0), 4), flush=True)
    save('out_v4.npz', out)

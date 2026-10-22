"""
ms.Sim with Starsim's run tools: sim.new(), ss.parallel, diff(), MultiSim, describe(), hash.
Runs on v3 or v4 (v4 with the shim). Each step prints OK/ERROR and its key number.
"""
import traceback
import numpy as np
import sciris as sc
import starsim as ss
print(ss.__version__, ss.__file__)
V4 = ss.__version__.startswith('4')
if V4:
    import malshim # noqa
import malariasim as ms

def nodes():
    return ms.NodeGrid(populations=[1500, 500], labels=['A', 'B'], eir=[20.0, 5.0])

def vaccine_campaign(sim, eir_only=False):
    waning = ms.WaningEffect(initial_efficacy=0.9, box_duration=ss.days(365), decay_time_constant=ss.days(730))
    return ms.Campaign(sim, [ms.InterventionDistributorScheduled(name='vacc', products=ms.MalariaVaccine(waning=waning), start=2000.5, coverage=0.8)])

def make(campaign=None, seed=0, **kw):
    return ms.Sim(human='sis', mosquito='eir_only', nodes=nodes(), start=2000, stop=2002, rand_seed=seed, verbose=0,
                  human_pars=dict(init_prev=ss.bernoulli(0.1)), build_campaign=campaign, **kw)

def cum_inf(sim):
    """ Total person-days infected in year 2: robust to how new_infections is counted """
    ni = np.asarray(sim.diseases[0].results.n_infected.values, dtype=float)
    return float(ni[-365:].sum())

def step(label, fn):
    try:
        out = fn()
        print(f'OK    {label}: {out}')
        return out
    except Exception as e:
        print(f'ERROR {label}: {type(e).__name__}: {str(e)[:300]}')
        return None

base = make()
step('run base', lambda: cum_inf(base.run()))
base2 = make()
step('run base again (same seed)', lambda: cum_inf(base2.run()))
vacc = make(vaccine_campaign).run()
step('vaccine campaign (direct)', lambda: cum_inf(vacc))

if V4:
    s = step('new() of an unrun ms.Sim: type, n_agents, modules', lambda: (lambda n: (type(n).__name__, n.pars.n_agents, list(n.pars.diseases.keys()) if hasattr(n.pars.diseases,'keys') else n.pars.diseases))(make().new()))
    step('new() of a run sim, then run: same as base?', lambda: (lambda n: (cum_inf(n), cum_inf(base)))(base.new().run()))
    step('new(rand_seed=1).run()', lambda: cum_inf(base.new(rand_seed=1).run()))
    step('new() of the vaccine sim, run: same as direct?', lambda: (cum_inf(vacc.new().run()), cum_inf(vacc)))
    step('new(pars=dict(prob_human_infected_on_bite=0.1))', lambda: cum_inf(base.new(pars=dict(prob_human_infected_on_bite=0.1)).run()))
    m = step('ss.parallel(base, n_runs=3)', lambda: ss.parallel(make(), n_runs=3))
    if m is not None:
        step('  per-run year-2 infected-days', lambda: [cum_inf(s) for s in m.sims])
    m2 = step('ss.parallel(baseline=, vaccine=, n_runs=3)', lambda: ss.parallel(baseline=make(), vaccine=make(vaccine_campaign), n_runs=3))
    if m2 is not None:
        step('  per-run year-2 infected-days', lambda: [(s.label, cum_inf(s)) for s in m2.sims])
        step('  diff("baseline")', lambda: str(m2.diff('baseline'))[:600])
    step('ss.diff(base, vacc)', lambda: str(ss.diff(base, vacc))[:400])
    step('hash(base)', lambda: hash(base))
    step('base.describe()', lambda: base.describe(output=True)[:800] if base.describe(output=True) else None)
    step('ms.MalariaMultiSim([make(), make(vaccine_campaign)]).run()', lambda: [cum_inf(s) for s in ms.MalariaMultiSim([make(), make(vaccine_campaign)]).run().sims])
else:
    m = step('ss.parallel(make(), make(vaccine_campaign))', lambda: [cum_inf(s) for s in ss.parallel(make(), make(vaccine_campaign)).sims])
    step('ss.MultiSim(make(), n_runs=3).run()', lambda: [cum_inf(s) for s in ss.MultiSim(make(), n_runs=3).run().sims])
    step('ms.MalariaMultiSim([make(), make(vaccine_campaign)]).run()', lambda: [cum_inf(s) for s in ms.MalariaMultiSim([make(), make(vaccine_campaign)]).run().sims])

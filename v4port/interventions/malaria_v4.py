""" A declared EIR-driven malaria SIS in Starsim v4, with malariasim's interventions as v4 products and deliveries """
import numpy as np
import starsim as ss
from scen import *

def box_exp(e0, box, tau):
    """ malariasim's WaningEffect: full efficacy for `box` days, then exponential decay with time constant `tau`, zero below 1% """
    def w(t):
        t = np.asarray(t, dtype=float)
        if not np.isfinite(box):
            return np.ones_like(t)
        decayed = np.exp(-np.maximum(t - box, 0)/tau) if tau > 0 else (t <= box).astype(float)
        return np.where(decayed < 0.01, 0.0, decayed)
    w.e0 = e0
    return w

def itn_waning():
    """ The bednet's indoor protection, 1 - (1 - usage*repelling)(1 - usage*blocking), as initial value times a fraction of it """
    rep, blk, use = [box_exp(*ITN[k]) for k in ['rep', 'block', 'usage']]
    def combined(t):
        u = ITN['usage'][0]*use(t)
        return 1 - (1 - u*ITN['rep'][0]*rep(t))*(1 - u*ITN['block'][0]*blk(t))
    c0 = float(combined(0.0))
    return c0, (lambda t: combined(t)/c0)

def make_disease(eir=EIR):
    S, I = ss.states(2)
    eir_by = ss.by(patch=[ss.peryear(e) for e in eir])
    mal = ss.Disease(
        infection_in  = S >> I @ eir_by * F_INDOOR,        # indoor bites: what a bednet blocks
        infection_out = S >> I @ eir_by * (1 - F_INDOOR),  # outdoor bites
        recovery      = I >> S @ ss.lognorm_ex(mean=ss.days(DUR_INF), std=ss.days(DUR_INF)),  # as malariasim; an ODE keeps it as 1 Erlang stage
        name = 'malaria',
    )
    return mal.stratify(patch=['n0', 'n1'], split=[p/sum(POPS) for p in POPS], kind='space')

def make_interventions(name, mal):
    at = [ss.days(T_INTV)]
    if name in ['vax', 'vax_node0', 'boost']:
        # efficacy=0.8 (against 'acquisition', i.e. rel_sus) does nothing here: rel_sus scales only contact-mediated hazards (repro_acquisition_external_foi.py)
        vax = ss.Vaccine(efficacy=dict(infection_in=VAX['eff'], infection_out=VAX['eff']), waning=box_exp(VAX['eff'], VAX['box'], VAX['tau']))
        if name == 'boost':
            return ss.vaccinate(vax, coverage=VAX['cov'], at=[ss.days(BOOST['day1']), ss.days(BOOST['day2'])])
        where = (mal.patch == 'n0') if name == 'vax_node0' else None
        return ss.vaccinate(vax, coverage=VAX['cov'], at=at, where=where)
    elif name == 'mda':
        drug = ss.Treatment(cure=1.0, transition='recovery') # without transition=, UnknownEffect: no transition out of the infected states
        return ss.treat(drug, coverage=DRUG['cov'], at=at, where=mal.I) # without where=, nobody is eligible (probe_mda.py)
    elif name == 'itn':
        c0, w = itn_waning()
        itn = ss.Vaccine(efficacy=dict(infection_in=c0), waning=w)
        return ss.vaccinate(itn, coverage=ITN['cov'], at=at)
    elif name == 'seed':
        return ss.Delivery(move='S -> I', coverage=SEED['cov'], at=[ss.days(SEED['day'])], where=mal.patch == f"n{SEED['node']}")
    return None

def make(name='base', solver='abm', seed=1, **kwargs):
    mal = make_disease(eir=[0.0, 0.0] if name == 'seed' else EIR)
    intv = make_interventions(name, mal)
    n = sum(POPS)
    kw = dict(n_agents=n) if solver == 'abm' else dict(total_pop=n)
    kw.update(kwargs)
    return ss.Sim(mal, start='2000-01-01', stop=ss.days(DAYS), dt=ss.days(1), solver=solver, rand_seed=seed, interventions=intv, verbose=0, **kw)

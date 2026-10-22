""" The population part of malariasim, declared in Starsim v4: nodes as a patch dimension with per-patch seasonal EIR
and indoor fraction, steady-state births and deaths, results by age and patch """
import numpy as np, pandas as pd, sciris as sc
import starsim as ss
from common import *

def eir_series(row):
    """ A monthly EIR profile (bites/person/year), linearly interpolated and repeated every year, as malariasim's interpolate_monthly """
    return ss.timeseries({k/12: ss.peryear(v) for k, v in enumerate(list(row) + [row[0]])}, mode='linear', period=ss.years(1))

def make_malaria(b=0.4, eir=None, f_indoor=F_INDOOR, labels=LABELS, pops=POPS, age=AGE_BIN_EDGES[:-1]):
    """ Human SIS driven by an external per-patch EIR, split into indoor and outdoor bites (so bednets can act on one) """
    eir = node_eir_profiles() if eir is None else eir
    S, I = ss.states('S I')
    eir = ss.par(ss.by(patch=[eir_series(r) for r in eir]), name='eir')
    fin = ss.par(ss.by(patch=list(f_indoor)), name='f_indoor')
    mal = ss.Disease(
        bite_indoor  = S >> I @ b * fin * eir,
        bite_outdoor = S >> I @ b * (1 - fin) * eir,
        recovery     = I >> S @ ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)),
        prevalence   = I / (S + I),
        people       = S + I,
        name = 'malaria',
    )
    mal.stratify(patch=list(labels), kind='space', split=np.array(pops)/np.sum(pops))
    return mal

def steady_state(kind='declared', age=AGE_BIN_EDGES[:-1]):
    """ Births balancing deaths at malariasim's pyramid. kind: 'modules' (malariasim's ss.Births + ss.Deaths(table), verbatim);
    'declared' (ss.birth + ss.death by single year of age); 'lifetable' (ss.birth + ss.death(lifetable=)) """
    mu, n, br, max_age = compute_mortality_schedule()
    if kind == 'modules':
        df = pd.DataFrame({'Time': 2000, 'AgeGrpStart': np.arange(max_age), 'mx': mu})
        return [ss.Births(birth_rate=ss.peryear(br*1000)), ss.Deaths(death_rate=df, rate_units=1, metadata={'data_cols': dict(year='Time', age='AgeGrpStart', value='mx')})]
    elif kind == 'declared':
        edges = np.asarray(age, float)
        mu_bin = np.array([np.average(mu[int(lo):int(hi)], weights=n[int(lo):int(hi)]) for lo, hi in zip(edges, np.r_[edges[1:], max_age])]) # Mean hazard in each age group, weighted by the pyramid
        return [ss.birth(rate=ss.peryear(br)), ss.death(rate=ss.by(age=[ss.peryear(m) for m in mu_bin]))]
    elif kind == 'lifetable':
        surv = np.r_[1, np.exp(-np.cumsum(mu))] # Survival to each integer age
        dx = -np.diff(surv)
        return [ss.birth(rate=ss.peryear(br)), ss.death(lifetable=ss.lifetable(np.arange(max_age), dx))]
    elif kind == 'mixed': # Declared births with malariasim's ss.Deaths(table) module
        df = pd.DataFrame({'Time': 2000, 'AgeGrpStart': np.arange(max_age), 'mx': mu})
        return [ss.birth(rate=ss.peryear(br)), ss.Deaths(death_rate=df, rate_units=1, metadata={'data_cols': dict(year='Time', age='AgeGrpStart', value='mx')})]
    elif kind == 'fast': # malariasim's FastBirths/FastDeaths (branch Interventions2-ck), unchanged
        import v3_fast_demog as fd
        return fd.make_steady_state_demographics()
    raise ValueError(kind)

def age_data():
    _, n, _, max_age = compute_mortality_schedule()
    return pd.DataFrame({'age': np.arange(max_age, dtype=int), 'value': n})

def make_sim(seed=0, solver='abm', demog='declared', years=STOP-START, n_agents=sum(POPS), age=AGE_BIN_EDGES[:-1], **kw):
    mal = make_malaria(age=age)
    demographics = steady_state(demog, age=age) if demog else None
    stratify = dict(age=list(age)) if age is not None else None
    extra = dict(people=ss.People(n_agents, age_data=sc.dataframe(age_data()))) # Under every solver: the ODE takes its starting pyramid from the agents' ages
    return ss.Sim(mal, demographics=demographics, stratify=stratify, solver=solver, start=START, stop=START+years, dt=ss.days(DT_DAYS), rand_seed=seed, verbose=0, **extra, **kw)

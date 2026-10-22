""" The human side of malariasim, declared in Starsim v4: SIS or SEIS driven by an external, per-node, seasonal EIR, with biting heterogeneity and age-dependent biting """
import numpy as np
import scipy.stats as sps
import starsim as ss

def eir_series(profile):
    """ 12 annual EIR values anchored at the start of each twelfth of the year, linearly interpolated and repeating (malariasim's interpolate_monthly) """
    return ss.timeseries({k/12: ss.peryear(v) for k,v in enumerate(profile)}, mode='linear', period=ss.years(1))

def gamma_levels(shape, n):
    """ n equal-probability bins of Gamma(shape, scale=1/shape) (mean 1), each represented by its conditional mean """
    edges = sps.gamma.ppf(np.linspace(0, 1, n+1), a=shape, scale=1/shape)
    upper = sps.gamma.cdf(edges, a=shape+1, scale=1/shape) # E[X; X < x] = mean * F_{shape+1}(x)
    return np.diff(upper)*n

def age_levels(ages, values, age_data, step=1):
    """ Age bins every `step` years up to the last breakpoint, each with malariasim's interpolated biting factor at its midpoint, normalized so the population mean is 1 """
    edges = list(np.arange(ages[0], ages[-1], step)) + [ages[-1]]
    mids = [(lo + hi)/2 for lo,hi in zip(edges[:-1], edges[1:])] + [ages[-1]]
    f = np.interp(mids, ages, values)
    w = np.asarray(age_data['value'], float); a = np.asarray(age_data['age'], float)
    share = np.array([w[(a >= lo) & (a < hi)].sum() for lo,hi in zip(edges, edges[1:] + [np.inf])])/w.sum()
    return edges, f/(share*f).sum(), share

def make_human(model='sis', eir=([10.0]*12,), pops=(10_000,), b=0.4, dur_inf=200, std_inf=200, dur_exp=(12, 2), shape=0, n_het=10,
               age_ages=None, age_vals=None, age_data=None, name='malaria'):
    """ Build the declared human disease; returns it, stratified by node (patch), biting heterogeneity, and age as needed """
    b = ss.par(b, name='b', bounds=[0, 1])
    eirs = [eir_series(p) for p in eir]
    EIR = eirs[0] if len(eirs) == 1 else ss.by(patch=eirs) # Per person per year, in the person's own node
    dur = ss.lognorm_ex(mean=ss.days(dur_inf), std=ss.days(std_inf))
    if model == 'sis':
        S, I = ss.states('S I')
        dis = ss.Disease(infection=S >> I @ b * EIR, recovery=I >> S @ dur, prevalence=I/(S + I), init=dict(I=0.0), name=name)
    else:
        S, E, I = ss.states('S E I')
        dis = ss.Disease(infection=S >> E @ b * EIR, latency=E >> I @ ss.normal(ss.days(dur_exp[0]), ss.days(dur_exp[1])), recovery=I >> S @ dur, prevalence=I/(S + E + I), init=dict(I=0.0), name=name)
    if len(pops) > 1:
        dis.stratify(patch=[f'node{i}' for i in range(len(pops))], split=list(np.array(pops)/sum(pops)))
    if shape > 0:
        dis.stratify(biting=[f'b{k}' for k in range(n_het)], split=[1/n_het]*n_het, adjust={'infection': list(gamma_levels(shape, n_het))})
    if age_ages is not None:
        edges, f, share = age_levels(age_ages, age_vals, age_data)
        dis.stratify(age=edges, split=list(share), adjust={'infection': list(f)}) # split= so a compartmental run has malariasim's ages too
    return dis

""" Ross-Macdonald malaria declared in Starsim v4: humans (S, I) as agents or groups, mosquitoes (Sv, Ev, Iv) as counts, with bednets and a vaccine """
import numpy as np
import starsim as ss
from scen import *
from malaria_v4 import box_exp, itn_waning

def make_disease(name='base', double_count=True):
    br, km = itn_factors(name)
    br_h = br if double_count else 1.0  # malariasim lowers indoor bites for everyone by the mean block, and again for each holder
    S, I = ss.states(2)
    Sv, Ev, Iv = [ss.State(n, people=False) for n in ['Sv', 'Ev', 'Iv']]  # with ss.states(3), init=dict(Sv=...) makes the mosquitoes the agents (repro_mosquitoes_become_people.py)
    Nh = (S + I).n
    a, b, c, f = ss.perday(RM['a']), RM['b'], RM['c'], F_INDOOR
    mal = ss.Disease(
        infection_in  = S >> I @ a * b * f * Iv.n / Nh * br_h,      # indoor bites: what a bednet blocks (malariasim lowers them for everyone by the mean block too)
        infection_out = S >> I @ a * b * (1 - f) * Iv.n / Nh,
        recovery      = I >> S @ ss.lognorm_ex(mean=ss.days(DUR_INF), std=ss.days(DUR_INF)),
        emergence     = ss.null >> Sv @ ss.perday(RM['em'] * RM['pop']),   # new females per human per day, times the humans
        biting_in     = Sv >> Ev @ a * c * f * I.n / Nh * br,     # mosquitoes infected biting indoors
        biting_out    = Sv >> Ev @ a * c * (1 - f) * I.n / Nh,
        incubation    = Ev >> Iv @ ss.days(RM['eip']),
        mortality     = Sv + Ev + Iv >> ss.null @ ss.perday(1/RM['life']) * km,
        init          = dict(S=1.0, Sv=RM['em'] * RM['pop'] * RM['life']),   # mosquitoes at disease-free equilibrium, as malariasim; S=1.0 avoids an IndexError (repro_mosquitoes_become_people.py)
        name = 'malaria',
    )
    return mal

# Bednet efficacies over time since delivery, per person: what malariasim's BednetUpdater computes
def itn_parts(t, kill=True):
    rep, blk, kil, use = [box_exp(*ITN[k]) for k in ['rep', 'block', 'kill', 'usage']]
    u = ITN['usage'][0]*use(t)
    r, bl, k = ITN['rep'][0]*rep(t), ITN['block'][0]*blk(t), ITN['kill'][0]*kil(t)*kill
    combined = 1 - (1 - u*r)*(1 - u*bl)
    u_kill = u*(1 - r)*bl*k
    return combined, u_kill

def node_means(sim, product, kill):
    """ malariasim's NodeState fields: the means over all humans of the combined block and the kill probability of their nets """
    product = sim.interventions.campaign.product # The sim's copy: the product in the closure is not the one that runs (its arrays are empty)
    has = np.asarray(sim.people[product.dim].raw, dtype=bool) & np.asarray(sim.people.alive.raw, dtype=bool)
    if not has.any():
        return 0.0, 0.0
    t = (sim.ti - product.ti_delivered.raw[has])*1.0  # days since delivery (dt = 1 day)
    cb, uk = itn_parts(t, kill)
    n = sim.people.alive.sum()
    return cb.sum()/n, uk.sum()/n

def make_interventions(name, mal):
    at = [ss.days(T_INTV)]
    seed = ss.Delivery(move='S -> I', coverage=RM['seed_cov'], at=[ss.days(RM['seed_day'])], name='seed')
    if name == 'vax':
        vax = ss.Vaccine(efficacy=dict(infection_in=VAX['eff'], infection_out=VAX['eff']), waning=box_exp(VAX['eff'], VAX['box'], VAX['tau']))
        return [seed, ss.vaccinate(vax, coverage=VAX['cov'], at=at, name='campaign')]
    if name in ['itn', 'itn_nokill']:
        c0, w = itn_waning()
        itn = ss.Vaccine(efficacy=dict(infection_in=c0), waning=w)          # the human side: indoor bites on the people who hold a net
        return [seed, ss.vaccinate(itn, coverage=ITN['cov'], at=at, name='campaign')]
    return [seed]

def itn_factors(name):
    """ The mosquito side of bednets, as functions of the sim (agents only): ss.modify on these transitions is ignored as agents (repro_modify_count_transition.py) """
    kill = name == 'itn'
    a, f, mu = RM['a'], F_INDOOR, 1/RM['life']
    def means(sim):
        if name not in ['itn', 'itn_nokill'] or 'campaign' not in sim.interventions:
            return 0.0, 0.0
        return node_means(sim, None, kill)
    br = ss.par(lambda sim: 1 - means(sim)[0], name='itn_biting')         # fewer indoor bites land, for mosquitoes ...
    km = ss.par(lambda sim: 1 + a*f*means(sim)[1]/mu, name='itn_killing') # ... and nets kill: mu + a*f*kill_prob
    return br, km

def make(name='base', solver='abm', seed=1, double_count=True, **kwargs):
    mal = make_disease(name, double_count)
    intv = make_interventions(name, mal)
    kw = dict(n_agents=RM['pop']) if solver == 'abm' else dict(total_pop=RM['pop'])
    kw.update(kwargs)
    return ss.Sim(mal, start='2000-01-01', stop=ss.days(DAYS), dt=ss.days(1), solver=solver, rand_seed=seed, interventions=intv, verbose=0, **kw)

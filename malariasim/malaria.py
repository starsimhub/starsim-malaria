"""
Malaria in people and mosquitoes, declared from states and transitions, so that one
model runs as agents or as an ODE (`ss.Sim(solver='abm')` or `solver='ode'`).

People are susceptible (S), exposed (E, in the SEIS model only), or infectious (I).
They are infected by the bites of infectious mosquitoes, indoors and outdoors, at
a rate set by the entomological inoculation rate (EIR). The EIR comes from one of:

- 'ode': mosquitoes as counts (Sv, Ev, Iv), which emerge, feed on people, and die
  (Ross-Macdonald); as agents, people are agents and the mosquitoes are counts
- 'eir_only': an EIR given directly (a number, or 12 monthly values)
- 'agents': individual mosquitoes, written as code (`ms.MosquitoAgents`), whose EIR
  the people's rates read

Nodes (places) are a patch stratification: give `nodes=` an `ss.geo.geography`, and
any of `eir`, `emergence`, and `f_indoor` one value (or monthly profile) per node.
"""

import numpy as np
import sciris as sc
import starsim as ss
from . import defaults as msd

__all__ = ['Malaria', 'monthly', 'per_node', 'gamma_levels', 'age_biting']


def monthly(profile, unit=ss.peryear):
    """
    A value given for each month, as a series that repeats every year: the 12 values
    are at the start of each month (from the start of the sim), linearly interpolated
    between them, and from December back to January

    Args:
        profile (list): 12 values, or one value for every month
        unit (type): the unit of the values, e.g. `ss.peryear` (the default) or `ss.perday`
    """
    profile = np.atleast_1d(np.asarray(profile, dtype=float))
    if profile.size == 1 or np.all(profile == profile[0]): # The same all year
        return unit(float(profile[0]))
    if profile.shape != (12,):
        raise ValueError(f'A monthly profile has 12 values, one per month, not {profile.shape}')
    vals = list(profile) + [profile[0]] # Back to January
    return ss.timeseries({ss.years(k/12): unit(v) for k,v in enumerate(vals)}, mode='linear', period=ss.years(1))


def per_node(value, n_nodes, unit=ss.peryear):
    """
    A value for every node, or one for each: a number or 12 monthly values (the same in every node),
    or a list with one of these per node, which is a value by node (`ss.by(patch=...)`)

    With exactly 12 nodes, a list of 12 numbers is one per node.
    """
    listlike = isinstance(value, (list, tuple, np.ndarray))
    nested = listlike and any(isinstance(v, (list, tuple, np.ndarray)) for v in value) # One profile per node
    if n_nodes > 1 and listlike and len(value) == n_nodes and (nested or n_nodes != 12 or unit is None):
        return ss.by(patch=[per_node(v, 1, unit) for v in value])
    if unit is None:
        return float(value)
    return monthly(value, unit)


def gamma_levels(shape, n):
    """
    The mean of each of n equally likely levels of a gamma distribution with mean 1 and the
    given shape (variance 1/shape): the levels of biting heterogeneity
    """
    import scipy.stats as sps
    edges = sps.gamma.ppf(np.linspace(0, 1, n+1), a=shape, scale=1/shape)
    upper = sps.gamma.cdf(edges, a=shape+1, scale=1/shape) # E[X; X < x] = mean*F_{shape+1}(x)
    return np.diff(upper)*n


def age_biting(ages, values, age_data=None, edges=None):
    """
    The relative biting rate by age, interpolated linearly between the ages given and held beyond them, scaled so
    that its mean over the ages of `age_data` (the people at the start; by default malariasim's) is 1

    By single year of age, or with `edges`, averaged over each age group (as a compartmental run needs it)
    """
    age_data = msd.make_age_data() if age_data is None else age_data
    years = np.asarray(age_data['age'], dtype=float)
    weights = np.asarray(age_data['value'], dtype=float)
    rel = np.interp(years, np.asarray(ages, dtype=float), np.asarray(values, dtype=float))
    rel = rel/np.average(rel, weights=weights)
    if edges is None:
        return ss.by(dict(age=years.tolist(), value=rel.tolist()))
    edges = np.asarray(edges, dtype=float)
    group = np.clip(np.searchsorted(edges, years, side='right') - 1, 0, len(edges) - 2)
    vals = [np.average(rel[group == g], weights=weights[group == g]) if (group == g).any() else 1.0 for g in range(len(edges) - 1)]
    return ss.by(age=dict(zip(edges[:-1].tolist(), vals)))


class Malaria(ss.Disease):
    """
    Malaria in people (SIS or SEIS), driven by mosquito bites

    Args:
        human (str): 'sis' (susceptible, infectious, susceptible again) or 'seis' (with an exposed, pre-infectious period)
        mosquito (str): 'ode' (mosquitoes as counts), 'eir_only' (an EIR given), or 'agents' (individual mosquitoes, `ms.MosquitoAgents`)
        nodes (geography): the nodes, as an `ss.geo.geography`; if None, one node
        bednets (list): the nets (`ms.Bednet`) whose blocking and killing the mosquitoes (as counts) feel; `ms.Sim` gives those delivered
        init_prev (float): the share of people infectious at the start
        age_data (dataframe): the ages at the start, to scale the age-dependent biting to a mean of 1 (by default, malariasim's)
        age_bins (list): the edges of the sim's age groups, if any, over which the age-dependent biting is averaged (as an ODE needs)
        kwargs (dict): the parameters (below)

    Parameters:
        prob_human_infected_on_bite (float): the probability that an infectious bite infects (Ross-Macdonald *b*); default 1
        prob_mosquito_infected_on_bite (float): the probability that a bite on an infectious person infects the mosquito (*c*); default 1
        infection_duration (dur/dist): the infectious period; default lognormal with mean and standard deviation 200 days (an exponential with that mean as an ODE)
        dur_exp (dur/dist): the exposed (incubation) period, for 'seis'; default normal, 12 days with SD 2 days
        default_infectiousness (float): how infectious to mosquitoes infectious people are, from 0 to 1; default 1
        heterogeneity_shape (float): the gamma shape of each person's relative biting rate (variance 1/shape); 0 (the default) for none
        n_biting_levels (int): the number of levels of biting heterogeneity; default 10
        age_biting_ages (list): ages for the age-dependent biting rate (None for none)
        age_biting_values (list): the relative biting rate at those ages, interpolated linearly
        eir (float/list): for 'eir_only', the annual EIR (infectious bites per person per year): a number, 12 monthly values, or one of these per node; default 10
        f_indoor (float/list): the share of bites indoors, or one per node; default 0.7
        bites_per_female_per_day (rate): bites per mosquito (Ross-Macdonald *a*); default 0.3 per day
        life_expectancy (dur): the mean life of an adult mosquito; default 10 days
        eip_days (dur): the mean extrinsic incubation period of mosquitoes (exponential); default 12 days
        emergence (list): new adult female mosquitoes per person per day, by month (12 values), or a profile per node; default malariasim's

    The results are the count in each state and the flow of each transition (e.g. `bite_indoor`), plus
    `infections` (by either route), `prevalence` (the infectious share of people), and for mosquitoes as counts,
    `mosquitoes` (the total), and `eir` (infectious bites per person per day).

    Examples:
        ```python
        mal = ms.Malaria(human='seis', mosquito='eir_only', eir=[2, 2, 5, 15, 30, 50, 50, 30, 15, 5, 2, 2])
        sim = ss.Sim(mal, n_agents=10_000, start=2000, stop=2005, dt=ss.days(1)).run()
        ```
    """
    defaults = dict(
        prob_human_infected_on_bite    = ss.par(1.0, bounds=[0, 1], doc='Probability that an infectious bite infects (b)'),
        prob_mosquito_infected_on_bite = ss.par(1.0, bounds=[0, 1], doc='Probability that a bite on an infectious person infects the mosquito (c)'),
        infection_duration             = ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)),
        dur_exp                        = ss.normal(ss.days(12), ss.days(2)),
        default_infectiousness         = 1.0,
        heterogeneity_shape            = 0.0,
        n_biting_levels                = 10,
        age_biting_ages                = None,
        age_biting_values              = None,
        eir                            = 10.0,
        f_indoor                       = 0.7,
        bites_per_female_per_day       = ss.perday(0.3),
        life_expectancy                = ss.days(10),
        eip_days                       = ss.days(12),
        emergence                      = msd.DEFAULT_EMERGENCE,
    )
    structural = ['heterogeneity_shape', 'n_biting_levels', 'age_biting_ages', 'age_biting_values', 'eir', 'f_indoor', 'emergence'] # Used to build the model, not as parameters of it

    def __init__(self, human='sis', mosquito='ode', nodes=None, bednets=None, init_prev=0.0, age_data=None, age_bins=None, name='malaria', **kwargs):
        if human not in ['sis', 'seis']:
            raise ValueError(f"Unknown human model '{human}': 'sis' or 'seis' (for an ODE, run with solver='ode')")
        if mosquito not in ['ode', 'eir_only', 'agents']:
            raise ValueError(f"Unknown mosquito model '{mosquito}': 'ode', 'eir_only', or 'agents'")
        given = {k:v for k,v in kwargs.items() if k in self.structural}
        p = ss.pars({k:v for k,v in self.defaults.items() if k not in self.structural}, **{k:v for k,v in kwargs.items() if k not in self.structural})
        s = sc.mergedicts({k:self.defaults[k] for k in self.structural}, given)
        n_nodes = 1 if nodes is None else len(nodes.names)
        bednets = sc.tolist(bednets)
        weights = gamma_levels(s['heterogeneity_shape'], int(s['n_biting_levels'])) if s['heterogeneity_shape'] > 0 else None # Each person's relative biting rate is one of n equally likely levels

        # People: susceptible, exposed (SEIS), and infectious
        S, E, I = ss.states('S E I')
        infected = E if human == 'seis' else I
        people = (S + E + I) if human == 'seis' else (S + I)
        N = people.n
        b, c, a = p.prob_human_infected_on_bite, p.prob_mosquito_infected_on_bite, p.bites_per_female_per_day
        f_in = ss.par(per_node(s['f_indoor'], n_nodes, unit=None), name='f_indoor')

        # The EIR indoors and outdoors (infectious bites per person), and so the infection of people
        tr = sc.objdict()
        if mosquito == 'ode':
            Sv, Ev, Iv = ss.states('Sv Ev Iv', people=False)
            eir = a*Iv.n/N
        elif mosquito == 'eir_only':
            eir = ss.par(per_node(s['eir'], n_nodes), name='eir')
        else: # Read from the mosquito agents, which are written as code
            eir = ss.by(patch=mosquito_eir)
        if s['age_biting_ages'] is not None: # Relative to the mean over people at the start
            eir = eir*age_biting(s['age_biting_ages'], s['age_biting_values'], age_data, age_bins)
        tr.bite_indoor  = ss.Transition(S, infected, rate=b*f_in*eir, acquisition=True)
        tr.bite_outdoor = ss.Transition(S, infected, rate=b*(1 - f_in)*eir, acquisition=True)
        if human == 'seis':
            tr.onset = E >> I @ p.dur_exp
        tr.recovery = I >> S @ p.infection_duration

        # The mosquitoes, as counts: they emerge, are infected by biting infectious people (each counted by how much they are bitten), and become infectious after the EIP
        init = dict(I=init_prev)
        if mosquito == 'ode':
            em = ss.par(per_node(s['emergence'], n_nodes, unit=ss.perday), name='emergence')
            infectious = (ss.by(biting=list(weights))*I).n if weights is not None else I.n
            unblocked, killed = 1, 0 # Indoor bites on people under nets are blocked, and some of the mosquitoes are killed
            for net in bednets:
                blocks, kills = net.on_mosquitoes()
                unblocked, killed = unblocked - f_in*blocks, killed + a*f_in*kills
            tr.emerge = ss.null >> Sv @ em*N # New adult females per person per day
            tr.feeding = Sv >> Ev @ a*c*p.default_infectiousness*unblocked*infectious/N
            mortality = 1/p.life_expectancy + killed if bednets else p.life_expectancy # With nets, a rate: the natural one plus the nets' killing
            tr.mortality = Sv + Ev + Iv >> ss.null @ mortality
            tr.eip = Ev >> Iv @ p.eip_days
            tr.mosquitoes = Sv + Ev + Iv
            tr.eir = a*Iv/people # Infectious bites per person per day (in each node)
            init.update(Sv=self.mosquito_start)
        tr.infections = ss.outcome(tr.bite_indoor + tr.bite_outdoor, label='Infections')
        tr.prevalence = I/people
        super().__init__(**tr, init=init, first='S', name=name)
        self.define_pars(**{k:v for k,v in p.items() if k not in self.pars}) # Those the declared model does not use (e.g. the mosquitoes' as agents), for code to read
        self.human, self.mosquito, self.bednets, self.nodes = human, mosquito, bednets, nodes
        self.emergence0 = self.first_month(s['emergence'], n_nodes) if mosquito == 'ode' else None # Per person per day, at the start, in each node
        self.biting_weights = weights # The relative biting rate at each level of biting heterogeneity, if any

        # Nodes, and biting heterogeneity
        if nodes is not None:
            self.stratify(patch=nodes, kind='space')
        if weights is not None:
            n = len(weights)
            self.stratify(biting=[f'b{k}' for k in range(n)], split=[1/n]*n, adjust={'bite_indoor':list(weights), 'bite_outdoor':list(weights)})
        return

    @staticmethod
    def first_month(value, n_nodes):
        """ The value of a profile (or one per node, as `per_node()` reads them) at the start, in each node """
        item = per_node(value, n_nodes, unit=ss.perday)
        first = lambda v: float((v.values[0] if isinstance(v, ss.timeseries) else v).value)
        return np.array([first(v) for v in item.items]) if isinstance(item, ss.by) else np.full(n_nodes, first(item))

    def f_indoor_by_node(self):
        """ The share of bites indoors in each node (or the one value) """
        f = getattr(self.pars.f_indoor, 'raw', self.pars.f_indoor)
        return np.array([float(getattr(v, 'value', v)) for v in f.items]) if isinstance(f, ss.by) else float(f)

    def mosquito_start(self, sim):
        """ The mosquitoes at the start: in each node, at the disease-free equilibrium of the emergence at the start """
        per_person = self.emergence0*self.pars.life_expectancy.to('day').value
        n = sim.pars.total_pop or sim.pars.n_agents*(sim.pars.pop_scale or 1)
        if self.nodes is None:
            return float(np.mean(per_person))*n
        pops = np.asarray(self.nodes.pop, dtype=float)
        return ss.by(patch=list(per_person*n*pops/pops.sum()))


def mosquito_eir(sim):
    """ The EIR (infectious bites per person per day) in each node, from `ms.MosquitoAgents` """
    return ss.perday(sim.get_module('mosquitoes').eir)

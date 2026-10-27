"""
Sim: a malaria simulation, assembled from ``ms.Malaria`` and, for mosquito agents, ``ms.MosquitoAgents``
"""

import numpy as np
import sciris as sc
import starsim as ss

from .malaria import Malaria
from .mosquito_agents import MosquitoAgents
from .interventions import Bednet

__all__ = ['Sim']


class Sim(ss.Sim):
    """
    Create a Starsim malaria simulation

    Args:
        human (str): 'sis' (the default) or 'seis' (with an incubation period of about 12 days)
        mosquito (str): 'agents' (individual mosquitoes, the default), 'ode' (mosquitoes as counts), or 'eir_only' (an EIR given, no mosquitoes)
        solver (str): 'abm' (people as agents, the default) or 'ode' (people as compartments too); with mosquito agents, agents only
        nodes (geography/list): the nodes, as an `ss.geo.geography` (e.g. `ss.geo.geography(names=['Urban', 'Rural'], pop=[6000, 4000])`), or a list of populations; by default one node of 10,000 people
        age_data (dataframe): the ages of people at the start (columns `age` and `value`); by default uniform
        age_bin_edges (list): age groups (e.g. `ms.AGE_BIN_EDGES`), for results by age and for the ages of an ODE
        start (int/str): the start (default 2000)
        stop (int/str): the end (default 2010)
        dt (dur): the time step (default one day)
        common_pars (CommonParams): a canonical set of biological parameters (`ms.CommonParams`)
        human_pars (dict): parameters of `ms.Malaria` (also given directly by keyword)
        mosquito_pars (dict): the same, for the mosquitoes (e.g. `emergence`, `life_expectancy`, `eir`)
        init_prev (float): the share of people infectious at the start
        interventions (list/func): the interventions, or a function of the disease that returns them, so that they can name its states (e.g. `lambda mal: ss.Delivery(move=mal.S >> mal.I, at='2000-02-01', coverage=0.15)`)
        kwargs (dict): parameters of `ms.Malaria` by name (e.g. `eir=20`), or passed to `ss.Sim` (e.g. `rand_seed`, `demographics`)

    Examples:
        ```python
        sim = ms.Sim(mosquito='eir_only', eir=[2, 2, 5, 15, 30, 50, 50, 30, 15, 5, 2, 2])
        sim = ms.Sim(human='seis', mosquito='ode', solver='ode', init_prev=0.1)
        nodes = ss.geo.geography(names=['Urban', 'Rural'], pop=[6000, 4000])
        sim = ms.Sim(nodes=nodes, mosquito='ode', emergence=[ms.URBAN_EMERGENCE, ms.RURAL_EMERGENCE], init_prev=0.1)
        ```
    """
    def __init__(self, human='sis', mosquito='agents', solver='abm', nodes=None, age_data=None, age_bin_edges=None,
                 start=2000, stop=2010, dt=None, common_pars=None, human_pars=None, mosquito_pars=None, init_prev=0.0,
                 interventions=None, **kwargs):
        mal_kw = sc.mergedicts(common_pars.pars() if common_pars is not None else None, human_pars, mosquito_pars,
                               {k:kwargs.pop(k) for k in list(kwargs) if k in Malaria.defaults})

        # Nodes: one of 10,000 people, unless given
        if nodes is None:
            pops = [10_000]
        elif isinstance(nodes, ss.geo.geography):
            pops = list(nodes.pop)
        else:
            pops = list(nodes)
            nodes = ss.geo.geography(names=[f'Node {i}' for i in range(len(pops))], pop=pops) if len(pops) > 1 else None
        n = int(np.sum(pops))
        if nodes is not None and len(nodes.names) == 1:
            nodes = None

        # The disease, and its interventions; the mosquitoes feel the bednets delivered (made again with them if the interventions are a function of the disease)
        def make(bednets=None):
            mal = Malaria(human=human, mosquito=mosquito, nodes=nodes, init_prev=init_prev, age_data=age_data, age_bins=age_bin_edges, bednets=bednets, **mal_kw)
            intvs = sc.tolist(interventions(mal) if callable(interventions) and not isinstance(interventions, ss.Module) else interventions)
            return mal, intvs
        mal, intvs = make()
        bednets = [intv.product for intv in intvs if isinstance(getattr(intv, 'product', None), Bednet)]
        if bednets:
            mal, intvs = make(bednets)

        # People, and the population by age for results (and the ages of an ODE)
        pop = dict(people=ss.People(n, age_data=age_data)) if solver == 'abm' else dict(people=ss.People(n, age_data=age_data), total_pop=n)
        stratify = dict(age=list(age_bin_edges[:-1])) if age_bin_edges is not None else None
        connectors = sc.tolist(kwargs.pop('connectors', None))
        if mosquito == 'agents':
            connectors.append(MosquitoAgents(emergence=mal_kw.get('emergence')))
        super().__init__(diseases=mal, connectors=connectors or None, interventions=intvs or None, solver=solver, stratify=stratify,
                         start=start, stop=stop, dt=ss.days(1) if dt is None else dt, **pop, **kwargs)
        return

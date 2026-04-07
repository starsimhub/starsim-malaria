"""
Sim: malaria simulation class.

Thin ``ss.Sim`` subclass that wires together a chosen human disease module
and a chosen mosquito module into a ready-to-run simulation.
"""

import starsim as ss

from .human_sis import MalariaHumanSIS
from .mosquito_eir_only import MosquitoesEirOnly
from .mosquito_agents import MosquitoesAdultsOnly
from .node_grid import NodeGrid
from .node_state import NodeState


class Sim(ss.Sim):
    """
    Create a Starsim malaria simulation.

    Parameters
    ----------
    human : str
        Which human disease module: 'sis' (default).
    mosquito : str
        Which mosquito module: 'eir_only' (default) or 'agents'.
    start : int or str
        Simulation start (year or ISO date). Default: 2000.
    stop : int or str
        Simulation stop. Default: 2010.
    dt : float or None
        Timestep in years. Default: 1 day.
    human_pars : dict or None
        Parameter overrides for the human module.
    mosquito_pars : dict or None
        Parameter overrides for the mosquito module.
    nodes : NodeGrid or None
        Spatial structure. Default: 1 node with 10k people.
    **sim_kwargs
        Forwarded to ss.Sim (e.g. rand_seed, verbose, label).
    """

    def __init__(
        self,
        human='sis',
        mosquito='eir_only',
        start=2000,
        stop=2010,
        dt=None,
        human_pars=None,
        mosquito_pars=None,
        nodes=None,
        **sim_kwargs,
    ):
        if dt is None:
            dt = ss.days(1)

        if nodes is None:
            nodes = NodeGrid(populations=[10_000])
        n_agents = nodes.total_population
        people = ss.People(n_agents)

        # Create the human disease module
        if human == 'sis':
            human_mod = MalariaHumanSIS(pars=human_pars)
        else:
            raise ValueError(f"Unknown human model '{human}'. Valid: 'sis'.")

        # Create the mosquito module
        if mosquito == 'eir_only':
            mos = MosquitoesEirOnly(pars=mosquito_pars)
        elif mosquito == 'agents':
            mos = MosquitoesAdultsOnly(pars=mosquito_pars)
        else:
            raise ValueError(f"Unknown mosquito model '{mosquito}'. Valid: 'eir_only', 'agents'.")

        node_state = NodeState(nodes)

        super().__init__(
            people=people,
            start=start,
            stop=stop,
            dt=dt,
            diseases=human_mod,
            connectors=[node_state, mos],
            **sim_kwargs,
        )

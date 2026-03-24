"""
Sim: malaria simulation class.

This module provides a ``Sim`` subclass of ``ss.Sim`` that wires together a
chosen human disease module and a chosen mosquito module into a ready-to-run
simulation. It is intentionally thin — all real logic lives in the individual
modules.
"""

import starsim as ss

from .human_sis         import MalariaHumanSIS
from .human_seis        import MalariaHumanSEIS
from .mosquito_eir_only import MosqitoesEirOnly
from .mosquito_agents   import MosquitoesAdultFemaleAgentsOnly
from .node_grid         import NodeGrid
from .node_state        import NodeState


class Sim(ss.Sim):
    """
    Create a Starsim malaria simulation.

    Parameters
    ----------
    human : str
        Which human disease module to use:

        - ``'sis'``  : :class:`MalariaHumanSIS` — susceptible → infected →
          susceptible. No incubation period. Default.
        - ``'seis'`` : :class:`MalariaHumanSEIS` — susceptible → exposed →
          infected → susceptible. Adds an explicit ~12-day incubation period
          before agents become infectious.

    mosquito : str
        Which mosquito module to use:

        - ``'agents'``   : :class:`MosquitoesAdultFemaleAgentsOnly` — SEI
          agent-based mosquito population with seasonal emergence. This is
          the default and gives the most realistic dynamics.
        - ``'eir_only'`` : :class:`MosqitoesEirOnly` — fixed or seasonally-
          varying EIR with no explicit mosquito agents. Faster and simpler;
          useful for calibration runs or when mosquito dynamics are not of
          interest.

    age_data : pd.DataFrame or None
        Age distribution passed to ``ss.People``.  DataFrame with ``age`` and
        ``value`` columns (value = relative weight per single-year age group).
        When *None* agents are initialised with a uniform age distribution.

    start : int or str
        Simulation start. Accepts an integer year (e.g. ``2000``) or an ISO
        date string (e.g. ``'2000-01-01'``). Default: 2000.

    stop : int or str
        Simulation stop. Same format as *start*. Default: 2010.

    dt : float or None
        Timestep in years. Default: ``ss.days(1)`` (one day). The model is
        designed to work at any timestep, but daily resolution is recommended
        to accurately represent the mosquito EIP (~12 days) and lifespan
        (~10 days). Use ``ss.days(7)`` for weekly or ``1/12`` for monthly if
        speed is a priority.

    human_pars : dict or None
        Parameter overrides forwarded to the chosen human module. Any
        parameter listed in that class's docstring can be set here, e.g.::

            human_pars=dict(
                prob_human_infected_on_bite=0.4,
                prob_mosquito_infected_on_bite=0.3,
            )

        For ``'seis'``, the incubation duration can also be overridden::

            human_pars=dict(dur_exp=ss.normal(loc=10/365, scale=1/365))

    mosquito_pars : dict or None
        Parameter overrides forwarded to the chosen mosquito module, e.g.::

            # for 'agents' mode
            mosquito_pars=dict(
                emergence=[0.1, 0.1, 0.2, 0.5, 1.5, 3.0, 3.0, 1.5, 0.5, 0.2, 0.1, 0.1],
                life_expectancy=14.0,
                eip_days=10.0,
            )

            # for 'eir_only' mode
            mosquito_pars=dict(eir=[2, 2, 5, 15, 30, 50, 50, 30, 15, 5, 2, 2])

    **sim_kwargs
        Any additional keyword arguments are passed directly to ``ss.Sim``
        (e.g. ``rand_seed``, ``verbose``, ``label``).

    Examples
    --------
    Quickstart with defaults (SIS human, agent mosquitoes)::

        import malariasim as ms
        sim = ms.Sim()
        sim.run()
        sim.plot()

    SEIS human model with EIR-only mosquitoes::

        sim = ms.Sim(
            human='seis',
            mosquito='eir_only',
            mosquito_pars=dict(eir=[2, 2, 5, 15, 30, 50, 50, 30, 15, 5, 2, 2]),
        )

    SEIS with a shorter incubation period::

        import starsim as ss
        sim = ms.Sim(
            human='seis',
            human_pars=dict(dur_exp=ss.normal(loc=10/365, scale=1/365)),
        )

    Agent-based mosquitoes, weekly timestep for speed::

        sim = ms.Sim(mosquito='agents', dt=ss.days(7), n_agents=5_000)

    High-transmission site, 5-year run::

        sim = ms.Sim(
            mosquito='agents',
            n_agents=20_000,
            start=2010,
            stop=2015,
            human_pars=dict(
                prob_human_infected_on_bite=0.5,
                prob_mosquito_infected_on_bite=0.5,
            ),
            mosquito_pars=dict(
                emergence=[0.5, 0.5, 1.0, 2.0, 4.0, 6.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.5],
            ),
        )
    """

    def __init__(
        self,
        human          = 'sis',
        mosquito       = 'agents',
        age_data       = None,
        start          = 2000,
        stop           = 2010,
        dt             = None,
        human_pars     = None,
        mosquito_pars  = None,
        nodes          = None,
        **sim_kwargs,
    ):
        #-------------------------------------------------------------
        # default time step is one day
        # Starsim times are in years, so convert from days if needed.
        #-------------------------------------------------------------
        if dt is None:
            dt = ss.days(1)

        #-------------------------------------------------------------
        # If nodes aren't provided, create a single-NodeGrid with
        # the entire population.
        #-------------------------------------------------------------
        if nodes is None:
            nodes = NodeGrid(populations=[10_000])
        n_agents = nodes.total_population
        people   = ss.People(n_agents, age_data=age_data)

        #-------------------------------------------------------------
        # Create the malaria disease module
        #-------------------------------------------------------------
        if human == 'sis':
            human_mod = MalariaHumanSIS(pars=human_pars)
        elif human == 'seis':
            human_mod = MalariaHumanSEIS(pars=human_pars)
        else:
            raise ValueError(
                f"Unknown human model '{human}'. "
                f"Valid options are 'sis' or 'seis'."
            )

        #-------------------------------------------------------------
        # Create the mosquito module
        #-------------------------------------------------------------
        if mosquito == 'agents':
            mos = MosquitoesAdultFemaleAgentsOnly(pars=mosquito_pars)
        elif mosquito == 'eir_only':
            mos = MosqitoesEirOnly(pars=mosquito_pars)
        else:
            raise ValueError(
                f"Unknown mosquito model '{mosquito}'. "
                f"Valid options are 'agents' or 'eir_only'."
            )

        node_state = NodeState(nodes)

        super().__init__(
            people     = people,
            start      = start,
            stop       = stop,
            dt         = dt,
            diseases   = human_mod,
            connectors = [node_state, mos],
            **sim_kwargs,
        )

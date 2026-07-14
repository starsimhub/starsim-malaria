"""
Sim: malaria simulation class.

This module provides a ``Sim`` subclass of ``ss.Sim`` that wires together a
chosen human disease module and a chosen mosquito module into a ready-to-run
simulation. It is intentionally thin — all real logic lives in the individual
modules.
"""

import numpy as np
import sciris as sc
import starsim as ss

from .human_sis         import MalariaHumanSIS
from .human_seis        import MalariaHumanSEIS
from .human_ode         import MalariaHumanODE
from .mosquito_eir_only import MosquitoesEirOnly
from .mosquito_agents   import MosquitoesAdultFemaleAgentsOnly
from .mosquito_ode      import MosquitoesODE
from .node_grid         import NodeGrid
from .node_state        import NodeState
from .updater           import Updater
from .updaters          import bednet as _bednet_module  # ensure BednetUpdater registers


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
        - ``'ode'``  : :class:`MalariaHumanODE` — SIS compartmental (ODE)
          human population. Deterministic, no per-agent heterogeneity.
          Fixed population; demographics are ignored.

    mosquito : str
        Which mosquito module to use:

        - ``'agents'``   : :class:`MosquitoesAdultFemaleAgentsOnly` — SEI
          agent-based mosquito population with seasonal emergence. This is
          the default and gives the most realistic dynamics.
        - ``'ode'``      : :class:`MosquitoesODE` — SEI compartmental (ODE)
          mosquito population. Deterministic, much faster than the ABM.
          Uses the same parameter names as ``'agents'``.
        - ``'eir_only'`` : :class:`MosquitoesEirOnly` — fixed or seasonally-
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

    Vaccine campaign starting in 2005::

        def build_campaign(sim, eir_only=False):
            waning  = ms.WaningEffect(initial_efficacy=0.80,
                                      box_duration=ss.days(180),
                                      decay_time_constant=ss.days(730))
            vaccine = ms.MalariaVaccine(waning=waning)
            return ms.Campaign(sim, [
                ms.InterventionDistributorScheduled(
                    name='vacc_2005', products=vaccine, start=2005.0, coverage=0.80),
            ])

        sim = ms.Sim(start=2000, stop=2015, build_campaign=build_campaign)

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
        age_bin_edges  = None,
        start          = 2000,
        stop           = 2010,
        dt             = None,
        common_pars   = None,
        human_pars     = None,
        mosquito_pars  = None,
        nodes          = None,
        build_campaign = None,
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

        # ODE human uses ss.People with one slot per (node, age_bin) compartment.
        # Demographics are stripped so Starsim aging is disabled (use_aging=False
        # when no demographics are present).  MalariaHumanODE.init_post() sets
        # people.age[uid] = bin_min_age and populates node_id, enabling uniform
        # eligibility filtering via sim.people for both ABM and ODE.
        if human == 'ode':
            demos = sim_kwargs.pop('demographics', None) or []
            if not isinstance(demos, list):
                demos = [demos]

            # Extract crude birth/death rates from ss.Births / ss.Deaths modules.
            birth_rate = 0.0
            death_rate = 0.0
            for d in demos:
                if isinstance(d, ss.Deaths):
                    try:
                        # ss.peryear(20) → float gives 20.0 (per 1 000 per year)
                        death_rate = float(d.pars.death_rate) / 1000.0
                    except Exception:
                        pass

            # Match birth rate to death rate so the ODE population stays roughly
            # constant while still turning over the age distribution realistically.
            birth_rate = death_rate

            # Compute age_bin_fractions from age_data if edges are provided.
            age_bin_fracs = None
            if age_bin_edges is not None and age_data is not None:
                edges = np.asarray(age_bin_edges)
                ages  = age_data['age'].values
                vals  = age_data['value'].values
                fracs = np.zeros(len(edges) - 1)
                for i in range(len(edges) - 1):
                    mask     = (ages >= edges[i]) & (ages < edges[i + 1])
                    fracs[i] = vals[mask].sum()
                age_bin_fracs = fracs  # will be normalised inside MalariaHumanODE

            # Merge into human_pars (caller-supplied values take precedence).
            ode_extra = dict(
                age_bin_edges       = age_bin_edges,
                age_bin_fractions   = age_bin_fracs,
                birth_rate_per_year = birth_rate,
                death_rate_per_year = death_rate,
            )
            if human_pars:
                ode_extra.update(human_pars)
            human_pars = ode_extra

        if human == 'ode':
            n_bins_for_people = len(age_bin_edges) - 1 if age_bin_edges is not None else 1
            n_agents = nodes.n_nodes * n_bins_for_people
            # age_data is not passed — MalariaHumanODE.init_post() sets
            # people.age[uid] = bin_min_age directly so that HumanEligibility
            # can filter ODE compartments by age using the same sim.people.age
            # path as the ABM.  Starsim aging is already off (no demographics).
            people = ss.People(n_agents)
        else:
            n_agents = nodes.total_population
            people   = ss.People(n_agents, age_data=age_data)

        #-------------------------------------------------------------
        # Create the malaria disease module
        #-------------------------------------------------------------
        if human == 'sis':
            human_mod = MalariaHumanSIS(common_pars=common_pars, pars=human_pars)
        elif human == 'seis':
            human_mod = MalariaHumanSEIS(common_pars=common_pars, pars=human_pars)
        elif human == 'ode':
            human_mod = MalariaHumanODE(common_pars=common_pars, pars=human_pars)
        else:
            raise ValueError(
                f"Unknown human model '{human}'. "
                f"Valid options are 'sis', 'seis', or 'ode'."
            )

        #-------------------------------------------------------------
        # Create the mosquito module
        #-------------------------------------------------------------
        # Seed mosquito pars from common_pars so they match what eir_to_params
        # used, then apply any explicit mosquito_pars overrides on top.
        mos_pars = {}
        if common_pars is not None:
            mos_pars['bites_per_female_per_day'] = common_pars.bites_per_female_per_day
            mos_pars['life_expectancy']           = ss.days(common_pars.mosquito_life_expectancy_days)
            mos_pars['eip_days']                  = ss.days(common_pars.mosquito_eip_days)
        if mosquito_pars:
            mos_pars.update(mosquito_pars)

        if mosquito == 'agents':
            mos = MosquitoesAdultFemaleAgentsOnly(pars=mos_pars)
        elif mosquito == 'ode':
            mos = MosquitoesODE(pars=mos_pars)
        elif mosquito == 'eir_only':
            mos = MosquitoesEirOnly(pars=mosquito_pars)  # eir_only doesn't use these pars
        else:
            raise ValueError(
                f"Unknown mosquito model '{mosquito}'. "
                f"Valid options are 'agents', 'ode', or 'eir_only'."
            )

        node_state = NodeState(nodes)

        #-------------------------------------------------------------
        # If the caller provided a build_campaign function, run it now
        # inside the Updater registration window so Products can call
        # Updater.get().register().  We pass a thin time-only object so
        # Campaign.add() can validate distributor start years immediately.
        #
        # Active updaters (those with at least one registered prefix) are
        # inserted BEFORE node_state and mos in the connectors list.
        # Insertion order within _possible_updaters reflects import order
        # (waning.py before vaccine.py), which is the required step
        # sequence: WaningEffectUpdater must update efficacy before
        # MalariaVaccineUpdater writes prob_acquire.
        #-------------------------------------------------------------
        extra_updaters     = []
        extra_interventions = []

        if build_campaign is not None:
            dt_yrs   = dt.years if isinstance(dt, ss.TimePar) else float(dt)
            start_yr = float(start)
            stop_yr  = float(stop)
            yearvec  = np.arange(start_yr, stop_yr + dt_yrs * 0.5, dt_yrs)
            _thin_sim = sc.objdict(t=sc.objdict(yearvec=yearvec), diseases=[human_mod])

            for updater in Updater._possible_updaters.values():
                updater.clear()

            Updater.activate()
            try:
                campaign = build_campaign(_thin_sim, eir_only=(mosquito == 'eir_only'))
            finally:
                Updater.deactivate()

            extra_interventions = campaign.as_interventions()
            
            # Include all registered updaters. Consumer updaters such as
            # MalariaVaccineUpdater have no _entries of their own but still
            # need to run each step — they return immediately when there is
            # nothing to do, so the overhead is negligible.
            extra_updaters = list(Updater._possible_updaters.values())

        super().__init__(
            people        = people,
            start         = start,
            stop          = stop,
            dt            = dt,
            diseases      = human_mod,
            # node_state must be first: its step() resets itn_biting_reduction and
            # itn_kill_prob to 0 each tick so that BednetUpdater (in extra_updaters)
            # writes into a clean slate.  mos must be last so it reads the fully
            # updated NodeState fields written by the updaters ahead of it.
            connectors    = [node_state] + extra_updaters + [mos],
            interventions = extra_interventions or None,
            **sim_kwargs,
        )

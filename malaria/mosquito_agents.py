"""
MosquitoesAdultFemaleAgentsOnly: SEI agent-based adult female mosquito module.

Female Anopheles mosquitoes are modelled as individual agents with three
states: Susceptible, Exposed, and Infectious. The mosquito population is
managed as numpy arrays within this module — mosquitoes are kept separate
from the human People object to avoid mixing species.

Decoupling from the human model
--------------------------------
This module discovers the human disease module at runtime by duck-typing —
it searches sim.modules for any object that implements:

    get_p_inf_mosquito_indoor(sim)   -> float
    get_p_inf_mosquito_outdoor(sim)  -> float

Any human model (current or future) that exposes these two methods will work
without any changes to this module. Similarly, the human model discovers this
module by searching for get_eir_indoor(sim), so neither module imports or
directly references the other.

Population dynamics
-------------------
The mosquito population is dynamic and seasonal, driven by a monthly emergence
rate (new adult females per human per day). Values between months are linearly
interpolated; December wraps to January.

Each timestep, new susceptible mosquitoes emerge at:

    n_new ~ Poisson(emergence_rate * n_humans * dt_days)

At equilibrium the expected mosquito-to-human ratio is:

    N_mosquitoes / N_humans ≈ emergence_rate / (1 / life_expectancy)

When a mosquito dies (pre-scheduled from an exponential lifespan at birth),
its array slot is freed and reused by new arrivals. Arrays are pre-allocated
to 3× the maximum equilibrium population to avoid runtime resizing.

Human → mosquito transmission
------------------------------
Each susceptible mosquito is assigned indoors or outdoors (Bernoulli with
p = f_indoor), bites with probability bites_per_female_per_day * dt_days, and
if it bites becomes exposed with probability from the human module:

    P(exposed | indoors)  = get_p_inf_mosquito_indoor()   = I_h * c
    P(exposed | outdoors) = get_p_inf_mosquito_outdoor()  = I_h * c

Both return the same value now and will differ once interventions are added.

Exposed → infectious transition
---------------------------------
Scheduled at ti_exposed + eip_days / dt_days (EIP in timestep units).

Mosquito → human transmission (EIR)
-------------------------------------
Exposes the same get_eir_indoor / get_eir_outdoor interface as
MosqitoesEirOnly, so swapping between modules requires no changes to the
human model.
"""

import numpy as np
import starsim as ss

DAYS_PER_YEAR = 365.25

# Default monthly emergence profile: seasonal with a wet-season peak.
# Values are new adult female mosquitoes per human per day.
# With default life_expectancy=10 days (mu=0.1/day):
#   peak (1.0)  → ~10 mosquitoes per human
#   trough (0.2) → ~2 mosquitoes per human
_DEFAULT_EMERGENCE = np.array(
    [0.005, 0.005, 0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.005, 0.005],
    dtype=float,
)


class MosquitoesAdultFemaleAgentsOnly(ss.Module):
    """
    SEI agent-based adult female mosquito module for malaria.

    The mosquito population is dynamic and seasonal, driven by a monthly
    emergence rate. Mosquitoes are individual agents with Susceptible,
    Exposed, and Infectious states, kept separate from the human People
    object.

    This module is decoupled from any specific human disease model. It
    discovers the human module at runtime by searching for the interface
    methods get_p_inf_mosquito_indoor() and get_p_inf_mosquito_outdoor().

    Parameters
    ----------
    emergence : array-like of length 12
        Monthly adult female emergence rate in new mosquitoes per human per
        day. One value per calendar month (index 0 = January). Values between
        months are linearly interpolated; December wraps to January.

        At equilibrium the expected mosquito-to-human ratio is
        emergence / (1 / life_expectancy). Default: seasonal profile
        peaking at 1.0 in Jul-Aug (ratio ≈ 10) and troughing at 0.2 in
        Jan-Feb (ratio ≈ 2), assuming default life_expectancy=10.

    bites_per_female_per_day : float
        Average number of bites per adult female per day. Default: 0.3.

    life_expectancy : float
        Mean adult female lifespan in days. Default: 10.0 (corresponding to
        a daily mortality rate of 0.1).

    eip_days : float
        Extrinsic incubation period (EIP) in days — the time from a mosquito
        becoming exposed to becoming infectious. Default: 12.0.

    f_indoor : float
        Fraction of bites occurring indoors. Must be in [0, 1]. Default: 0.7.

    Examples
    --------
    Default seasonal profile::

        sim = ss.Sim(
            diseases=MalariaHumanSIS(),
            connectors=MosquitoesAdultFemaleAgentsOnly(),
        )

    Custom high-transmission wet-season profile::

        sim = ss.Sim(
            diseases=MalariaHumanSIS(),
            connectors=MosquitoesAdultFemaleAgentsOnly(
                emergence=[0.1, 0.1, 0.2, 0.5, 1.5, 3.0, 3.0, 1.5, 0.5, 0.2, 0.1, 0.1],
                life_expectancy=14.0,
                eip_days=10.0,
            ),
        )
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            emergence                = _DEFAULT_EMERGENCE.copy(),  # new females/human/day, by month
            bites_per_female_per_day = 0.3,                        # biting rate (bites/female/day)
            life_expectancy          = 10.0,                       # mean adult lifespan (days)
            eip_days                 = 12.0,                       # extrinsic incubation period (days)
            f_indoor                 = 0.7,                        # fraction of bites indoors
        )
        self.update_pars(pars, **kwargs)
        self._validate()

        # Mosquito agent arrays — allocated on the first step.
        self._max_mos      = 0
        self.alive         = None   # bool: slot is occupied by a live mosquito
        self.susceptible   = None   # bool: susceptible
        self.exposed       = None   # bool: exposed (not yet infectious)
        self.infectious    = None   # bool: infectious
        self.ti_infectious = None   # float: timestep when exposed → infectious
        self.ti_dead       = None   # float: timestep when mosquito dies

        self._initialized = False
        return

    # ------------------------------------------------------------------
    # Interface required by the human model
    # ------------------------------------------------------------------

    def get_eir(self, sim):
        """
        Total per-timestep EIR (indoor + outdoor).

        Returns
        -------
        float
            Expected infectious bites per person this timestep.
        """
        if not self._initialized:
            return 0.0
        n_humans = sim.people.alive.sum()
        if n_humans == 0:
            return 0.0
        dt_days = sim.t.dt_year * DAYS_PER_YEAR
        n_inf   = float((self.alive & self.infectious).sum())
        return n_inf * self.pars.bites_per_female_per_day * dt_days / float(n_humans)

    def get_eir_indoor(self, sim):
        """
        Indoor per-timestep EIR — the component affected by ITNs and IRS.

        Returns
        -------
        float
        """
        return self.get_eir(sim) * self.pars.f_indoor

    def get_eir_outdoor(self, sim):
        """
        Outdoor per-timestep EIR — unaffected by ITNs and IRS.

        Returns
        -------
        float
        """
        return self.get_eir(sim) * (1.0 - self.pars.f_indoor)

    # ------------------------------------------------------------------
    # Starsim module lifecycle
    # ------------------------------------------------------------------

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_susceptible',      dtype=int,   label='Mosquitoes — susceptible'),
            ss.Result('n_exposed',          dtype=int,   label='Mosquitoes — exposed'),
            ss.Result('n_infectious',       dtype=int,   label='Mosquitoes — infectious'),
            ss.Result('n_total',            dtype=int,   label='Mosquitoes — total alive'),
            ss.Result('eir_annual',         dtype=float, label='EIR — total (bites/person/year)'),
            ss.Result('eir_indoor_annual',  dtype=float, label='EIR — indoor (bites/person/year)'),
            ss.Result('eir_outdoor_annual', dtype=float, label='EIR — outdoor (bites/person/year)'),
        )
        return

    def step(self):
        """
        Advance the mosquito population by one timestep.

        Order of operations:
          1. Allocate arrays on the first step (once n_humans is known).
          2. Kill mosquitoes whose scheduled death time has arrived.
          3. Birth new susceptible mosquitoes from the current emergence rate.
          4. Transition exposed → infectious when EIP has elapsed.
          5. Apply human → mosquito transmission.
          6. Record results.
        """
        sim = self.sim

        if not self._initialized:
            self._init_mosquitoes()
            self._initialized = True

        p       = self.pars
        ti      = sim.ti
        dt_days = sim.t.dt_year * DAYS_PER_YEAR

        # 2. Deaths
        dying = self.alive & (self.ti_dead <= ti)
        if dying.any():
            self._kill_mosquitoes(dying)

        # 3. Births
        n_humans = int(sim.people.alive.sum())
        self._birth_mosquitoes(ti, dt_days, n_humans)

        # 4. Exposed → infectious
        newly_infectious = self.alive & self.exposed & (self.ti_infectious <= ti)
        if newly_infectious.any():
            self.exposed[newly_infectious]    = False
            self.infectious[newly_infectious] = True

        # 5. Human → mosquito transmission
        human = self._human_mod()
        if human is not None:
            self._infect_mosquitoes(human, ti, dt_days)

        # 6. Results
        alive_mask = self.alive
        n_alive    = int(alive_mask.sum())
        n_inf      = int((alive_mask & self.infectious).sum())
        annual_eir = (
            n_inf * p.bites_per_female_per_day * DAYS_PER_YEAR / float(n_humans)
            if n_humans > 0 else 0.0
        )
        self.results.n_susceptible[ti]      = int((alive_mask & self.susceptible).sum())
        self.results.n_exposed[ti]          = int((alive_mask & self.exposed).sum())
        self.results.n_infectious[ti]       = n_inf
        self.results.n_total[ti]            = n_alive
        self.results.eir_annual[ti]         = annual_eir
        self.results.eir_indoor_annual[ti]  = annual_eir * p.f_indoor
        self.results.eir_outdoor_annual[ti] = annual_eir * (1.0 - p.f_indoor)
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_mosquitoes(self):
        """
        Allocate mosquito agent arrays sized to 3× the maximum equilibrium
        population. Seed with the equilibrium count for the simulation start
        month; lifespan of each is drawn from an exponential distribution.
        """
        sim     = self.sim
        p       = self.pars
        dt_days = sim.t.dt_year * DAYS_PER_YEAR
        mu      = 1.0 / p.life_expectancy  # daily death rate

        n_humans = int(sim.people.alive.sum())

        # Maximum equilibrium population per human = max(emergence) * life_expectancy.
        # Multiply by 3 for a safety buffer above the seasonal peak.
        max_equil_per_human = float(np.max(p.emergence)) * p.life_expectancy
        self._max_mos = max(100, int(max_equil_per_human * n_humans * 3))

        self.alive         = np.zeros(self._max_mos, dtype=bool)
        self.susceptible   = np.zeros(self._max_mos, dtype=bool)
        self.exposed       = np.zeros(self._max_mos, dtype=bool)
        self.infectious    = np.zeros(self._max_mos, dtype=bool)
        self.ti_infectious = np.full(self._max_mos,  np.inf)
        self.ti_dead       = np.full(self._max_mos,  np.inf)

        # Seed starting population at the equilibrium level for the start month.
        start_emergence = self._emergence_rate(sim.t.year % 1)
        n_start         = int(round(start_emergence * p.life_expectancy * n_humans))
        n_start         = min(n_start, self._max_mos)

        mean_life_steps = p.life_expectancy / dt_days
        self.alive[:n_start]       = True
        self.susceptible[:n_start] = True
        self.ti_dead[:n_start]     = np.random.exponential(mean_life_steps, n_start).astype(int)
        return

    def _kill_mosquitoes(self, dying):
        """Mark dead mosquitoes as inactive, freeing their slots for new births."""
        self.alive[dying]         = False
        self.susceptible[dying]   = False
        self.exposed[dying]       = False
        self.infectious[dying]    = False
        self.ti_infectious[dying] = np.inf
        self.ti_dead[dying]       = np.inf
        return

    def _birth_mosquitoes(self, ti, dt_days, n_humans):
        """
        Emerge new susceptible mosquitoes from the current emergence rate.

        Count is Poisson-distributed with mean emergence_rate * n_humans * dt_days.
        New mosquitoes are placed into free (dead) array slots. If the buffer
        is exhausted, a warning is printed and births are capped.
        """
        p = self.pars

        emergence_rate = self._emergence_rate(self.sim.t.year % 1)  # per human per day
        n_new = int(np.random.poisson(emergence_rate * n_humans * dt_days))

        if n_new == 0:
            return

        free_slots = np.where(~self.alive)[0]
        if len(free_slots) < n_new:
            print(
                f'Warning (MosquitoesAdultFemaleAgentsOnly): mosquito buffer full at '
                f'timestep {ti}. {n_new - len(free_slots)} births dropped. '
                f'Consider increasing the buffer by raising the emergence values.'
            )
            n_new = len(free_slots)
            if n_new == 0:
                return

        idx             = free_slots[:n_new]
        mean_life_steps = p.life_expectancy / dt_days

        self.alive[idx]         = True
        self.susceptible[idx]   = True
        self.exposed[idx]       = False
        self.infectious[idx]    = False
        self.ti_infectious[idx] = np.inf
        self.ti_dead[idx]       = ti + np.random.exponential(mean_life_steps, n_new).astype(int)
        return

    def _infect_mosquitoes(self, human, ti, dt_days):
        """
        Apply human → mosquito transmission for all live susceptible mosquitoes.

        Each susceptible mosquito:
          1. Is assigned to bite indoors (Bernoulli p = f_indoor) or outdoors.
          2. Bites with probability bites_per_female_per_day * dt_days
             (at most one bite per step).
          3. If it bites, becomes exposed with probability from the human module:
               - get_p_inf_mosquito_indoor()  if biting indoors
               - get_p_inf_mosquito_outdoor() if biting outdoors

        Newly exposed mosquitoes have their infectious time scheduled at
        ti + eip_days / dt_days (EIP converted from days to timesteps).
        """
        sus_idx = np.where(self.alive & self.susceptible)[0]
        n_sus   = len(sus_idx)
        if n_sus == 0:
            return

        p   = self.pars
        sim = self.sim

        p_inf_indoor  = human.get_p_inf_mosquito_indoor(sim)
        p_inf_outdoor = human.get_p_inf_mosquito_outdoor(sim)

        # Probability of biting at all this timestep (at most one bite per step).
        p_bite = min(p.bites_per_female_per_day * dt_days, 1.0)

        # Each susceptible mosquito bites indoors or outdoors.
        bites_indoor = np.random.random(n_sus) < p.f_indoor

        # Combined probability of exposure: P(bites) × P(exposed | bite location).
        p_exposed = np.where(
            bites_indoor,
            p_bite * p_inf_indoor,
            p_bite * p_inf_outdoor,
        )

        gets_exposed      = np.random.random(n_sus) < p_exposed
        newly_exposed_idx = sus_idx[gets_exposed]

        if len(newly_exposed_idx) == 0:
            return

        # Transition S → E and schedule E → I after the EIP.
        eip_steps = p.eip_days / dt_days
        self.susceptible[newly_exposed_idx]   = False
        self.exposed[newly_exposed_idx]       = True
        self.ti_infectious[newly_exposed_idx] = ti + eip_steps
        return

    def _emergence_rate(self, t):
        """
        Linearly interpolate the monthly emergence profile at fractional year t.

        Monthly values are anchored at the start of each month:
          month 0 (Jan) → t = 0/12, ..., month 11 (Dec) → t = 11/12.

        The January value is appended at t = 1.0 to allow smooth wrapping
        from December back to January.

        Parameters
        ----------
        t : float
            Fractional position within the current year, in [0, 1).

        Returns
        -------
        float
            Interpolated emergence rate (new females per human per day).
        """
        em       = self.pars.emergence
        x_months = np.arange(12) / 12
        x_wrap   = np.append(x_months, 1.0)
        y_wrap   = np.append(em, em[0])
        return float(np.interp(t, x_wrap, y_wrap))

    def _human_mod(self):
        """
        Return the first module that implements the human-to-mosquito interface.

        Searches sim.modules for any object with get_p_inf_mosquito_indoor(),
        making this module compatible with any current or future human disease
        model that exposes that interface — no import of a specific class needed.
        """
        for mod in self.sim.modules:
            if hasattr(mod, 'get_p_inf_mosquito_indoor'):
                return mod
        return None

    def _validate(self):
        """Raise informative errors for malformed parameters."""
        em = np.asarray(self.pars.emergence, dtype=float)
        if em.shape != (12,):
            raise ValueError(
                f'emergence must be a 12-element array (one per calendar month), '
                f'got shape {em.shape}'
            )
        if np.any(em < 0):
            raise ValueError('All emergence values must be non-negative')
        self.pars.emergence = em

        if self.pars.life_expectancy <= 0:
            raise ValueError(f'life_expectancy must be positive, got {self.pars.life_expectancy}')
        if self.pars.eip_days <= 0:
            raise ValueError(f'eip_days must be positive, got {self.pars.eip_days}')
        if self.pars.bites_per_female_per_day < 0:
            raise ValueError(f'bites_per_female_per_day must be non-negative, got {self.pars.bites_per_female_per_day}')
        f = self.pars.f_indoor
        if not (0.0 <= f <= 1.0):
            raise ValueError(f'f_indoor must be in [0, 1], got {f}')
        return

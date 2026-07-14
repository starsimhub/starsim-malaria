"""
mock_objects.py — shared mock classes and helpers for the malariasim test suite.

Import from here rather than defining local copies in individual test files.
"""

import numpy as np
import starsim as ss

from malariasim.updater import Updater, Target
from malariasim.updaters.waning import WaningEffect, WaningEffectUpdater, _WaningEntry
from malariasim.human_base import BiteTarget


# ---------------------------------------------------------------------------
# build_campaign() window helpers
# ---------------------------------------------------------------------------

def activate():
    """Open the build_campaign() window. Returns a restore callable."""
    Updater.activate()
    return Updater.deactivate


def make_waning(**kw):
    """Construct a WaningEffect with sensible defaults."""
    defaults = dict(
        initial_efficacy    = 0.80,
        box_duration        = ss.days(180),
        decay_time_constant = ss.days(730),
    )
    defaults.update(kw)
    return WaningEffect(**defaults)


# ---------------------------------------------------------------------------
# MockDisease — minimal disease module mock with waning state arrays
# ---------------------------------------------------------------------------

class MockArr:
    """Minimal mock for a ss.FloatArr state array (no dead-agent separation)."""

    def __init__(self, n, default):
        self.values = np.full(n, default, dtype=float)

    @property
    def raw(self):
        return self.values

    def __setitem__(self, key, value):
        self.values[key] = value

    def __getitem__(self, key):
        return self.values[key]


class MockDisease:
    """
    Minimal mock for a disease module with waning state arrays and the full
    intervention interface (pre_administer, sync_vaccinated_fraction, etc.).

    Usage
    -----
    d = MockDisease(n=10)
    d.add_prefix('vaccine')   # adds vaccine_efficacy0/efficacy/box_end/decay_end
    d._prob_acquire            # _Arr initialised to 1.0, added by add_prefix
    """

    _supports_intervention_effects = True

    def __init__(self, n=10, prefix=None):
        self.n = n
        if prefix is not None:
            self.add_prefix(prefix)

    def add_prefix(self, prefix):
        """Add the four waning state arrays and _prob_acquire for a given prefix."""
        n = self.n
        setattr(self, f'{prefix}_efficacy0', ss.FloatArr(f'{prefix}_efficacy0', raw=np.zeros(n),      default=0.0,  skip_init=True))
        setattr(self, f'{prefix}_efficacy',  ss.FloatArr(f'{prefix}_efficacy',  raw=np.zeros(n),      default=0.0,  skip_init=True))
        setattr(self, f'{prefix}_box_end',   ss.FloatArr(f'{prefix}_box_end',   raw=np.full(n, -1.0), default=-1.0, skip_init=True))
        setattr(self, f'{prefix}_decay_end', ss.FloatArr(f'{prefix}_decay_end', raw=np.full(n, -1.0), default=-1.0, skip_init=True))
        if not hasattr(self, '_prob_acquire_indoor'):
            self._prob_acquire_indoor  = MockArr(self.n, 1.0)
            self._prob_acquire_outdoor = MockArr(self.n, 1.0)

    # -- intervention interface stubs (ABM-style) --

    def pre_administer(self, uids, weights, t_now):
        return np.full(len(uids), t_now, dtype=float)

    def sync_vaccinated_fraction(self, uids, efficacy):
        pass

    def get_vaccinated_weights(self, uids):
        return np.ones(len(uids), dtype=float)

    def set_acquisition_probability(self, uids, efficacy, weights, target=BiteTarget.ALL):
        reduction = 1.0 - efficacy * weights
        if target in (BiteTarget.ALL, BiteTarget.INDOOR):
            self._prob_acquire_indoor.values[uids]  *= reduction
        if target in (BiteTarget.ALL, BiteTarget.OUTDOOR):
            self._prob_acquire_outdoor.values[uids] *= reduction

    def store_itn_fraction(self, uids, weights):
        pass

    def get_itn_weights(self, uids):
        return np.ones(len(uids), dtype=float)

    def aggregate_itn_to_node_state(self, uids, combined_block, u_kill, ns):
        pass

    def validate_eligibility(self, eligibility):
        pass


def make_waning_updater_for_disease(disease, prefix='vaccine'):
    """Return a WaningEffectUpdater with a pre-resolved entry pointing at disease."""
    updater = WaningEffectUpdater()
    entry = _WaningEntry(prefix, Target.DISEASE)
    entry.module = disease
    updater._entries[prefix] = entry
    return updater


# ---------------------------------------------------------------------------
# MockProduct — records administer() calls
# ---------------------------------------------------------------------------

class MockProduct:
    """Records all administer() calls for assertion in tests."""

    def __init__(self):
        self.calls = []

    def administer(self, targets, weights=None):
        self.calls.append(ss.uids(np.asarray(targets)))

    def init_pre(self, sim, force=False):
        pass

    @property
    def all_administered(self):
        """Flat sorted array of all UIDs ever administered."""
        if not self.calls:
            return np.array([], dtype=int)
        return np.sort(np.concatenate([np.asarray(c) for c in self.calls]))


# ---------------------------------------------------------------------------
# MockSim — single configurable sim mock covering all test use cases
# ---------------------------------------------------------------------------

class MockAgeArr:
    """Minimal mock for ss.FloatArr that supports __getitem__ by UID array."""

    def __init__(self, ages):
        self._data = np.array(ages, dtype=float)

    def __getitem__(self, uids):
        return self._data[np.asarray(uids, dtype=int)]


class MockPeople:
    """Minimal mock for ss.People with auids and age."""

    def __init__(self, ages, alive_mask=None):
        ages = np.array(ages, dtype=float)
        if alive_mask is None:
            alive_mask = np.ones(len(ages), dtype=bool)
        self.auids = ss.uids(np.where(alive_mask)[0])
        self.age   = MockAgeArr(ages)


class MockGrid:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes


class MockNodeState:
    _is_node_state = True

    def __init__(self, n_nodes):
        self.grid = MockGrid(n_nodes)


class MockTime:
    """
    Sim time mock carrying both year (scalar) and yearvec (array).

    Parameters
    ----------
    year  : float — current sim year (used by vaccine tests)
    start : float or None — if provided along with stop, builds yearvec
    stop  : float or None
    dt    : float — timestep in years (default 1.0)
    """
    def __init__(self, year=2010, start=None, stop=None, dt=1.0):
        self.year    = float(year)
        self.yearvec = (np.arange(float(start), float(stop) + dt * 0.5, dt)
                        if start is not None else np.array([float(year)]))


class MockSim:
    """
    Single configurable sim mock covering all test use cases.

    Parameters
    ----------
    ages       : list or None  — agent ages; if provided, creates self.people
    alive_mask : array or None — boolean alive mask for people
    n_nodes    : int or None   — if provided, adds MockNodeState to self.connectors
    ti         : int or None   — current timestep index (distributor tests)
    year       : float         — current sim year (vaccine tests); default 2010
    start      : float or None — sim start year; if provided with stop, builds yearvec
    stop       : float or None — sim stop year
    dt         : float         — timestep in years (default 1.0)
    """

    def __init__(self, ages=None, alive_mask=None, n_nodes=None,
                 ti=None, year=2010, start=2010, stop=2020, dt=1.0,
                 diseases=None):
        if ages is not None or alive_mask is not None:
            self.people = MockPeople(ages or [], alive_mask)
        self.connectors = [MockNodeState(n_nodes)] if n_nodes is not None else []
        self.diseases   = diseases if diseases is not None else []
        self.ti = ti
        self.t  = MockTime(year=year, start=start, stop=stop, dt=dt)

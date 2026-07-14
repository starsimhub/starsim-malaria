"""
updater.py — Updater base class and Target enum.

An Updater is a Starsim connector that runs once per timestep to maintain
per-person (or per-node) intervention state for all agents that have ever
received an intervention, regardless of which distributor gave it to them.

Updaters are framework internals — they are created automatically by ms.Sim
and are invisible to the user.

Auto-registration
-----------------
Each Updater subclass registers itself in ``Updater._possible_updaters`` via
``__init_subclass__`` the moment its class body is executed.  The
``malariasim/updaters/`` package imports every module in the directory
automatically so all subclasses are registered when ``import malariasim``
runs.

build_campaign() window
-----------------------
``ms.Sim`` calls ``Updater.activate()`` immediately before invoking
``build_campaign()`` and ``Updater.deactivate()`` immediately after.
Products and interventions that need to register with an Updater call
``Updater.get(key)`` to retrieve the singleton and call its ``register()``
method.  ``Updater.get()`` raises ``RuntimeError`` when called outside the
active window so construction errors surface at campaign-build time rather
than silently at run time.
"""

import enum
import starsim as ss


class Target(enum.Enum):
    """Where per-intervention state arrays live."""
    DISEASE    = 'disease'
    NODE_STATE = 'node_state'


class Updater(ss.Module):
    """
    Base class for all per-step intervention updaters.

    Class attributes
    ----------------
    _possible_updaters : dict[str, Updater]
        Populated automatically by ``__init_subclass__``.  Maps each
        ``_updater_key`` to the singleton instance of that updater.
    _updater_key : str or None
        Key identifying this updater.  Must be overridden by every concrete
        subclass.
    _active : bool
        True only while ``ms.Sim`` is executing ``build_campaign()``.
        Enforced by ``Updater.get()``.
    """

    _possible_updaters: dict = {}
    _updater_key = None
    _active = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._updater_key is not None:
            Updater._possible_updaters[cls._updater_key] = cls()

    def __init__(self):
        super().__init__()
        self._entries = {}

    # ------------------------------------------------------------------
    # build_campaign() window management (called by ms.Sim)
    # ------------------------------------------------------------------

    @classmethod
    def activate(cls):
        """
        Open the registration window.

        Called by ``ms.Sim`` immediately before ``build_campaign()``.
        """
        cls._active = True

    @classmethod
    def deactivate(cls):
        """
        Close the registration window.

        Called by ``ms.Sim`` immediately after ``build_campaign()``.
        """
        cls._active = False

    @classmethod
    def get(cls, key):
        """
        Return the updater singleton for ``key``.

        Raises ``RuntimeError`` if called outside the ``build_campaign()``
        window, so mis-placed product construction surfaces immediately.

        Parameters
        ----------
        key : str
            The ``_updater_key`` of the desired updater.
        """
        if not cls._active:
            raise RuntimeError(
                f"Updater.get({key!r}) called outside the build_campaign() "
                "window.  Products and interventions that register with an "
                "Updater must be created inside the build_campaign() callback "
                "passed to ms.Sim."
            )
        if key not in cls._possible_updaters:
            raise KeyError(
                f"No Updater registered for key {key!r}.  "
                f"Available keys: {list(cls._possible_updaters)}"
            )
        return cls._possible_updaters[key]

    # ------------------------------------------------------------------
    # Framework hooks
    # ------------------------------------------------------------------

    def clear(self):
        """
        Reset all registered entries.

        Called by ms.Sim before each ``build_campaign()`` so that a
        fresh Sim does not inherit state from a previous one.
        """
        self._entries = {}

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def register(self, *args, **kwargs):
        """
        Record that a new intervention effect needs state arrays.

        Called by products during ``build_campaign()``.  The signature
        varies by updater subclass.
        """
        pass

    def define_states(self):
        """
        Create per-person (or per-node) state arrays for all registered entries.

        The default is a no-op.  Subclasses that inject states into other
        modules (e.g. ``WaningEffectUpdater``) override this and call it from
        their own ``init_pre()`` — before the target module's ``init_pre()``
        runs and connects states to People.
        """
        pass

    def step(self):
        """Run per-step maintenance for all affected agents."""
        raise NotImplementedError

"""
eligibility.py — Eligibility strategy objects for intervention distributors.

An Eligibility object is a callable: ``eligibility(sim)`` → target IDs.
It is passed to an ``InterventionDistributor`` and called each time the
distributor fires to determine the pool of eligible targets before coverage
is applied.

``HumanEligibility`` returns ``ss.uids`` of alive agents within an optional
age range, compatible with ``ss.bernoulli.filter()``.

``NodeEligibility`` returns node IDs from ``NodeState`` (default: all nodes),
wrapped as ``ss.uids`` so they too are compatible with ``ss.bernoulli.filter()``.

Both classes implement ``__call__(sim)``, making them drop-in callables for
``ss.Intervention.check_eligibility()``.
"""

import numpy as np
import starsim as ss


class Eligibility:
    """
    Abstract base class for eligibility strategies.

    An eligibility object is a callable that takes a sim and returns the pool
    of candidate target IDs before coverage sampling is applied by the
    distributor.  Subclasses implement ``__call__(sim)``.
    """

    def __call__(self, sim):
        """
        Return eligible target IDs for this timestep.

        Parameters
        ----------
        sim : ss.Sim

        Returns
        -------
        ss.uids
            IDs of eligible targets (human UIDs or node IDs).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement __call__(sim)."
        )


class HumanEligibility(Eligibility):
    """
    Eligibility strategy for human agents, optionally filtered by age and node.

    Returns ``ss.uids`` of alive agents within ``[min_age, max_age)`` and, when
    ``node_ids`` is supplied, within the specified spatial nodes only.  All
    constraints are optional — omitting one removes that constraint.

    Parameters
    ----------
    min_age : ss.years, optional
        Minimum age (inclusive).  Agents younger than this are excluded.
    max_age : ss.years, optional
        Maximum age (exclusive).  Agents this age or older are excluded.
    node_ids : list of int or None, optional
        Spatial node IDs to include.  When *None* (default) agents in all
        nodes are eligible.

    Examples
    --------
    All alive agents (no age or node filter)::

        ms.HumanEligibility()

    Children aged one to seven years in nodes 2 and 3::

        ms.HumanEligibility(
            min_age=ss.years(1),
            max_age=ss.years(7),
            node_ids=[2, 3],
        )
    """

    def __init__(self, min_age=None, max_age=None, node_ids=None):
        if min_age is not None and min_age.years < 0:
            raise ValueError(
                f"min_age must be >= 0, got {min_age}."
            )
        if max_age is not None and max_age.years < 0:
            raise ValueError(
                f"max_age must be >= 0, got {max_age}."
            )
        if min_age is not None and max_age is not None and min_age.years >= max_age.years:
            raise ValueError(
                f"min_age ({min_age}) must be less than max_age ({max_age})."
            )
        self.min_age  = min_age    # ss.years or None
        self.max_age  = max_age    # ss.years or None
        self.node_ids = None if node_ids is None else np.asarray(node_ids, dtype=int)

    def __call__(self, sim):
        """Return ss.uids of eligible agents filtered by age and node."""
        alive = sim.people.auids
        mask  = np.ones(len(alive), dtype=bool)

        if self.min_age is not None or self.max_age is not None:
            ages = np.asarray(sim.people.age[alive], dtype=float)
            if self.min_age is not None:
                mask &= ages >= self.min_age.years
            if self.max_age is not None:
                mask &= ages <  self.max_age.years

        if self.node_ids is not None:
            disease     = self._find_disease(sim)
            agent_nodes = np.asarray(disease.node_id[alive], dtype=int)
            mask &= np.isin(agent_nodes, self.node_ids)

        if mask.all():
            return alive
        return alive[mask]

    @staticmethod
    def _find_disease(sim):
        """Return the disease module that exposes per-agent node_id."""
        for mod in sim.diseases.values():
            if hasattr(mod, '_supports_intervention_effects'):
                return mod
        raise RuntimeError(
            "HumanEligibility: no disease module with "
            "'_supports_intervention_effects' found. "
            "Ensure MalariaHumanBase (or a subclass) is the human disease module."
        )


class NodeEligibility(Eligibility):
    """
    Eligibility strategy for spatial nodes.

    Returns node IDs from ``NodeState``.  Defaults to all nodes when no
    explicit list is supplied.

    Parameters
    ----------
    node_ids : array-like of int or None
        Explicit list of node IDs to include in the eligible pool.  When
        *None* (default), all nodes in the simulation are eligible.

    Examples
    --------
    All nodes::

        ms.NodeEligibility()

    Specific nodes only::

        ms.NodeEligibility(node_ids=[0, 2])
    """

    def __init__(self, node_ids=None):
        self.node_ids = node_ids

    def __call__(self, sim):
        """Return eligible node IDs as ss.uids."""
        node_state = None
        for mod in sim.connectors:
            if hasattr(mod, '_is_node_state'):
                node_state = mod
                break

        if node_state is None:
            raise RuntimeError(
                "NodeEligibility: no NodeState connector found in sim.connectors. "
                "Ensure a NodeState is registered before using NodeEligibility."
            )

        if self.node_ids is not None:
            ids = np.asarray(self.node_ids, dtype=int)
            valid = np.arange(node_state.grid.n_nodes)
            invalid = ids[~np.isin(ids, valid)]
            if len(invalid):
                raise ValueError(
                    f"NodeEligibility: node_ids {invalid.tolist()} do not exist "
                    f"in NodeState (valid range: 0–{node_state.grid.n_nodes - 1})."
                )
            return ss.uids(ids)

        return ss.uids(np.arange(node_state.grid.n_nodes, dtype=int))

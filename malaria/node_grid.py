"""
NodeGrid: static spatial node configuration.

The number of nodes is inferred from the length of ``populations``.  All
other per-node parameters (labels, emergence, eir, f_indoor) must have the
same length when provided.

Per-node overrides
------------------
emergence, eir, and f_indoor are optional.  When omitted the companion
mosquito module falls back to its own module-level parameter, applying it
uniformly across all nodes.  When provided they must match the node count:

    emergence  — shape (n_nodes, 12)          : monthly emergence per node
    eir        — shape (n_nodes,) or (n_nodes, 12) : per-node EIR
    f_indoor   — shape (n_nodes,)             : indoor bite fraction per node
"""

import numpy as np


class NodeGrid:
    """
    Static spatial node configuration.

    The number of nodes is inferred from ``populations``.  When
    ``populations`` is omitted a single node is created.

    Parameters
    ----------
    populations : array-like of shape (n_nodes,) or None
        Initial population size for each node (absolute counts, not fractions).
        Used to compute birth-assignment probabilities so that node populations
        remain stable over time.  Values are normalised internally; only the
        ratios matter.  When *None* a single node is created.

    labels : list of str or None
        Human-readable name for each node.
        Defaults to ``['Node 0', 'Node 1', ...]``.

    emergence : array-like of shape (n_nodes, 12) or None
        Monthly adult-female mosquito emergence rate (new females per human
        per day) for each node.  When *None* the mosquito-agents module uses
        its own ``emergence`` parameter uniformly across all nodes.

    eir : array-like of shape (n_nodes,) or (n_nodes, 12) or None
        Per-node EIR for MosqitoesEirOnly.

        - Shape ``(n_nodes,)``    : constant annual EIR per node.
        - Shape ``(n_nodes, 12)`` : monthly EIR profile per node.
        - *None*                  : use the module-level ``eir`` parameter
          uniformly across all nodes.

    f_indoor : array-like of shape (n_nodes,) or None
        Indoor bite fraction for each node.  When *None* the mosquito module
        uses its own ``f_indoor`` parameter uniformly across all nodes.

    Examples
    --------
    Single node (default)::

        grid = NodeGrid()

    Three nodes with different population sizes::

        grid = NodeGrid(populations=[5_000, 3_000, 2_000])

    Two nodes with per-node EIR, emergence, and indoor fraction::

        grid = NodeGrid(
            populations = [4_000, 6_000],
            labels      = ['Urban', 'Rural'],
            eir = [
                [ 5,  5,  8, 10, 14, 18, 20, 14,  8,  6,  5,  5],
                [ 1,  1,  2,  3,  5,  8, 10,  6,  3,  2,  1,  1],
            ],
            emergence = [
                [0.008, 0.008, 0.015, 0.030, 0.045, 0.060, 0.060, 0.045, 0.030, 0.015, 0.008, 0.008],
                [0.002, 0.002, 0.005, 0.010, 0.018, 0.025, 0.025, 0.018, 0.010, 0.005, 0.002, 0.002],
            ],
            f_indoor = [0.80, 0.55],
        )
    """

    def __init__(
        self,
        populations = None,
        labels      = None,
        emergence   = None,
        eir         = None,
        f_indoor    = None,
    ):
        # --- n_nodes from populations ---
        if populations is None:
            self.n_nodes          = 1
            self.pop_fractions    = np.ones(1)
            self.total_population = None
        else:
            pops = np.asarray(populations, dtype=float)
            if pops.ndim != 1 or len(pops) == 0:
                raise ValueError('populations must be a non-empty 1-D array')
            if np.any(pops <= 0):
                raise ValueError('All populations must be positive')
            self.n_nodes          = len(pops)
            self.pop_fractions    = pops / pops.sum()
            self.total_population = int(pops.sum())

        # --- labels ---
        if labels is None:
            self.labels = [f'Node {i}' for i in range(self.n_nodes)]
        else:
            self.labels = list(labels)
            if len(self.labels) != self.n_nodes:
                raise ValueError(
                    f'len(labels)={len(self.labels)} must equal '
                    f'n_nodes={self.n_nodes}'
                )

        # --- emergence ---
        if emergence is None:
            self.emergence = None
        else:
            self.emergence = np.asarray(emergence, dtype=float)
            if self.emergence.shape != (self.n_nodes, 12):
                raise ValueError(
                    f'emergence must have shape ({self.n_nodes}, 12), '
                    f'got {self.emergence.shape}'
                )
            if np.any(self.emergence < 0):
                raise ValueError('All emergence values must be non-negative')

        # --- eir ---
        if eir is None:
            self.eir = None
        else:
            arr = np.asarray(eir, dtype=float)
            if arr.shape == (self.n_nodes,):
                self.eir = arr
            elif arr.shape == (self.n_nodes, 12):
                self.eir = arr
            else:
                raise ValueError(
                    f'eir must have shape ({self.n_nodes},) or '
                    f'({self.n_nodes}, 12), got {arr.shape}'
                )
            if np.any(self.eir < 0):
                raise ValueError('All eir values must be non-negative')

        # --- f_indoor ---
        if f_indoor is None:
            self.f_indoor = None
        else:
            self.f_indoor = np.asarray(f_indoor, dtype=float)
            if self.f_indoor.shape != (self.n_nodes,):
                raise ValueError(
                    f'f_indoor must have shape ({self.n_nodes},), '
                    f'got {self.f_indoor.shape}'
                )
            if np.any((self.f_indoor < 0) | (self.f_indoor > 1)):
                raise ValueError('All f_indoor values must be in [0, 1]')
        return

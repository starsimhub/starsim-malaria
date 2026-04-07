"""
NodeGrid: static spatial node configuration.

The number of nodes is inferred from the length of ``populations``.  All
other per-node parameters (labels, eir, f_indoor) are optional and must
match the node count when provided.
"""

import numpy as np


class NodeGrid:
    """
    Static spatial node configuration.

    Parameters
    ----------
    populations : array-like of shape (n_nodes,) or None
        Initial population size for each node. Values are normalised
        internally to compute birth-assignment probabilities.
        When *None* a single node is created.
    labels : list of str or None
        Human-readable name for each node.
    eir : array-like of shape (n_nodes,) or (n_nodes, 12) or None
        Per-node EIR for MosquitoesEirOnly (LASER ``eir_month_XX`` units: per-timestep
        rate after monthly interpolation).
    f_indoor : array-like of shape (n_nodes,) or None
        Indoor bite fraction for each node.
    """

    def __init__(self, populations=None, labels=None, eir=None, f_indoor=None):
        if populations is None:
            self.n_nodes = 1
            self.pop_fractions = np.ones(1)
            self.total_population = None
        else:
            pops = np.asarray(populations, dtype=float)
            if pops.ndim != 1 or len(pops) == 0:
                raise ValueError('populations must be a non-empty 1-D array')
            if np.any(pops <= 0):
                raise ValueError('All populations must be positive')
            self.n_nodes = len(pops)
            self.pop_fractions = pops / pops.sum()
            self.total_population = int(pops.sum())

        if labels is None:
            self.labels = [f'Node {i}' for i in range(self.n_nodes)]
        else:
            self.labels = list(labels)
            if len(self.labels) != self.n_nodes:
                raise ValueError(f'len(labels)={len(self.labels)} must equal n_nodes={self.n_nodes}')

        if eir is None:
            self.eir = None
        else:
            arr = np.asarray(eir, dtype=float)
            if arr.shape not in [(self.n_nodes,), (self.n_nodes, 12)]:
                raise ValueError(f'eir must have shape ({self.n_nodes},) or ({self.n_nodes}, 12), got {arr.shape}')
            if np.any(arr < 0):
                raise ValueError('All eir values must be non-negative')
            self.eir = arr

        if f_indoor is None:
            self.f_indoor = None
        else:
            self.f_indoor = np.asarray(f_indoor, dtype=float)
            if self.f_indoor.shape != (self.n_nodes,):
                raise ValueError(f'f_indoor must have shape ({self.n_nodes},), got {self.f_indoor.shape}')
            if np.any((self.f_indoor < 0) | (self.f_indoor > 1)):
                raise ValueError('All f_indoor values must be in [0, 1]')
        return

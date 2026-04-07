"""
NodeState: dynamic per-node communication bus for malaria simulation.

Registered as the first connector so it sits between the human disease step
and the mosquito connector step each timestep:

    1. Human disease step  — reads eir (written last step), writes p_inf
    2. NodeState.step()    — no-op ordering anchor
    3. Mosquito step       — reads p_inf, writes eir for the next step
"""

import numpy as np
import starsim as ss

from .node_grid import NodeGrid


class NodeState(ss.Module):
    """
    Dynamic per-node communication bus.

    Parameters
    ----------
    grid : NodeGrid
        The static node configuration.
    """

    _is_node_state = True  # duck-type marker for module discovery

    def __init__(self, grid: NodeGrid):
        super().__init__()
        self.grid = grid
        n = grid.n_nodes

        # Written by mosquito module each step; read by human next step
        self.eir_indoor = np.zeros(n)
        self.eir_outdoor = np.zeros(n)

        # Written by human module each step; read by mosquito same step
        self.p_inf_indoor = np.zeros(n)
        self.p_inf_outdoor = np.zeros(n)
        self.n_humans = np.zeros(n, dtype=int)
        return

    def step(self):
        """Ordering anchor — no state changes needed."""
        return

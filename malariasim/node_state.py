"""
NodeState: dynamic per-node communication bus for malaria simulation.

NodeState is registered as the first connector so that it runs in the slot
between the human disease step and the mosquito connector step each timestep:

    1. Human disease step  — reads eir_indoor/outdoor (written last step),
                             writes p_inf_indoor/outdoor and n_humans.
    2. NodeState.step()    — no-op placeholder (ordering anchor).
    3. Mosquito step       — reads p_inf_indoor/outdoor and n_humans,
                             writes eir_indoor/outdoor for the next step.

Because diseases run before connectors, there is a one-step lag on the EIR
seen by the human module.  This is acceptable for daily timesteps.

All fields are plain numpy arrays indexed by node ID.  They are set directly
by whichever module owns them (single-writer); there is no accumulation logic.
"""

import numpy as np
import starsim as ss

from .node_grid import NodeGrid


class NodeState(ss.Module):
    """
    Dynamic per-node communication bus.

    Holds the per-node fields that human and mosquito modules share each
    timestep.  Register this as the **first** connector so that it occupies
    the correct position in the execution loop.

    Parameters
    ----------
    grid : NodeGrid
        The static node configuration.  NodeState keeps a reference so that
        every module that holds a NodeState pointer can also access grid
        metadata (n_nodes, labels, pop_fractions, emergence, eir, f_indoor).
    """

    _is_node_state = True   # duck-type marker for module discovery

    def __init__(self, grid: NodeGrid):
        super().__init__()
        self.grid = grid
        n = grid.n_nodes

        # Written by the mosquito module each step; read by human next step.
        self.eir_indoor    = np.zeros(n)
        self.eir_outdoor   = np.zeros(n)

        # Written by the human module each step; read by mosquito same step.
        self.p_inf_indoor  = np.zeros(n)
        self.p_inf_outdoor = np.zeros(n)
        self.n_humans      = np.zeros(n, dtype=int)

        # Written by BednetUpdater each step (via disease.aggregate_itn_to_node_state());
        # read by mosquito modules same step.
        # Both default to 0.0 (no ITN effect) and are reset each step before BednetUpdater runs.
        # itn_biting_reduction : fraction of bite attempts blocked per node ∈ [0,1]
        # itn_kill_prob        : kill probability per bite attempt ∈ [0,1]
        #                        — mosquito module multiplies by a*dt to get per-step kill fraction
        self.itn_biting_reduction = np.zeros(n)
        self.itn_kill_prob        = np.zeros(n)
        return

    def step(self):
        """Reset ITN fields and serve as ordering anchor between human and mosquito steps."""
        self.itn_biting_reduction[:] = 0.0
        self.itn_kill_prob[:]        = 0.0
        return

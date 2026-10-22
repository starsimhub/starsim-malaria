""" Arithmetic only (no Starsim): how well does a 5-year-group ODE hold malariasim's pyramid over 20 years under two aging rules?
1/width (v4's rule for a given split) against density-at-the-edge / number-in-group (v4's rule for a named population), with the same death rates by group """
import numpy as np
from common import *
mu, n, br, max_age = compute_mortality_schedule()
fr = np.array(AGE_BIN_FRACTIONS)/np.sum(AGE_BIN_FRACTIONS)
e = np.array(AGE_BIN_EDGES[:-1])
mu5 = np.array([np.average(mu[lo:lo+5], weights=n[lo:lo+5]) for lo in e])
dens = n/n.sum() # density per year of age
def run(aging):
    N = fr.copy(); dt = 7/365
    for _ in range(int(20/dt)):
        out = aging*N; out[-1] = 0
        dN = -mu5*N - out + np.r_[br*N.sum(), out[:-1]]
        N = N + dt*dN
    return N
a_width = np.full(20, 1/5.)
a_edge  = np.array([dens[min(lo+5, 99)-1]/fr[k] if k < 19 else 0 for k, lo in enumerate(e)]) # density just below the upper edge over the share in the group
for label, a in [('1/width', a_width), ('density at edge / group', a_edge)]:
    N = run(a); print(f'{label:24s}: N after 20 y {N.sum():.4f} of 1; pyramid TVD from start {tvd(N/N.sum(), fr):.4f}; under 5 {N[0]/N.sum():.3f} (start {fr[0]:.3f})')

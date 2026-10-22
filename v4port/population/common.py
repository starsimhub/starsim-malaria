""" Shared settings for the population port: the same grid, EIRs, pyramid and run length on v3 and v4 """
import os
import numpy as np

POPS      = [2000, 1500, 1500]                    # three nodes (Urban, Peri-urban, Rural)
LABELS    = ['urban', 'peri', 'rural']
ANNUAL    = [4.0, 6.0, 8.0]                        # target annual EIR per node (as run_malariasim.py)
F_INDOOR  = [0.80, 0.70, 0.55]
START, STOP = 2000, 2020
DT_DAYS   = int(os.environ.get('DT_DAYS', 7)) # weekly by default; DT_DAYS=1 for malariasim's daily step
TAG       = '' if DT_DAYS == 7 else f'_dt{DT_DAYS}'
SEEDS     = list(range(6))

URBAN_EIR      = [1,  2,  3,  5,  8, 10,  6,  3,  2,  1,  1,  1]
PERI_URBAN_EIR = [2,  2,  5,  6,  8, 10, 12,  8,  5,  3,  2,  2]
RURAL_EIR      = [5,  5,  5,  8, 10, 14, 18, 20, 14,  8,  6,  5]
SHAPES = [URBAN_EIR, PERI_URBAN_EIR, RURAL_EIR]

def node_eir_profiles():
    """ Monthly EIR (bites/person/year) per node, mean = ANNUAL (as ms.eir_to_params with a seasonal profile) """
    out = []
    for target, shape in zip(ANNUAL, SHAPES):
        s = np.asarray(shape, float)
        out.append(list(target * s / s.mean()))
    return np.array(out)

AGE_BIN_EDGES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
AGE_BIN_FRACTIONS = [0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.006, 0.0036, 0.0022, 0.0013]

def compute_mortality_schedule(age_bin_edges=AGE_BIN_EDGES, age_bin_fractions=AGE_BIN_FRACTIONS):
    """ Copy of malariasim.demographics.compute_mortality_schedule (so v4 scripts need not import malariasim) """
    edges = np.asarray(age_bin_edges, dtype=float)
    fracs = np.asarray(age_bin_fractions, dtype=float)
    fracs = fracs/fracs.sum()
    max_age   = int(edges[-1])
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    widths    = np.diff(edges)
    densities = fracs / widths
    ages_int = np.arange(max_age, dtype=float)
    n = np.exp(np.interp(ages_int, midpoints, np.log(densities)))
    mu = np.append(np.log(n[:-1] / n[1:]), 2.0)
    p_die_1yr = 1.0 - np.exp(-mu)
    cap_rate  = np.exp(-mu[-1]) * n[-1] / n.sum()
    birth_rate = float(np.dot(p_die_1yr, n) / n.sum() + cap_rate)
    return mu, n, birth_rate, max_age

def pyramid(ages, edges=AGE_BIN_EDGES):
    """ Share of people in each 5-year bin """
    c = np.bincount(np.minimum((np.asarray(ages)/5).astype(int), len(edges)-2), minlength=len(edges)-1)
    return c/c.sum()

def tvd(p, q):
    """ Total variation distance between two pyramids """
    return 0.5*np.abs(np.asarray(p) - np.asarray(q)).sum()

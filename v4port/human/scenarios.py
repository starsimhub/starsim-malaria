""" The scenarios shared by the v3 reference (run_v3.py) and the v4 port (run_v4.py); no Starsim import here """
import numpy as np

URBAN_EIR      = [1,  2,  3,  5,  8, 10,  6,  3,  2,  1,  1,  1] # Shapes from malariasim.defaults (normalized to mean 1 below)
PERI_URBAN_EIR = [2,  2,  5,  6,  8, 10, 12,  8,  5,  3,  2,  2]
RURAL_EIR      = [5,  5,  5,  8, 10, 14, 18, 20, 14,  8,  6,  5]

def seasonal(annual, shape):
    w = np.asarray(shape, float)
    return list(w/w.mean()*annual)

b       = 0.4   # CommonParams.prob_human_infected_on_bite
c       = 0.3   # CommonParams.prob_mosquito_infected_on_bite
dur_inf = 200.0 # days, mean (and std, for agents) of the infectious period
dur_exp = (12.0, 2.0) # days, mean and sd of the SEIS incubation period
years   = 5

scenarios = dict(
    A = dict(label='1 node, constant EIR 10/yr', pops=[10_000], eir=[[10.0]*12]),
    B = dict(label='3 nodes, seasonal EIR 5, 20, 50/yr', pops=[3_000, 3_000, 4_000], eir=[seasonal(5, URBAN_EIR), seasonal(20, PERI_URBAN_EIR), seasonal(50, RURAL_EIR)]),
    C = dict(label='B + biting heterogeneity (gamma shape 1) and age biting [0,5,15] -> [0.5,0.8,1]', pops=[3_000, 3_000, 4_000], eir=[seasonal(5, URBAN_EIR), seasonal(20, PERI_URBAN_EIR), seasonal(50, RURAL_EIR)],
             shape=1.0, age_ages=[0, 5, 15], age_vals=[0.5, 0.8, 1.0]),
)

def save(path, out):
    """ Save {key: dict(I=, E=, N=, time=)} compactly (float32, compressed) """
    arrs = {}
    for k,r in out.items():
        for f in ['I', 'E', 'N']: arrs[f'{k}|{f}'] = np.asarray(r[f], dtype=np.float32)
        arrs[f'{k}|time'] = np.array(r['time'])
    np.savez_compressed(path, **arrs)

def load(path):
    """ Load what save() wrote, as {key: dict(I=, E=, N=, time=)} """
    z = np.load(path)
    out = {}
    for name in z.files:
        k, f = name.rsplit('|', 1)
        out.setdefault(k, {})[f] = z[name].tolist() if f != 'time' else float(z[name])
    return out

""" Shared scenario constants for the v3 and v4 intervention comparisons (no Starsim import) """
POPS     = [2000, 2000]     # two nodes
EIR      = [2.0, 4.0]       # annual EIR per node (constant), EIR-only mosquito
F_INDOOR = 0.7
DUR_INF  = 200              # days, mean infectious period
T_INTV   = 365              # day of the campaign (2001-01-01)
DAYS     = 730              # 2000-01-01 to 2002-01-01
SEEDS    = [1, 2, 3, 4, 5]
REPORT   = [364, 372, 395, 455, 545, 729]  # days to report n_infected

VAX   = dict(eff=0.8, box=180, tau=365, cov=0.8)
DRUG  = dict(mean=1.5, cov=0.9)
ITN   = dict(rep=(0.5, 0, 730), block=(0.9, 365, 365), kill=(0.6, 180, 365), usage=(0.8, 365, 365), cov=0.7)  # (initial efficacy, box days, tau days)
SEED  = dict(day=30, cov=0.5, node=1)   # outbreak on node 1 with no EIR
BOOST = dict(day1=200, day2=500)        # two vaccine rounds, to test re-dosing

SCENARIOS = ['base', 'vax', 'vax_node0', 'mda', 'itn', 'seed', 'boost']

# Ross-Macdonald comparison (human SIS + mosquito ODE in malariasim; one declared disease with mosquito counts in v4)
RM = dict(pop=4000, em=0.01, a=0.3, b=1.0, c=1.0, life=10, eip=12, seed_day=10, seed_cov=0.2)
RM_SCENARIOS = ['base', 'itn', 'itn_nokill', 'vax']
RM_REPORT = [364, 395, 455, 545, 729]

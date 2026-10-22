"""
Head-to-head wall-clock benchmark: malariasim on Starsim v3 or v4 (with the shim), and LASER (malaria_model).

Reconstructed from the disassembly of malaria_benchmarks/__pycache__/run_benchmarks.cpython-313.pyc (the source
is not in the repo): same constants, same build_starsim/build_laser, crn=False, outbreak only, serial runs,
numba warmed up first. Scenarios a-d: 3 nodes (6000/3000/1000 humans, target EIR 4/6/8, f_indoor 0.7);
e: single node, human seis + mosquito agents, (humans, mosquitoes) pairs; f: sis/seis + mosquito ODE at 10k/30k/100k.

Usage: python bench.py <framework: starsim|laser> <scenario ids, e.g. abcd,e10,f> <reps> <years> <out.json>
Timings are run() only (as in the original), best of `reps`.
"""
import sys, os, time, json
import numpy as np
import pandas as pd
import sciris as sc

FW = sys.argv[1]
if FW == 'starsim':
    import starsim as ss
    print(ss.__version__, ss.__file__, flush=True)
    if ss.__version__.startswith('4'):
        import malshim # noqa
    ss.options(crn=False)
import malariasim as ms # Also needed for laser: eir_to_params and the age data, as in the original
if FW == 'laser':
    import starsim as ss # Only for ss.days in nothing; malariasim imports it anyway
    from malaria_model.defaults import default_params
    from malaria_model.sim import Sim as LaserSim
    from malaria_model.interventions.campaign import Campaign as LaserCampaign
    from malaria_model.interventions.distributor import InterventionDistributorScheduled as LaserDistributor
    from malaria_model.interventions.outbreak import MalariaOutbreak
    print('laser', flush=True)

START, SEED, OUTBREAK_COV, OUTBREAK_DAY = 2000, 0, 0.15, 30
POPS_AD, LABELS_AD, TARGET_EIR_AD, F_INDOOR = [6000, 3000, 1000], ['Node-A', 'Node-B', 'Node-C'], [4.0, 6.0, 8.0], 0.7
E_PAIRS = {'e5': (5000, 100000), 'e10': (10000, 200000), 'e20': (20000, 400000)}
F_TARGET_EIR, MOS_LIFE_DAYS = 6.0, 10.0
MOS_NAME = dict(eir=dict(starsim='eir_only', laser='eir_only'), ode=dict(starsim='ode', laser='ode_linear'), agents=dict(starsim='agents', laser='agents'))
SCENARIOS_AD = dict(a=('human-ode + mos-eir', 'ode', 'eir'), b=('human-ode + mos-ode', 'ode', 'ode'), c=('human-seis + mos-ode', 'seis', 'ode'), d=('human-ode + mos-agents', 'ode', 'agents'))
COMMON_PARS = ms.CommonParams()
AGE_BIN_EDGES = ms.AGE_BIN_EDGES
AGE_DATA = sc.dataframe(ms.make_age_data(edges=ms.AGE_BIN_EDGES, fractions=ms.AGE_BIN_FRACTIONS))

def epi_profiles(target_annual_eir):
    eir, emergence = ms.eir_to_params(target_annual_eir, common_pars=COMMON_PARS, seasonal_profile=None)
    return list(eir), list(emergence)

def _starsim_campaign(sim, eir_only=False):
    c = ms.Campaign(sim)
    c.add(ms.InterventionDistributorScheduled(name='Outbreak', products=ms.Outbreak(), start=START + OUTBREAK_DAY/365.0, coverage=OUTBREAK_COV))
    return c

def build_starsim(human, mos_key, populations, labels, eir_profiles, emergence_profiles, years):
    nodes = ms.NodeGrid(populations=populations, labels=labels, eir=eir_profiles, emergence=emergence_profiles, f_indoor=[F_INDOOR]*len(populations))
    return ms.Sim(age_data=AGE_DATA, age_bin_edges=AGE_BIN_EDGES, start=START, stop=START+years, dt=ss.days(1), rand_seed=SEED,
                  demographics=ms.make_steady_state_demographics(), common_pars=COMMON_PARS, human=human, mosquito=MOS_NAME[mos_key]['starsim'],
                  nodes=nodes, build_campaign=_starsim_campaign, verbose=0, use_aging=False) # people_results=False (in the original) fails here: ss.Births reads results.n_alive; the original used malariasim's FastBirths, not in this commit

def starsim_mosquitoes(sim):
    for conn in sim.connectors.values():
        if isinstance(conn, ms.MosquitoesAdultFemaleAgentsOnly) and conn.alive is not None:
            return int(conn.alive.sum())
    return 0

def _laser_campaign(sim, eir_only=False):
    c = LaserCampaign(sim)
    if not eir_only:
        c.add(LaserDistributor(name='Outbreak', products=MalariaOutbreak(), start_tick=OUTBREAK_DAY, coverage=OUTBREAK_COV))
    return c

def build_laser(human, mos_key, populations, labels, eir_profiles, emergence_profiles, years):
    params = default_params()
    params.nticks = int(round(years*365))
    params.seed = SEED
    emerge_scale = float(params.mos_emergence_scale)
    mu = 1.0/MOS_LIFE_DAYS
    n_nodes = len(populations)
    scenario = pd.DataFrame(dict(nodeid=list(range(n_nodes)), label=list(labels), population=list(populations)))
    all_emergence = []
    for node in range(n_nodes):
        eir_m, eme_m = eir_profiles[node], emergence_profiles[node]
        for m in range(12):
            scenario.at[node, f'eir_month_{m+1:02d}'] = eir_m[m]/365.0
            scenario.at[node, f'mos_emergence_month_{m+1:02d}'] = eme_m[m]*1000.0/emerge_scale
        all_emergence.extend(eme_m)
    params.mos_init_females_per_1000 = float(np.mean(all_emergence))/mu*1000.0
    return LaserSim(label=f'{human}+{mos_key}', scenario=scenario, params=params, human=human, mosquito=MOS_NAME[mos_key]['laser'], build_campaign=_laser_campaign)

def laser_mosquitoes(sim):
    mos = getattr(sim.model, 'mosquitoes_adults', None)
    if mos is None:
        return 0
    return int((np.asarray(mos.mos_alive) == 1).sum())

BUILD = dict(starsim=(build_starsim, starsim_mosquitoes), laser=(build_laser, laser_mosquitoes))[FW]

def time_run(reps, n_humans, **kwargs):
    build_fn, mos_fn = BUILD
    times, build_times, n_m = [], [], 0
    for _ in range(reps):
        t0 = time.perf_counter()
        sim = build_fn(**kwargs)
        build_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        sim.run()
        times.append(time.perf_counter() - t0)
        n_m = mos_fn(sim)
        del sim
    return dict(time_s=float(min(times)), times=[float(t) for t in times], build_s=float(min(build_times)), n_humans=int(n_humans), n_mosquitoes=int(n_m))

def warmup():
    eir, eme = epi_profiles(6.0)
    for human, mos in [('ode', 'agents'), ('sis', 'ode'), ('seis', 'ode')]:
        try:
            BUILD[0](human=human, mos_key=mos, populations=[200], labels=['w'], eir_profiles=[eir], emergence_profiles=[eme], years=0.1).run()
        except Exception as e:
            print(f'  ({FW} warmup {human}+{mos} skipped: {e})', flush=True)

if __name__ == '__main__':
    ids, reps, years, out = sys.argv[2].split(','), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    warmup()
    prof = [epi_profiles(e) for e in TARGET_EIR_AD]
    results = {}
    for sid in ids:
        if sid in SCENARIOS_AD:
            label, human, mos = SCENARIOS_AD[sid]
            r = time_run(reps, sum(POPS_AD), human=human, mos_key=mos, populations=POPS_AD, labels=LABELS_AD,
                         eir_profiles=[p[0] for p in prof], emergence_profiles=[p[1] for p in prof], years=years)
        elif sid in E_PAIRS:
            n_h, n_m = E_PAIRS[sid]
            emergence = n_m/n_h/MOS_LIFE_DAYS
            r = time_run(reps, n_h, human='seis', mos_key='agents', populations=[n_h], labels=['Node-0'],
                         eir_profiles=[epi_profiles(6.0)[0]], emergence_profiles=[[emergence]*12], years=years)
            label = f'seis+agents {n_h}/{n_m}'
        elif sid.startswith('f'): # e.g. fsis10, fseis100
            human = 'seis' if sid.startswith('fseis') else 'sis'
            n_h = int(sid.split(human)[1])*1000
            r = time_run(reps, n_h, human=human, mos_key='ode', populations=[n_h], labels=['Node-0'],
                         eir_profiles=[epi_profiles(F_TARGET_EIR)[0]], emergence_profiles=[epi_profiles(F_TARGET_EIR)[1]], years=years)
            label = f'{human}+ode {n_h}'
        r['label'] = label
        results[sid] = r
        print(f'{FW} {sid} {label}: {r["time_s"]:.3f} s  (all {np.round(r["times"], 3).tolist()}; build {r["build_s"]:.3f} s; M={r["n_mosquitoes"]})', flush=True)
    sc.savejson(out, results)

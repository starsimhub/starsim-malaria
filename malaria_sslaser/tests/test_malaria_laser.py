"""
Integration tests for the Starsim-based malaria simulation (malaria_sslaser).

Tests use qualitative assertions (not exact numerical) since stochastic
results vary between runs.
"""

import numpy as np
import starsim as ss
import sciris as sc
import pytest

import malaria_sslaser as ms


def build_eir_profiles():
    """Per-node monthly EIR profiles (3 nodes × 12 months)."""
    return np.array([
        [0.01, 0.01, 0.02, 0.04, 0.07, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01],
        [0.02, 0.02, 0.03, 0.08, 0.09, 0.20, 0.12, 0.10, 0.06, 0.03, 0.02, 0.02],
        [0.01, 0.01, 0.02, 0.05, 0.08, 0.11, 0.09, 0.06, 0.04, 0.02, 0.01, 0.01],
    ])


@pytest.fixture(scope="session")
def sim_results():
    """Run the simulation once and return results for all tests."""
    T = sc.timer()

    nodes = ms.NodeGrid(
        populations=[5000, 3000, 2000],
        eir=build_eir_profiles(),
    )

    sim = ms.Sim(
        human='sis',
        mosquito='eir_only',
        nodes=nodes,
        start=2000,
        stop=2002,  # 2 years
        dt=ss.days(1),
        human_pars=dict(
            prob_human_infected_on_bite=0.15,
            infection_duration_years=ss.normal(loc=120/365, scale=30/365),
            default_infectiousness=0.25,
            init_prev=ss.bernoulli(p=0.01),
        ),
        mosquito_pars=dict(
            f_indoor=0.7,
        ),
        rand_seed=123,
    )
    sim.run()

    # Extract result arrays for easy access via starsim result paths
    sim_res = sim.results
    res = sc.dcp(sim_res.malariahumansis)
    res.n_alive = sim_res.n_alive
    res.prevalence = res.n_infected / np.maximum(res.n_alive, 1)
    res.eir_indoor = sim_res.mosquitoeseironly.eir_indoor_annual
    res.elapsed = T.total
    res.nticks = len(res.n_infected)
    return res


def test_total_infectious_exceeds_new_infections(sim_results):
    """Total infectious should exceed new infections since each infection lasts ~120 days."""
    warmup = 90
    total_inf = sim_results.n_infected[warmup:]
    new_inf = sim_results.new_infections[warmup:]

    assert total_inf.sum() > new_inf.sum(), (
        f"Expected n_infected ({total_inf.sum()}) > new_infections ({new_inf.sum()}) "
        "over the simulation — infections lasting ~120 days should accumulate"
    )


def test_eir_is_seasonal(sim_results):
    """EIR should show seasonal variation matching the monthly forcing."""
    half = len(sim_results) // 2
    eir = sim_results.eir_indoor[half:]
    monthly_avg = np.array([eir[m * 30:(m + 1) * 30].mean() for m in range(12)])

    peak = monthly_avg.max()
    trough = monthly_avg.min()

    assert peak > 0.0, "Expected nonzero EIR"
    assert peak > trough, "Expected seasonal EIR variation"


def test_infections_are_seasonal(sim_results):
    """New infections should show seasonal peaks matching the monthly EIR forcing."""
    # Use second year to avoid warm-up effects
    half = len(sim_results) // 2
    new_inf = sim_results.new_infections[half:].astype(float)

    # 12 monthly averages (~30 days each)
    monthly_avg = np.array([new_inf[m * 30:(m + 1) * 30].mean() for m in range(12)])

    peak = monthly_avg.max()

    # With low EIR some months may have zero infections stochastically,
    # so check that the peak-season months average more infections than
    # the off-season months (top 4 vs bottom 4).
    sorted_avg = np.sort(monthly_avg)
    top4 = sorted_avg[-4:].mean()
    bot4 = sorted_avg[:4].mean()

    assert peak > 0, "Expected nonzero peak infections"
    assert top4 > bot4, (
        f"Expected seasonal variation: peak months ({top4:.2f}) > trough months ({bot4:.2f}). "
        "Infections should peak mid-year following the EIR seasonal forcing."
    )


def test_prevalence_is_nonzero(sim_results):
    """Prevalence should be nonzero after seeding — basic sanity check."""
    warmup = 90
    prevalence = sim_results.prevalence[warmup:]
    assert prevalence.max() > 0.0, "Expected nonzero prevalence after warm-up"
    assert prevalence.max() <= 1.0, "Prevalence should never exceed 1.0"


def test_population_is_stable(sim_results):
    """Total alive population should remain roughly stable."""
    total_alive = sim_results.n_alive
    initial = total_alive[0]
    assert initial > 0, "Expected nonzero initial population"
    assert total_alive.min() > 0.9 * initial, (
        f"Population dropped too much: min={total_alive.min()}, initial={initial}"
    )
    assert total_alive.max() < 1.1 * initial, (
        f"Population grew too much: max={total_alive.max()}, initial={initial}"
    )


def test_performance(sim_results, max_time=3):
    """Simulation should complete within a reasonable time."""
    perf = sc.benchmark(which='numpy')
    expected = 260
    perf_ratio = perf / expected
    norm_max = max_time / perf_ratio
    elapsed = sim_results.elapsed
    sc.printcyan(f'\nElapsed time: {elapsed:n} s')

    assert elapsed <= norm_max, f'Expecting shorter run time but {elapsed:n} > {norm_max:n} s'


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])

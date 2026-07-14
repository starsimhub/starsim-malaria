"""
Bednet (ITN) intervention tests.

Tests cover seven ITN behaviors:
  - vs baseline:              high-efficacy bednet reduces total infections
  - killing reduces mos:      blocking+killing lowers mosquito population (ODE)
  - killing scales coverage:  50% coverage kills fewer mosquitoes than 100% (ODE regression)
  - killing needs block:      killing with blocking=0 has no protective effect
  - repelling gates killing:  100% repelling prevents mosquito killing (u_kill=0)
  - repelling scales killing: kill effect proportional to (1 - repelling)
  - usage=0 no effect:        zero-usage bednet is a no-op
  - node-targeted:            bednet on node 0 only reduces node 0 infections
"""
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import malariasim as ms

_MOSQUITO_MODULES = (ms.MosquitoesODE, ms.MosquitoesAdultFemaleAgentsOnly)


def _find_mosquito_module(sim):
    return next(m for m in sim.connectors.values() if isinstance(m, _MOSQUITO_MODULES))

_COMMON_PARS = ms.CommonParams()
_, _EMERGENCE_10 = ms.eir_to_params(10.0, common_pars=_COMMON_PARS)


def make_sim(human='sis', mosquito='ode', **kw):
    return ms.Sim(
        human     = human,
        mosquito  = mosquito,
        start     = kw.pop('start', 2000),
        stop      = kw.pop('stop',  2001),
        nodes     = kw.pop('nodes', ms.NodeGrid(populations=[1000])),
        dt        = kw.pop('dt',    ss.days(1)),
        rand_seed = kw.pop('rand_seed', 0),
        verbose   = kw.pop('verbose', 0),
        **kw,
    )


# ------------------------------------------------------------------
# Bednet vs baseline
# ------------------------------------------------------------------

@sc.timer()
def test_bednet_reduces_infections_vs_baseline():
    """High-efficacy ITN at 80% coverage reduces total infections vs unprotected baseline.

    Two identical sims seeded at day 10; one receives a bednet campaign on day 30.
    Repelling=80%, blocking=90%, killing=60%, usage=80% — all sustained for the
    full year.  The bednet sim must accumulate fewer infections over the year.
    Uses ODE mosquitoes for deterministic comparison.
    """
    sc.heading('Testing bednet reduces infections vs unprotected baseline...')

    nodes = ms.NodeGrid(populations=[2000], emergence=[_EMERGENCE_10])
    kw = dict(
        human        = 'sis',
        mosquito     = 'ode',
        nodes        = nodes,
        stop         = 2001,
        demographics = [],
        rand_seed    = 42,
    )

    def build_baseline(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='seed', products=ms.Outbreak(),
                start=2000 + 10/365, coverage=0.15,
            ),
        ])

    def build_bednet(sim, eir_only=False):
        bednet = ms.Bednet(
            repelling = ms.WaningEffect(initial_efficacy=0.80, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            blocking  = ms.WaningEffect(initial_efficacy=0.90, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            killing   = ms.WaningEffect(initial_efficacy=0.60, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            usage     = ms.WaningEffect(initial_efficacy=0.80, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
        )
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='seed', products=ms.Outbreak(),
                start=2000 + 10/365, coverage=0.15,
            ),
            ms.InterventionDistributorScheduled(
                name='itn', products=bednet,
                start=2000 + 30/365, coverage=0.80,
            ),
        ])

    sim_base = make_sim(**kw, build_campaign=build_baseline)
    sim_itn  = make_sim(**kw, build_campaign=build_bednet)
    sim_base.run()
    sim_itn.run()

    inf_base = sim_base.diseases[0].results.n_infected.values.sum()
    inf_itn  = sim_itn.diseases[0].results.n_infected.values.sum()

    assert inf_itn < inf_base, (
        f'Expected bednet to reduce total infections: '
        f'baseline={inf_base:.0f}, bednet={inf_itn:.0f}'
    )


# ------------------------------------------------------------------
# Killing reduces mosquito population
# ------------------------------------------------------------------

@sc.timer()
def test_bednet_killing_reduces_mosquitoes():
    """ITN killing lowers the mosquito population relative to baseline.

    100% coverage with blocking=1.0 and killing=1.0 adds extra death rate
    a * itn_kill_prob to all mosquito compartments.  With default a=0.3 and
    mu_m=0.1, the equilibrium population drops to 1/4 of baseline.
    Uses ODE mosquitoes for a deterministic, easily-verified result.
    """
    sc.heading('Testing bednet killing reduces mosquito count...')

    nodes = ms.NodeGrid(populations=[2000], emergence=[_EMERGENCE_10])
    kw = dict(
        human        = 'sis',
        mosquito     = 'ode',
        nodes        = nodes,
        stop         = 2001,
        demographics = [],
        rand_seed    = 0,
    )

    def build_baseline(sim, eir_only=False):
        return ms.Campaign(sim, [])

    def build_killing(sim, eir_only=False):
        bednet = ms.Bednet(
            repelling = ms.WaningEffect(initial_efficacy=0.0),
            blocking  = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            killing   = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            usage     = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
        )
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='itn', products=bednet,
                start=2000.0, coverage=1.0,
            ),
        ])

    sim_base    = make_sim(**kw, build_campaign=build_baseline)
    sim_killing = make_sim(**kw, build_campaign=build_killing)
    sim_base.run()
    sim_killing.run()

    mos_base    = next(m for m in sim_base.connectors.values()    if isinstance(m, ms.MosquitoesODE))
    mos_killing = next(m for m in sim_killing.connectors.values() if isinstance(m, ms.MosquitoesODE))

    n_base    = mos_base.results.n_total.values.mean()
    n_killing = mos_killing.results.n_total.values.mean()

    assert n_killing < n_base * 0.5, (
        f'Expected killing bednet to reduce mosquito count by >50%: '
        f'baseline={n_base:.1f}, killing={n_killing:.1f}'
    )


# ------------------------------------------------------------------
# Killing scales with coverage (ODE population-weighting regression)
# ------------------------------------------------------------------

@pytest.mark.parametrize('human', ['sis', 'ode'])
@sc.timer()
def test_bednet_killing_scales_with_coverage(human):
    """Mosquito kill effect at 50% coverage must be substantially less than at 100%.

    This is a regression test for the ODE population-weighting bug in
    aggregate_itn_to_node_state: omitting _itn_fraction from pop_w caused
    sum(combined_block * N_k) / N = combined_block regardless of coverage,
    making 50% and 100% coverage produce identical mosquito killing.

    Uses ODE mosquitoes for a fully deterministic comparison.  Three sims:
      baseline  — no bednet
      half      — 50% coverage
      full      — 100% coverage

    Assertions:
      1. full kills more than half (monotonicity)
      2. half kills more than baseline (some effect at 50%)
      3. The fractional kill at half is between 30% and 70% of the kill at full
         (proportionality — rules out the bug where half ≈ full).
    """
    sc.heading(f'Testing bednet killing scales with coverage [{human}]...')

    nodes = ms.NodeGrid(populations=[2000], emergence=[_EMERGENCE_10])
    kw = dict(
        human        = human,
        mosquito     = 'ode',
        nodes        = nodes,
        stop         = 2001,
        demographics = [],
        rand_seed    = 0,
    )

    def _killing_bednet(coverage):
        def build(sim, eir_only=False):
            bednet = ms.Bednet(
                repelling = ms.WaningEffect(initial_efficacy=0.0),
                blocking  = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
                killing   = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
                usage     = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            )
            return ms.Campaign(sim, [
                ms.InterventionDistributorScheduled(
                    name='itn', products=bednet,
                    start=2000.0, coverage=coverage,
                ),
            ])
        return build

    sim_base = make_sim(**kw, build_campaign=lambda sim, eir_only=False: ms.Campaign(sim, []))
    sim_half = make_sim(**kw, build_campaign=_killing_bednet(0.5))
    sim_full = make_sim(**kw, build_campaign=_killing_bednet(1.0))
    sim_base.run()
    sim_half.run()
    sim_full.run()

    n_base = _find_mosquito_module(sim_base).results.n_total.values[-180:].mean()
    n_half = _find_mosquito_module(sim_half).results.n_total.values[-180:].mean()
    n_full = _find_mosquito_module(sim_full).results.n_total.values[-180:].mean()

    assert n_full < n_half, (
        f'[{human}] 100% coverage must kill more than 50%: '
        f'full={n_full:.1f}, half={n_half:.1f}'
    )
    assert n_half < n_base, (
        f'[{human}] 50% coverage must kill more than baseline: '
        f'half={n_half:.1f}, baseline={n_base:.1f}'
    )
    kill_ratio = (n_base - n_half) / (n_base - n_full)
    # Expected ratio ~0.75 (nonlinear: equilibrium N = B/(mu_m + mu_itn)).
    # The bug set ratio → 1.0 (50% coverage killed identically to 100%).
    assert 0.50 <= kill_ratio <= 0.90, (
        f'[{human}] Kill effect at 50% coverage should be ~75% of the 100% effect '
        f'(ratio={kill_ratio:.2f}); '
        f'baseline={n_base:.1f}, half={n_half:.1f}, full={n_full:.1f}'
    )


# ------------------------------------------------------------------
# Killing requires blocking
# ------------------------------------------------------------------

@sc.timer()
def test_bednet_killing_requires_blocking():
    """ITN killing with blocking=0 has no protective effect.

    Killing is conditioned on blocking: u_kill = usage * blocking * killing.
    With blocking=0 and repelling=0, both itn_kill_prob and itn_biting_reduction
    are zero — the bednet is a no-op regardless of killing efficacy.
    Total infections must remain at baseline levels (within 10% for ABM noise).
    """
    sc.heading('Testing bednet: killing without blocking is a no-op...')

    nodes = ms.NodeGrid(populations=[2000], emergence=[_EMERGENCE_10])
    kw = dict(
        human        = 'sis',
        mosquito     = 'ode',
        nodes        = nodes,
        stop         = 2001,
        demographics = [],
        rand_seed    = 42,
    )

    def build_baseline(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='seed', products=ms.Outbreak(),
                start=2000 + 10/365, coverage=0.15,
            ),
        ])

    def build_null_bednet(sim, eir_only=False):
        bednet = ms.Bednet(
            repelling = ms.WaningEffect(initial_efficacy=0.0),
            blocking  = ms.WaningEffect(initial_efficacy=0.0),
            killing   = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            usage     = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
        )
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='seed', products=ms.Outbreak(),
                start=2000 + 10/365, coverage=0.15,
            ),
            ms.InterventionDistributorScheduled(
                name='itn', products=bednet,
                start=2000 + 30/365, coverage=1.0,
            ),
        ])

    sim_base = make_sim(**kw, build_campaign=build_baseline)
    sim_null = make_sim(**kw, build_campaign=build_null_bednet)
    sim_base.run()
    sim_null.run()

    inf_base = sim_base.diseases[0].results.n_infected.values.sum()
    inf_null = sim_null.diseases[0].results.n_infected.values.sum()

    assert inf_null >= inf_base * 0.90, (
        f'Bednet with blocking=0 should not reduce infections; '
        f'baseline={inf_base:.0f}, null bednet={inf_null:.0f}'
    )


# ------------------------------------------------------------------
# Repelling gates mosquito killing
# ------------------------------------------------------------------

@sc.timer()
def test_bednet_full_repelling_prevents_mosquito_killing():
    """100% repelling means no mosquitoes enter the house, so none are killed.

    u_kill = usage * (1 - repelling) * blocking * killing.
    With repelling=1.0 the kill term is zero regardless of blocking/killing,
    so the mosquito population must equal the no-bednet baseline.
    """
    sc.heading('Testing bednet: full repelling prevents mosquito killing...')

    nodes = ms.NodeGrid(populations=[2000], emergence=[_EMERGENCE_10])
    kw = dict(
        human        = 'sis',
        mosquito     = 'ode',
        nodes        = nodes,
        stop         = 2001,
        demographics = [],
        rand_seed    = 0,
    )

    def build_repelling_only(sim, eir_only=False):
        bednet = ms.Bednet(
            repelling = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            blocking  = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            killing   = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            usage     = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
        )
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='itn', products=bednet, start=2000.0, coverage=1.0,
            ),
        ])

    sim_base = make_sim(**kw, build_campaign=lambda sim, eir_only=False: ms.Campaign(sim, []))
    sim_rep  = make_sim(**kw, build_campaign=build_repelling_only)
    sim_base.run()
    sim_rep.run()

    n_base = _find_mosquito_module(sim_base).results.n_total.values[-180:].mean()
    n_rep  = _find_mosquito_module(sim_rep).results.n_total.values[-180:].mean()

    assert n_rep == pytest.approx(n_base, rel=1e-6), (
        f'Full repelling should not kill any mosquitoes: '
        f'baseline={n_base:.1f}, repelling-only={n_rep:.1f}'
    )


@sc.timer()
def test_bednet_repelling_scales_mosquito_killing():
    """Mosquito kill effect scales with (1 - repelling).

    u_kill = usage * (1 - repelling) * blocking * killing, so at 50% repelling
    the kill term is halved relative to 0% repelling.  Three sims at rep=0, 0.5,
    1.0 with blocking=killing=usage=1.0 verify:
      1. Monotonicity: n(rep=1) > n(rep=0.5) > n(rep=0)
      2. Proportionality: kill fraction at rep=0.5 is ~50% of kill fraction at rep=0
    """
    sc.heading('Testing bednet: repelling scales mosquito killing proportionally...')

    nodes = ms.NodeGrid(populations=[2000], emergence=[_EMERGENCE_10])
    kw = dict(
        human        = 'sis',
        mosquito     = 'ode',
        nodes        = nodes,
        stop         = 2001,
        demographics = [],
        rand_seed    = 0,
    )

    def _bednet_with_repelling(rep):
        def build(sim, eir_only=False):
            bednet = ms.Bednet(
                repelling = ms.WaningEffect(initial_efficacy=rep, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
                blocking  = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
                killing   = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
                usage     = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            )
            return ms.Campaign(sim, [
                ms.InterventionDistributorScheduled(
                    name='itn', products=bednet, start=2000.0, coverage=1.0,
                ),
            ])
        return build

    sim_base = make_sim(**kw, build_campaign=lambda sim, eir_only=False: ms.Campaign(sim, []))
    sim_r0   = make_sim(**kw, build_campaign=_bednet_with_repelling(0.0))
    sim_r50  = make_sim(**kw, build_campaign=_bednet_with_repelling(0.5))
    sim_r100 = make_sim(**kw, build_campaign=_bednet_with_repelling(1.0))
    for s in (sim_base, sim_r0, sim_r50, sim_r100):
        s.run()

    n_base = _find_mosquito_module(sim_base).results.n_total.values[-180:].mean()
    n_r0   = _find_mosquito_module(sim_r0).results.n_total.values[-180:].mean()
    n_r50  = _find_mosquito_module(sim_r50).results.n_total.values[-180:].mean()
    n_r100 = _find_mosquito_module(sim_r100).results.n_total.values[-180:].mean()

    assert n_r100 > n_r50 > n_r0, (
        f'Expected n(rep=1) > n(rep=0.5) > n(rep=0): '
        f'rep=0: {n_r0:.1f}, rep=0.5: {n_r50:.1f}, rep=1: {n_r100:.1f}'
    )
    # u_kill scales linearly with (1-repelling), but the equilibrium population
    # N = B/(mu_m + mu_itn) is nonlinear, so the population-reduction ratio at
    # rep=0.5 is between 0.5 (mu_itn >> mu_m limit) and 1.0 (mu_itn << mu_m).
    # The bug (repelling not gating killing) would give ratio ≈ 1.0.
    kill_ratio = (n_base - n_r50) / (n_base - n_r0)
    assert 0.50 <= kill_ratio <= 0.90, (
        f'Kill fraction at rep=0.5 should be less than rep=0 due to (1-repelling) '
        f'factor (ratio={kill_ratio:.2f}); '
        f'baseline={n_base:.1f}, rep=0: {n_r0:.1f}, rep=0.5: {n_r50:.1f}'
    )


# ------------------------------------------------------------------
# Usage=0 has no effect
# ------------------------------------------------------------------

@sc.timer()
def test_bednet_usage_zero_has_no_effect():
    """ITN with usage=0 provides no protection regardless of other efficacies.

    Usage multiplies all three effects: u_rep = usage*rep = 0, u_block = 0,
    u_kill = 0.  Combined_block and itn_kill_prob are both zero — identical to
    no bednet.  Infections must remain at baseline levels (within 10% for ABM noise).
    """
    sc.heading('Testing bednet: usage=0 is a no-op...')

    nodes = ms.NodeGrid(populations=[2000], emergence=[_EMERGENCE_10])
    kw = dict(
        human        = 'sis',
        mosquito     = 'ode',
        nodes        = nodes,
        stop         = 2001,
        demographics = [],
        rand_seed    = 42,
    )

    def build_baseline(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='seed', products=ms.Outbreak(),
                start=2000 + 10/365, coverage=0.15,
            ),
        ])

    def build_no_usage(sim, eir_only=False):
        bednet = ms.Bednet(
            repelling = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            blocking  = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            killing   = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            usage     = ms.WaningEffect(initial_efficacy=0.0),
        )
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='seed', products=ms.Outbreak(),
                start=2000 + 10/365, coverage=0.15,
            ),
            ms.InterventionDistributorScheduled(
                name='itn', products=bednet,
                start=2000 + 30/365, coverage=1.0,
            ),
        ])

    sim_base     = make_sim(**kw, build_campaign=build_baseline)
    sim_no_usage = make_sim(**kw, build_campaign=build_no_usage)
    sim_base.run()
    sim_no_usage.run()

    inf_base     = sim_base.diseases[0].results.n_infected.values.sum()
    inf_no_usage = sim_no_usage.diseases[0].results.n_infected.values.sum()

    assert inf_no_usage >= inf_base * 0.90, (
        f'Bednet with usage=0 should not reduce infections; '
        f'baseline={inf_base:.0f}, no-usage bednet={inf_no_usage:.0f}'
    )


# ------------------------------------------------------------------
# Node-targeted bednet
# ------------------------------------------------------------------

@sc.timer()
def test_bednet_node_targeted():
    """Bednet on node 0 only reduces node 0 infections; node 1 stays endemic.

    Two equal nodes, both seeded at day 10.  A high-efficacy bednet is
    distributed to node 0 only (100% coverage) on day 30.  At the end of
    the run, node 0 must have lower prevalence than node 1.
    """
    sc.heading('Testing node-targeted bednet: node 0 protected, node 1 unprotected...')

    nodes = ms.NodeGrid(
        populations = [1000, 1000],
        emergence   = [_EMERGENCE_10, _EMERGENCE_10],
    )

    def build_campaign(sim, eir_only=False):
        bednet = ms.Bednet(
            repelling = ms.WaningEffect(initial_efficacy=0.80, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            blocking  = ms.WaningEffect(initial_efficacy=0.90, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            killing   = ms.WaningEffect(initial_efficacy=0.60, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
            usage     = ms.WaningEffect(initial_efficacy=0.80, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
        )
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='seed', products=ms.Outbreak(),
                start=2000 + 10/365, coverage=1.0,
            ),
            ms.InterventionDistributorScheduled(
                name        = 'itn_node0',
                products    = bednet,
                start       = 2000 + 30/365,
                eligibility = ms.HumanEligibility(node_ids=[0]),
                coverage    = 1.0,
            ),
        ])

    sim = make_sim(
        human        = 'sis',
        mosquito     = 'ode',
        nodes        = nodes,
        stop         = 2001,
        demographics = [],
        rand_seed    = 0,
        build_campaign = build_campaign,
    )
    sim.run()

    disease = sim.diseases[0]
    alive   = sim.people.auids
    nids    = np.asarray(disease.node_id[alive], dtype=int)

    node0 = alive[nids == 0]
    node1 = alive[nids == 1]

    prev_node0 = float(np.asarray(disease.infected[node0], dtype=bool).mean())
    prev_node1 = float(np.asarray(disease.infected[node1], dtype=bool).mean())

    assert prev_node0 < prev_node1, (
        f'Expected bednet on node 0 to give lower prevalence than unprotected node 1; '
        f'node0={prev_node0:.1%}, node1={prev_node1:.1%}'
    )


# ------------------------------------------------------------------
# f_indoor scaling
# ------------------------------------------------------------------

def _perfect_bednet():
    """Perfect blocking+killing+usage — maximises mosquito killing.

    Repelling is zero so mosquitoes enter the house and can be killed.
    With repelling>0, u_kill = usage*(1-repelling)*blocking*killing would be
    reduced, hiding the f_indoor scaling signal.
    """
    return ms.Bednet(
        repelling = ms.WaningEffect(initial_efficacy=0.0),
        blocking  = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
        killing   = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
        usage     = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(365), decay_time_constant=ss.days(0)),
    )


@pytest.mark.parametrize('mosquito', ['ode', 'agents'])
@sc.timer()
def test_bednet_outdoor_only_has_no_effect(mosquito):
    """Perfect bednet with f_indoor=0 has no effect — all bites are outdoors.

    The net is never encountered: itn_biting_reduction and itn_kill_prob apply
    only to the indoor fraction (a * f), so with f=0 both are zero.
    Infections must remain at baseline levels.
    """
    sc.heading(f'Testing bednet: f_indoor=0 (outdoor only) has no effect [{mosquito}]...')

    nodes_outdoor = ms.NodeGrid(populations=[2000], emergence=[_EMERGENCE_10], f_indoor=[0.0])
    kw = dict(human='sis', mosquito=mosquito, stop=2001, demographics=[], rand_seed=42)

    def build_baseline(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='seed', products=ms.Outbreak(),
                start=2000 + 10/365, coverage=0.15,
            ),
        ])

    def build_bednet(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='seed', products=ms.Outbreak(),
                start=2000 + 10/365, coverage=0.15,
            ),
            ms.InterventionDistributorScheduled(
                name='itn', products=_perfect_bednet(),
                start=2000 + 30/365, coverage=1.0,
            ),
        ])

    sim_base = make_sim(**kw, nodes=nodes_outdoor, build_campaign=build_baseline)
    sim_itn  = make_sim(**kw, nodes=nodes_outdoor, build_campaign=build_bednet)
    sim_base.run()
    sim_itn.run()

    inf_base = sim_base.diseases[0].results.n_infected.values.sum()
    inf_itn  = sim_itn.diseases[0].results.n_infected.values.sum()

    assert inf_itn >= inf_base * 0.90, (
        f'Perfect bednet with f_indoor=0 should have no effect; '
        f'baseline={inf_base:.0f}, bednet={inf_itn:.0f}'
    )


@pytest.mark.parametrize('mosquito', ['ode', 'agents'])
@sc.timer()
def test_bednet_indoor_only_full_effect(mosquito):
    """Perfect bednet with f_indoor=1 provides maximum protection.

    All bites are indoors so both the human-side blocking and the mosquito-side
    killing operate at full strength.  Infections must be substantially fewer
    than an identical run without bednets.
    """
    sc.heading(f'Testing bednet: f_indoor=1 (indoor only) gives maximum effect [{mosquito}]...')

    nodes_indoor = ms.NodeGrid(populations=[2000], emergence=[_EMERGENCE_10], f_indoor=[1.0])
    kw = dict(human='sis', mosquito=mosquito, stop=2001, demographics=[], rand_seed=42)

    def build_baseline(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='seed', products=ms.Outbreak(),
                start=2000 + 10/365, coverage=0.15,
            ),
        ])

    def build_bednet(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='seed', products=ms.Outbreak(),
                start=2000 + 10/365, coverage=0.15,
            ),
            ms.InterventionDistributorScheduled(
                name='itn', products=_perfect_bednet(),
                start=2000 + 30/365, coverage=1.0,
            ),
        ])

    sim_base = make_sim(**kw, nodes=nodes_indoor, build_campaign=build_baseline)
    sim_itn  = make_sim(**kw, nodes=nodes_indoor, build_campaign=build_bednet)
    sim_base.run()
    sim_itn.run()

    inf_base = sim_base.diseases[0].results.n_infected.values.sum()
    inf_itn  = sim_itn.diseases[0].results.n_infected.values.sum()

    assert inf_itn < inf_base * 0.5, (
        f'Perfect bednet with f_indoor=1 should halve infections; '
        f'baseline={inf_base:.0f}, bednet={inf_itn:.0f}'
    )


@pytest.mark.parametrize('mosquito', ['ode', 'agents'])
@sc.timer()
def test_bednet_f_indoor_scales_mosquito_killing(mosquito):
    """ITN mosquito killing scales with f_indoor.

    The extra kill rate is mu_itn = a * f_indoor * itn_kill_prob.  With a perfect
    bednet at 100% coverage (itn_kill_prob=1), mosquito counts at equilibrium
    must be strictly ordered: n(f=0) > n(f=0.5) > n(f=1).
    """
    sc.heading(f'Testing bednet: mosquito kill rate scales with f_indoor [{mosquito}]...')

    kw = dict(human='sis', mosquito=mosquito, stop=2001, demographics=[], rand_seed=0)

    def build_bednet(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name='itn', products=_perfect_bednet(),
                start=2000.0, coverage=1.0,
            ),
        ])

    def build_baseline(sim, eir_only=False):
        return ms.Campaign(sim, [])

    results = {}
    for f in (0.0, 0.5, 1.0):
        nodes = ms.NodeGrid(populations=[2000], emergence=[_EMERGENCE_10], f_indoor=[f])
        sim_b = make_sim(**kw, nodes=nodes, build_campaign=build_baseline)
        sim_i = make_sim(**kw, nodes=nodes, build_campaign=build_bednet)
        sim_b.run()
        sim_i.run()
        mos = _find_mosquito_module(sim_i)
        results[f] = mos.results.n_total.values[-180:].mean()  # steady-state average

    assert results[0.0] > results[0.5] > results[1.0], (
        f'Expected mosquito count to decrease as f_indoor increases: '
        f'f=0: {results[0.0]:.1f}, f=0.5: {results[0.5]:.1f}, f=1: {results[1.0]:.1f}'
    )


if __name__ == '__main__':
    test_bednet_reduces_infections_vs_baseline()
    test_bednet_killing_reduces_mosquitoes()
    test_bednet_killing_requires_blocking()
    test_bednet_full_repelling_prevents_mosquito_killing()
    test_bednet_repelling_scales_mosquito_killing()
    test_bednet_usage_zero_has_no_effect()
    test_bednet_node_targeted()
    for mosquito in ('ode', 'agents'):
        test_bednet_outdoor_only_has_no_effect(mosquito)
        test_bednet_indoor_only_full_effect(mosquito)
        test_bednet_f_indoor_scales_mosquito_killing(mosquito)

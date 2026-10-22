# malariasim on Starsim v4: scripts, and malariasim's own bugs found along the way

These folders hold the scripts and outputs behind the Starsim v4 review of malaria (`v4_design/review_oct17/malaria_review.md` in the [starsim](https://github.com/starsimhub/starsim) repo, `v4` branch). They used malariasim `a1351fc` (this repo's `v4` branch), Starsim v4 at `1a5b7a77`, and Starsim 3 at `f40175b4`, all on one interpreter (the `ngm` env, which also has LASER and `malaria_model` from `laser-ngm`).

| Folder | Part |
|---|---|
| `human/` | The human disease (SIS, SEIS, biting heterogeneity, the age-binned ODE), declared in v4 (`human_v4.py`) |
| `vector/` | The mosquitoes and the coupling: `malaria_hybrid` (saved in `hybrid_v3/`) declared in v4 (`hybrid_v4.py`), malariasim's mosquitoes declared (`ms_mosquito_v4.py`) and as coded agents (`mosquito_agents_v4.py`), species and patches |
| `interventions/` | Vaccine, bednet, drug, outbreak and campaigns as v4 products on declared models (`malaria_v4.py`, `malaria_rm_v4.py`) |
| `population/` | Nodes as a patch dimension, steady-state demography, results by age and node (`mal_v4.py`), and the population tests ported |
| `framework/` | The unchanged test suite with the shims (`malshim.py`, `malshim_full.py`), `ms.Sim` with v4's run tools, calibration, and the speed benchmark against v3 and LASER (`bench.py`) |

Each script prints the Starsim it ran on. The `env.sh` and `run*.sh` files point at a temporary worktree of Starsim v4 and a copy of Starsim 3 in a temporary folder, which may be gone. To rerun, make them again (`git -C <starsim> worktree add --detach <dir> 1a5b7a77`, and `git -C <starsim> archive f40175b4 starsim | tar -x -C <dir>`) and set `PYTHONPATH` to one of them.

**Which Starsim 3.** malariasim passes all 284 tests on Starsim 3.5.2 and on `f40175b4` (the 3.6.0 performance refactor, before release). Released 3.6.0 to 3.7.2 fail 76:
- 72 on `float(p.death_rate_per_year)` at `human_ode.py:205`, since `float()` of a rate is now refused (use `.value`);
- 4 on `new_infections` being 0, since Starsim now counts new infections in `Infection.set_prognoses()`, which malariasim's overrides don't call (`super().set_prognoses(uids, sources)`).

**On v4,** unchanged malariasim fails 123 tests, because `ss.Disease` and `ss.SIS` now mean declared classes. `framework/malshim.py` (27 lines, no change to malariasim) brings it to 273 passing, and `framework/malshim_full.py` to all 284. What the full shim fixes, malariasim should fix itself: write results produced during `step()` at `self.ti + 1` (v4 moves `ti` before `update_results()`), and don't assume in an analyzer that modules have stepped (v4 runs analyzers at the initial point).

## malariasim bugs

Found while comparing. Each happens the same way under Starsim 3 and v4 unless noted.

**Transmission and the coupling**
1. **The documented lag is the wrong way round.** `node_state.py`, `CLAUDE.md` and the comment in `_init_mosquitoes` say diseases run before connectors, so humans see last step's EIR. In fact the mosquito connector runs before the disease's `step()`: humans see this step's EIR, and mosquitoes see human infectiousness from the previous step's `step_state()`, missing that step's new infections (`vector/v3_probe_order.out`, `human/probe_loop_order.py`).
2. **The infection probability is linear and clipped:** `p = clip(b × EIR_dt × rel, 0, 1)` rather than `1 − exp(−x)`. At high EIR and high `rel` it overstates infection (by 5.6% at 0.11 a day), and clipping caps the most-bitten people (`human/diag_v3_expprob.py`).
3. **`MalariaHumanSEIS.new_infections` counts onsets of infectiousness, about 12 days late, not inoculations** (on `f40175b4`, where `update_results()` counted `round(ti_infected) == ti`). The first 8 days show 0 against about 100 a day entering exposed; the year's total is 15,859 against 16,260 (`human/mal_bug_seis_newinf.py`). On Starsim 3.6+ and v4 it is 0, since `set_prognoses()` doesn't call `super()`.
4. **Heterogeneity, node assignment, newborns' nodes and the mosquito agents use the global `np.random`** (`human_base.py:286`, `315`, `321`, `326`; `mosquito_agents.py:289` on), so runs aren't common-random-number safe: any other user of the global generator changes who bites more.

**Mosquitoes**

5. **Agent mosquitoes start from zero with agent humans.** `n_humans` is first written by the human `step()`, after the mosquito step, so `_init_mosquitoes` seeds 0 per node: 0 mosquitoes on day 0 and 229.8 on day 30, against 250 for the ODE mosquitoes (`vector/ms_v3_ref.json`).
6. **Agent mosquito lifespans are floored:** `ti_dead = ti + exponential(...).astype(int)` gives a mean life of 9.6 days instead of 10, so 4% fewer mosquitoes than the ODE (879 against 917), an EIR 6% lower, and prevalence 0.461 against 0.480.
7. **Agent mosquitoes collapse at a weekly step, which the docs recommend for speed.** `p_bite = a*dt` is clipped to 1, so a mosquito bites at most once per step (2.1 expected at dt = 7): prevalence 0.014 against 0.485 with ODE mosquitoes (`vector/lag_dt_v3.out`).
8. (Removed code.) `malaria_hybrid`'s `MosquitoABM` gives 2% more mosquitoes than its `MosquitoODE` (3.060 against 3.001), from scheduling deaths at `ti_dead <= ti` on a continuous draw.

**Interventions**

9. **Bednets cut indoor bites twice.** The mosquito modules multiply each node's indoor EIR by (1 − mean block) for everyone, and `BednetUpdater` then multiplies each holder's `_prob_acquire_indoor` by (1 − their block). On the step after distribution, humans receive 0.422 of the infectious indoor bites the mosquitoes land, and a holder's multiplier is 0.071 against 0.422 for a non-holder (`interventions/malariasim_bugs_v3.log`). The infections averted are overstated by 14% with killing and 34% without (`interventions/itn_double_count_v4.log`). `docs/interventions_design.md` describes the same reads.
10. **A second vaccine round on the ODE human removes everyone's protection.** `_age_vaccinated_state` zeroes the dose time of age bin 0 every step; with the default single bin that is everyone. At the next round `pre_administer` averages 0 and 2001.4 to 333.6, so the box ends in year 334, protection expires at the next step, and `sync_vaccinated_fraction` zeroes it. Two rounds give more infections than one: 2,329 against 1,995 at day 729, where agents give 1,330 against 2,033 (`interventions/malariasim_ode_redose_v3.log`). With several bins, the youngest bin breaks the same way.
11. **The ODE and agents disagree on what a second dose means:** the ODE adds coverage only to the unvaccinated (f + c(1 − f)) and averages dose times; agents overwrite. Even with bug 10 fixed, the two forms of one campaign differ.

**Population and the ODE**

12. **The ODE human ignores the death table malariasim builds.** `ms.Sim(human='ode', demographics=make_steady_state_demographics())` reads the death rate with `float(d.pars.death_rate)` inside `try/except Exception: pass` (`sim.py`). With the `ss.Deaths(table)` that malariasim builds, `float()` fails, so the ODE gets a death rate of 0 for bin 0 (against 0.008 a year as agents) and solves the other bins to hold the pyramid. The agent and ODE humans don't share a mortality schedule (`population/out_v3_ode.txt`).
13. **`MalariaHumanODE` with no age bins and no demography loses about 1% of its people a year.** The single 100-year bin still ages people out at `dt/100`, with no births: 10,000 becomes 9,512 after 5 years and 9,048 after 10 for the default `ms.Sim(human='ode')`. `NodeState.n_humans` follows, so the mosquitoes see it too, and a fit to counts was biased to `b` = 0.359 (`human/mal_bug_ode_shrinks.py`).
14. **`defaults.py` labels `URBAN_EIR` and the others "bites / person / month",** but `NodeGrid`, `MosquitoesEirOnly` and `eir_to_params` read them as annual or as shapes. `run_sweep_b_and_c.py` passes `ms.URBAN_EIR` as an EIR directly. Taken at its label, the EIR would be 12 times too low.

**Framework**

15. **`PerformanceMonitor`'s per-module totals are always 0.** `loop.to_df()['cpu_time']` is filled only by `sim.run(profile=True)`, and a bare `try/except` hides it (`framework/perf_monitor.py`).
16. **The July benchmark script exists only as a `.pyc`** in `malaria_benchmarks/__pycache__/` (rebuilt as `framework/bench.py`), and the July Numba work (`FastBirths`, `FastDeaths`, the node-aggregation and disease kernels) isn't on `main` or `v4`. `FastBirths`/`FastDeaths` are on branch `Interventions2-ck`.
17. **`PerformanceMonitor` needs `psutil`, which isn't in `pyproject.toml`,** and `CLAUDE.md` describes an `env/` folder and entry points (`run_malaria_demo.py`, `run_malaria_hybrid.py`, `run_malaria_sslaser.py`) that no longer exist.

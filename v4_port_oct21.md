# malariasim on Starsim v4: recheck of October 21

This rechecks the October 17 port work (`v4port/`, summarized in `v4port/README.md`) against Starsim v4 as it is now: commit `ee5d1fb0` on the `v4` branch of the [starsim](https://github.com/starsimhub/starsim) repo, 220 commits and a large simplification later. malariasim itself is unchanged (`a1351fc`). What v4 should change is written up in the v4 design notes (`v4_design/oct21/malaria_recheck.md` in the starsim repo, `v4` branch). This note covers malariasim's side: what a port needs today, which of malariasim's own bugs remain, and how to rerun everything.

## In brief

- **Unchanged malariasim now stops at import,** with a message naming the fix: `MalariaHumanBase defines set_prognoses, which a declared disease never calls ... subclasses ssl.AgentSIS instead`. On October 17, 123 of 284 tests failed in several different ways.
- **With a shim, nothing has changed:** the smallest shim (now 24 lines) leaves the same 11 failures as before, and the full shim passes all 284. Results agree with Starsim 3 within 0.7% in all 9 human × mosquito combinations; the ODE humans are bit-identical to October 17.
- **v4's migration script now does most of the port:** `python -m starsim.interop.migrate malariasim tests` makes 21 changes and lists 8 hand edits. After those, 199 of 284 tests pass. The 85 failures come from gaps in the script, written up for v4. About 16 more lines by hand (below) make all 284 pass with no shim.
- **The declared versions keep most of their gains,** and v4 fixed many of the October 17 problems: bednets and vaccines now act on bite-driven infection, MDA reaches the infected, mosquitoes no longer die out at a monthly step, and newborns keep their node. But the declared model has new traps, chiefly that `init=dict(S=1.0)` now means one person, and that mosquito states must be declared `people=False`.
- **Speed:** unchanged malariasim is about 15% slower than on October 17's v4. Declared agents with mosquito counts are 1.5–1.9 times slower. The declared ODEs are as fast or faster (2.3 times faster with nodes, ages and a seasonal EIR).

## Porting malariasim today

What the migration script does on a copy of `malariasim/` and `tests/` (`v4port/oct21/migrate/`):

| Change | Done by |
|---|---|
| `ss.SIS` → `ssl.AgentSIS` for `MalariaHumanBase`, and `import starsim.library as ssl` | Script |
| Results written in `step()` at `self.ti + 1` (`human_ode.py`, `mosquito_ode.py`, `mosquito_eir_only.py`; 13 lines) | Script |
| `ResultsAnalyzer.step()`: return at `self.ti == 0`, since v4 runs analyzers at the start | Listed; 2 lines by hand |
| `float(rate)` → `float(rate.value)` (`human_ode.py:205`, `216`; `sim.py:214`) | Listed; 3 lines by hand |
| `MalariaHumanODE(ss.Disease)` → `ss.Infection` | Script, but it fails: `ss.Infection` has its own `susceptible` and `infected`, so 72 tests error. v4 needs a bare coded base; until then the shim's `V3Disease` is the workaround |
| `eligible=` → `where=` in `tests/test_distributor_scheduled.py` | Script, wrongly: that's malariasim's own helper, and 9 tests break. Revert it |
| `set_prognoses()` in `MalariaHumanBase` and `MalariaHumanSEIS` must count new infections, or `new_infections` stays 0: add `ss.Infection.set_prognoses(self, uids, sources)` (not `super()`, since the SEIS override must not go through the SIS one) | Not listed by the script; v4 warns at run time. 4 tests fail. This is malariasim's own bug since Starsim 3.6 |
| `MalariaHumanODE.step()`: with results at `self.ti + 1`, also write the starting point (index 0) at `self.ti == 0` | Not listed; 3 lines. Without it the ODE's prevalence at the start is 0/0 (1 test) |
| `FastBirths` (branch `Interventions2-ck`): `ss.prob.array_to_prob()` is gone | Not listed: use `ss.prob(rate, ss.years(1)).to_prob(self.t.dt)` |

Checked on a copy: the script, the five listed lines, reverting the test rename, the two `new_infections` lines, the three initial-point lines, and a 10-line bare base (`CodedDisease`, as in the shim) for the ODE human. All 284 tests pass with no shim, from 37 lines added and 18 removed in 7 files (`v4port/oct21/migrate/migrated_full.diff`). Once v4 has a public bare coded base and the script's rules are fixed, the script alone should get there.

**If writing malariasim's model declared, as `v4port/` did,** know these changes since October 17:
- **Mosquito states must say `people=False`** (`Sv, Ev, Iv = ss.states(3, people=False)`). Without it they are agents, silently in some cases: 3,000 of 5,000 agents become mosquitoes. v4 should catch this (MA35).
- **Don't write `init=dict(S=1.0, ...)`.** A value of 1.0 is now one person; give the infected share instead (`init=dict(I=0.15, Sv=...)`), and the rest start in the first state (MA34).
- **Waning by a function runs as agents only.** malariasim's box-then-decay vaccine, bednet and drug can't run as an ODE now (MA38, a decision for v4).
- **Arrow strings are gone.** `ss.transmission('S -> I', beta=b)` is `S >> I @ b * I`, `ss.geo(names=, pop=)` is `ss.geo.location(...)`, and `ssl.CodedSIS` is `ssl.AgentSIS`.
- **Re-dosing, a net's killing as a rate, biting-weighted counts and several species** still have no declared spelling (deferred in v4 until the malaria port needs them). Those parts stay coded for now.

## malariasim's own bugs

The 17 bugs listed in `v4port/README.md` are all still in malariasim, since its code hasn't changed. Two now interact differently with v4:
- **Bug 3** (`new_infections` 0 on Starsim 3.6+, since `set_prognoses()` skips `super()`): v4 now warns, naming the fix, every time a sim runs.
- **Bug 12** (the ODE human ignores the death table, via `float()` inside a bare `try/except`): the migration script's `.value` doesn't fix it, since `ss.Deaths` holds a table, not a rate. It still silently gives the ODE a death rate of 0.

Fixing bugs 1–4 (the coupling order, the clipped linear infection probability, SEIS `new_infections`, and the global `np.random`) is still the most valuable change to malariasim, whichever Starsim it runs on.

## Results and speed against October 17

| Check | October 17 | October 21 |
|---|---|---|
| Unchanged malariasim with the full shim, 9 combinations × 5 seeds, year-3 prevalence | Within 1% of Starsim 3 | Within 0.7% and 1.5 SE of Starsim 3; ODE humans identical to October 17 |
| Unchanged malariasim, SEIS + ODE mosquitoes, 10k, 2 years (best of 3, after warm-up) | 0.95 s | 1.09 s |
| Declared SEIS + mosquito counts, agents | 2.19 s | 4.19 s |
| Declared human ODE, 3 nodes × 20 ages, seasonal EIR, births and deaths | 1.53 s | 0.67 s |
| Fit of `b` on the declared human ODE | 3.2 s, 35 runs | 7.1 s, 89 runs (v4 now checks convergence with three searches); 65–80 s on first use, while Numba compiles, with nothing printed at `verbose=0` |

The machine was shared (load 18 to 31 on 20 cores), so differences under about 30% are noise.

## Rerunning

Everything for this recheck is in `v4port/oct21/`:

| Path | What |
|---|---|
| `malshim.py` | The smallest shim, updated for v4 `ee5d1fb0` (24 lines) |
| `logs/` | Every October 17 script rerun on v4 `ee5d1fb0`, unchanged (`<part>__<script>.log`), the suite runs (`suite_*.log`), and the 9-combination comparison (`compare_v4_oct21.*`) |
| `logs_people_false/` | The 30 scripts that needed `people=False` or a renamed spelling, rerun with `people_false_and_renames.patch` applied |
| `people_false_and_renames.patch` | The edits to the October 17 scripts: `people=False` on mosquito states, operators for arrow strings, `ss.geo.location`, `ssl.AgentSIS`, and the `array_to_prob` replacement |
| `probes/` | New probes: `init_rule.py` (MA34), `rm_init.py` (MA35), `order0.py` (the coupling order with valid initial values), `minis.py` (small items) |
| `bench/` | The old-against-new benchmark (`ab_declared.py`, `ab_fseis.py`, `malshim_either.py` for either v4, `ab.log`) and profiles |
| `migrate/` | The migration script's report (`migrate.log`), the diff after the script and its listed edits and the suite on it (85 failures), and the diff after all edits (`migrated_full.diff`) with the suite on it (284 passed) |
| `wrap.py`, `run_one.sh`, `run_suite.sh`, `cmp.sh`, `torun*.txt` | The runners: each script in a fresh process, loading the shim when it uses malariasim. They point at a temporary worktree and scratch folder that may be gone; recreate with `git -C <starsim> worktree add --detach <dir> ee5d1fb0` (and `1a5b7a77` for the old v4) and edit the paths |

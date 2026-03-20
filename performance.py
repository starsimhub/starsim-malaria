"""
performance.py — wall-clock and memory monitoring for Starsim simulation runs.

Usage
-----
    monitor = PerformanceMonitor()

    for sc in scenarios:
        sim = make_sim(...)
        monitor.start()
        sim.run()
        monitor.stop(sim)

    monitor.plot()
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import psutil
import time


class PerformanceMonitor:
    """
    Track wall-clock duration and RSS memory for a sequence of simulation runs.

    Each start() / stop(sim) pair records one entry.  Call plot() after all
    runs complete to write per-sim CPU charts and text summaries to disk.
    """

    def __init__(self):
        self._proc     = psutil.Process(os.getpid())
        self._t0       = None
        self._mem_before = None
        self._records  = []   # list of {'sim': ..., 'elapsed': ..., ...}

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def start(self):
        """Snapshot time and memory — call immediately before sim.run()."""
        self._mem_before = self._proc.memory_info().rss
        self._t0         = time.perf_counter()

    def stop(self, sim):
        """Record elapsed time and memory delta — call immediately after sim.run()."""
        elapsed   = time.perf_counter() - self._t0
        mem_after = self._proc.memory_info().rss
        self._records.append(dict(
            sim        = sim,
            elapsed    = elapsed,
            mem_before = self._mem_before / 1024**2,
            mem_after  = mem_after        / 1024**2,
            mem_delta  = (mem_after - self._mem_before) / 1024**2,
        ))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def plot(self):
        """Write a CPU bar chart and a text performance summary for each sim."""
        for rec in self._records:
            sim  = rec['sim']
            slug = _slug(sim.label)

            fig = sim.loop.plot_cpu()
            if fig is not None:
                cpu_fname = f'malaria_cpu_{slug}.png'
                fig.savefig(cpu_fname, dpi=150)
                plt.close(fig)
                print(f'CPU plot saved to {cpu_fname}')

            loop_df      = sim.loop.to_df()
            module_times = loop_df.groupby('module')['cpu_time'].sum().sort_values(ascending=False)

            disease_name    = type(sim.diseases[0]).__name__.lower()
            connector_names = [type(c).__name__.lower() for c in sim.connectors] if sim.connectors else []

            W = 52
            lines = [
                f'{"=" * W}',
                f'  {sim.label}',
                f'{"=" * W}',
                f'  Runtime        : {rec["elapsed"]:.2f}s',
                f'  Memory (RSS)   : {rec["mem_before"]:.1f} → {rec["mem_after"]:.1f} MB  (Δ {rec["mem_delta"]:+.1f} MB)',
                f'  Human  ({disease_name})',
                f'    CPU          : {module_times.get(disease_name, 0.0):.3f}s',
            ]
            for cname in connector_names:
                lines += [
                    f'  Mosquito ({cname})',
                    f'    CPU          : {module_times.get(cname, 0.0):.3f}s',
                ]
            lines.append(f'  Top modules by CPU:')
            for mod, t in module_times.head(8).items():
                lines.append(f'    {mod:<38}  {t:.3f}s')

            summary    = '\n'.join(lines)
            perf_fname = f'malaria_perf_{slug}.txt'
            with open(perf_fname, 'w', encoding='utf-8') as f:
                f.write(summary + '\n')
            print(f'Performance summary saved to {perf_fname}')


def _slug(label):
    return label.lower().replace(' ', '_').replace('+', 'plus').replace('-', '').strip('_')

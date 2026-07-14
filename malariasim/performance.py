"""
performance.py — wall-clock and memory monitoring for Starsim simulation runs.

Usage
-----
    monitor = PerformanceMonitor()

    for sc in scenarios:
        sim = ms.Sim(...)
        monitor.start()
        sim.run()
        monitor.stop(sim)

    monitor.plot(out_prefix='malaria_results/perf')

Output
------
    <out_prefix>_performance.png   — multi-sim bar-chart grid (one subplot per
                                     module + total wall time + max memory)
    <out_prefix>_summary.txt       — text table matching laser-ngm format
"""

import os
import psutil
import time
import numpy as np
import matplotlib.pyplot as plt


# One color per scenario — mirrors laser-ngm SCENARIO_COLORS
SCENARIO_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf',
]


class PerformanceMonitor:
    """
    Track wall-clock duration and RSS memory for a sequence of simulation runs.

    Each start() / stop(sim) pair records one entry.  Call plot() after all
    runs complete to write a multi-sim performance chart to disk — one subplot
    per module (total CPU time across all scenarios), plus subplots for total
    wall time and max RSS memory.  Layout mirrors laser-ngm's
    multisim_plotting._plot_performance.
    """

    def __init__(self):
        self._proc    = psutil.Process(os.getpid())
        self._t0      = None
        self._records = []   # list of dicts, one per sim

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def start(self):
        """Snapshot start time — call immediately before sim construction."""
        self._t0 = time.perf_counter()

    def stop(self):
        """Record elapsed wall time. Call immediately after sim.run(), before release."""
        self._elapsed = time.perf_counter() - self._t0

    def record(self, sim):
        """Capture RSS and store the record. Call after release() so RSS reflects freed memory."""
        max_rss = self._proc.memory_info().rss / (1024.0 * 1024.0)
        module_totals = self._module_totals(sim)
        self._records.append(dict(
            sim           = sim,
            elapsed       = self._elapsed,
            max_rss_mb    = max_rss,
            module_totals = module_totals,
        ))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def plot(self, out_prefix='malaria_results/perf', show_plots=False, show=False, do_save=True):
        """Write multi-sim performance chart and text summary."""
        if not self._records:
            return
        _show = show_plots or show
        os.makedirs(os.path.dirname(out_prefix) if os.path.dirname(out_prefix) else '.', exist_ok=True)
        paths = []
        if do_save or _show:
            p = self._plot_performance(out_prefix, _show)
            if p:
                paths.append(p)
            p = self._write_summary(out_prefix)
            if p:
                paths.append(p)
        if paths:
            print(f"Saved: {', '.join(paths)}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _module_totals(sim):
        """Return dict of module_name -> total cpu_time (seconds)."""
        try:
            loop_df = sim.loop.to_df()
            grouped = loop_df.groupby('module')['cpu_time'].sum()
            return dict(grouped)
        except Exception:
            return {}

    def _plot_performance(self, out_prefix, show):
        """Multi-sim bar-chart grid — one subplot per module, plus wall time and memory.

        Mirrors laser-ngm multisim_plotting._plot_performance:
          - Total wall time   (one bar per scenario, no shared y-limit)
          - One subplot per module (union of all sims, in first-seen order)
          - Max RSS memory   (one bar per scenario, no shared y-limit)
        """
        # Union of module names in first-seen order
        module_names: list[str] = []
        for rec in self._records:
            for name in rec['module_totals']:
                if name not in module_names:
                    module_names.append(name)

        labels = [
            rec['sim'].label.replace('human-', 'h:').replace(' + mosquito-', ' m:')
            for rec in self._records
        ]
        colors = [SCENARIO_COLORS[i % len(SCENARIO_COLORS)] for i in range(len(self._records))]
        x      = np.arange(len(self._records))
        bar_w  = 0.5

        def _bar(ax, values, title, ylabel, ylim=None):
            bars = ax.bar(x, values, width=bar_w, color=colors)
            ax.set_title(title, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
            ax.grid(True, axis='y', alpha=0.4)
            if ylim is not None:
                ax.set_ylim(0, ylim)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)

        # Pre-compute shared y-limit for module subplots
        if module_names:
            shared_max = max(
                rec['module_totals'].get(name, 0.0)
                for rec in self._records for name in module_names
            )
            shared_ylim = shared_max * 1.15
        else:
            shared_ylim = 1.0

        # Build subplot specs: total wall time, modules..., max memory
        subplot_specs = (
            [('Total wall time', 'total', 's',  None)]
            + [(name, name, 's', shared_ylim) for name in module_names]
            + [('Max memory',   'mem',   'MB', None)]
        )

        n_plots = len(subplot_specs)
        ncols   = 4
        nrows   = max(1, int(np.ceil(n_plots / ncols)))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.0 * ncols, 3.5 * nrows),
            squeeze=False,
        )

        for plot_idx, (title, key, unit, ylim) in enumerate(subplot_specs):
            ax = axes[plot_idx // ncols][plot_idx % ncols]
            if key == 'total':
                values = [rec['elapsed'] for rec in self._records]
            elif key == 'mem':
                values = [rec['max_rss_mb'] for rec in self._records]
            else:
                values = [rec['module_totals'].get(key, 0.0) for rec in self._records]
            _bar(ax, values, title, unit, ylim=ylim)

        for k in range(n_plots, nrows * ncols):
            axes[k // ncols][k % ncols].axis('off')

        fig.suptitle('Performance by scenario', fontsize=12)
        fig.tight_layout()

        path = f'{out_prefix}_performance.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        return path

    def _write_summary(self, out_prefix):
        """Text summary table matching laser-ngm format."""
        W = 60
        lines = []
        for rec in self._records:
            sim = rec['sim']
            lines += [
                '=' * W,
                f'  {sim.label}',
                '=' * W,
                f'  Wall time      : {rec["elapsed"]:.3f} s',
                f'  Max RSS        : {rec["max_rss_mb"]:.1f} MB',
                '',
                'Module totals (CPU seconds):',
            ]
            for mod, t in sorted(rec['module_totals'].items(), key=lambda x: -x[1]):
                lines.append(f'  {mod:<45}  {t:.6f} s')
            lines.append('')

        path = f'{out_prefix}_summary.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        return path

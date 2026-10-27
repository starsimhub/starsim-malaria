"""
MalariaMultiSim: a MultiSim with plots that compare malaria scenarios, from the results the sims keep
(by node and age, through their stratifications)
"""

import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

__all__ = ['MalariaMultiSim']


def levels(res, dim):
    """ A result's values at each level of a dimension, as {label: values}; empty if it is not stratified by it """
    try:
        vals = res.by(dim)
    except Exception:
        return {}
    return {label:np.asarray(vals[:, k]) for k,label in enumerate(vals.columns)}


def by_node(res):
    """ A result's values in each node, as {label: values}, or {'Total': values} with one node """
    return levels(res, 'patch') or {'Total': np.asarray(res)}


class MalariaMultiSim(ss.MultiSim):
    """
    A MultiSim of malaria sims, with plots of each sim side by side

    Examples:
        ```python
        msim = ms.MalariaMultiSim([ms.Sim(mosquito=m, label=m) for m in ['eir_only', 'ode', 'agents']])
        msim.run()
        msim.plot_all(prefix='malaria_results/')
        ```
    """
    def grid(self, title, key, ylabel, fn, prefix='', show_plots=False, do_save=True):
        """ One panel per sim, each plotting fn(sim) -> {label: values} """
        nrows, ncols = sc.getrowscols(len(self.sims))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), sharex=True, squeeze=False)
        fig.suptitle(title, fontsize=13)
        for ax, sim in zip(axes.flat, self.sims):
            years = sim.results.timevec.years if hasattr(sim.results.timevec, 'years') else sim.t.yearvec
            lines = fn(sim)
            if not lines:
                ax.text(0.5, 0.5, 'Not in this model', ha='center', va='center', transform=ax.transAxes, color='gray')
            for label, vals in lines.items():
                x = years if len(vals) == len(years) else np.linspace(years[0], years[-1], len(vals)) # A result kept at fewer points
                ax.plot(x, vals, lw=1.5, label=label)
            ax.set_title(sim.label)
            ax.set_xlabel('Year')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if len(lines) > 1:
                ax.legend(fontsize=7)
        for ax in axes.flat[len(self.sims):]:
            ax.set_visible(False)
        plt.tight_layout()
        if do_save:
            sc.savefig(f'{prefix}malaria_{key}.png', dpi=150)
        if show_plots:
            plt.show()
        plt.close(fig)
        return fig

    def plot_prevalence(self, **kwargs):
        """ Prevalence (%) over all nodes, and in each """
        def fn(sim):
            res = sim.results.malaria.prevalence
            out = {'Total': 100*np.asarray(res)}
            out.update({k:100*v for k,v in levels(res, 'patch').items()})
            return out
        return self.grid('Malaria prevalence', 'prevalence', 'Prevalence (%)', fn, **kwargs)

    def plot_mosquito_by_node(self, **kwargs):
        """ Mosquitoes in each node """
        def fn(sim):
            for mod in ['malaria', 'mosquitoes']:
                if mod in sim.results and 'mosquitoes' in sim.results[mod]:
                    return by_node(sim.results[mod].mosquitoes)
            return {}
        return self.grid('Mosquitoes by node', 'mosquitoes_by_node', 'Mosquitoes', fn, **kwargs)

    def plot_eir(self, **kwargs):
        """ The annual EIR (infectious bites per person per year) over all nodes """
        def fn(sim):
            for mod in ['malaria', 'mosquitoes']:
                if mod in sim.results and 'eir' in sim.results[mod]:
                    return {'EIR': 365*np.asarray(sim.results[mod].eir)}
            return {}
        return self.grid('EIR', 'eir', 'EIR (bites/person/year)', fn, **kwargs)

    def plot_age_bins(self, **kwargs):
        """ People in each age group, for sims with age groups """
        def fn(sim):
            res = sim.results.malaria.people if 'people' in sim.results.malaria else None
            return levels(res, 'age') if res is not None else {}
        return self.grid('Population by age group', 'age_bins', 'People', fn, **kwargs)

    def plot_all(self, do_pop_plot=False, do_mosquito_by_node=True, **kwargs):
        """ Every plot; kwargs (`prefix`, `show_plots`, `do_save`) are passed to each """
        self.plot_prevalence(**kwargs)
        if do_mosquito_by_node:
            self.plot_mosquito_by_node(**kwargs)
        self.plot_eir(**kwargs)
        if do_pop_plot:
            self.plot_age_bins(**kwargs)
        return

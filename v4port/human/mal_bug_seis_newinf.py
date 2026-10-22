""" malariasim (v3): what does MalariaHumanSEIS's new_infections count? Infection.update_results counts round(ti_infected) == ti, and SEIS sets ti_infected to the
future onset of infectiousness, so new_infections counts onsets (shifted ~12 days), not inoculations; compare with a count of entries to exposed """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc, malariasim as ms

class CountExposed(ss.Analyzer):
    def init_results(self):
        super().init_results(); self.n = []
    def step(self):
        d = self.sim.diseases[0]
        self.n.append(int(np.count_nonzero(d.exposed & (d.ti_exposed == self.sim.ti))))

sim = ms.Sim(human='seis', mosquito='eir_only', nodes=ms.NodeGrid(populations=[10_000], eir=[[10.0]*12]), common_pars=ms.CommonParams(),
             start=2000, stop=2001, dt=ss.days(1), rand_seed=0, verbose=0, analyzers=CountExposed())
sim.run()
r = sim.diseases[0].results
n = np.array(sim.analyzers[0].n)
print('new_infections, first 15 days:', r.new_infections[:15].astype(int).tolist())
print('entries to exposed, first 15 days:', n[:15].tolist())
print('totals over the year: new_infections', int(r.new_infections.sum()), 'entries to exposed', int(n.sum()))

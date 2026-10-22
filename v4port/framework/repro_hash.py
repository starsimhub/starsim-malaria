""" hash() of unchanged ms.Sims (v4 + shim): does it see a change to a coded module's parameter, to the node EIR, to the campaign? """
import starsim as ss; print(ss.__version__, ss.__file__)
import malshim # noqa
import malariasim as ms

def camp(sim, eir_only=False):
    return ms.Campaign(sim, [ms.InterventionDistributorScheduled(name='vacc', products=ms.MalariaVaccine(), start=2000.5, coverage=0.8)])
def mk(b=0.4, eir=10.0, campaign=None, seed=0):
    return ms.Sim(human='sis', mosquito='eir_only', nodes=ms.NodeGrid(populations=[500], eir=[eir]), start=2000, stop=2001, verbose=0,
                  human_pars=dict(prob_human_infected_on_bite=b), build_campaign=campaign, rand_seed=seed)
h0 = mk().hash()
print('same sim built twice:           ', mk().hash() == h0)
print('human b 0.4 -> 0.1:             ', 'differs' if mk(b=0.1).hash() != h0 else 'SAME HASH')
print('node EIR 10 -> 100:             ', 'differs' if mk(eir=100.0).hash() != h0 else 'SAME HASH')
print('with a vaccine campaign:        ', 'differs' if mk(campaign=camp).hash() != h0 else 'SAME HASH')
print('rand_seed 0 -> 1:               ', 'differs' if mk(seed=1).hash() != h0 else 'SAME HASH')
a, b = mk(eir=10.0).run(), mk(eir=100.0).run()
print('n_infected at end, EIR 10 vs 100:', a.diseases[0].results.n_infected[-2], b.diseases[0].results.n_infected[-2])

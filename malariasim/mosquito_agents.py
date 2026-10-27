"""
Individual adult female mosquitoes (SEI), written as code, for ``ms.Malaria(mosquito='agents')``

A declared state that is not people is a count, so mosquitoes as individuals are a module of
their own, with arrays of its own (kept apart from ``sim.people``) [PROPOSAL MS1: several
populations of agents, specified in Starsim's ``v4_design/oct26/populations.md`` but not yet
built]. Each step, mosquitoes emerge in each node in proportion to the people there, susceptible
mosquitoes are infected by biting infectious people, exposed ones become infectious, and some die
(more where bednets kill them); each step works on the live mosquitoes only, and the arrays grow
as needed. The people's infection rates read the EIR this module works out, by node.
"""

import numpy as np
import starsim as ss

from . import defaults as msd

__all__ = ['MosquitoAgents']


class MosquitoAgents(ss.Module):
    """
    SEI adult female mosquitoes as agents, in each node

    It reads the people, the parameters, and the nodes of the sim's `ms.Malaria`, which must be made with
    `mosquito='agents'` (`ms.Sim(mosquito='agents')` adds this module). As agents only.

    Args:
        emergence (list): new adult females per person per day, by month (12 values), or one profile per node; default malariasim's
        disease (str): the name of the malaria disease
        kwargs (dict): passed to `ss.Module`

    The results are `Sv`, `Ev`, `Iv` (mosquitoes in each state), `mosquitoes` (their total), and `eir`
    (infectious bites per person per day, over every node); `eir_by_node` holds the EIR in each node
    at each time point.
    """
    def __init__(self, emergence=None, disease='malaria', name='mosquitoes', **kwargs):
        super().__init__(name=name, **kwargs)
        self.define_pars(emergence=np.array(msd.DEFAULT_EMERGENCE if emergence is None else emergence, dtype=float))
        self.disease = disease
        self.eir = 0.0 # Infectious bites per person per day in each node, which the people's rates read
        return

    def init_results(self):
        super().init_results()
        self.define_results(*[ss.Result(k, dtype=float, label=label) for k,label in
                              [('Sv', 'Susceptible mosquitoes'), ('Ev', 'Exposed mosquitoes'), ('Iv', 'Infectious mosquitoes'),
                               ('mosquitoes', 'Mosquitoes'), ('eir', 'EIR (infectious bites per person per day)')]])
        return

    @property
    def mal(self):
        return self.sim.diseases[self.disease]

    def par(self, key):
        """ A parameter of the malaria disease, as a number (rates per day, durations in days) """
        val = self.mal.pars[key]
        val = getattr(val, 'raw', val)
        if isinstance(val, ss.dur):
            return val.days
        if isinstance(val, ss.TimePar):
            return val.to('day').value
        return float(val)

    def emergence_now(self):
        """ The emergence in each node now, per person per day (interpolated between months, as `ms.monthly()`) """
        em = np.atleast_2d(self.pars.emergence)
        t = (self.t.now('year') - self.sim.t.yearvec[0]) % 1
        x = np.append(np.arange(12)/12, 1.0)
        out = np.array([np.interp(t, x, np.append(row, row[0])) for row in em])
        return out if len(out) == self.n_nodes else np.full(self.n_nodes, out[0])

    def people_by_node(self):
        """ The people alive in each node, the share of them (weighted by biting) that are infectious, and the node of each person """
        people = self.sim.people
        alive = people.alive.raw
        patch = np.asarray(people.patch.raw, dtype=int) if 'patch' in people else np.zeros(len(alive), dtype=int)
        n = np.bincount(patch[alive], minlength=self.n_nodes).astype(float)
        infectious = self.mal.I.raw & alive
        w = np.ones(len(alive))
        if self.mal.biting_weights is not None:
            w = self.mal.biting_weights[np.asarray(people.biting.raw, dtype=int)]
        n_inf = np.bincount(patch[infectious], weights=w[infectious], minlength=self.n_nodes)
        return n, np.divide(n_inf, n, out=np.zeros(self.n_nodes), where=n > 0), patch

    def bednets_by_node(self, n, patch):
        """ The share of indoor bites blocked by the bednets delivered, and the chance per indoor bite that a mosquito is killed by them: in each node, means over the people there """
        from .interventions import Bednet
        sim = self.sim
        blocked, killed = np.zeros(self.n_nodes), np.zeros(self.n_nodes)
        for net in [p for p in sim.products() if isinstance(p, Bednet)]:
            uids = np.flatnonzero((sim.people[net.dim].raw > 0) & sim.people.alive.raw)
            t = ss.days((sim.ti - net.ti_delivered.raw[uids])*sim.t.dt.days) # The time since delivery, as Starsim gives it to the waning
            for out,fn in [(blocked, net.protection), (killed, net.kill)]:
                out += np.bincount(patch[uids], weights=np.broadcast_to(fn(t), len(uids)), minlength=self.n_nodes)
        return blocked/np.maximum(n, 1), killed/np.maximum(n, 1)

    def init_post(self):
        super().init_post()
        mal = self.mal
        if mal.mosquito != 'agents':
            raise ValueError(f'ms.MosquitoAgents drives a malaria disease made with mosquito="agents", not "{mal.mosquito}"')
        self.n_nodes = 1 if mal.nodes is None else len(mal.nodes.names)
        self.rng = np.random.default_rng(np.random.SeedSequence([self.sim.pars.rand_seed, 2])) # The mosquitoes' own random numbers
        n, _, _ = self.people_by_node()
        n0 = self.rng.poisson(self.emergence_now()*self.par('life_expectancy')*n) # At the disease-free equilibrium of the emergence at the start
        self.state = np.zeros(max(100, int(1.5*n0.sum())), dtype=np.int8) # 0 a free slot, 1 susceptible, 2 exposed, 3 infectious; more are added as needed
        self.node = np.zeros(len(self.state), dtype=np.int32)
        self.add(n0)
        self.update_eir(n)
        return

    def add(self, n_new):
        """ Add new susceptible mosquitoes in each node, in free slots (adding slots if needed) """
        total = int(n_new.sum())
        free = np.flatnonzero(self.state == 0)
        if len(free) < total:
            extra = max(total - len(free), len(self.state))
            self.state = np.concatenate([self.state, np.zeros(extra, dtype=np.int8)])
            self.node = np.concatenate([self.node, np.zeros(extra, dtype=np.int32)])
            free = np.flatnonzero(self.state == 0)
        slots = free[:total]
        self.state[slots] = 1
        self.node[slots] = np.repeat(np.arange(self.n_nodes), n_new)
        return slots

    def step(self):
        """ Deaths, the end of the EIP, infection by biting infectious people, and emergence, each as a hazard over the step """
        dt = self.t.dt.days
        a, c, inf = self.par('bites_per_female_per_day'), self.par('prob_mosquito_infected_on_bite'), self.par('default_infectiousness')
        mu, sigma = 1/self.par('life_expectancy'), 1/self.par('eip_days')
        f_in = np.broadcast_to(self.mal.f_indoor_by_node(), (self.n_nodes,))
        n, frac_inf, patch = self.people_by_node()
        blocked, killed = self.bednets_by_node(n, patch)
        mu_all = mu + a*f_in*killed # Mortality in each node, with the bednets' killing

        # Deaths, among the live mosquitoes only
        live = np.flatnonzero(self.state > 0)
        node = self.node[live]
        dying = self.rng.random(len(live)) < -np.expm1(-mu_all*dt)[node]
        self.state[live[dying]] = 0
        live, node = live[~dying], node[~dying]

        # Infection, and the end of the EIP
        lam = a*c*inf*(1 - f_in*blocked)*frac_inf # The force of infection on a susceptible mosquito, per day, in each node
        p_inf = -np.expm1(-lam*dt) # Infected within the step...
        p_both = p_inf - lam*(np.exp(-sigma*dt) - np.exp(-lam*dt))/np.where(lam == sigma, np.inf, lam - sigma) # ...and infectious by its end (two exponential stages), so that the step's new infections incubate as in continuous time
        state = self.state[live]
        u = self.rng.random(len(live))
        sus = state == 1
        to_inf = (sus & (u < p_both[node])) | ((state == 2) & (u < -np.expm1(-sigma*dt)))
        to_exp = sus & (u >= p_both[node]) & (u < p_inf[node])
        self.state[live[to_inf]] = 3
        self.state[live[to_exp]] = 2

        # Emergence: of those who emerge during the step, the expected number alive at its end, so that the mean population is emergence/mortality, as in the ODE
        new = self.add(self.rng.poisson(self.emergence_now()*n*(-np.expm1(-mu_all*dt))/mu_all))
        bitten = new[self.rng.random(len(new)) < -np.expm1(-lam[self.node[new]]*dt/2)] # Those who emerged during the step were exposed for half of it, on average
        self.state[bitten] = 2
        self.update_eir(n)
        return

    def update_eir(self, n):
        """ The EIR in each node: infectious bites per person per day """
        a = self.par('bites_per_female_per_day')
        n_inf = np.bincount(self.node[self.state == 3], minlength=self.n_nodes)
        eir = np.divide(a*n_inf, n, out=np.zeros(self.n_nodes), where=n > 0)
        self.eir = eir if self.n_nodes > 1 else float(eir[0])
        self.n_people = n.sum()
        return

    def update_results(self):
        super().update_results()
        res, ti = self.results, self.ti
        counts = [np.count_nonzero(self.state == k) for k in [1, 2, 3]]
        res.Sv[ti], res.Ev[ti], res.Iv[ti] = counts
        res.mosquitoes[ti] = sum(counts)
        res.eir[ti] = self.par('bites_per_female_per_day')*counts[2]/max(self.n_people, 1)
        if ti == 0:
            self.eir_by_node = np.zeros((len(res.timevec), self.n_nodes))
        self.eir_by_node[ti] = self.eir
        return

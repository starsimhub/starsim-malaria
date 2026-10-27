"""
Malaria interventions as Starsim products, delivered by Starsim's deliveries

- ``ms.Bednet``: an insecticide-treated net, with four effects (repelling, blocking, killing,
  usage), each of which can wane, which protects its holder from indoor bites and blocks and
  kills mosquitoes

A vaccine is ``ss.Vaccine``, waning as a box then an exponential decay with ``ss.box()``, e.g.
``ss.Vaccine(efficacy=0.8, waning=ss.box(dur=ss.days(180), decay=ss.days(730)))``. They are
delivered by ``ss.vaccinate()`` (or ``ss.Delivery``), with targeting by ``where=`` (e.g.
``where=(ss.ppl.age < 5) & (ss.ppl.patch == 'Rural')``). A drug that clears infection is
``ss.Treatment(cure=dict(I=1.0))``, delivered by ``ss.treat()``, and an outbreak (seeding) is
``ss.Delivery(move=mal.S >> mal.I)``.
"""

import numpy as np
import starsim as ss

__all__ = ['Bednet']


class Bednet(ss.Vaccine):
    """
    An insecticide-treated bed net, with four effects, each of which can wane

    Its holder is protected from indoor bites by `1 - (1 - usage*repelling)*(1 - usage*blocking)`. In a model
    made with `ms.Malaria(bednets=...)` (as `ms.Sim` makes it when nets are delivered), mosquitoes also feel the
    nets: in each node, the mean of the same over people blocks indoor feeding on them, and
    `usage*(1 - repelling)*blocking*killing` is the chance per indoor bite that the mosquito is killed.

    Waning by one `ss.box()` without a decay for every effect (a net that lasts for a time) runs under every solver;
    any other waning, as agents only.

    Args:
        repelling (float): the probability that a mosquito is repelled from the house
        blocking (float): the probability that one that enters is blocked from biting
        killing (float): the probability that one blocked is killed (none that is repelled is)
        usage (float): the probability that the net is used, which scales the three effects
        waning (box/func/dict): how the effects wane, as a function of the time since delivery (as `ss.Vaccine(waning=...)`
            takes it): one for every effect (e.g. `ss.box(dur=ss.years(3))`), or a dict by effect, with None (or an effect
            left out) for none; by default, none
        name (str): the name of the product
        kwargs (dict): passed to `ss.Vaccine`

    Examples:
        ```python
        itn = ms.Bednet(repelling=0.5, blocking=0.9, killing=0.6, usage=0.8, waning=ss.box(dur=ss.years(3)))
        itn = ms.Bednet(blocking=0.9, killing=0.6, waning=dict(blocking=ss.box(dur=ss.days(365), decay=ss.days(365)), killing=ss.box(dur=ss.days(180), decay=ss.days(365))))
        intv = ss.vaccinate(itn, at='2005-01-01', coverage=0.5)
        ```
    """
    def __init__(self, repelling=1.0, blocking=1.0, killing=1.0, usage=1.0, waning=None, name=None, **kwargs):
        self.net = dict(repelling=repelling, blocking=blocking, killing=killing, usage=usage)
        for key,val in self.net.items():
            if not 0 <= val <= 1:
                raise ValueError(f'{key} is a probability, from 0 to 1, not {val}')
        fns = waning if isinstance(waning, dict) else dict.fromkeys(self.net, waning)
        unknown = set(fns) - set(self.net)
        if unknown:
            raise ValueError(f'waning= by effect has the effects {", ".join(self.net)}, not {", ".join(sorted(unknown))}')
        self.net_waning = {key:fns.get(key) for key in self.net}
        self.protection0 = float(self.protection(ss.days(0))) # Against the holder's indoor bites, at delivery
        super().__init__(efficacy=dict(bite_indoor=self.protection0), waning=self.protection_waning(), name=name, **kwargs)
        return

    def protection_waning(self):
        """ The waning of the holder's protection: none, the one box of every effect (exactly, if it has no decay), or as a share of that at delivery """
        fns = list(self.net_waning.values())
        if all(fn is None for fn in fns):
            return None
        if all(isinstance(fn, ss.box) and fn.decay is None and fn == fns[0] for fn in fns): # Every effect is full, then none
            return fns[0]
        return self.relative_protection

    def effects_at(self, t):
        """ The usage-scaled repelling, blocking, and killing, `t` after delivery (a duration) """
        val = {key:eff*(fn(t) if fn is not None else 1.0) for (key,eff),fn in zip(self.net.items(), self.net_waning.values())}
        u = val['usage']
        return u*val['repelling'], u*val['blocking'], u*val['killing']

    def protection(self, t):
        """ The share of the holder's indoor bites prevented, `t` after delivery """
        rep, blk, _ = self.effects_at(t)
        return 1 - (1 - rep)*(1 - blk)

    def kill(self, t):
        """ The chance per indoor bite on the holder that the mosquito is killed, `t` after delivery """
        rep, blk, kill = self.effects_at(t)
        return (1 - rep)*blk*kill

    def relative_protection(self, t):
        """ The protection as a share of that at delivery: the waning of `ss.Vaccine` """
        return self.protection(t)/self.protection0 if self.protection0 > 0 else 0*self.protection(t)

    def on_mosquitoes(self):
        """
        The share of indoor bites blocked by the net, and the chance per indoor bite that a mosquito is killed by it, each the
        mean over the people in the same node, as quantities in the mosquitoes' rates (from its coverage, `self.coverage`)
        """
        if not callable(self.waning) or isinstance(self.waning, ss.box): # Each holder counts the same: as much as at delivery, as long as they hold it
            return self.protection0*self.coverage, float(self.kill(ss.days(0)))*self.coverage
        return self.coverage(self.protection), self.coverage(self.kill)

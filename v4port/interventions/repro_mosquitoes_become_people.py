""" Repro: the spec's Ross-Macdonald model (06-routes.md), given a starting number of mosquitoes, makes the mosquitoes the agents and the humans counts, silently """
import starsim as ss; print(ss.__version__, ss.__file__)

def make(init):
    Sh, Eh, Ih, Rh = ss.states(4)
    Sv, Ev, Iv     = ss.states(3)
    Nh = (Sh + Eh + Ih + Rh).n
    a = ss.perday(0.3); b, c = 0.5, 0.5
    malaria = ss.Disease(
        infection  = Sh >> Eh @ a * b * Iv.n / Nh,
        latency    = Eh >> Ih @ ss.days(12),
        recovery   = Ih >> Rh @ ss.days(60),
        emergence  = ss.null >> Sv @ ss.perday(5000),
        biting     = Sv >> Ev @ a * c * Ih.n / Nh,
        incubation = Ev >> Iv @ ss.days(10),
        mortality  = Sv + Ev + Iv >> ss.null @ ss.days(14),
        init = init, name='malaria',
    )
    return ss.Sim(malaria, n_agents=5_000, unit='day', stop=100, verbose=0)

for init in [None, dict(Sv=3_000), dict(Sv=70_000), dict(Sv=70_000, Ih=0.01)]:
    try:
        sim = make(init).init()
    except Exception as e:
        print(f'init={init}: {type(e).__name__}: {e}'); continue
    g = sim.diseases.malaria.graph
    sim.run()
    r = sim.results.malaria
    print(f'init={init}: people = {[s.name for s in g.people_states]}, counts = {[s.name for s in g.count_states]}; at day 100 n_Sh {r.n_Sh[-1]:.0f}, n_Ih {r.n_Ih[-1]:.0f}, n_Sv {r.n_Sv[-1]:.0f}, n_Iv {r.n_Iv[-1]:.0f}, people alive {sim.results.n_alive[-1]:.0f}')

# Saying outright that the mosquitoes are not people, as 06-routes.md suggests, crashes when only they have initial values
Sh, Eh, Ih, Rh = ss.states(4)
Sv, Ev, Iv = [ss.State(n, people=False) for n in ['Sv', 'Ev', 'Iv']]
Nh = (Sh + Eh + Ih + Rh).n
a = ss.perday(0.3)
for init in [dict(Sv=3_000), dict(Sv=3_000, Sh=1.0)]:
    try:
        malaria = ss.Disease(infection=Sh >> Eh @ a * 0.5 * Iv.n / Nh, latency=Eh >> Ih @ ss.days(12), recovery=Ih >> Rh @ ss.days(60),
                             emergence=ss.null >> Sv @ ss.perday(5000), biting=Sv >> Ev @ a * 0.5 * Ih.n / Nh, incubation=Ev >> Iv @ ss.days(10),
                             mortality=Sv + Ev + Iv >> ss.null @ ss.days(14), init=init, name='malaria')
        print(f'people=False, init={init}: people = {[s.name for s in malaria.graph.people_states]}')
    except Exception as e:
        print(f'people=False, init={init}: {type(e).__name__}: {e}')

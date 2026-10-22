"""
disease.R0() on the spec's own Ross-Macdonald example (06-routes.md, "Vector-borne transmission"), with constant
emergence. Analytic: R0 = m a^2 b c exp(-mu*EIP)... here with exponential dwells, R0 = (a b M/Nh)(1/recovery) x (a c)(sigma/(sigma+mu))(1/mu)
"""
import starsim as ss; print(ss.__version__, ss.__file__)
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
    init = dict(Ih=0.01), name='malaria')
sim = ss.Sim(malaria, total_pop=5_000, unit='day', stop=365, verbose=0).run()
mu, sigma, r, M, N = 1/14, 1/10, 1/60, 5000*14, 5000
R0 = (0.3*b*M/N/r) * (0.3*c*sigma/(sigma + mu)/mu)
print(f'malaria.R0() = {malaria.R0()}; analytic (squared form, the next-generation matrix gives its square root) = {R0:.2f}, sqrt = {R0**0.5:.2f}')
print(f'prevalence in humans at the end: {(sim.results.malaria.n_Ih[-1] + sim.results.malaria.n_Eh[-1])/5000:.3f}')

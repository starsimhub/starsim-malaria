"""
Script to test hybrid compartmental/ABM modeling
"""

import malaria_hybrid as mh

if __name__ == '__main__':
    Nh = 10000 # Number of humans
    hum_pars = dict(b=0.15, r=1/120, Ih0=10)
    mos_pars = dict(c=0.15, mu_m=0.10, sigma_m=1/12, m0=5, Im0=100)

    combos = [
        ('ode', mh.HumanODE,  'ode', mh.MosquitoODE),
        ('abm', mh.HumanABM,  'ode', mh.MosquitoODE),
        ('ode', mh.HumanODE,  'abm', mh.MosquitoABM),
        ('abm', mh.HumanABM,  'abm', mh.MosquitoABM),
    ]

    sims = []
    for hl, H, ml, M in combos:
        label = f'Human={hl.upper()}, Mosquito={ml.upper()}'
        sim_pars = dict(a=0.24, Nh=Nh, start=0, stop=365*5, dt=1, label=label)
        sim = mh.Sim(pars=sim_pars, human=H(**hum_pars), mosquito=M(**mos_pars))
        sims.append(sim)
    
    # Run the sims
    msim = mh.MultiSim(sims).run(debug=True)

    # Plot
    msim.plot_malaria()
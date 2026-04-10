# Prompt 1 (Claude, unsuccessful)

implement a python file for the ross-macdonald malaria model based on this paper:
/home/cliffk/idm/ngm/starsim-ngm/malaria_hybrid/ross-macdonald.pdf

use your judgement about what version to implement, but keep it simple. the code should have the following:
1) a single class called RossMacdonaldModel() that handles everything
2) self.pars = sc.objdict() to store the parameters; within the code, you can shorten to p = self.pars, with individual parameters accessed as e.g. p.M, p.H
3) a run() method that uses euler integration to solve the model with a given start, stop, dt (plus initial conditions etc)
4) a plot() method that plots key dynamics from the model
5) a main block that includes rmm = RossMacdonaldModel(); rm.run(); rm.plot()

# Prompt 2 (ChatGPT, successful)

write a python script implementing the ross-macdonald model. include seasonality. write it as a class, with a separate dict of parameters, and a run() method and a plot() method. the plot() method should show human and mosquito prevalence on one panel. also show R0 in a separate panel. you can add other panels for any other quantities you think are important.
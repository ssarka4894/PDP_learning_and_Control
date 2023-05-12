# inspired from,
# @article{jin2019pontryagin,
#   title={Pontryagin Differentiable Programming: An End-to-End Learning and Control Framework},
#   author={Jin, Wanxin and Wang, Zhaoran and Yang, Zhuoran and Mou, Shaoshuai},
#   journal={arXiv preprint arXiv:1912.12970},
#   year={2019}
# }

import matplotlib.pyplot as plt
import scipy.io as sio

class Loss_Computation:
   def __init__(self, pend, pop_pendulum, lqr_solver):
       self.system = pend
       self.pop = pop_pendulum
       self.aop = lqr_solver
        
   def plot_loss(self):
       pdp_loss_data = []
       for i in range(1):
           load_pdp_results = sio.loadmat('data_pdp_pendulum/PDP_results_trial_' + str(i))
           loss_data = load_pdp_results['results']['ioc_loss_trace'][0, 0].flatten()
           pdp_loss_data += [loss_data]
       

       params = {'axes.labelsize': 30,
                 'axes.titlesize': 30,
                 'xtick.labelsize':20,
                 'ytick.labelsize':20,
                 'legend.fontsize':16}
       plt.rcParams.update(params)
       
       # plot
       fig = plt.figure( figsize=(15, 8))
       ax = fig.subplots()
       
       for iml_loss in pdp_loss_data:
           ax.set_yscale('log')
           ax.set_xlim(-500, 10000)
       
           ax.set_xlabel('Iteration')
           ax.tick_params(axis='both', which='major')
           ax.set_facecolor('#E6E6E6')
           ax.grid()
           line_iml,=ax.plot(iml_loss, marker='o', markevery=1000, color = [0.6350, 0.0780, 0.1840], linewidth=4,markersize=10)
           ax.set_title('Trial #'+str(1))
           ax.legend([line_iml],['PDP'],facecolor='white',framealpha=0.5)
       ax.set_ylabel('Pendulum imitation loss')
       plt.show()
       
        


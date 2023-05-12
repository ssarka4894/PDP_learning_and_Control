# @article{jin2019pontryagin,
#   title={Pontryagin Differentiable Programming: An End-to-End Learning and Control Framework},
#   author={Jin, Wanxin and Wang, Zhaoran and Yang, Zhuoran and Mou, Shaoshuai},
#   journal={arXiv preprint arXiv:1912.12970},
#   year={2019}
# }

from casadi import *
import numpy
from scipy import interpolate

# initializing primal optimization program
class POP:

    def __init__(self, weights=None, state = None, state_lb=[], state_ub=[], control = None, control_lb=[], control_ub=[], ode = None,
                 stage_cost = None, terminal_cost = None):
        self.weights = weights
        self.n_weights = self.weights.numel()
        self.state = state
        self.n_state = self.state.numel()
        self.state_lb = self.n_state * [-1e20]
        self.state_ub = self.n_state * [1e20]
        self.control = control
        self.n_control = self.control.numel()
        self.control_lb = self.n_control * [-1e20]
        self.control_ub = self.n_control * [1e20]
        self.dyn = ode
        self.dyn_fn = casadi.Function('dynamics', [self.state, self.control, self.weights], [self.dyn])
        self.stage_cost = stage_cost
        self.stage_cost_fn = casadi.Function('stage_cost', [self.state, self.control, self.weights], [self.stage_cost])
        self.terminal_cost = terminal_cost
        self.terminal_cost_fn = casadi.Function('terminal_cost', [self.state, self.weights], [self.terminal_cost])

    def popSolver(self, initial_state, horizon, weights_value=1, print_level=0):

        if type(initial_state) == numpy.ndarray:
            initial_state = initial_state.flatten().tolist()

        # Start with an empty NLP
        traj = []
        traj0 = []
        lb_traj = []
        ub_traj = []
        Cost = 0
        constraint = []
        lb_constraint = []
        ub_constraint = []

        # "Lift" initial conditions
        Xk = MX.sym('X0', self.n_state)
        traj += [Xk]
        lb_traj += initial_state
        ub_traj += initial_state
        traj0 += initial_state

        
        for k in range(horizon):
            Uk = MX.sym('U_' + str(k), self.n_control)
            traj += [Uk]
            lb_traj += self.control_lb
            ub_traj += self.control_ub
            traj0 += [0.5 * (x + y) for x, y in zip(self.control_lb, self.control_ub)]

            Xnext = self.dyn_fn(Xk, Uk, weights_value)
            Ck = self.stage_cost_fn(Xk, Uk, weights_value)
            Cost = Cost + Ck

            Xk = MX.sym('X_' + str(k + 1), self.n_state)
            traj += [Xk]
            lb_traj += self.state_lb
            ub_traj += self.state_ub
            traj0 += [0.5 * (x + y) for x, y in zip(self.state_lb, self.state_ub)]

            # Add equality constraint
            constraint += [Xnext - Xk]
            lb_constraint += self.n_state * [0]
            ub_constraint += self.n_state * [0]

        # Adding the final cost
        Cost = Cost + self.terminal_cost_fn(Xk, weights_value)

        # USing IPOPT to solve for trajectories based on quadratic cost structure
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': Cost, 'x': vertcat(*traj), 'g': vertcat(*constraint)}
        solver = nlpsol('solver', 'ipopt', prob, opts)
        # Solve the NLP
        sol = solver(x0=traj0, lbx=lb_traj, ubx=ub_traj, lbg=lb_constraint, ubg=ub_constraint)
        traj_opt = sol['x'].full().flatten()

        # take the optimal control and state
        sol_traj = numpy.concatenate((traj_opt, self.n_control * [0]))
        sol_traj = numpy.reshape(sol_traj, (-1, self.n_state + self.n_control))
        state_trajectories = sol_traj[:, 0:self.n_state]
        control_trajectories = numpy.delete(sol_traj[:, self.n_state:], -1, 0)
        time = numpy.array([k for k in range(horizon + 1)])
        
        # Solving using IPOPT
        costate_trajectories = numpy.reshape(sol['lam_g'].full().flatten(), (-1, self.n_state))

        # output
        opt_sol = {"state_trajectories": state_trajectories,
                   "control_trajectories": control_trajectories,
                   "costate_trajectories": costate_trajectories,
                   'weights_value': weights_value,
                   "time": time,
                   "horizon": horizon,
                   "cost": sol['f'].full()}

        return opt_sol

    # adapted from https://github.com/wanxinjin/Pontryagin-Differentiable-Programming/blob/master/PDP/PDP.py
    def differential_PMP(self):
        # assert hasattr(self, 'state'), "Define the state variable first!"
        # assert hasattr(self, 'control'), "Define the control variable first!"
        # assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        # assert hasattr(self, 'stage_cost'), "Define the running cost/reward function first!"
        # assert hasattr(self, 'terminal_cost'), "Define the final cost/reward function first!"

        # Define the Hamiltonian function
        self.costate = casadi.SX.sym('lambda', self.state.numel())
        self.stage_Hamiltonian = self.stage_cost + dot(self.dyn, self.costate)  # path Hamiltonian
        self.terminal_Hamiltonian = self.terminal_cost  # final Hamiltonian

        # Differentiating dynamics; notations here are consistent with the PDP paper
        self.dfx = jacobian(self.dyn, self.state)
        self.dfx_fn = casadi.Function('dfx', [self.state, self.control, self.weights], [self.dfx])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfu_fn = casadi.Function('dfu', [self.state, self.control, self.weights], [self.dfu])
        self.dfe = jacobian(self.dyn, self.weights)
        self.dfe_fn = casadi.Function('dfe', [self.state, self.control, self.weights], [self.dfe])

        # First-order derivative of path Hamiltonian
        self.dHx = jacobian(self.stage_Hamiltonian, self.state).T
        self.dHx_fn = casadi.Function('dHx', [self.state, self.control, self.costate, self.weights], [self.dHx])
        self.dHu = jacobian(self.stage_Hamiltonian, self.control).T
        self.dHu_fn = casadi.Function('dHu', [self.state, self.control, self.costate, self.weights], [self.dHu])

        # Second-order derivative of path Hamiltonian
        self.ddHxx = jacobian(self.dHx, self.state)
        self.ddHxx_fn = casadi.Function('ddHxx', [self.state, self.control, self.costate, self.weights], [self.ddHxx])
        self.ddHxu = jacobian(self.dHx, self.control)
        self.ddHxu_fn = casadi.Function('ddHxu', [self.state, self.control, self.costate, self.weights], [self.ddHxu])
        self.ddHxe = jacobian(self.dHx, self.weights)
        self.ddHxe_fn = casadi.Function('ddHxe', [self.state, self.control, self.costate, self.weights], [self.ddHxe])
        self.ddHux = jacobian(self.dHu, self.state)
        self.ddHux_fn = casadi.Function('ddHux', [self.state, self.control, self.costate, self.weights], [self.ddHux])
        self.ddHuu = jacobian(self.dHu, self.control)
        self.ddHuu_fn = casadi.Function('ddHuu', [self.state, self.control, self.costate, self.weights], [self.ddHuu])
        self.ddHue = jacobian(self.dHu, self.weights)
        self.ddHue_fn = casadi.Function('ddHue', [self.state, self.control, self.costate, self.weights], [self.ddHue])

        # First-order derivative of final Hamiltonian
        self.dhx = jacobian(self.terminal_Hamiltonian, self.state).T
        self.dhx_fn = casadi.Function('dhx', [self.state, self.weights], [self.dhx])

        # second order differential of path Hamiltonian
        self.ddhxx = jacobian(self.dhx, self.state)
        self.ddhxx_fn = casadi.Function('ddhxx', [self.state, self.weights], [self.ddhxx])
        self.ddhxe = jacobian(self.dhx, self.weights)
        self.ddhxe_fn = casadi.Function('ddhxe', [self.state, self.weights], [self.ddhxe])

    def getAuxSys(self, state_trajectories, control_trajectories, costate_trajectories, weights_value=1):
        statement = [hasattr(self, 'dfx_fn'), hasattr(self, 'dfu_fn'), hasattr(self, 'dfe_fn'),
                     hasattr(self, 'ddHxx_fn'), \
                     hasattr(self, 'ddHxu_fn'), hasattr(self, 'ddHxe_fn'), hasattr(self, 'ddHux_fn'),
                     hasattr(self, 'ddHuu_fn'), \
                     hasattr(self, 'ddHue_fn'), hasattr(self, 'ddhxx_fn'), hasattr(self, 'ddhxe_fn'), ]
        if not all(statement):
            self.differential_PMP()

        # Initialize the coefficient matrices of the auxiliary control system: note that all the notations used here are
        # consistent with the notations defined in the PDP paper.
        dynF, dynG, dynE = [], [], []
        matHxx, matHxu, matHxe, matHux, matHuu, matHue, mathxx, mathxe = [], [], [], [], [], [], [], []

        # Solve the above coefficient matrices
        for t in range(numpy.size(control_trajectories, 0)):
            curr_x = state_trajectories[t, :]
            curr_u = control_trajectories[t, :]
            next_lambda = costate_trajectories[t, :]
            dynF += [self.dfx_fn(curr_x, curr_u, weights_value).full()]
            dynG += [self.dfu_fn(curr_x, curr_u, weights_value).full()]
            dynE += [self.dfe_fn(curr_x, curr_u, weights_value).full()]
            matHxx += [self.ddHxx_fn(curr_x, curr_u, next_lambda, weights_value).full()]
            matHxu += [self.ddHxu_fn(curr_x, curr_u, next_lambda, weights_value).full()]
            matHxe += [self.ddHxe_fn(curr_x, curr_u, next_lambda, weights_value).full()]
            matHux += [self.ddHux_fn(curr_x, curr_u, next_lambda, weights_value).full()]
            matHuu += [self.ddHuu_fn(curr_x, curr_u, next_lambda, weights_value).full()]
            matHue += [self.ddHue_fn(curr_x, curr_u, next_lambda, weights_value).full()]
        mathxx = [self.ddhxx_fn(state_trajectories[-1, :], weights_value).full()]
        mathxe = [self.ddhxe_fn(state_trajectories[-1, :], weights_value).full()]

        auxSys = {"dynF": dynF,
                  "dynG": dynG,
                  "dynE": dynE,
                  "Hxx": matHxx,
                  "Hxu": matHxu,
                  "Hxe": matHxe,
                  "Hux": matHux,
                  "Huu": matHuu,
                  "Hue": matHue,
                  "hxx": mathxx,
                  "hxe": mathxe}
        return auxSys

 # adapted from https://github.com/wanxinjin/Pontryagin-Differentiable-Programming/blob/master/PDP/PDP.py
class LQR:

    def __init__(self, project_name="LQR system"):
        self.project_name = project_name

    def setDyn(self, dynF, dynG, dynE=None):
        if type(dynF) is numpy.ndarray:
            self.dynF = [dynF]
            self.n_state = numpy.size(dynF, 0)
        elif type(dynF[0]) is numpy.ndarray:
            self.dynF = dynF
            self.n_state = numpy.size(dynF[0], 0)
        else:
            assert False, "Type of dynF matrix should be numpy.ndarray  or list of numpy.ndarray"

        if type(dynG) is numpy.ndarray:
            self.dynG = [dynG]
            self.n_control = numpy.size(dynG, 1)
        elif type(dynG[0]) is numpy.ndarray:
            self.dynG = dynG
            self.n_control = numpy.size(self.dynG[0], 1)
        else:
            assert False, "Type of dynG matrix should be numpy.ndarray  or list of numpy.ndarray"

        if dynE is not None:
            if type(dynE) is numpy.ndarray:
                self.dynE = [dynE]
                self.n_batch = numpy.size(dynE, 1)
            elif type(dynE[0]) is numpy.ndarray:
                self.dynE = dynE
                self.n_batch = numpy.size(dynE[0], 1)
            else:
                assert False, "Type of dynE matrix should be numpy.ndarray, list of numpy.ndarray, or None"
        else:
            self.dynE = None
            self.n_batch = None

    def initialize_stage_cost(self, Hxx, Huu, Hxu=None, Hux=None, Hxe=None, Hue=None):

        if type(Hxx) is numpy.ndarray:
            self.Hxx = [Hxx]
        elif type(Hxx[0]) is numpy.ndarray:
            self.Hxx = Hxx
        else:
            assert False, "Type of path cost Hxx matrix should be numpy.ndarray or list of numpy.ndarray, or None"

        if type(Huu) is numpy.ndarray:
            self.Huu = [Huu]
        elif type(Huu[0]) is numpy.ndarray:
            self.Huu = Huu
        else:
            assert False, "Type of path cost Huu matrix should be numpy.ndarray or list of numpy.ndarray, or None"

        if Hxu is not None:
            if type(Hxu) is numpy.ndarray:
                self.Hxu = [Hxu]
            elif type(Hxu[0]) is numpy.ndarray:
                self.Hxu = Hxu
            else:
                assert False, "Type of path cost Hxu matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hxu = None

        if Hux is not None:
            if type(Hux) is numpy.ndarray:
                self.Hux = [Hux]
            elif type(Hux[0]) is numpy.ndarray:
                self.Hux = Hux
            else:
                assert False, "Type of path cost Hux matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hux = None

        if Hxe is not None:
            if type(Hxe) is numpy.ndarray:
                self.Hxe = [Hxe]
            elif type(Hxe[0]) is numpy.ndarray:
                self.Hxe = Hxe
            else:
                assert False, "Type of path cost Hxe matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hxe = None

        if Hue is not None:
            if type(Hue) is numpy.ndarray:
                self.Hue = [Hue]
            elif type(Hue[0]) is numpy.ndarray:
                self.Hue = Hue
            else:
                assert False, "Type of path cost Hue matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hue = None

    def initialize_terminal_cost(self, hxx, hxe=None):

        if type(hxx) is numpy.ndarray:
            self.hxx = [hxx]
        elif type(hxx[0]) is numpy.ndarray:
            self.hxx = hxx
        else:
            assert False, "Type of final cost hxx matrix should be numpy.ndarray or list of numpy.ndarray"

        if hxe is not None:
            if type(hxe) is numpy.ndarray:
                self.hxe = [hxe]
            elif type(hxe[0]) is numpy.ndarray:
                self.hxe = hxe
            else:
                assert False, "Type of final cost hxe matrix should be numpy.ndarray, list of numpy.ndarray, or None"
        else:
            self.hxe = None

    def lqrSolver(self, initial_state, horizon):

        # Data pre-processing
        n_state = numpy.size(self.dynF[0], 1)
        if type(initial_state) is list:
            self.ini_x = numpy.array(initial_state, numpy.float64)
            if self.ini_x.ndim == 2:
                self.n_batch = numpy.size(self.ini_x, 1)
            else:
                self.n_batch = 1
                self.ini_x = self.ini_x.reshape(n_state, -1)
        elif type(initial_state) is numpy.ndarray:
            self.ini_x = initial_state
            if self.ini_x.ndim == 2:
                self.n_batch = numpy.size(self.ini_x, 1)
            else:
                self.n_batch = 1
                self.ini_x = self.ini_x.reshape(n_state, -1)
        else:
            assert False, "Initial state should be of numpy.ndarray type or list!"

        self.horizon = horizon

        if self.dynE is not None:
            assert self.n_batch == numpy.size(self.dynE[0],
                                              1), "Number of data batch is not consistent with column of dynE"

        # Check the time horizon
        if len(self.dynF) > 1 and len(self.dynF) != self.horizon:
            assert False, "time-varying dynF is not consistent with given horizon"
        elif len(self.dynF) == 1:
            F = self.horizon * self.dynF
        else:
            F = self.dynF

        if len(self.dynG) > 1 and len(self.dynG) != self.horizon:
            assert False, "time-varying dynG is not consistent with given horizon"
        elif len(self.dynG) == 1:
            G = self.horizon * self.dynG
        else:
            G = self.dynG

        if self.dynE is not None:
            if len(self.dynE) > 1 and len(self.dynE) != self.horizon:
                assert False, "time-varying dynE is not consistent with given horizon"
            elif len(self.dynE) == 1:
                E = self.horizon * self.dynE
            else:
                E = self.dynE
        else:
            E = self.horizon * [numpy.zeros(self.ini_x.shape)]

        if len(self.Hxx) > 1 and len(self.Hxx) != self.horizon:
            assert False, "time-varying Hxx is not consistent with given horizon"
        elif len(self.Hxx) == 1:
            Hxx = self.horizon * self.Hxx
        else:
            Hxx = self.Hxx

        if len(self.Huu) > 1 and len(self.Huu) != self.horizon:
            assert False, "time-varying Huu is not consistent with given horizon"
        elif len(self.Huu) == 1:
            Huu = self.horizon * self.Huu
        else:
            Huu = self.Huu

        hxx = self.hxx

        if self.hxe is None:
            hxe = [numpy.zeros(self.ini_x.shape)]

        if self.Hxu is None:
            Hxu = self.horizon * [numpy.zeros((self.n_state, self.n_control))]
        else:
            if len(self.Hxu) > 1 and len(self.Hxu) != self.horizon:
                assert False, "time-varying Hxu is not consistent with given horizon"
            elif len(self.Hxu) == 1:
                Hxu = self.horizon * self.Hxu
            else:
                Hxu = self.Hxu

        if self.Hux is None:  # Hux is the transpose of Hxu
            Hux = self.horizon * [numpy.zeros((self.n_control, self.n_state))]
        else:
            if len(self.Hux) > 1 and len(self.Hux) != self.horizon:
                assert False, "time-varying Hux is not consistent with given horizon"
            elif len(self.Hux) == 1:
                Hux = self.horizon * self.Hux
            else:
                Hux = self.Hux

        if self.Hxe is None:
            Hxe = self.horizon * [numpy.zeros((self.n_state, self.n_batch))]
        else:
            if len(self.Hxe) > 1 and len(self.Hxe) != self.horizon:
                assert False, "time-varying Hxe is not consistent with given horizon"
            elif len(self.Hxe) == 1:
                Hxe = self.horizon * self.Hxe
            else:
                Hxe = self.Hxe

        if self.Hue is None:
            Hue = self.horizon * [numpy.zeros((self.n_control, self.n_batch))]
        else:
            if len(self.Hue) > 1 and len(self.Hue) != self.horizon:
                assert False, "time-varying Hue is not consistent with given horizon"
            elif len(self.Hue) == 1:
                Hue = self.horizon * self.Hue
            else:
                Hue = self.Hue

        # Solve the Riccati equations: the notations used here are consistent with Lemma 4.2 in the PDP paper
        I = numpy.eye(self.n_state)
        PP = self.horizon * [numpy.zeros((self.n_state, self.n_state))]
        WW = self.horizon * [numpy.zeros((self.n_state, self.n_batch))]
        PP[-1] = self.hxx[0]
        WW[-1] = self.hxe[0]
        for t in range(self.horizon - 1, 0, -1):
            P_next = PP[t]
            W_next = WW[t]
            invHuu = numpy.linalg.inv(Huu[t])
            GinvHuu = numpy.matmul(G[t], invHuu)
            HxuinvHuu = numpy.matmul(Hxu[t], invHuu)
            A_t = F[t] - numpy.matmul(GinvHuu, numpy.transpose(Hxu[t]))
            R_t = numpy.matmul(GinvHuu, numpy.transpose(G[t]))
            M_t = E[t] - numpy.matmul(GinvHuu, Hue[t])
            Q_t = Hxx[t] - numpy.matmul(HxuinvHuu, numpy.transpose(Hxu[t]))
            N_t = Hxe[t] - numpy.matmul(HxuinvHuu, Hue[t])

            temp_mat = numpy.matmul(numpy.transpose(A_t), numpy.linalg.inv(I + numpy.matmul(P_next, R_t)))
            P_curr = Q_t + numpy.matmul(temp_mat, numpy.matmul(P_next, A_t))
            W_curr = N_t + numpy.matmul(temp_mat, W_next + numpy.matmul(P_next, M_t))

            PP[t - 1] = P_curr
            WW[t - 1] = W_curr

        # Compute the trajectory using the Raccti matrices obtained from the above: the notations used here are
        # consistent with the PDP paper in Lemma 4.2
        state_trajectories = (self.horizon + 1) * [numpy.zeros((self.n_state, self.n_batch))]
        control_trajectories = (self.horizon) * [numpy.zeros((self.n_control, self.n_batch))]
        costate_trajectories = (self.horizon) * [numpy.zeros((self.n_state, self.n_batch))]
        state_trajectories[0] = self.ini_x
        for t in range(self.horizon):
            P_next = PP[t]
            W_next = WW[t]
            invHuu = numpy.linalg.inv(Huu[t])
            GinvHuu = numpy.matmul(G[t], invHuu)
            A_t = F[t] - numpy.matmul(GinvHuu, numpy.transpose(Hxu[t]))
            M_t = E[t] - numpy.matmul(GinvHuu, Hue[t])
            R_t = numpy.matmul(GinvHuu, numpy.transpose(G[t]))

            x_t = state_trajectories[t]
            u_t = -numpy.matmul(invHuu, numpy.matmul(numpy.transpose(Hxu[t]), x_t) + Hue[t]) \
                  - numpy.linalg.multi_dot([invHuu, numpy.transpose(G[t]), numpy.linalg.inv(I + numpy.dot(P_next, R_t)),
                                            (numpy.matmul(numpy.matmul(P_next, A_t), x_t) + numpy.matmul(P_next,
                                                                                                         M_t) + W_next)])

            x_next = numpy.matmul(F[t], x_t) + numpy.matmul(G[t], u_t) + E[t]
            lambda_next = numpy.matmul(P_next, x_next) + W_next

            state_trajectories[t + 1] = x_next
            control_trajectories[t] = u_t
            costate_trajectories[t] = lambda_next
        time = [k for k in range(self.horizon + 1)]

        opt_sol = {'state_trajectories': state_trajectories,
                   'control_trajectories': control_trajectories,
                   'costate_trajectories': costate_trajectories,
                   'time': time}
        return opt_sol



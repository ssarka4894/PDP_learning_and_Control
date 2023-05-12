# IOC using PDP Learning

## Brief
In this study, we investigate the use of Pontryagin
Differential Programming (PDP) towards performing Inverse
Optimal Control (IOC) of discrete-time, nonlinear systems. In
particular, for given system dynamics with control objective
parameterized by weights w, we consider the problem of estab-
lishing optimal values of the weights w, minimizing the imitation
loss L between the resulting system trajectories and some desired
system trajectory. In order to find such weight, we use the PDP
methodology to compute the gradient dL
dw of the loss function with
respect to the weights. Using this gradient, we can then apply
a steepest descent algorithm to compute improved estimates of
the weights, iteratively converging to a local minimizer of the
loss function. The proposed methodology is examined across a
diverse range of dynamic systems, verifying the results for a cart-
pole system presented in earlier works, as well as successfully
applying the method to perform IOC of an inverted pendulum
and a quadrotor system.

## Requirements
This project depends on the python modules `numpy`, `scipy`, `matplotlib`, `casadi`, `ffmpeg-python`.

```bash
pip install numpy scipy matplotlib casadi ffmpeg-python
```
You may need FFMPEG locally installed in Windows and from [FFMPEG](https://ffmpeg.org/download.html) and add the binary folder to the windows environment variables _PATH_.

## Step-by-Step instructions
There are 3 models implemented for the project.

- [Cart-Pole System](cart_pole_system/)
- [Quadcopter System](quad_copter_system/)
- [Simple Inverted Pendulum System](simple_pendulum_system/)


Go to file 'run_pdp_user_interface.py' and change the parameter variable
'user_choice' from 1 to 5 successively to go through successive stages of
computing pdp loss and achieving inverse optimal control.

## Steps to execute the sequence of PDP (you can skip step 2 and move on to 3, 4, 5 for viewing obtained results if needed)

1. `user_choice = 1` : Generate optimal trajectories with respect to assumed quadratic cost (Primal Optimization Program).
2. `user_choice = 2` : pdp iteration (Computing PDP loss, weight updates using Auxillary Optimization Program and Differential PMP).
3. `user_choice = 3` : plot loss trace vs iteration.
4. `user_choice = 4` : plot trajectories vs ground truth, plot weights vs iteration.
5. `user_choice = 5` : generate animations for compare primal optimizaton program and auxillary optimization program.

## Warnings:
While switching from one system model to another model, please restart the kernel or clear the variables of the previously
used system model and run it step wise with above choices in succession to avoid runtime errors.

## Virtual Environment (Optional)
To Run in a virual env, follow the instructions below. (_You need to have the python venv package installed_)

```bash
python -m venv .PDPvenv
.PDPvenv/Scripts/Activate
pip install -r requiments.txt
```

## Tested on
- Ubuntu and Windows PC
- Python 3.9.13
- Python 3.10.9
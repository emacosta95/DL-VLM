import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from src.qutip_lab.qutip_class import SpinOperator, SpinHamiltonian, SteadyStateSolver
from scipy.fft import fft, ifft
from scipy.sparse.linalg import eigsh
import qutip
from qutip.metrics import fidelity
from typing import List
from qutip import propagator
import os
from datetime import datetime
from scipy.sparse.linalg import eigsh,expm
from scipy.interpolate import interp1d
from src.tddft_methods.tddft_solver import Driving,second_derivative_formula
import argparse

parser = argparse.ArgumentParser()


parser.add_argument(
    "--diagnostic",
    type=bool,
    help="if activated, it disconnects every double check on the TDDFT reconstruction. It is used for debugging of the TDDFT method.",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--initial_state_ground_state",
    type=bool,
    help="if activated, it constrains the initial state to the ground state of H(t=0).",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--pbc",
    type=bool,
    help="if activated, it sets the periodic boundary conditions",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--condition_initial_state",
    type=str,
    help="it could be either 'fixed' -same initial state for each realization- or 'variable' ",
    default="fixed",
)

parser.add_argument(
    "--ndata",
    type=int,
    help="number of realization",
    default=100,
)

parser.add_argument(
    "--j",
    type=float,
    help="coupling of the nearest neighbour interaction",
    default=-1,
)

parser.add_argument(
    "--omega",
    type=float,
    help="strength of the longitudinal field",
    default=1,
)

parser.add_argument(
    "--rate_mean",
    type=float,
    help="the mean value of the rate distribution for the sampling of the driving",
    default=1.5,
)


parser.add_argument(
    "--rate_sigma",
    type=float,
    help="the standard deviation of the rate distribution for the sampling of the driving",
    default=1.5,
)

parser.add_argument(
    "--amplitude_max",
    type=float,
    help="the maximum value of the amplitude of the random driving",
    default=2.,
)

parser.add_argument(
    "--amplitude_min",
    type=float,
    help="the minimum value of the amplitude of the random driving",
    default=0.,
)

parser.add_argument(
    "--derivative_formula",
    type=str,
    help="finite difference formula for computing the second-order time derivative in the TDDFT method. it can be either '3-points', '5-points', '7-points' or '9-points' ",
    default='9-points',
)

parser.add_argument(
    "--tf",
    type=float,
    help="final time of the time evolution",
    default=20.,
)

parser.add_argument(
    "--steps",
    type=int,
    help="number of time steps for the time evolution",
    default=800,
)

parser.add_argument(
    "--final_steps",
    type=int,
    help="number of time steps for the final realizations",
    default=200,
)

parser.add_argument(
    "--l",
    type=int,
    help="length of the spin chain",
    default=8,
)

parser.add_argument(
    "--dz_threshold",
    type=float,
    help="threshold of the error for the TDDFT simulation",
    default=0.01,
)

args=parser.parse_args()


# Get the current date and time
now = datetime.now()

# Format the date and time into a string
# Example format: YYYY-MM-DD_HH-MM
formatted_date_time = now.strftime("%Y-%m-%d_%H-%M")


# hyperaparameters

# if this is true, we are making datasets just to check the feasibility of the TDDFT method
diagnostic=args.diagnostic
derivative_formula=args.derivative_formula #it can be either 3-points,5-points,7-points,9-points

# parameters

nbatch = 1
batch_size =args.ndata

initial_state_ground_state=args.initial_state_ground_state
pbc=args.pbc
condition_initial_state=args.condition_initial_state #'fixed'

# j coupling
j = args.j
# omega auxiliary field
omega = args.omega


rate_mean=args.rate_mean
rate_sigma=args.rate_sigma

amplitude_max=args.amplitude_max
amplitude_min=args.amplitude_min

tf = args.tf

steps = args.steps
steps_tddft=args.steps
final_steps= args.final_steps


time = np.linspace(0.0, tf, steps)
time_tddft=np.linspace(0.0, tf, steps_tddft)
time_final=np.linspace(0.0, tf, final_steps)

l=args.l #np.random.randint(2,10)


# we fix the seed (thi is to fix just for checking the algorithm)
if diagnostic:
    np.random.seed(42)


info=f'xx-z-x model with omega={omega:.1f}, coupling={j: .1f} external field with rate mean={rate_mean:.1f} and rate sigma={rate_sigma:.1f} amplitude max={amplitude_max:.1f} amplitude min={amplitude_min:.1f} tf={tf:.0f} steps={steps} l variable ndata={batch_size} initial state option={initial_state_ground_state} pbc={pbc}'
comments=condition_initial_state+f' Initial state ground state, with a diagonostic is {diagnostic} dataset. 2nd order time derivative with'+derivative_formula+' formula'
z_qutip_tot = np.zeros(( batch_size , final_steps,l))
z_auxiliary=np.zeros((batch_size,final_steps,l))
h_eff_tot = np.zeros(( batch_size , final_steps+1,l))
h_tot = np.zeros(( batch_size , final_steps+1,l))
current_qutip_tot = np.zeros(( batch_size , final_steps,l))
current_derivative_tot = np.zeros(( batch_size , final_steps,l))
x_sp_tot = np.zeros(( batch_size , final_steps,l))
ls=[]


idx=0
while(idx<batch_size):

    
    ham0 = SpinHamiltonian(
    direction_couplings=[("x", "x")],
    pbc=pbc,
    coupling_values=[j],
    size=l,
    )

    hamExtX = SpinOperator(index=[("x", i) for i in range(l)], coupling=[omega] * l, size=l)

    obs: List[qutip.Qobj] = []
    
    for i in range(l):
        z_op = SpinOperator(index=[("z", i)], coupling=[1.0], size=l, verbose=1)
        obs.append(z_op.qutip_op)
    
    
    initial_value=np.random.uniform(-1,1)
    if condition_initial_state=='variable':
        if initial_state_ground_state:
            hi=initial_value*np.ones((time.shape[0], l))
        else:
            hi=initial_value*np.zeros((time.shape[0], l))
    elif condition_initial_state=='fixed':
        hi=initial_value*np.zeros((time.shape[0], l))


    rate_cutoff = 10
    rate=rate_mean+rate_sigma*np.random.normal(size=rate_cutoff)
    delta = np.random.uniform(amplitude_min,amplitude_max,size=(rate_cutoff))
    
    h = (
        delta[:, None, None]
        * ((np.cos(np.pi+time[None, :, None] * rate[:, None, None])+1))
        + hi
    )

    h = np.average(h, axis=0)

    hamExtZ = SpinOperator(index=[("z", i) for i in range(l)], coupling=h[0], size=l)

    if initial_state_ground_state:
        eng, psi0 = (ham0.qutip_op + hamExtZ.qutip_op + hamExtX.qutip_op).eigenstates(
            eigvals=1
        )
        psi0 = psi0[
            0
        ]  # qutip.Qobj(psi0[:, 0], shape=psi0.shape, dims=([[2 for i in range(l)], [1]]))


    else:
        # we can build our own psi0
        psi_plus=qutip.basis(2,0)
        psi_minus=qutip.sigmam()*qutip.basis(2,0)
        
        p=np.random.uniform(0.,0.5,)
    
        
        for i in range(l):
            if i==0:
                psi0=psi_plus*np.sqrt(p)+np.sqrt(1-p)*psi_minus
            else:
                psi0=qutip.tensor(psi0,psi_plus*np.sqrt(p)+np.sqrt(1-p)*psi_minus)

    
    hamiltonian = [ham0.qutip_op + hamExtX.qutip_op]

    for i in range(l):
        drive_z = Driving(
            h=h,
            dt=time[1] - time[0],
            idx=i,
        )

        hamiltonian.append([obs[i], drive_z.field])

    # evolution and

    output = qutip.sesolve(hamiltonian, psi0, time, e_ops=obs )
    #current_exp = np.zeros((steps, l))
    z_exp = np.zeros((steps, l))

    for r in range(l):
        z_exp[:, r] = output.expect[r]
        #current_exp[:, r] = output.expect[l + r]



    # compute the effective field
    psi=np.zeros((2,l))
    psi[0] = np.sqrt((1 + z_exp[0]) / 2)
    psi[1] = np.sqrt((1 - z_exp[0]) / 2)


    # build up the operators
    x_op=np.array([[0.,1.],[1.,0]])
    z_op=np.array([[1.,0.],[0.,-1.]])

    
    # extrapolate the fields
    #f=interp1d(time,z_exp,axis=0)
    #z_tddft=f(time_tddft)
    z_tddft=z_exp
    current_derivative_tddft=second_derivative_formula(z_tddft,dt=time_tddft[1]-time_tddft[0],derivative_formula=derivative_formula)

    

    dt=time_tddft[1]-time_tddft[0]    

    z_reconstruction=np.zeros((steps_tddft,l))
    h_eff_vector=np.zeros((steps_tddft,l))
    for i in trange(steps_tddft):
        psi_r=psi.copy()
        for f in range(1):
            x_ave=np.einsum('al,ab,bl->l',np.conj(psi_r),x_op,psi_r)
            z_ave=np.einsum('al,ab,bl->l',np.conj(psi_r),z_op,psi_r)
            
            if pbc:
                #pbc
                nonlinear_term=np.abs(j)*(np.roll(x_ave,shift=1)+np.roll(x_ave,shift=-1))+omega
            
            else:
                #obc
                shift_plus=np.zeros(l)
                shift_plus[1:]=x_ave[1:] #np.roll(x_sp,shift=1,axis=-1)
                shift_minus=np.zeros(l)
                shift_minus[:-1]=x_ave[:-1] #np.roll(x_sp,shift=-1,axis=-1)
                #print(shift_minus,shift_plus)
                nonlinear_term=np.abs(j)*(shift_plus+shift_minus)+omega+10**-10
            h_eff=(0.25*current_derivative_tddft[i]/nonlinear_term+z_tddft[i]*nonlinear_term)/(x_ave+10**-10)
            h_eff_vector[i]=h_eff
            hamiltonian_t=nonlinear_term[:,None,None]*x_op[None,:,:]+h_eff[:,None,None]*z_op[None,:,:]
            exp_h_t=np.zeros((l,2,2),dtype=np.complex128)
            
            for r in range(l):
                exp_h_t[r]=expm(-1j*dt*hamiltonian_t[r])
            #print(exp_h_t)    
            psi_r=np.einsum('lab,bl->al',exp_h_t,psi)
            psi_r=psi_r/np.linalg.norm(psi_r,axis=0)
            
            
        psi=np.einsum('lab,bl->al',exp_h_t,psi)
        psi=psi/np.linalg.norm(psi,axis=0)
        
        z_reconstruction[i]=np.einsum('al,ab,bl->l',np.conj(psi),z_op,psi)

    dz=np.average(np.abs(z_reconstruction-z_tddft))
    print('dz=',dz,'idx=',idx,'p=',p)
    if diagnostic:
        dz=0.
    
    if dz<args.dz_threshold:
        # update the database
        h_eff_tot[idx,1:]=np.array([np.interp(time_final, time_tddft, h_eff_vector[:, i]) for i in range(h_eff_vector.shape[1])]).T
        h_tot[idx,1:]=np.array([np.interp(time_final, time, h[:, i]) for i in range(h.shape[1])]).T
        h_tot[idx,0]=z_exp[0]
        
        z_qutip_tot[idx,]=np.array([np.interp(time_final, time_tddft, z_tddft[:, i]) for i in range(z_tddft.shape[1])]).T
        z_auxiliary[idx,]= np.array([np.interp(time_final, time_tddft, z_reconstruction[:, i]) for i in range(z_reconstruction.shape[1])]).T
        current_derivative_tot[idx]=np.array([np.interp(time_final, time_tddft, current_derivative_tddft[:, i]) for i in range(current_derivative_tddft.shape[1])]).T

        idx=idx+1


    if idx % 20==0:
        print(f'Loading... dataset complete at the {100*idx/batch_size:.2f} % \n')

    if idx % 100 == 0:
        np.savez(
            f"data/dataset_h_eff/new_analysis_xxzx_model/dataset_{formatted_date_time}",
            z=z_qutip_tot[:,:,l//2-1:l//2],
            z_auxiliary=z_auxiliary[:,:,l//2-1:l//2],
            h_eff=h_eff_tot[:,:,l//2-1:l//2],
            current_derivative=current_derivative_tot[:,:,l//2-1:l//2],
            h=h_tot[:,:,l//2-1:l//2],
            time=time_final,
            info=info,
            comments=comments,
            l=ls,
        )

np.savez(
            f"data/dataset_h_eff/new_analysis_xxzx_model/dataset_{formatted_date_time}",
            z=z_qutip_tot[:,:,l//2-1:l//2],
            z_auxiliary=z_auxiliary[:,:,l//2-1:l//2],
            h_eff=h_eff_tot[:,:,:1],
            current_derivative=current_derivative_tot[:,:,l//2-1:l//2],
            h=h_tot[:,:,l//2-1:l//2],
            time=time_final,
            info=info,
            comments=comments,
            l=ls,
        )

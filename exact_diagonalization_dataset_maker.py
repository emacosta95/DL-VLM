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



def second_derivative_formula(arr, dt,derivative_formula:str):
    """
    Computes the second-order derivative using a 9-point finite difference stencil 
    along the time axis of a 2D array.

    Parameters:
        arr (numpy.ndarray): Input 2D array with shape [time, space].
        dt (float): Time step between successive data points.

    Returns:
        numpy.ndarray: Second derivative array with the same shape as input.
    """

    if derivative_formula=='9-points':

        # Ensure time axis (axis=0) has at least 9 points
        if arr.shape[0] < 9:
            raise ValueError("Time dimension must have at least 9 points for a 9-point stencil.")

        # 9-point stencil coefficients
        coeffs = np.array([
            -1/560,   # f(t-4h)
        8/315,   # f(t-3h)
            -1/5,     # f(t-2h)
        8/5,     # f(t-h)
            -205/72,  # f(t)
        8/5,     # f(t+h)
            -1/5,     # f(t+2h)
        8/315,   # f(t+3h)
            -1/560    # f(t+4h)
        ])

        # Initialize output array
        d2_arr = np.zeros_like(arr)

        # Apply stencil only on valid indices (excluding boundaries)
        for i in range(4, arr.shape[0] - 4):
            d2_arr[i, :] = (
                coeffs[0] * arr[i - 4, :] +
                coeffs[1] * arr[i - 3, :] +
                coeffs[2] * arr[i - 2, :] +
                coeffs[3] * arr[i - 1, :] +
                coeffs[4] * arr[i, :] +
                coeffs[5] * arr[i + 1, :] +
                coeffs[6] * arr[i + 2, :] +
                coeffs[7] * arr[i + 3, :] +
                coeffs[8] * arr[i + 4, :]
            )

        # Convert to second derivative by dividing by dt^2
        d2_arr /= dt ** 2

        # Handle boundary conditions using np.gradient as fallback
        d2_arr[:4, :] = np.gradient(np.gradient(arr, dt, axis=0), dt, axis=0)[:4, :]
        d2_arr[-4:, :] = np.gradient(np.gradient(arr, dt, axis=0), dt, axis=0)[-4:, :]

        return d2_arr
    
    if derivative_formula=='5-points':

        # Ensure time axis (axis=0) has at least 5 points
        if arr.shape[0] < 5:
            raise ValueError("Time dimension must have at least 5 points for a 5-point stencil.")

        # 5-point stencil coefficients for the second derivative
        coeffs = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])

        # Initialize output array
        d2_arr = np.zeros_like(arr)

        # Apply stencil only on valid indices (excluding boundaries)
        for i in range(2, arr.shape[0] - 2):
            d2_arr[i, :] = (
                coeffs[0] * arr[i - 2, :] +
                coeffs[1] * arr[i - 1, :] +
                coeffs[2] * arr[i, :] +
                coeffs[3] * arr[i + 1, :] +
                coeffs[4] * arr[i + 2, :]
            )

        # Convert to second derivative by dividing by dt^2
        d2_arr /= dt ** 2

        # Handle boundary conditions using np.gradient as fallback
        d2_arr[:2, :] = np.gradient(np.gradient(arr, dt, axis=0), dt, axis=0)[:2, :]
        d2_arr[-2:, :] = np.gradient(np.gradient(arr, dt, axis=0), dt, axis=0)[-2:, :]

        return d2_arr
        
    if derivative_formula=='3-points':

        # Ensure time axis (axis=0) has at least 3 points
        if arr.shape[0] < 3:
            raise ValueError("Time dimension must have at least 3 points for a 3-point stencil.")

        # Initialize output array
        d2_arr = np.zeros_like(arr)

        # Apply the 3-point stencil (central difference)
        for i in range(1, arr.shape[0] - 1):
            d2_arr[i, :] = (arr[i + 1, :] - 2 * arr[i, :] + arr[i - 1, :]) / (dt ** 2)

        # Handle boundary conditions using forward/backward difference
        d2_arr[0, :] = (arr[1, :] - 2 * arr[0, :] + arr[2, :]) / (dt ** 2)
        d2_arr[-1, :] = (arr[-2, :] - 2 * arr[-1, :] + arr[-3, :]) / (dt ** 2)

        return d2_arr
    
    

# Get the current date and time
now = datetime.now()

# Format the date and time into a string
# Example format: YYYY-MM-DD_HH-MM
formatted_date_time = now.strftime("%Y-%m-%d_%H-%M")

def generate_smooth_gaussian_noise(
    time: np.ndarray,
    tau: float,
    tf: float,
    mean: float,
    sigma: float,
    min_range: float,
    max_range: float,
    shift: float,
):
    a_omegas = np.random.normal(mean, sigma, size=time.shape[0])
    omegas = np.linspace(0, time.shape[0] * 2 * np.pi / tf, time.shape[0])
    driving = np.zeros(time.shape[0])

    for tr in range(time.shape[0]):
        if omegas[tr] < 2 * np.pi / tau:
            driving = driving + a_omegas[tr] * np.cos(omegas[tr] * time)

        else:
            break

    max_driving = np.max(driving)
    min_driving = np.min(driving)

    old_interval = max_driving - min_driving
    driving = (
        (driving - min_driving) * (max_range - min_range) / old_interval
        + min_range
        + shift
    )

    return driving


class Driving:
    def __init__(self, h: np.array, idx: int, dt: float) -> None:
        self.h = h
        # self.tf=tf
        self.idx: int = idx
        self.dt: float = dt

    def field(self, t: float, args):
        return self.h[int(t / self.dt), self.idx]

    def get_the_field(
        self,
    ):
        return self.h


# hyperaparameters

# if this is true, we are making datasets just to check the feasibility of the TDDFT method
diagnostic=True
derivative_formula='9-points' #it can be either 3-points,5-points,7-points,9-points

# parameters

nbatch = 1

batch_size =100

initial_state_ground_state=True
pbc=True
condition_initial_state='fixed'
# rates = [0.1, 0.5, 0.8, 1.0]

# j coupling
j = -1
# omega auxiliary field
omega = 1


rate_mean=1.5
rate_sigma=1.5

amplitude_max=2.
amplitude_min=0.

steps = 800
tf = 20.0

steps_tddft=steps*2

final_steps=200


time = np.linspace(0.0, tf, steps)
time_tddft=np.linspace(0.0, tf, steps_tddft)
time_final=np.linspace(0.0, tf, final_steps)

l=8 #np.random.randint(2,10)


# we fix the seed (thi is to fix just for checking the algorithm)
if diagnostic:
    np.random.seed(42)


info=f'xx-z-x model with omega={omega:.1f}, coupling={j: .1f} external field with rate mean={rate_mean:.1f} and rate sigma={rate_sigma:.1f} amplitude max={amplitude_max:.1f} amplitude min={amplitude_min:.1f} tf={tf:.0f} steps={steps} l variable ndata={batch_size} initial state option={initial_state_ground_state} pbc={pbc}'
comments=condition_initial_state+f' Initial state ground state, with a diagonostic is {diagnostic} dataset. 2nd order time derivative with'+derivative_formula+' formula'
# z_qutip_tot = np.zeros((nbatch * nbatch * batch_size, steps, l))
z_qutip_tot = np.zeros(( batch_size , final_steps,l))
z_auxiliary=np.zeros((batch_size,final_steps,l))
h_eff_tot = np.zeros(( batch_size , final_steps+1,l))
h_tot = np.zeros(( batch_size , final_steps+1,l))
current_qutip_tot = np.zeros(( batch_size , final_steps,l))
current_derivative_tot = np.zeros(( batch_size , final_steps,l))
x_sp_tot = np.zeros(( batch_size , final_steps,l))
ls=[]

p=np.random.uniform(0.,0.5,size=(batch_size))
#hi = np.ones((time.shape[0], l))  # we fix the initial field to be 1J
idx=0
while(idx<batch_size):
#for idx in trange(0, batch_size):
    
    # l=np.random.randint(3,9)
    # ls.append(l)
    
    ham0 = SpinHamiltonian(
    direction_couplings=[("x", "x")],
    pbc=pbc,
    coupling_values=[j],
    size=l,
    )

    hamExtX = SpinOperator(index=[("x", i) for i in range(l)], coupling=[omega] * l, size=l)

    obs: List[qutip.Qobj] = []
    #current_obs: List[qutip.Qobj] = []

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
        
        
        for i in range(l):
            if i==0:
                psi0=psi_plus*np.sqrt(p[idx])+np.sqrt(1-p[idx])*psi_minus
            else:
                psi0=qutip.tensor(psi0,psi_plus*np.sqrt(p[idx])+np.sqrt(1-p[idx])*psi_minus)

    
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
    # # x_sp = np.sqrt(1 - z_exp**2) * np.cos(
    # #     np.arcsin(-1 * (current_exp) / (2 * np.sqrt(1 - z_exp**2)))
    # # )
    

    # compute the effective field
    psi=np.zeros((2,l))
    psi[0] = np.sqrt((1 + z_exp[0]) / 2)
    psi[1] = np.sqrt((1 - z_exp[0]) / 2)


    # build up the operators
    x_op=np.array([[0.,1.],[1.,0]])
    z_op=np.array([[1.,0.],[0.,-1.]])

    
    # extrapolate the fields
    f=interp1d(time,z_exp,axis=0)
    z_tddft=f(time_tddft)
    current_tddft=np.gradient(z_tddft,time_tddft,axis=0)
    #current_derivative_tddft= np.gradient(current_tddft, time_tddft, axis=0)
    current_derivative_tddft=second_derivative_formula(z_tddft,dt=time_tddft[1]-time_tddft[0],derivative_formula=derivative_formula)

    

    dt=time_tddft[1]-time_tddft[0]    

    z_reconstruction=np.zeros((steps_tddft,l))
    h_eff_vector=np.zeros((steps_tddft,l))
    for i in range(steps_tddft):
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
            h_eff=(0.25*current_derivative_tddft[i]/nonlinear_term+z_tddft[i]*nonlinear_term)/(x_ave+1.e-10)
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
    if diagnostic:
        dz=0.
    
    if dz<0.01:
        # update the database
        h_eff_tot[idx,1:]=np.array([np.interp(time_final, time_tddft, h_eff_vector[:, i]) for i in range(h_eff_vector.shape[1])]).T
        h_tot[idx,1:]=np.array([np.interp(time_final, time, h[:, i]) for i in range(h.shape[1])]).T
        h_tot[idx,0]=z_exp[0]
        
        z_qutip_tot[idx,]=np.array([np.interp(time_final, time_tddft, z_tddft[:, i]) for i in range(z_tddft.shape[1])]).T
        z_auxiliary[idx,]= np.array([np.interp(time_final, time_tddft, z_reconstruction[:, i]) for i in range(z_reconstruction.shape[1])]).T
        current_qutip_tot[idx,]=np.array([np.interp(time_final, time_tddft, current_tddft[:, i]) for i in range(current_tddft.shape[1])]).T
        current_derivative_tot[idx]=np.array([np.interp(time_final, time_tddft, current_derivative_tddft[:, i]) for i in range(current_derivative_tddft.shape[1])]).T

        idx+=1


    if idx % 20==0:
        print(f'Loading... dataset complete at the {100*idx/batch_size:.2f} % \n')

    if idx % 100 == 0:
        np.savez(
            f"data/dataset_h_eff/new_analysis_xxzx_model/dataset_{formatted_date_time}",
            current=current_qutip_tot[:,:,l//2-1:l//2],
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
            current=current_qutip_tot[:,:,l//2-1:l//2],
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

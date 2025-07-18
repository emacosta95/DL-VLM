#### wrap the tddft_solver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from src.qutip_lab.qutip_class import SpinOperator, SpinHamiltonian, SteadyStateSolver
from scipy.fft import fft, ifft
from scipy.sparse.linalg import eigsh
import qutip
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

        current_arr=np.gradient(arr,dt,axis=0)
        derivative_current_arr=np.gradient(current_arr,dt,axis=0)

        return derivative_current_arr
 
 
 
 
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


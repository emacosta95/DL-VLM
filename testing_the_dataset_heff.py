# %%
import matplotlib.pyplot as plt
import numpy as np


data = np.load(
    "data/dataset_h_eff/periodic/dataset_periodic_random_rate_JUSTEFFECTIVEFIELD_nbatch_100_batchsize_1000_steps_200_tf_20.0_l_8_240227.npz"
)


z = data["z"]
h_eff = data["h_eff"]
h = data["h"]

print(h.shape)
# %%
# for i in range(0, 10):
#     for j in range(h.shape[-1]):
#         plt.plot(h[i, :, j])
#         plt.show()

# %%
idx=np.random.randint(0,h.shape[0])
for j in range(h.shape[-1]):
    plt.plot(z[idx, :, j])
plt.show()

# %%

for j in range(h.shape[-1]):
    plt.plot(h_eff[idx, :, j])
plt.show()

# %%
z_ave_values=np.average(np.abs(z),axis=-1)


plt.hist(z_ave_values.reshape(-1),bins=100)
plt.show()



# %% The fourier analysis
from numpy.fft import fft,ifft
import scipy
idx_batch=3
idx=50
batch_size=100
print(h.shape)

steps=1000
tf=30
dt=tf/steps
omega=np.linspace(0,1/(2*dt),steps//2)

h_fft=fft(h[(batch_size*(idx_batch)):(batch_size*(idx_batch+1)),:,:],norm='forward',axis=1)
h_eff_fft=fft(h_eff[(batch_size*(idx_batch)):(batch_size*(idx_batch+1)),:,:],norm='forward',axis=1)
omega_vector=scipy.fft.fftfreq(steps,dt)

print(omega_vector[499],omega[-1])
#%%
site=4
jdx=np.random.randint(0,idx)

plt.plot(h[jdx,:,site],label='driving')
plt.plot(h_eff[jdx,:,site],label='effective driving')
plt.show()


plt.title('real')
plt.plot(omega,np.real(h_fft[jdx,:steps//2,site]),label='driving')
plt.plot(omega,np.real(h_eff_fft[jdx,:steps//2,site]),label='effective driving')
plt.legend()
plt.show()

plt.title('imag')
plt.plot(omega,np.imag(h_fft[jdx,:steps//2,site]),label='driving')
plt.plot(omega,np.imag(h_eff_fft[jdx,:steps//2,site]),label='effective driving')
plt.legend()
plt.show()


# %% we test the number of indipendent frequencies
omega_index_cut=200
# cutoff of the frequencies

print(h_fft[jdx,omega_index_cut,site])
h_fft_cutoff=h_fft.copy()


h_fft_cutoff[:,omega_index_cut:-omega_index_cut]=0.

h_fft_cutoff[:,omega_index_cut:-omega_index_cut] =h_fft_cutoff[:,omega_index_cut:-omega_index_cut]* np.linspace(1, 0, h_fft_cutoff.shape[1] - 2 * omega_index_cut)[None,:,None]

#window=np.hamming(h_fft.shape[1])

#h_fft_cutoff=window[None,:,None]*h_fft_cutoff


# plt.title('real')
# plt.plot(omega,np.real(h_fft[:steps//2]),label='driving')
# plt.plot(omega,np.real(h_eff_fft[:steps//2]),label='effective driving')
# plt.legend()
# plt.show()

#new_norm_h_fft=np.linalg.norm(h_fft_cutoff,axis=1)


h_reconstruction_cutoff=ifft(h_fft_cutoff,norm='forward',axis=1)

h_reconstruction=ifft(h_fft,norm='forward',axis=1)



plt.plot(h_reconstruction[jdx,:,site],label='reconstruction')

print('h_recon=',h_reconstruction[jdx,:,site])
plt.plot(h_reconstruction_cutoff[jdx,:,site],label='cutoff')

plt.plot(h[jdx+(batch_size*(idx_batch)),:,site],label='exact')
plt.legend()
plt.show()


# %% Reconstruct the frequency at longer times
new_steps=2*steps
omega_tilde=np.linspace(0,1/dt,new_steps)

h_fft_extrapolated=np.zeros((h_fft.shape[0],new_steps,h_fft.shape[-1]))
for n in range(steps//2):
    
    n_tilde=int(n*(new_steps/steps))
    h_fft_extrapolated[:,n_tilde,:]=h_fft[:,n,:]
    h_fft_extrapolated[:,-n_tilde,:]=h_fft[:,-n,:]
    

h_extrapolated=ifft(h_fft_extrapolated,norm='forward',axis=1)

plt.plot(h_extrapolated[jdx,:,site])
plt.plot(h_reconstruction[jdx,:,site])
plt.show()

    
# %% Create a fourier transform dataset
import matplotlib.pyplot as plt
import numpy as np


data = np.load(
    "data/dataset_h_eff/periodic/dataset_periodic_random_rate_JUSTEFFECTIVEFIELD_nbatch_100_batchsize_1000_steps_200_tf_20.0_l_8_240227.npz"
)



z = data["z"]
z=np.einsum('bti->bit',z)
h_eff = data["h_eff"]
h_eff=np.einsum('bti->bit',h_eff)
h = data["h"]
h=np.einsum('bti->bit',h)



#%% check the distribution of values
print(z.shape)

plt.hist(z.reshape(-1),bins=200)
plt.show()

plt.hist(h_eff.reshape(-1),bins=200)
plt.show()

plt.hist(h.reshape(-1),bins=200)
plt.show()

nan_indices = np.where(np.isnan(h_eff))[0]

print(nan_indices)
print(h_eff[2148])
plt.plot(h_eff[2148])
plt.show()

#%% random samples
idx=np.random.randint(0,h_eff.shape[0])
for i in range(h_eff.shape[1]):
    plt.plot(h_eff[idx,i])
plt.show()

for i in range(h_eff.shape[1]):
    plt.plot(z[idx,i])
plt.show()
    
for i in range(h_eff.shape[1]):
    plt.plot(h[idx,i])
plt.show()



#%% We remove the nan values
h_eff_without_nan = h_eff[~np.isnan(h_eff).any(axis=(1,2))]

z_without_nan=z[~np.isnan(h_eff).any(axis=(1,2))]

h_without_nan=h[~np.isnan(h_eff).any(axis=(1,2))]



print(h_eff_without_nan.shape)
print(h_without_nan.shape)


nan_indices = np.where(np.isnan(h_eff_without_nan))[0]

print(nan_indices)


np.savez('data/dataset_h_eff/train_dataset_periodic_driving_ndata_5000_steps_200_240226.npz',potential=h_eff_without_nan,h=h_without_nan,density=z_without_nan)


#%% we train the puntual neural network
import matplotlib.pyplot as plt
import numpy as np


data = np.load(
    "data/dataset_h_eff/periodic/dataset_periodic_random_rate_JUSTEFFECTIVEFIELD_nbatch_100_batchsize_1000_steps_200_tf_20.0_l_8_240227.npz"
)



z = data["z"]
h_eff = data["h_eff"]
h = data["h"]

# z=z.reshape(-1,z.shape[-1])

# h_eff=h_eff.reshape(-1,z.shape[-1])

# h=h.reshape(-1,z.shape[-1])

# p=np.random.permutation(h.shape[0])

# np.savez(f'data/dataset_h_eff/train_puntual_mapping_ndata_{h.shape[0]}',density=z[p],h=h[p],potential=h_eff[p])



#%% Fourier transform
steps=z.shape[1]

z_fft=np.fft.fft(z_without_nan,axis=1,norm='forward')
h_fft=np.fft.fft(h_without_nan,axis=1,norm='forward')
h_eff_fft=np.fft.fft(h_eff_without_nan,axis=1,norm='forward')

h_fourier=np.zeros((z_fft.shape[0],2,z_fft.shape[1],z_fft.shape[-1]))
h_fourier[:,0]=np.real(h_fft)
h_fourier[:,1]=np.imag(h_fft)


h_eff_fourier=np.zeros((z_fft.shape[0],2,z_fft.shape[1],z_fft.shape[-1]))
h_eff_fourier[:,0]=np.real(h_eff_fft)
h_eff_fourier[:,1]=np.imag(h_eff_fft)

z_fourier=np.zeros((h_eff_without_nan.shape[0],2,h_eff_without_nan.shape[1],h_eff_without_nan.shape[-1]))
z_fourier[:,0]=np.real(z_fft)
z_fourier[:,1]=np.imag(z_fft)

np.savez('data/dataset_h_eff/periodic/fourier_transform/dataset_periodic_fourier_random_rate_JUSTEFFECTIVEFIELD_nbatch_100_batchsize_1000_steps_200_tf_20.0_l_8_240227.npz',density=z_fourier[:,:],h=h_fourier[:,:],potential=h_eff_fourier[:,:])


# %%

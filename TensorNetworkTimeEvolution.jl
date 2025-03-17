#Imports
using ITensors
using Random, Distributions, LinearAlgebra


# Function to measure global magnetization
function magnetization(psi, sites)
    x_sum = 0.0
    for i in 1:L
        x_sum += expect(psi, "X"; site=i)
    end
    return x_sum / length(sites)
end

global h 

# Parameters
L = 20             # System size
J = 1.0            # Ising interaction
omega= 0.5           # Transverse field strength
h0 = 0.       # Initial longitudinal field
dt = 0.1          # Time step
tmax = 10.0       # Total simulation time
rate_max=2.
rate_min=0.
amplitude_max=2.
amplitude_min=0.
rate_cutoff=10

# Build up the longitudinal field
# Generate random values for rate and delta
rate = rand(Uniform(rate_min, rate_max), rate_cutoff)
delta = rand(Uniform(amplitude_min, amplitude_max), rate_cutoff)

# Compute h using broadcasting
h= zeros(Int(tmax/dt))
#print(h)
for i in 1:rate_cutoff
    h =h+ delta[i] .* (cos.(π .+ time .* rate[i]) .+ 1) 
    print(i)
end
h=h/rate_cutoff
print("h shape=$(size(h))")
# Initialize the site indices for spin-1/2 systems
sites = siteinds("S=1/2", L)

# initial state
state=["Up" for n in 1:L]
psi=MPS(sites,state)

# Hyperparameters of the time evolution
sweeps = Sweeps(5)   # Number of sweeps
maxdim!(sweeps, 10, 20, 40, 100)  # Increase max bond dimension
cutoff!(sweeps, 1E-10)


# Time evolution loop
t = 0.0
x_in_time=Float64[]


while t < tmax
    h_current = h_t(t)  # Compute h(t)

    # Update the Hamiltonian with new h(t)
    ampo = OpSum()
    for i in 1:L-1
        ampo += -J, "Z", i, "Z", i+1
    end
    for i in 1:L
        ampo += -omega, "Z", i
        ampo += -h_current, "X", i
    end
    H = MPO(ampo, sites)

    # Evolve the state using TDVP
    psi = tdvp(psi, H, -im * dt; sweeps=sweeps)

    # Compute expectation values
    x_ave = magnetization(psi, sites)
    x_center = expect(psi, "X"; site=L÷2)
    push!(x_in_time,x_ave)
    println("t = $t, x_ave = $x_ave, Z_center = $x_center")

    t += dt
end



#Imports
using ITensors
using Random, Distributions, LinearAlgebra
using ITensorTDVP


# Function to measure global magnetization
function magnetization(psi, sites)
    x_sum = 0.0
    for i in 1:L
        x_sum += mean(expect(psi, "X"; site=i))
    end
    return x_sum / length(sites)
end



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
num_steps = Int(tmax/dt)
# Build up the longitudinal field
# Generate random values for rate and delta
rate = rand(Uniform(rate_min, rate_max), rate_cutoff)
delta = rand(Uniform(amplitude_min, amplitude_max), rate_cutoff)

time = range(0, step=dt, length=num_steps)
# Compute h using broadcasting
h= zeros(Int(tmax/dt))
#print(h)
for i in 1:rate_cutoff
    global h =h+ delta[i] .* (cos.(π .+ time .* rate[i]) .+ 1) 
end
h=h/rate_cutoff
print("h shape=$(size(h))")
# Initialize the site indices for spin-1/2 systems
sites = siteinds("S=1/2", L;conserve_qns=false)

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

ampo_0 = OpSum()
for i in 1:L-1
    global ampo_0 += -J, "Z", i, "Z", i+1
end
for i in 1:L
    global ampo_0 += -omega, "Z", i
    
end

ampo_1 =OpSum()
for i in 1:L
    global ampo_1 += -1, "X", i
end

while t < tmax
    h_current = h[Int(round(t/dt)) + 1]  # Compute h(t)

    print("h_current=$h_current")
    # Update the Hamiltonian with new h(t)
    ampo=ampo_0 + h_current*ampo_1
    hamiltonian=MPO(ampo,sites)
    


    # Evolve the state using TDVP
    global psi = tdvp( hamiltonian, -im * dt, psi; maxdim=30,cutoff=1e-8,normalize=true, reverse_step=false,outputlevel=1)
 
    # Compute expectation values
    x_center = mean(expect(psi, "X"; site=L÷2))
    print(x_center)
    x_ave = magnetization(psi, sites)
    push!(x_in_time,x_ave)
    println("t = $t, x_ave = $x_ave, Z_center = $x_center")

    global t += dt
end



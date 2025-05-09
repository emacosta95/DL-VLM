#Imports
using ITensors
using Random, Distributions, LinearAlgebra
using ITensorTDVP
using NPZ

function Expand_D(MPS,D,sites)
    N=length(MPS)
    Exp=truncate!(randomMPS(sites,linkdims=D))*(1+0im)  #MPS(sites)#linkdim(der,nn)
    for n in 1:N
      if n==1
        Bd=2
        α,β=siteind(MPS,n),linkind(MPS,n)
        #i,j=Index(2,"Site,n=$n"),Index(Bd,"Link,l=$n")
        i,j=siteind(Exp,n),linkind(Exp,n)
        temp=ITensor(i,j)*(1+0im)
        for o in 1:ITensors.dim(sites[1])
          for p in 1:linkdim(MPS,n)
              temp[i=>o,j=>p] = MPS[n][α=>o,β=>p]
          end
        end
        Exp[n]=temp
  
      elseif n==N
        Bd=2
        α,β=siteind(MPS,n),linkind(MPS,n-1)
        #i,j=Index(2,"Site,n=$n"),Index(Bd,"Link,l=$(n-1)")
        i,j=siteind(Exp,n),linkind(Exp,n-1)
        temp=ITensor(i,j)*(1+0im)
        for o in 1:ITensors.dim(sites[1])
          for p in 1:linkdim(MPS,n-1)
              temp[i=>o,j=>p] = MPS[n][α=>o,β=>p]
          end
        end
        Exp[n]=temp
      else 
        #Bd1=min(D,2^(n-1),2^(N-(n-1)))
        #Bd2=min(D,2^n,2^(N-n))
  
        α,β,γ=siteind(MPS,n),linkind(MPS,n-1),linkind(MPS,n)
        #i,j,k=Index(2,"Site,n=$n"),Index(Bd1,"Link,l=$(n-1)"),Index(Bd2,"Link,l=$n")
        i,j,k=siteind(Exp,n),linkind(Exp,n-1),linkind(Exp,n)
        temp=ITensor(i,j,k)*(1+0im)
        for o in 1:ITensors.dim(sites[1])
          for p in 1:linkdim(MPS,n-1)
            for q in 1:linkdim(MPS,n)
              temp[i=>o,j=>p,k=>q] = MPS[n][α=>o,β=>p,γ=>q]
            end
          end
        end
        Exp[n]=temp
  
  
      end
    end
    return Exp
  end


# Function to measure global magnetization
function magnetization(psi, sites)
    x_sum = 0.0
    for i in 1:L
        x_sum += mean(expect(psi, "X"; site=i))
    end
    return x_sum / length(sites)
end

# Open the file once before the time loop
open("x_ave_vs_time.txt", "w") do io
    # Optionally, write a header
    println(io, "# time x_ave")

# Parameters
L = 30             # System size
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

psi = Expand_D(psi, 5, siteinds(psi))

# Hyperparameters of the time evolution
sweeps = Sweeps(8)   # Number of sweeps
maxdim!(sweeps, 10, 20, 40, 200)  # Increase max bond dimension
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
    h_current = h[Int(round(t/dt))]  # Compute h(t)

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
    print("\n")
    println(" t = $t, x_ave = $x_ave, x_center = $x_center")
    println(io, "$t $x_ave")
    global t += dt
end
# Add a blank line at the end
println(io)



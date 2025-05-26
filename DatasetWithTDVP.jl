using ITensors, Random, Distributions, LinearAlgebra, ITensorTDVP, NPZ, Interpolations,Dates

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

function Expand_D(MPS, D, sites)
    N = length(MPS)
    Exp = truncate!(randomMPS(sites, linkdims=D)) * (1+0im)
    for n in 1:N
        α = siteind(MPS,n)
        if n == 1
            β = linkind(MPS,n)
            i, j = siteind(Exp,n), linkind(Exp,n)
            temp = ITensor(i,j) * (1+0im)
            for o in 1:ITensors.dim(sites[1]), p in 1:linkdim(MPS,n)
                temp[i=>o,j=>p] = MPS[n][α=>o,β=>p]
            end
            Exp[n] = temp
        elseif n == N
            β = linkind(MPS,n-1)
            i, j = siteind(Exp,n), linkind(Exp,n-1)
            temp = ITensor(i,j) * (1+0im)
            for o in 1:ITensors.dim(sites[1]), p in 1:linkdim(MPS,n-1)
                temp[i=>o,j=>p] = MPS[n][α=>o,β=>p]
            end
            Exp[n] = temp
        else
            β, γ = linkind(MPS,n-1), linkind(MPS,n)
            i, j, k = siteind(Exp,n), linkind(Exp,n-1), linkind(Exp,n)
            temp = ITensor(i,j,k) * (1+0im)
            for o in 1:ITensors.dim(sites[1]), p in 1:linkdim(MPS,n-1), q in 1:linkdim(MPS,n)
                temp[i=>o,j=>p,k=>q] = MPS[n][α=>o,β=>p,γ=>q]
            end
            Exp[n] = temp
        end
    end
    return Exp
end

function magnetization(psi, sites)
    sum(mean(expect(psi, "X"; site=i)) for i in 1:length(sites)) / length(sites)
end

# ----------------- Main Config --------------------
L = 30
J = -1.0
omega = 1
dt = 0.05
tmax = 10.0
num_steps = Int(tmax/dt)
rate_min, rate_max = 0.0, 4.0
amplitude_min, amplitude_max = 0.0, 2.0
rate_cutoff = 10
ndata = 10  # <- You control how many samples

time = range(0, step=dt, length=num_steps)



# Static Hamiltonian part
ampo_0 = OpSum()
for i in 1:L-1
    global ampo_0 += J, "Z", i, "Z", i+1
end
for i in 1:L
    global ampo_0 += omega, "Z", i
end
ampo_1 = OpSum()
for i in 1:L
    global ampo_1 += 1, "X", i
end

# Output data containers
all_drivings = []
all_zs = []

for sample_id in 1:ndata
    println("Generating sample $sample_id / $ndata")

    # Generate random driving parameters
    rate = rand(Uniform(rate_min, rate_max), rate_cutoff)
    delta = rand(Uniform(amplitude_min, amplitude_max), rate_cutoff)

    h = zeros(num_steps)
    for i in 1:rate_cutoff
        h .+= delta[i] .* (cos.(π .+ time .* rate[i]) .+ 1)
    end
    h ./= rate_cutoff

 

    # Setup spin system
    sites = siteinds("S=1/2", L; conserve_qns=false)
    state = ["Up" for _ in 1:L]
    psi = MPS(sites, state)
    psi = Expand_D(psi, 5, siteinds(psi))

    sweeps = Sweeps(30)
    maxdim!(sweeps, 300)
    cutoff!(sweeps, 1E-10)
    maxdim = 400



    # TDVP loop
    t = 0.0
    x_in_time = Float64[]
    while t < tmax - 1e-10
        h_current = h[Int(round(t/dt)) + 1]
        global ampo = ampo_0 + h_current * ampo_1
        H = MPO(ampo, sites)
        psi = tdvp(H, -im * dt, psi; maxdim=maxdim, cutoff=1e-8, outputlevel=1)
        push!(x_in_time, magnetization(psi, sites))
        t += dt
    end

    # Store each sample
    push!(all_drivings, h)
    push!(all_zs, x_in_time)
end

# Save to npz
npzwrite("data/transfer_learning_tdvp/$(timestamp)_generated_tdvp_lstm_dataset_ndata_$ndata.npz", Dict(
    "drivings" => hcat(all_drivings...)',
    "z_values" => hcat(all_zs...)',
    "time" => collect(time)
))

println("Dataset saved with $ndata samples.")
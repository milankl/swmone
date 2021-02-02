using PyPlot

const L = 1
const ω = 100.0
const N = 100
const Δx = L/(N-1)
const κ = 1.0
const Ntime = 1000
const Δt = Δx^2/κ/2
x = collect(0:Δx:L);

function ∂ₓ²(u::Vector,t::Real)
    n = length(u)
    ∂ₓ²u = zeros(n)
    for i in 2:n-1
        ∂ₓ²u[i] = (u[i+1] - 2u[i] + u[i-1])/Δx^2
    end
    return ∂ₓ²u
end

boundary(x::Int,t::Real) = x==0 ? 0 : sin(2π*ω*t)

# initial conditions
u = zeros(N)
t = 0.0

fig,ax = subplots(figsize=(6,3))
l1, = ax.plot(u,x)
ax.set_xlim(-1,1)
ax.set_ylim(0,L)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("temperature")
ax.set_ylabel("depth")

for i in 1:Ntime
    t += Δt
    
    # set the boundary conditions
    u[1] = boundary(0,t)
    u[end] = boundary(1,t)
    
    u += κ*Δt*∂ₓ²(u,t)
    l1.set_data(u,x)
    pause(0.0005)
end
close(fig)
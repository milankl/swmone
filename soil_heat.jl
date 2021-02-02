using PyPlot
pygui(true)

const D = 1.0               # depth of domain
const ω = 100.0            # frequency of sin-wave boundary condition
const N = 30                # grid points in z
const Δz = D/(N-1)          # vertical grid spacing
const κ = 10.0               # heat conductivity [m^2/s]
const Ntime = 1000          # number of time steps
const Δt = Δz^2/κ/2         # time spacing (stability criteration = Δz^2/Δt/k < 1/2)
z = collect(0:Δz:D);        # vector of z values

function ∂z²(T::Vector)
    n = length(T)
    ∂z²T = zeros(n)
    for i in 2:n-1
        ∂z²T[i] = (T[i+1] - 2T[i] + T[i-1])/Δz^2
    end
    return ∂z²T
end

boundary(x::Int,t::Real) = x==0 ? 0 : sin(2π*ω*t)

# initial conditions
T = zeros(N)        # temperature T
t = 0.0             # 

fig,ax = subplots(figsize=(6,3))
l1, = ax.plot(T,z)
ax.set_xlim(-1,1)
ax.set_ylim(0,D)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("temperature")
ax.set_ylabel("depth")

for i in 1:Ntime
    t += Δt

    # set the boundary conditions
    T[1] = boundary(0,t)
    T[end] = boundary(1,t)
    
    # midpoint method
    dT = κ*Δt*∂z²(T)
    T += κ*Δt*∂z²(T + 1/2*dT)

    l1.set_data(T,z)
    pause(0.0005)
end
close(fig)
# swmone - An MPI-parallelised shallow water model in 1D.
![wave](figs/wave.gif?raw=true "1D non-linear wave")

Swmone is a fully-explicit finite difference non-linear shallow water model on a staggered grid, which is parallelised using domain decomposition and ghost-point communication via Julia's MPI wrapper. The 1D shallow water equations are

          ∂u/∂t = -u*∂u/∂x -g*∂η/∂x + ν∂²u/∂x² + F(x,t)   (1)
          ∂η/∂t = -∂(uh)/∂x.                              (2)

The terms on the right-hand side of (1) are non-linear advection, pressure gradient, viscosity and forcing. The layer thickness is h = η + H, with H being constant. Equation (1) is solved in the "vector invariant" form

          ∂u/∂t = ∂/∂x(-1/2*u² - g*η + ν∂u/∂x) + F(x,t)
          
with a Runge-Kutta 4th order time integration and 2nd order centred gradients and interpolations in space. 

# Domain decomposition

The domain of `N` grid points is split into chunks of equal size, which will be computed by individual processes separately. Each sub-domain is extend to the left and to the right with ghost points. Before every evaluation of the right-hand side the ghost points are updated via MPI.send/MPI.receive calls from adjacent processes. Only in the end a MPI.gather is executed to retrieve data from all sub-domains. Optional plotting with PyPlot (animation via updated figure).

# Documentation

There's a [blog article](http://www.milank.de/mpi-julia#main), describing the model, the domain decomposition and the parallelization.

# Requirements

MPI and PyPlot
          

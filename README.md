# StochasticForceInference	

StochasticForceInference is a package aimed at reconstructing
spatially variable force, diffusion and current fields and computing
entropy production in Brownian processes. It implements the method
presented in SFI_TheoryManual.pdf. Reference:

**Learning Force Fields from Stochastic Trajectories**, Anna Frishman and Pierre Ronceray, 
Phys. Rev. X 10, 021009, 2020.

Package author: Pierre Ronceray. Contact: pierre.ronceray@outlook.com. Web page: www.pierre-ronceray.net.

-----------------------------------------------------------------------

This is a force and diffusion inference package for  **overdamped** systems. For underdamped/inertial systems, check **StochasticInertialForceInference**: https://github.com/ronceray/StochasticInertialForceInference

-----------------------------------------------------------------------

Developed in Python 3.6. Dependencies:

- NumPy, SciPy

- Optional: Matplotlib (2D plotting, recommended); Mayavi2 (3D
  plotting)

-----------------------------------------------------------------------

##Contents:

**StochasticForceInference.py**: a front-end includer of all classes
   useful to the user.

**SFI_data.py**: contains a data wrapper class, StochasticTrajectoryData,
   which formats trajectories for force and diffusion inference. Also
   contains a number of plotting routines (of the trajectory, vector
   fields, tensor fields...). See this file for the different ways to
   initialize it using data.

**SFI_inference.py**: implements the core force, velocity and diffusion
   inference class, StochasticForceInference, that reconstructs the
   these fields, computes error on this reconstruction, and computes
   entropy production.  Takes as input a StochasticTrajectoryData
   instance, and inference parameters.

**SFI_langevin.py**: contains the class OverdampedLangevinProcess, which
   implements a simple Ito integration of Brownian dynamics, useful
   for testing the method with known models. It takes as input a force
   field and a diffusion tensor field. Also used by
   StochasticForceInference to bootstrap new trajectories with the
   inferred force field.

**SFI_projectors.py**: implements an internal class, TrajectoryProjectors,
   used by DiffusionInference and StochasticForceInference. Given a
   set of fitting functions, it orthonormalizes it as a premise to the
   inference.

**SFI_bases.py**: provides an internal dictionary of (more or less)
   standard fitting bases, such as polynomials. This dictionary is
   called by DiffusionInference and StochasticForceInference at
   initialization, unless a custom base is provided by the user.

**SFI_plotting_toolkit.py**: a few plotting functions for the convenience
   of the author.

**SFI_demo_Lorenz.py**: a fully commented example of force and diffusion
   inference on the example of a simple 3D process. Start here!
   
**SFI_demo_particles.py**: a more advanced example: active Brownian
   particles. Shows how to extract the pair interaction force.	       
   
-----------------------------------------------------------------------


Enjoy, and please send feedback at pierre.ronceray@outlook.com !

       	   	       				     PR
						
-----------------------------------------------------------------------

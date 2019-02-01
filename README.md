# StochasticForceInference	


StochasticForceInference is a package aimed at reconstructing
spatially variable force, diffusion and current fields and computing
entropy production in Brownian processes. It implements the method
presented in SFI_TheoryManual.pdf. 

Reference: Anna Frishman and Pierre Ronceray, Learning force fields
from stochastic trajectories, arXiv:1809.09650, 2018.

Author: Pierre Ronceray. Contact: pierre.ronceray@outlook.com. Web
page: www.pierre-ronceray.net.

-----------------------------------------------------------------------

Developed in Python 3.6. Dependencies:
- NumPy, SciPy
- Optional: Matplotlib (2D plotting, recommended); Mayavi2 (3D
  plotting)

-----------------------------------------------------------------------

Contents:

StochasticForceInference.py: a front-end includer of all classes
   useful to the user.

SFI_data.py: contains a data wrapper class, StochasticTrajectoryData,
   which formats trajectories for force and diffusion inference. Also
   contains a number of plotting routines (of the trajectory, vector
   fields, tensor fields...). See this file for the different ways to
   initialize it using data.

SFI_diffusion.py: implements the core diffusion inference class,
   DiffusionInference, that reconstructs the diffusion tensor
   field. Takes as input a StochasticTrajectoryData instance, and
   inference parameters.

SFI_forces.py: implements the core force inference class,
   StochasticForceInference, that reconstructs the force field.  Takes
   as input a StochasticTrajectoryData instance, inference parameters,
   and diffusion parameters (either a constant, a space-dependent
   function, or directly a DiffusionInference instance).

SFI_langevin.py: contains the class OverdampedLangevinProcess, which
   implements a simple Ito integration of Brownian dynamics, useful
   for testing the method with known models. It takes as input a force
   field and a diffusion tensor field. Also used by
   StochasticForceInference to bootstrap new trajectories with the
   inferred force field.

SFI_projectors.py: implements an internal class, TrajectoryProjectors,
   used by DiffusionInference and StochasticForceInference. Given a
   set of fitting functions, it orthonormalizes it as a premise to the
   inference.

SFI_bases.py: provides an internal dictionary of (more or less)
   standard fitting bases, such as polynomials. This dictionary is
   called by DiffusionInference and StochasticForceInference at
   initialization, unless a custom base is provided by the user.

SFI_plotting_toolkit.py: a few plotting functions for the convenience
   of the author.

SFI_demo.py: a fully commented example of force and diffusion
   inference on the example of a simple 3D process. Start here!
   
-----------------------------------------------------------------------


Enjoy, and please send feedback at pierre.ronceray@outlook.com !

       	   	       				     PR
						
-----------------------------------------------------------------------

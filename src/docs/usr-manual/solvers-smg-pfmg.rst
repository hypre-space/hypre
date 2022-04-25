.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


SMG
==============================================================================

SMG is a parallel semicoarsening multigrid solver for the linear systems arising
from finite difference, finite volume, or finite element discretizations of the
diffusion equation,

.. math::

   \nabla \cdot ( D \nabla u ) + \sigma u = f

on logically rectangular grids.  The code solves both 2D and 3D problems with
discretization stencils of up to 9-point in 2D and up to 27-point in 3D.  See
[Scha1998]_, [BrFJ2000]_, [FaJo2000]_ for details on the algorithm and its
parallel implementation/performance.

SMG is a particularly robust method.  The algorithm semicoarsens in the
z-direction and uses plane smoothing.  The xy plane-solves are effected by one
V-cycle of the 2D SMG algorithm, which semicoarsens in the y-direction and uses
line smoothing.


PFMG
==============================================================================

PFMG is a parallel semicoarsening multigrid solver similar to SMG.  See
[AsFa1996]_, [FaJo2000]_ for details on the algorithm and its parallel
implementation/performance.

The main difference between the two methods is in the smoother: PFMG uses simple
pointwise smoothing.  As a result, PFMG is not as robust as SMG, but is much
more efficient per V-cycle.


SysPFMG
==============================================================================

SysPFMG is a parallel semicoarsening multigrid solver for systems of elliptic
PDEs. It is a generalization of PFMG, with the interpolation defined only within
the same variable. The relaxation is of nodal type- all variables at a given
point location are simultaneously solved for in the relaxation.

Although SysPFMG is implemented through the SStruct interface, it can be used
only for problems with one grid part. Ideally, the solver should handle any of
the seven variable types (cell-, node-, xface-, yface-, zface-, xedge-, yedge-,
and zedge-based). However, it has been completed only for cell-based variables.


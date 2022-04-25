.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


Maxwell
==============================================================================

Maxwell is a parallel solver for edge finite element discretization of the
curl-curl formulation of the Maxwell equation

.. math::

   \nabla \times \alpha \nabla \times E + \beta E= f, \beta> 0

on semi-structured grids. Details of the algorithm can be found in [JoLe2006]_.
The solver can be viewed as an operator-dependent multiple-coarsening algorithm
for the Helmholtz decomposition of the error correction. Input to this solver
consist of only the linear system and a gradient operator. In fact, if the
orientation of the edge elements conforms to a lexicographical ordering of the
nodes of the grid, then the gradient operator can be generated with the routine
``HYPRE_MaxwellGrad``: at grid points :math:`(i,j,k)` and :math:`(i-1,j,k),` the
produced gradient operator takes values :math:`1` and :math:`-1` respectively,
which is the correct gradient operator for the appropriate edge
orientation. Since the gradient operator is normalized (i.e., :math:`h`
independent) the edge finite element must also be normalized in the
discretization.

This solver is currently developed for perfectly conducting boundary condition
(Dirichlet). Hence, the rows and columns of the matrix that corresponding to the
grid boundary must be set to the identity or zeroed off. This can be achieved
with the routines ``HYPRE_SStructMaxwellPhysBdy`` and
``HYPRE_SStructMaxwellEliminateRowsCols``. The former identifies the ranks of
the rows that are located on the grid boundary, and the latter adjusts the
boundary rows and cols. As usual, the rhs of the linear system must also be
zeroed off at the boundary rows. This can be done using
``HYPRE_SStructMaxwellZeroVector``.

With the adjusted linear system and a gradient operator, the user can form the
Maxwell multigrid solver using several different edge interpolation schemes. For
problems with smooth coefficients, the natural Nedelec interpolation operator
can be used. This is formed by calling ``HYPRE_SStructMaxwellSetConstantCoef``
with the flag :math:`>0` before setting up the solver, otherwise the default
edge interpolation is an operator-collapsing/element-agglomeration scheme. This
is suitable for variable coefficients.  Also, before setting up the solver, the
user must pass the gradient operator, whether user or ``HYPRE_MaxwellGrad``
generated, with ``HYPRE_SStructMaxwellSetGrad``. After these preliminary calls,
the Maxwell solver can be setup by calling ``HYPRE_SStructMaxwellSetup``.

There are two solver cycling schemes that can be used to solve the linear
system. To describe these, one needs to consider the augmented system operator

.. math::

   \bf{A}= \left [
     \begin{array}{ll}
        A_{ee} & A_{en}  \\
        A_{ne} & A_{nn}
     \end{array}
   \right ],

where :math:`A_{ee}` is the stiffness matrix corresponding to the above
curl-curl formulation, :math:`A_{nn}` is the nodal Poisson operator created by
taking the Galerkin product of :math:`A_{ee}` and the gradient operator, and
:math:`A_{ne}` and :math:`A_{en}` are the nodal-edge coupling operators (see
[JoLe2006]_). The algorithm for this Maxwell solver is based on forming a
multigrid hierarchy to this augmented system using the block-diagonal
interpolation operator

.. math::

   \bf{P}= \left[  \begin{array}{ll}
               P_e & 0  \\
               0   & P_n
            \end{array}
   \right],

where :math:`P_e` and :math:`P_n` are respectively the edge and nodal
interpolation operators determined individually from :math:`A_{ee}` and
:math:`A_{nn}.` Taking a Galerkin product between :math:`\bf{A}` and
:math:`\bf{P}` produces the next coarse augmented operator, which also has the
nodal-edge coupling operators. Applying this procedure recursively produces
nodal-edge coupling operators at all levels. Now, the first solver cycling
scheme, ``HYPRE_SStructMaxwellSolve``, keeps these coupling operators on all
levels of the V-cycle. The second, cheaper scheme,
``HYPRE_SStructMaxwellSolve2``, keeps the coupling operators only on the finest
level, i.e., separate edge and nodal V-cycles that couple only on the finest
level.


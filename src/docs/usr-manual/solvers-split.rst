.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


SplitSolve
==============================================================================

SplitSolve is a parallel block Gauss-Seidel solver for semi-structured problems
with multiple parts. For problems with only one variable, it can be viewed as a
domain-decomposition solver with no grid overlapping.

Consider a multiple part problem given by the linear system :math:`Ax=b`. Matrix
:math:`A` can be decomposed into a structured intra-variable block diagonal
component :math:`M` and a component :math:`N` consisting of the inter-variable
blocks and any unstructured connections between the parts. SplitSolve performs
the iteration

.. math::

   x_{k+1} = \tilde{M}^{-1} (b + N x_k),

where :math:`\tilde{M}^{-1}` is a decoupled block-diagonal V(1,1) cycle, a
separate cycle for each part and variable type. There are two V-cycle options,
SMG and PFMG.

.. Add blank lines to help with navigation pane formatting
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|

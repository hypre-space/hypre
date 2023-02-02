.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


LOBPCG Eigensolver
==============================================================================

LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) is a simple,
yet very efficient, algorithm suggested in [Knya2001]_, [KLAO2007]_, [BLOPEWeb]_
for computing several smallest eigenpairs of the symmetric generalized
eigenvalue problem :math:`Ax=\lambda Bx` with large, possibly sparse, symmetric
matrix :math:`A` and symmetric positive definite matrix :math:`B`. The matrix
:math:`A` is not assumed to be positive, which also allows one to use LOBPCG to
compute the largest eigenpairs of :math:`Ax=\lambda Bx` simply by solving
:math:`-Ax=\mu Bx` for the smallest eigenvalues :math:`\mu=-\lambda`.

LOBPCG simultaneously computes several eigenpairs together, which is controlled
by the ``blockSize`` parameter, see example ``ex11.c``. The LOBCPG also allows
one to impose constraints on the eigenvectors of the form :math:`x^T B y_i=0`
for a set of vectors :math:`y_i` given to LOBPCG as input parameters. This makes
it possible to compute, e.g., 50 eigenpairs by 5 subsequent calls to LOBPCG with
the ``blockSize=10``, using deflation.  LOBPCG can use preconditioning in two
different ways: by running an inner preconditioned PCG linear solver, or by
applying the preconditioner directly to the eigenvector residual (option
``-pcgitr 0``).  In all other respects, LOBPCG is similar to the PCG linear
solver.

The LOBPCG code is available for system interfaces: Struct, SStruct, and IJ.  It
is also used in the Auxiliary-space Maxwell Eigensolver (AME).  The LOBPCG setup
is similar to the setup for PCG.

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

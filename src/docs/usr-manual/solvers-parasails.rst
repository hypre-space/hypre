.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


ParaSails
==============================================================================

.. warning::
   ParaSails is not actively supported by the hypre development team. We recommend using
   :ref:`fsai` for parallel sparse approximate inverse algorithms. This new implementation
   includes NVIDIA/AMD GPU support through the CUDA/HIP backends.

ParaSails is a parallel implementation of a sparse approximate inverse
preconditioner, using *a priori* sparsity patterns and least-squares (Frobenius
norm) minimization.  Symmetric positive definite (SPD) problems are handled
using a factored SPD sparse approximate inverse.  General (nonsymmetric and/or
indefinite) problems are handled with an unfactored sparse approximate inverse.
It is also possible to precondition nonsymmetric but definite matrices with a
factored, SPD preconditioner.

ParaSails uses *a priori* sparsity patterns that are patterns of powers of
sparsified matrices.  ParaSails also uses a post-filtering technique to reduce
the cost of applying the preconditioner.  In advanced usage not described here,
the pattern of the preconditioner can also be reused to generate preconditioners
for different matrices in a sequence of linear solves.

For more details about the ParaSails algorithm, see [Chow2000]_.


Parameter Settings
------------------------------------------------------------------------------

The accuracy and cost of ParaSails are parametrized by the real ``thresh`` and
integer ``nlevels`` parameters, :math:`0 \le` ``thresh`` :math:`\le 1`, :math:`0
\le` ``nlevels``.  Lower values of ``thresh`` and higher values of ``nlevels``
lead to more accurate, but more expensive preconditioners.  More accurate
preconditioners are also more expensive per iteration.  The default values are
``thresh`` :math:`= 0.1` and ``nlevels`` :math:`= 1`.  The parameters are set
using ``HYPRE_ParaSailsSetParams``.

Mathematically, given a symmetric matrix :math:`A`, the pattern of the
approximate inverse is the pattern of :math:`\tilde{A}^m` where
:math:`\tilde{A}` is a matrix that has been sparsified from :math:`A`.  The
sparsification is performed by dropping all entries in a symmetrically
diagonally scaled :math:`A` whose values are less than ``thresh`` in magnitude.
The parameter ``nlevel`` is equivalent to :math:`m-1`.  Filtering is a
post-thresholding procedure.  For more details about the algorithm, see
[Chow2000]_.

The storage required for the ParaSails preconditioner depends on the parameters
``thresh`` and ``nlevels``.  The default parameters often produce a
preconditioner that can be stored in less than the space required to store the
original matrix.  ParaSails does not need a large amount of intermediate storage
in order to construct the preconditioner.

ParaSail's Create function differs from the synopsis in the following way:

.. code-block:: c

   int HYPRE_ParaSailsCreate(MPI_Comm comm, HYPRE_Solver *solver, int symmetry);

where ``comm`` is the MPI communicator.

The value of ``symmetry`` has the following meanings, to indicate the symmetry
and definiteness of the problem, and to specify the type of preconditioner to
construct:

   =====  =========================================================================
   value  meaning
   =====  =========================================================================
   0      nonsymmetric and/or indefinite problem, and nonsymmetric preconditioner
   1      SPD problem, and SPD (factored) preconditioner
   2      nonsymmetric, definite problem, and SPD (factored) preconditioner
   =====  =========================================================================

For more information about the final case, see section :ref:`nearly`.

Parameters for setting up the preconditioner are specified using

.. code-block:: c

   int HYPRE_ParaSailsSetParams(HYPRE_Solver solver, double thresh,
                                int nlevel, double filter);

The parameters are used to specify the sparsity pattern and filtering value (see
above), and are described with suggested values as follows:

   ==========  =======  =============  ==============  =======  =============================
   param       type     range          sug. values     default  meaning
   ==========  =======  =============  ==============  =======  =============================
   ``nlevel``  integer  :math:`\ge 0`  0, 1, 2         1        :math:`m=1+` ``nlevel``
   ``thresh``  real     :math:`\ge 0`  0, 0.1, 0.01    0.1      thresh :math:`=` ``thresh``
   \                    :math:`<  0`   -0.75, -0.90             thresh auto-selected
   ``filter``  real     :math:`\ge 0`  0, 0.05, 0.001  0.05     filter :math:`=` ``filter``
   \                    :math:`<  0`   -0.90                    filter auto-selected
   ==========  =======  =============  ==============  =======  =============================

When ``thresh`` :math:`< 0`, then a threshold is selected such that ``thresh``
represents the negative of the fraction of nonzero elements that are dropped.
For example, if ``thresh`` :math:`= -0.9` then :math:`\tilde{A}` will contain
approximately ten percent of the nonzeros in :math:`A`.

When ``filter`` :math:`< 0`, then a filter value is selected such that ``filter``
represents the negative of the fraction of nonzero elements that are dropped.
For example, if ``filter`` :math:`= -0.9` then approximately 90 percent of the
entries in the computed approximate inverse are dropped.


.. _nearly:

Preconditioning Nearly Symmetric Matrices
------------------------------------------------------------------------------

A nonsymmetric, but definite and nearly symmetric matrix :math:`A` may be
preconditioned with a symmetric preconditioner :math:`M`.  Using a symmetric
preconditioner has a few advantages, such as guaranteeing positive definiteness
of the preconditioner, as well as being less expensive to construct.

The nonsymmetric matrix :math:`A` must be definite, i.e., :math:`(A+A^T)/2` is
SPD, and the *a priori* sparsity pattern to be used must be symmetric.  The
latter may be guaranteed by 1) constructing the sparsity pattern with a
symmetric matrix, or 2) if the matrix is structurally symmetric (has symmetric
pattern), then thresholding to construct the pattern is not used (i.e., zero
value of the ``thresh`` parameter is used).

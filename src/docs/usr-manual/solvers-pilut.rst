.. Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


.. _PILUT:

PILUT: Parallel Incomplete Factorization
==============================================================================

**Note:** this code is no longer supported by the hypre team. We recommend to
use Euclid instead, which is more versatile and in general more efficient,
especially when used with many processors.

PILUT is a parallel preconditioner based on Saad's dual-threshold incomplete
factorization algorithm. The original version of PILUT was done by Karypis and
Kumar [KaKu1998]_ in terms of the Cray SHMEM library. The code was subsequently
modified by the hypre team: SHMEM was replaced by MPI; some algorithmic changes
were made; and it was software engineered to be interoperable with several
matrix implementations, including hypre's ParCSR format, PETSc's matrices, and
ISIS++ RowMatrix. The algorithm produces an approximate factorization :math:`L U`,
with the preconditioner :math:`M` defined by :math:`M = L U`.

**Note:** PILUT produces a nonsymmetric preconditioner even when the original
matrix is symmetric. Thus, it is generally inappropriate for preconditioning
symmetric methods such as Conjugate Gradient.


Parameters:
------------------------------------------------------------------------------

* ``SetMaxNonzerosPerRow( int LFIL ); (Default: 20)`` Set the maximum number of
  nonzeros to be retained in each row of :math:`L` and :math:`U`.  This
  parameter can be used to control the amount of memory that :math:`L` and
  :math:`U` occupy. Generally, the larger the value of ``LFIL``, the longer it
  takes to calculate the preconditioner and to apply the preconditioner and the
  larger the storage requirements, but this trades off versus a higher quality
  preconditioner that reduces the number of iterations.

* ``SetDropTolerance( double tol ); (Default: 0.0001)`` Set the tolerance
  (relative to the 2-norm of the row) below which entries in L and U are
  automatically dropped. PILUT first drops entries based on the drop tolerance,
  and then retains the largest LFIL elements in each row that remain.  Smaller
  values of ``tol`` lead to more accurate preconditioners, but can also lead to
  increases in the time to calculate the preconditioner.


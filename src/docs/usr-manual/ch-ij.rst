.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


.. _ch-IJ:

******************************************************************************
Linear-Algebraic System Interface (IJ)
******************************************************************************

The ``IJ`` interface described in this chapter is the lowest common
denominator for specifying linear systems in hypre.  This interface
provides access to general sparse-matrix solvers in hypre, not
to the specialized solvers that require more problem information.

IJ Matrix Interface
==============================================================================

As with the other interfaces in hypre, the ``IJ`` interface expects to get data
in distributed form because this is the only scalable approach for assembling
matrices on thousands of processes.  Matrices are assumed to be distributed by
blocks of rows as follows:

.. math::

   \left[
   \begin{array}{c}
   ~~~~~~~~~~ A_0 ~~~~~~~~~~ \\
   A_1 \\
   \vdots \\
   A_{P-1}
   \end{array}
   \right]

In the above example, the matrix is distributed across the :math:`P` processes,
:math:`0, 1, ..., P-1` by blocks of rows.  Each submatrix :math:`A_p` is "owned"
by a single process and its first and last row numbers are given by the global
indices ``ilower`` and ``iupper`` in the ``Create()`` call below.

The following example code illustrates the basic usage of the ``IJ`` interface
for building matrices:

.. code-block:: c
   
   MPI_Comm            comm;
   HYPRE_IJMatrix      ij_matrix;
   HYPRE_ParCSRMatrix  parcsr_matrix;
   int                 ilower, iupper;
   int                 jlower, jupper;
   int                 nrows;
   int                *ncols;
   int                *rows;
   int                *cols;
   double             *values;
   
   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &ij_matrix);
   HYPRE_IJMatrixSetObjectType(ij_matrix, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(ij_matrix);
   
   /* set matrix coefficients */
   HYPRE_IJMatrixSetValues(ij_matrix, nrows, ncols, rows, cols, values);
   ...
   /* add-to matrix cofficients, if desired */
   HYPRE_IJMatrixAddToValues(ij_matrix, nrows, ncols, rows, cols, values);
   ...
   
   HYPRE_IJMatrixAssemble(ij_matrix);
   HYPRE_IJMatrixGetObject(ij_matrix, (void **) &parcsr_matrix);

The ``Create()`` routine creates an empty matrix object that lives on the
``comm`` communicator.  This is a collective call (i.e., must be called on all
processes from a common synchronization point), with each process passing its
own row extents, ``ilower`` and ``iupper``.  The row partitioning must be
contiguous, i.e., ``iupper`` for process ``i`` must equal ``ilower``:math:`-1`
for process ``i``:math:`+1`.  Note that this allows matrices to have 0- or
1-based indexing.  The parameters ``jlower`` and ``jupper`` define a column
partitioning, and should match ``ilower`` and ``iupper`` when solving square
linear systems.  See Chapter :ref:`ch-API` for more information.

The ``SetObjectType()`` routine sets the underlying matrix object type to
``HYPRE_PARCSR`` (this is the only object type currently supported).  The
``Initialize()`` routine indicates that the matrix coefficients (or values) are
ready to be set.  This routine may or may not involve the allocation of memory
for the coefficient data, depending on the implementation.  The optional
``SetRowSizes()`` and ``SetDiagOffdSizes()`` routines mentioned later in this
chapter and in Chapter :ref:`ch-API`, should be called before this step.

The ``SetValues()`` routine sets matrix values for some number of rows
(``nrows``) and some number of columns in each row (``ncols``).  The actual row
and column numbers of the matrix ``values`` to be set are given by ``rows`` and
``cols``.  The coefficients can be modified with the ``AddToValues()``
routine. If ``AddToValues()`` is used to add to a value that previously didn't
exist, it will set this value.  Note that while ``AddToValues()`` will add to
values on other processors, ``SetValues()`` does not set values on other
processors. Instead if a user calls ``SetValues()`` on processor :math:`i` to
set a matrix coefficient belonging to processor :math:`j`, processor :math:`i`
will erase all previous occurrences of this matrix coefficient, so they will not
contribute to this coefficient on processor :math:`j`.  The actual coefficient
has to be set on processor :math:`j`.

The ``Assemble()`` routine is a collective call, and finalizes the matrix
assembly, making the matrix "ready to use".  The ``GetObject()`` routine
retrieves the built matrix object so that it can be passed on to hypre solvers
that use the ``ParCSR`` internal storage format.  Note that this is not an
expensive routine; the matrix already exists in ``ParCSR`` storage format, and
the routine simply returns a "handle" or pointer to it.  Although we currently
only support one underlying data storage format, in the future several different
formats may be supported.

One can preset the row sizes of the matrix in order to reduce the execution time
for the matrix specification.  One can specify the total number of coefficients
for each row, the number of coefficients in the row that couple the diagonal
unknown to (``Diag``) unknowns in the same processor domain, and the number of
coefficients in the row that couple the diagonal unknown to (``Offd``) unknowns
in other processor domains:

.. code-block:: c
   
   HYPRE_IJMatrixSetRowSizes(ij_matrix, sizes);
   HYPRE_IJMatrixSetDiagOffdSizes(matrix, diag_sizes, offdiag_sizes);

Once the matrix has been assembled, the sparsity pattern cannot be altered
without completely destroying the matrix object and starting from scratch.
However, one can modify the matrix values of an already assembled matrix.  To do
this, first call the ``Initialize()`` routine to re-initialize the matrix, then
set or add-to values as before, and call the ``Assemble()`` routine to
re-assemble before using the matrix.  Re-initialization and re-assembly are very
cheap, essentially a no-op in the current implementation of the code.

IJ Vector Interface
==============================================================================

The following example code illustrates the basic usage of the ``IJ`` interface
for building vectors:

.. code-block:: c
   
   MPI_Comm         comm;
   HYPRE_IJVector   ij_vector;
   HYPRE_ParVector  par_vector;
   int              jlower, jupper;
   int              nvalues;
   int             *indices;
   double          *values;
   
   HYPRE_IJVectorCreate(comm, jlower, jupper, &ij_vector);
   HYPRE_IJVectorSetObjectType(ij_vector, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_vector);
   
   /* set vector values */
   HYPRE_IJVectorSetValues(ij_vector, nvalues, indices, values);
   ...
   
   HYPRE_IJVectorAssemble(ij_vector);
   HYPRE_IJVectorGetObject(ij_vector, (void **) &par_vector);

The ``Create()`` routine creates an empty vector object that lives on the
``comm`` communicator.  This is a collective call, with each process passing its
own index extents, ``jlower`` and ``jupper``.  The names of these extent
parameters begin with a ``j`` because we typically think of matrix-vector
multiplies as the fundamental operation involving both matrices and vectors.
For matrix-vector multiplies, the vector partitioning should match the column
partitioning of the matrix (which also uses the ``j`` notation).  For linear
system solves, these extents will typically match the row partitioning of the
matrix as well.

The ``SetObjectType()`` routine sets the underlying vector storage type to
``HYPRE_PARCSR`` (this is the only storage type currently supported).  The
``Initialize()`` routine indicates that the vector coefficients (or values) are
ready to be set.  This routine may or may not involve the allocation of memory
for the coefficient data, depending on the implementation.

The ``SetValues()`` routine sets the vector ``values`` for some number
(``nvalues``) of ``indices``.  The values can be modified with the
``AddToValues()`` routine.  Note that while ``AddToValues()`` will add to values
on other processors, ``SetValues()`` does not set values on other
processors. Instead if a user calls ``SetValues()`` on processor :math:`i` to
set a value belonging to processor :math:`j`, processor :math:`i` will erase all
previous occurrences of this matrix coefficient, so they will not contribute to
this value on processor :math:`j`.  The actual value has to be set on processor
:math:`j`.

The ``Assemble()`` routine is a trivial collective call, and finalizes the
vector assembly, making the vector "ready to use".  The ``GetObject()`` routine
retrieves the built vector object so that it can be passed on to hypre solvers
that use the ``ParVector`` internal storage format.

Vector values can be modified in much the same way as with matrices by first
re-initializing the vector with the ``Initialize()`` routine.


A Scalable Interface
==============================================================================

As explained in the previous sections, problem data is passed to the hypre
library in its distributed form.  However, as is typically the case for a
parallel software library, some information regarding the global distribution of
the data will be needed for hypre to perform its function.  In particular, a
solver algorithm requires that a processor obtain "nearby" data from other
processors in order to complete the solve.  While a processor may easily
determine what data it needs from other processors, it may not know which
processor owns the data it needs.  Therefore, processors must determine their
communication partners, or neighbors.

The straightforward approach to determining neighbors involves constructing a
global partition of the data.  This approach, however, requires :math:`O(P)`
storage and computations and is not scalable for machines with tens of thousands
of processors.  The *assumed partition* algorithm was developed to address this
problem [BaFY2006]_.  It is the approach used in hypre.


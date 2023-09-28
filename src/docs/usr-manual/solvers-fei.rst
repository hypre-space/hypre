.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


.. _LSI_solvers:

FEI Solvers
==============================================================================

.. warning::
   FEI is not actively supported by the hypre development team. For similar
   functionality, we recommend using :ref:`sec-Block-Structured-Grids-FEM`, which
   allows the representation of block-structured grid problems via hypre's
   SStruct interface.

After the FEI has been used to assemble the global linear system (as described
in Chapter :ref:`ch-FEI`), a number of hypre solvers can be called to perform
the solution.  This is straightforward, if hypre's FEI has been used.  If an
external FEI is employed, the user needs to link with hypre's implementation of
the ``LinearSystemCore`` class, as described in Section :ref:`LSI_install`.

Solver parameters are specified as an array of strings, and a complete list of
the available options can be found in the FEI section of the reference manual.
They are passed to the FEI as in the following example:

.. code-block:: c++

   nParams = 5;
   paramStrings = new char*[nParams];
   for (i = 0; i < nParams; i++) }
      paramStrings[i] = new char[100];

   strcpy(paramStrings[0], "solver cg");
   strcpy(paramStrings[1], "preconditioner diag");
   strcpy(paramStrings[2], "maxiterations 100");
   strcpy(paramStrings[3], "tolerance 1.0e-6");
   strcpy(paramStrings[4], "outputLevel 1");

   feiPtr -> parameters(nParams, paramStrings);

To solve the linear system of equations, we call

.. code-block:: c++

   feiPtr -> solve(&status);

where the returned value ``status`` indicates whether the solve was successful.

Finally, the solution can be retrieved by the following function call:

.. code-block:: c++

   feiPtr -> getBlockNodeSolution(elemBlkID, nNodes, nodeIDList,
                                  solnOffsets, solnValues);

where ``nodeIDList`` is a list of nodes in element block ``elemBlkID``, and
``solnOffsets[i]`` is the index pointing to the first location where the
variables at node :math:`i` is returned in ``solnValues``.

Solvers Available Only through the FEI
------------------------------------------------------------------------------

While most of the solvers from the previous sections are available through the
FEI interface, there are number of additional solvers and preconditioners that
are accessible only through the FEI.  These solvers are briefly described in
this section (see also the reference manual).

Sequential and Parallel Solvers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

hypre currently has many iterative solvers. There is also internally a version
of the sequential ``SuperLU`` direct solver (developed at U.C.  Berkeley)
suitable to small problems (may be up to the size of :math:`10000`).  In the
following we list some of these internal solvers.

#. Additional Krylov solvers (FGMRES, TFQMR, symmetric QMR),
#. SuperLU direct solver (sequential),
#. SuperLU direct solver with iterative refinement (sequential),

Parallel Preconditioners
^^^^^^^^^^^^^^^^^^^^^^^^

The performance of the Krylov solvers can be improved by clever selection of
preconditioners. Besides those mentioned previously in this chapter, the
following preconditioners are available via the ``LinearSystemCore`` interface:

#. the modified version of MLI, which requires the finite element substructure
   matrices to construct the prolongation operators,
#. parallel domain decomposition with inexact local solves (``DDIlut``),
#. least-squares polynomial preconditioner,
#. :math:`2 \times 2` block preconditioner, and
#. :math:`2 \times 2` Uzawa preconditioner.

Some of these preconditioners can be tuned by a number of internal parameters
modifiable by users. A description of these parameters is given in the reference
manual.

Matrix Reduction
^^^^^^^^^^^^^^^^

For some structural mechanics problems with multi-point constraints the
discretization matrix is indefinite (eigenvalues lie in both sides of the
imaginary axis). Indefinite matrices are much more difficult to solve than
definite matrices. Methods have been developed to reduce these indefinite
matrices to definite matrices.  Two matrix reduction algorithms have been
implemented in hypre, as presented in the following subsections.

Schur Complement Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^
The incoming linear system of equations is assumed to be in the form:

.. math::

   \left[
   \begin{array}{cc}
      D   & B \\
      B^T & 0
   \end{array}
     \right]
     \left[
   \begin{array}{c}
      x_1 \\
      x_2
   \end{array}
     \right]
     =
     \left[
   \begin{array}{c}
      b_1 \\
      b_2
   \end{array}
     \right]

where :math:`D` is a diagonal matrix.  After Schur complement reduction is
applied, the resulting linear system becomes

.. math::
   - B^T D^{-1} B x_2 = b_2 - B^T D^{-1} b_1.

Slide Surface Reduction
^^^^^^^^^^^^^^^^^^^^^^^

With the presence of slide surfaces, the matrix is in the same form as in the
case of Schur complement reduction.  Here :math:`A` represents the relationship
between the master, slave, and other degrees of freedom.  The matrix block
:math:`[B^T 0]` corresponds to the constraint equations.  The goal of reduction
is to eliminate the constraints.  As proposed by Manteuffel, the trick is to
re-order the system into a :math:`3 \times 3` block matrix.

.. math::

   \left[
   \begin{array}{ccc}
      A_{11}  & A_{12} & N \\
      A_{21}  & A_{22} & D \\
      N_{T}   & D      & 0 \\
   \end{array}
   \right]
   =
   \left[
   \begin{array}{ccc}
      A_{11}       & \hat{A}_{12} \\
      \hat{A}_{21} & \hat{A}_{22}.
   \end{array}
   \right]

The reduced system has the form :

.. math::

   (A_{11} - \hat{A}_{21} \hat{A}_{22}^{-1} \hat{A}_{12}) x_1 =
   b_1 - \hat{A}_{21} \hat{A}_{22}^{-1} b_2,

which is symmetric positive definite (SPD) if the original matrix is PD.  In
addition, :math:`\hat{A}_{22}^{-1}` is easy to compute.

There are three slide surface reduction algorithms in hypre.  The first follows
the matrix formulation in this section.  The second is similar except that it
replaces the eliminated slave equations with identity rows so that the degree of
freedom at each node is preserved.  This is essential for certain block
algorithms such as the smoothed aggregation multilevel preconditioners.  The
third is similar to the second except that it is more general and can be applied
to problems with intersecting slide surfaces (sequential only for intersecting
slide surfaces).

Other Features
^^^^^^^^^^^^^^

To improve the efficiency of the hypre solvers, a few other features have been
incorporated.  We list a few of these features below :

#. Preconditioner reuse - For multiple linear solves with matrices that are
   slightly perturbed from each other, oftentimes the use of the same
   preconditioners can save preconditioner setup times but suffer little
   convergence rate degradation.
#. Projection methods - For multiple solves that use the same matrix, previous
   solution vectors can sometimes be used to give a better initial guess for
   subsequent solves.  Two projection schemes have been implemented in hypre -
   A-conjugate projection (for SPD matrices) and minimal residual projection
   (for both SPD and non-SPD matrices).
#. The sparsity pattern of the matrix is in general not destroyed after it has
   been loaded to an hypre matrix.  But if the matrix is not to be reused, an
   option is provided to clean up this pattern matrix to conserve memory usage.

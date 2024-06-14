.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


Euclid
==============================================================================

.. warning::
   Euclid is not actively supported by the hypre development team. We recommend using
   :ref:`ilu` for parallel ILU algorithms. This new ILU implementation includes
   64-bit integers support (for linear systems with more than 2,147,483,647 global
   unknowns) through both *mixedint* and *bigint* builds of hypre and NVIDIA/AMD GPUs
   support through the CUDA/HIP backends.

The Euclid library is a scalable implementation of the Parallel ILU algorithm
that was presented at SC99 [HyPo1999]_, and published in expanded form in the
SIAM Journal on Scientific Computing [HyPo2001]_.  By *scalable* we mean that
the factorization (setup) and application (triangular solve) timings remain
nearly constant when the global problem size is scaled in proportion to the
number of processors.  As with all ILU preconditioning methods, the number of
iterations is expected to increase with global problem size.

Experimental results have shown that PILU preconditioning is in general more
effective than Block Jacobi preconditioning for minimizing total solution time.
For scaled problems, the relative advantage appears to increase as the number of
processors is scaled upwards.  Euclid may also be used to good advantage as a
smoother within multigrid methods.


Overview
------------------------------------------------------------------------------

Euclid is best thought of as an "extensible ILU preconditioning framework."
*Extensible* means that Euclid can (and eventually will, time and contributing
agencies permitting) support many variants of ILU(:math:`k`) and ILUT
preconditioning.  (The current release includes Block Jacobi ILU(:math:`k`) and
Parallel ILU(:math:`k`) methods.)  Due to this extensibility, and also because
Euclid was developed independently of the hypre project, the methods by which
one passes runtime parameters to Euclid preconditioners differ in some respects
from the hypre norm.  While users can directly set options within their code,
options can also be passed to Euclid preconditioners via command line switches
and/or small text-based configuration files.  The latter strategies have the
advantage that users will not need to alter their codes as Euclid's capabilities
are extended.

The following fragment illustrates the minimum coding required to invoke Euclid
preconditioning within hypre application contexts.  The next subsection provides
examples of the various ways in which Euclid's options can be set.  The final
subsection lists the options, and provides guidance as to the settings that (in
our experience) will likely prove effective for minimizing execution time.

.. code-block:: c

   #include "HYPRE_parcsr_ls.h"

   HYPRE_Solver eu;
   HYPRE_Solver pcg_solver;
   HYPRE_ParVector b, x;
   HYPRE_ParCSRMatrix A;

   //Instantiate the preconditioner.
   HYPRE_EuclidCreate(comm, &eu);

   //Optionally use the following three methods to set runtime options.
   // 1. pass options from command line or string array.
   HYPRE_EuclidSetParams(eu, argc, argv);

   // 2. pass options from a configuration file.
   HYPRE_EuclidSetParamsFromFile(eu, "filename");

   // 3. pass options using interface functions.
   HYPRE_EuclidSetLevel(eu, 3);
   ...

   //Set Euclid as the preconditioning method for some
   //other solver, using the function calls HYPRE_EuclidSetup
   //and HYPRE_EuclidSolve.  We assume that the pcg_solver
   //has been properly initialized.
   HYPRE_PCGSetPrecond(pcg_solver,
                       (HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
                       (HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
                       eu);

   //Solve the system by calling the Setup and Solve methods for,
   //in this case, the HYPRE_PCG solver.  We assume that A, b, and x
   //have been properly initialized.
   HYPRE_PCGSetup(pcg_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
   HYPRE_PCGSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);

   //Destroy the Euclid preconditioning object.
   HYPRE_EuclidDestroy(eu);


Setting Options: Examples
------------------------------------------------------------------------------

For expositional purposes, assume you wish to set the ILU(:math:`k`)
factorization level to the value :math:`k = 3`.  There are several methods of
accomplishing this.  Internal to Euclid, options are stored in a simple database
that contains (name, value) pairs.  Various of Euclid's internal (private)
functions query this database to determine, at runtime, what action the user has
requested.  If you enter the option ``-eu_stats 1``, a report will be printed
when Euclid's destructor is called; this report lists (among other statistics)
the options that were in effect during the factorization phase.

**Method 1.** By default, Euclid always looks for a file titled ``database`` in
the working directory.  If it finds such a file, it opens it and attempts to
parse it as a configuration file.  Configuration files should be formatted as
follows.

.. code-block:: bash

   >cat database
   #this is an optional comment
   -level 3

Any line in a configuration file that contains a "``#``" character in the first
column is ignored.  All other lines should begin with an option *name*, followed
by one or more blanks, followed by the option *value*.  Note that option names
always begin with a ``-`` character.  If you include an option name that is not
recognized by Euclid, no harm should ensue.

**Method 2.** To pass options on the command line, call

.. code-block:: c

   HYPRE_EuclidSetParams(HYPRE_Solver solver, int argc, char *argv[]);

where ``argc`` and ``argv`` carry the usual connotation: ``main(int argc, char
*argv[])``.  If your hypre application is called ``phoo``, you can then pass
options on the command line per the following example.

.. code-block:: bash

   mpirun -np 2 phoo -level 3

Since Euclid looks for the ``database`` file when ``HYPRE_EuclidCreate`` is
called, and parses the command line when ``HYPRE_EuclidSetParams`` is called,
option values passed on the command line will override any similar settings that
may be contained in the ``database`` file.  Also, if same option name appears
more than once on the command line, the final appearance determines the setting.

Some options, such as ``-bj`` (see next subsection) are boolean.  Euclid always
treats these options as the value ``1`` (true) or ``0`` (false).  When passing
boolean options from the command line the value may be committed, in which case
it assumed to be ``1``.  Note, however, that when boolean options are contained
in a configuration file, either the ``1`` or ``0`` must stated explicitly.

**Method 3.** There are two ways in which you can read in options from a file
whose name is other than ``database``.  First, you can call
``HYPRE_EuclidSetParamsFromFile`` to specify a configuration filename.  Second,
if you have passed the command line arguments as described above in Method 2,
you can then specify the configuration filename on the command line using the
``-db_filename filename`` option, e.g.,

.. code-block:: bash

   mpirun -np 2 phoo -db_filename ../myConfigFile

**Method 4.** One can also set parameters via interface functions, e.g

.. code-block:: c

   int HYPRE_EuclidSetLevel(HYPRE_Solver solver, int level);

For a full set of functions, see the reference manual.


Options Summary
------------------------------------------------------------------------------

* **-level** :math:`\langle int \rangle` Factorization level for ILU(:math:`k`).
  Default: 1.  Guidance: for 2D convection-diffusion and similar problems,
  fastest solution time is typically obtained with levels 4 through 8.  For 3D
  problems fastest solution time is typically obtained with level 1.

* **-bj** Use Block Jacobi ILU preconditioning instead of PILU.  Default: 0
  (false). Guidance: if subdomains contain relatively few nodes (less than
  1,000), or the problem is not well partitioned, Block Jacobi ILU may give
  faster solution time than PILU.

* **-eu_stats** When Euclid's destructor is called a summary of runtime settings
  and timing information is printed to stdout.  Default: 0 (false).  The timing
  marks in the report are the maximum over all processors in the MPI
  communicator.

* **-eu_mem** When Euclid's destructor is called a summary of Euclid's memory
  usage is printed to stdout.  Default: 0 (false).  The statistics are for the
  processor whose rank in ``MPI_COMM_WORLD`` is 0.

* **-printTestData** This option is used in our autotest procedures, and should
  not normally be invoked by users.

* **-sparseA** :math:`\langle float \rangle` Drop-tolerance for ILU(:math:`k`)
  factorization.  Default: 0 (no dropping).  Entries are treated as zero if
  their absolute value is less than ``sparseA * max``, where ``max`` is the
  largest absolute value of any entry in the row. Guidance: try this in
  conjunction with -rowScale.  CAUTION: If the coefficient matrix :math:`A` is
  symmetric, this setting is likely to cause the filled matrix, :math:`F =
  L+U-I`, to be non-symmetric.  This setting has no effect when ILUT factorization
  is selected.

* **-rowScale** Scale values prior to factorization such that the largest value
  in any row is +1 or -1.  Default: 0 (false).  CAUTION: If the coefficient
  matrix :math:`A` is symmetric, this setting is likely to cause the filled
  matrix, :math:`F = L+U-I`, to be non-symmetric.  Guidance: if the matrix is
  poorly scaled, turning on row scaling may help convergence.

* **-ilut** :math:`\langle float \rangle` Use ILUT factorization instead of the
  default, ILU(:math:`k`).  Here, :math:`\langle float \rangle` is the drop
  tolerance, which is relative to the largest absolute value of any entry in the
  row being factored.  CAUTION: If the coefficient matrix :math:`A` is
  symmetric, this setting is likely to cause the filled matrix, :math:`F =
  L+U-I`, to be non-symmetric.  NOTE: this option can only be used sequentially!

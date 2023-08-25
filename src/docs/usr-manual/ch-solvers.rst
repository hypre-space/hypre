.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


.. _ch-Solvers:

******************************************************************************
Solvers and Preconditioners
******************************************************************************

There are several solvers available in hypre via different conceptual
interfaces:

   ===========  =======  =======  =======  =======
   :math:`\;`   System Interfaces
   -----------  ----------------------------------
   Solvers      Struct   SStruct    FEI      IJ
   ===========  =======  =======  =======  =======
   Jacobi          X        X
   SMG             X        X
   PFMG            X        X
   Split                    X
   SysPFMG                  X
   FAC                      X
   Maxwell                  X
   BoomerAMG                X        X        X
   AMS                      X        X        X
   ADS                      X        X        X
   MLI                      X        X        X
   MGR                                        X
   FSAI                                       X
   ParaSails                X        X        X
   ILU                                        X
   Euclid                   X        X        X
   PILUT                    X        X        X
   PCG             X        X        X        X
   GMRES           X        X        X        X
   FlexGMRES       X        X        X        X
   LGMRES          X        X                 X
   BiCGSTAB        X        X        X        X
   Hybrid          X        X        X        X
   LOBPCG          X        X                 X
   ===========  =======  =======  =======  =======

Note that there are a few additional solvers and preconditioners not mentioned
in the table that can be used only through the FEI interface and are described
in Paragraph 6.14.  The procedure for setup and use of solvers and
preconditioners is largely the same. We will refer to them both as solvers in
the sequel except when noted.  In normal usage, the preconditioner is chosen and
constructed before the solver, and then handed to the solver as part of the
solver's setup.  In the following, we assume the most common usage pattern in
which a single linear system is set up and then solved with a single righthand
side. We comment later on considerations for other usage patterns.


**Setup:**

#. **Pass to the solver the information defining the problem.** In the typical
   user cycle, the user has passed this information into a matrix through one of
   the conceptual interfaces prior to setting up the solver. In this situation,
   the problem definition information is then passed to the solver by passing
   the constructed matrix into the solver. As described before, the matrix and
   solver must be compatible, in that the matrix must provide the services
   needed by the solver. Krylov solvers, for example, need only a matrix-vector
   multiplication.  Most preconditioners, on the other hand, have additional
   requirements such as access to the matrix coefficients.

#. **Create the solver/preconditioner** via the ``Create()`` routine.

#. **Choose parameters for the preconditioner and/or solver.** Parameters are
   chosen through the ``Set()`` calls provided by the solver.  Throughout hypre,
   we have made our best effort to give all parameters reasonable defaults if
   not chosen.  However, for some preconditioners/solvers the best choices for
   parameters depend on the problem to be solved. We give recommendations in the
   individual sections on how to choose these parameters.  Note that in hypre,
   convergence criteria can be chosen after the preconditioner/solver has been
   setup.  For a complete set of all available parameters see Chapter
   :ref:`ch-API`.

#. **Pass the preconditioner to the solver.** For solvers that are not
   preconditioned, this step is omitted.  The preconditioner is passed through
   the ``SetPrecond()`` call.

#. **Set up the solver.** This is just the ``Setup()`` routine.  At this point
   the matrix and right hand side is passed into the solver or
   preconditioner. Note that the actual right hand side is not used until the
   actual solve is performed.

At this point, the solver/preconditioner is fully constructed and ready for use.


**Use:**

#. **Set convergence criteria.** Convergence can be controlled by the number of
   iterations, as well as various tolerances such as relative residual,
   preconditioned residual, etc.  Like all parameters, reasonable defaults are
   used.  Users are free to change these, though care must be taken.  For
   example, if an iterative method is used as a preconditioner for a Krylov
   method, a constant number of iterations is usually required.

#. **Solve the system.**  This is just the ``Solve()`` routine.


**Finalize:**

#. **Free the solver or preconditioner.** This is done using the ``Destroy()``
   routine.


**Synopsis**

In general, a solver (let's call it ``SOLVER``) is set up and run using the
following routines, where ``A`` is the matrix, ``b`` the right hand side and
``x`` the solution vector of the linear system to be solved:

.. code-block:: c

   /* Create Solver */
   int HYPRE_SOLVERCreate(MPI_COMM_WORLD, &solver);

   /* Set certain parameters if desired */
   HYPRE_SOLVERSetTol(solver, 1.e-8);
   ...

   /* Set up Solver */
   HYPRE_SOLVERSetup(solver, A, b, x);

   /* Solve the system */
   HYPRE_SOLVERSolve(solver, A, b, x);

   /* Destroy the solver */
   HYPRE_SOLVERDestroy(solver);

In the following sections, we will give brief descriptions of the available
hypre solvers with some suggestions on how to choose the parameters as well as
references for users who are interested in a more detailed description and
analysis of the solvers.  A complete list of all routines that are available can
be found in Chapter :ref:`ch-API`.


.. toctree::
   :maxdepth: 2

   solvers-smg-pfmg
   solvers-split
   solvers-fac
   solvers-maxwell
   solvers-hybrid
   solvers-boomeramg
   solvers-ams
   solvers-ads
   solvers-mli
   solvers-mgr
   solvers-fsai
   solvers-parasails
   solvers-ilu
   solvers-euclid
   solvers-pilut
   solvers-lobpcg
   solvers-fei

.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


.. _introduction:

******************************************************************************
Introduction
******************************************************************************

This manual describes hypre, a software library of high performance
preconditioners and solvers for the solution of large, sparse linear systems of
equations on massively parallel computers [FaJY2004]_.  The hypre library was
created with the primary goal of providing users with advanced parallel
preconditioners.  The library features parallel multigrid solvers for both
structured and unstructured grid problems.  For ease of use, these solvers are
accessed from the application code via hypre's conceptual linear system
interfaces [FaJY2005]_ (abbreviated to *conceptual interfaces* throughout much
of this manual), which allow a variety of natural problem descriptions.

This introductory chapter provides an overview of the various features in hypre,
discusses further sources of information on hypre, and offers suggestions on how
to get started.


.. _features:

Overview of Features
==============================================================================

**Scalable preconditioners provide efficient solution on today's and tomorrow's
systems:** hypre contains several families of preconditioner algorithms focused
on the scalable solution of *very large* sparse linear systems. (Note that small
linear systems, systems that are solvable on a sequential computer, and dense
systems are all better addressed by other libraries that are designed
specifically for them.)  hypre includes "grey box" algorithms that use more than
just the matrix to solve certain classes of problems more efficiently than
general-purpose libraries. This includes algorithms such as structured
multigrid.


**Suite of common iterative methods provides options for a spectrum of
problems:** hypre provides several of the most commonly used Krylov-based
iterative methods to be used in conjunction with its scalable
preconditioners. This includes methods for nonsymmetric systems such as GMRES
and methods for symmetric matrices such as Conjugate Gradient.

**Intuitive grid-centric interfaces obviate need for complicated data structures
and provide access to advanced solvers:** hypre has made a major step forward in
usability from earlier generations of sparse linear solver libraries in that
users do not have to learn complicated sparse matrix data structures.  Instead,
hypre does the work of building these data structures for the user through a
variety of conceptual interfaces, each appropriate to different classes of
users.  These include stencil-based structured/semi-structured interfaces most
appropriate for finite-difference applications; a finite-element based
unstructured interface; and a linear-algebra based interface.  Each conceptual
interface provides access to several solvers without the need to write new
interface code.

**User options accommodate beginners through experts:** hypre allows a spectrum
of expertise to be applied by users. The beginning user can get up and running
with a minimal amount of effort. More expert users can take further control of
the solution process through various parameters.

**Configuration options to suit your computing system:** hypre allows a simple
and flexible installation on a wide variety of computing systems.  Users can
tailor the installation to match their computing system. Options include debug
and optimized modes, the ability to change required libraries such as MPI and
BLAS, a sequential mode, and modes enabling threads for certain solvers.  On
most systems, however, hypre can be built by simply typing ``configure``
followed by ``make``, or by using CMake [CMakeWeb]_.

**Interfaces in multiple languages provide greater flexibility for
applications:** hypre is written in C (with the exception of the FEI interface,
which is written in C++) and provides an interface for Fortran users.


.. _more-info:

Getting More Information
==============================================================================

This user's manual consists of chapters describing each conceptual interface, a
chapter detailing the various linear solver options available, detailed
installation information, and the API reference.  In addition to this manual, a
number of other information sources for hypre are available.

* **Reference Manual:** This is equivalent to Chapter :ref:`ch-API` in this user
  manual, but it can also be built as a separate document.  The reference manual
  comprehensively lists all of the interface and solver functions available in
  hypre.  It is ideal for determining the various options available for a
  particular solver or for viewing the functions provided to describe a problem
  for a particular interface.

* **Example Problems:** A suite of example problems is provided with the hypre
  installation.  These examples reside in the ``examples`` subdirectory and
  demonstrate various features of the hypre library.  Associated documentation
  may be accessed by viewing the ``README.html`` file in that same directory.

* **Papers, Presentations, etc.:** Articles and presentations related to the
  hypre software library and the solvers available in the library are available
  from the hypre web page at `http://www.llnl.gov/CASC/hypre/`_.

* **Mailing List:** The mailing list ``hypre-announce`` can be subscribed to
  through the hypre web page at `http://www.llnl.gov/CASC/hypre/`_.  The
  development team uses this list to announce new releases of hypre.  It cannot
  be posted to by users.

.. _http://www.llnl.gov/CASC/hypre/: http://www.llnl.gov/CASC/hypre/


.. _getting-started:

How to get started
==============================================================================


.. _installing-hypre:

Installing hypre
------------------------------------------------------------------------------

As previously noted, on most systems hypre can be built by simply typing
``configure`` followed by ``make`` in the top-level source directory.
Alternatively, the CMake system [CMakeWeb]_ can be used, and is the best
approach for building hypre on Windows systems in particular.  For more detailed
instructions, read the ``INSTALL`` file provided with the hypre distribution or
the :ref:`ch-General` section of this manual.  Note the following requirements:

* To run in parallel, hypre requires an installation of MPI.

* Configuration of hypre with threads requires an implementation of OpenMP.
  Currently, only a subset of hypre is threaded.

* The hypre library currently does not directly support complex-valued systems.


.. _choosing-interface:

Choosing a conceptual interface
------------------------------------------------------------------------------

An important decision to make before writing any code is to choose an
appropriate conceptual interface.  These conceptual interfaces are intended to
represent the way that applications developers naturally think of their linear
problem and to provide natural interfaces for them to pass the data that defines
their linear system into hypre.  Essentially, these conceptual interfaces can be
considered convenient utilities for helping a user build a matrix data structure
for hypre solvers and preconditioners.  The top row of :ref:`fig-ls-interface`
illustrates a number of conceptual interfaces.  Generally, the conceptual
interfaces are denoted by different types of computational grids, but other
application features might also be used, such as geometrical information.  For
example, applications that use structured grids (such as in the left-most
interface in :ref:`fig-ls-interface`) typically view their linear problems in
terms of stencils and grids.  On the other hand, applications that use
unstructured grids and finite elements typically view their linear problems in
terms of elements and element stiffness matrices. Finally, the right-most
interface is the standard linear-algebraic (matrix rows/columns) way of viewing
the linear problem.

The hypre library currently supports four conceptual interfaces, and typically
the appropriate choice for a given problem is fairly obvious, e.g. a
structured-grid interface is clearly inappropriate for an unstructured-grid
application.

* **Structured-Grid System Interface (Struct):** This interface is appropriate
  for applications whose grids consist of unions of logically rectangular grids
  with a fixed stencil pattern of nonzeros at each grid point.  This interface
  supports only a single unknown per grid point.  See Chapter :ref:`ch-Struct`
  for details.

* **Semi-Structured-Grid System Interface (SStruct):** This interface is
  appropriate for applications whose grids are mostly structured, but with some
  unstructured features.  Examples include block-structured grids, composite
  grids in structured adaptive mesh refinement (AMR) applications, and overset
  grids.  This interface supports multiple unknowns per cell. See Chapter
  :ref:`ch-SStruct` for details.

* **Finite Element Interface (FEI):** This is appropriate for users who form
  their linear systems from a finite element discretization.  The interface
  mirrors typical finite element data structures, including element stiffness
  matrices.  Though this interface is provided in hypre, its definition was
  determined elsewhere (please send email to Alan Williams william@sandia.gov
  for more information). See Chapter :ref:`ch-FEI` for details.

* **Linear-Algebraic System Interface (IJ):** This is the traditional
  linear-algebraic interface.  It can be used as a last resort by users for whom
  the other grid-based interfaces are not appropriate.  It requires more work on
  the user's part, though still less than building parallel sparse data
  structures.  General solvers and preconditioners are available through this
  interface, but not specialized solvers which need more information.  Our
  experience is that users with legacy codes, in which they already have code
  for building matrices in particular formats, find the IJ interface relatively
  easy to use. See Chapter :ref:`ch-IJ` for details.

.. _fig-ls-interface:

.. figure:: figConcepIface.*
   :align: center

   Figure 1

   Graphic illustrating the notion of conceptual linear system interfaces.

Generally, a user should choose the most specific interface that matches their
application, because this will allow them to use specialized and more efficient
solvers and preconditioners without losing access to more general solvers.  For
example, the second row of Figure :ref:`fig-ls-interface` is a set of linear
solver algorithms.  Each linear solver group requires different information from
the user through the conceptual interfaces.  So, the geometric multigrid
algorithm (GMG) listed in the left-most box, for example, can only be used with
the left-most conceptual interface.  On the other hand, the ILU algorithm in the
right-most box may be used with any conceptual interface.  Matrix requirements
for each solver and preconditioner are provided in Chapter :ref:`ch-Solvers` and
in Chapter :ref:`ch-API`.  Your desired solver strategy may influence your
choice of conceptual interface.  A typical user will select a single Krylov
method and a single preconditioner to solve their system.

The third row of Figure :ref:`fig-ls-interface` is a list of data layouts or
matrix/vector storage schemes.  The relationship between linear solver and
storage scheme is similar to that of the conceptual interface and linear solver.
Note that some of the interfaces in hypre currently only support one
matrix/vector storage scheme choice.  The conceptual interface, the desired
solvers and preconditioners, and the matrix storage class must all be
compatible.


.. _writing-your-code:

Writing your code
------------------------------------------------------------------------------

As discussed in the previous section, the following decisions should be made
before writing any code:

* Choose a conceptual interface.
* Choose your desired solver strategy.
* Look up matrix requirements for each solver and preconditioner.
* Choose a matrix storage class that is compatible with your solvers and
  preconditioners and your conceptual interface.

Once the previous decisions have been made, it is time to code your application
to call hypre.  At this point, reviewing the previously mentioned example codes
provided with the hypre library may prove very helpful.  The example codes
demonstrate the following general structure of the application calls to hypre:

* **Build any necessary auxiliary structures for your chosen conceptual
  interface.** This includes, e.g., the grid and stencil structures if you are
  using the structured-grid interface.

* **Build the matrix, solution vector, and right-hand-side vector through your
  chosen conceptual interface.**  Each conceptual interface provides a series of
  calls for entering information about your problem into hypre.

* **Build solvers and preconditioners and set solver parameters (optional).**
  Some parameters like convergence tolerance are the same across solvers, while
  others are solver specific.

* **Call the solve function for the solver.**

* **Retrieve desired information from solver.** Depending on your application,
  there may be different things you may want to do with the solution vector.
  Also, performance information such as number of iterations is typically
  available, though it may differ from solver to solver.

The subsequent chapters of this User's Manual provide the details needed to more
fully understand the function of each conceptual interface and each solver.
Remember that a comprehensive list of all available functions is provided in
Chapter :ref:`ch-API`, and the provided example codes may prove helpful as
templates for your specific application.


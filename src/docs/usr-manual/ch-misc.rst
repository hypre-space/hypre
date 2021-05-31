.. Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


.. _ch-General:

******************************************************************************
General Information
******************************************************************************


Getting the Source Code
==============================================================================

The hypre distribution tar file is available from the Software link of the hypre
web page, http://www.llnl.gov/CASC/hypre/.  The hypre Software distribution page
allows access to the tar files of the latest and previous general and beta
distributions as well as documentation.


Building the Library
==============================================================================

In this and the following several sections, we discuss the steps to install and
use hypre on a Unix-like operating system, such as Linux, AIX, and Mac OS X.
Alternatively, the CMake build system [CMakeWeb]_ can be used, and is the best
approach for building hypre on Windows systems in particular (see the
``INSTALL`` file for details).

After unpacking the hypre tar file, the source code will be in the ``src``
sub-directory of a directory named hypre-VERSION, where VERSION is the current
version number (e.g., hypre-1.8.4, with a "b" appended for a beta release).

Move to the ``src`` sub-directory to build hypre for the host platform.  The
simplest method is to configure, compile and install the libraries in
``./hypre/lib`` and ``./hypre/include`` directories, which is accomplished by:

.. code-block:: bash

   ./configure
   make

NOTE: when executing on an IBM platform ``configure`` must be executed under the
nopoe script (``./nopoe ./configure <option> ...<option>``) to force a single
task to be run on the log-in node.

There are many options to ``configure`` and ``make`` to customize such things as
installation directories, compilers used, compile and load flags, etc.

Executing ``configure`` results in the creation of platform specific files that
are used when building the library. The information may include such things as
the system type being used for building and executing, compilers being used,
libraries being searched, option flags being set, etc.  When all of the
searching is done two files are left in the ``src`` directory; ``config.status``
contains information to recreate the current configuration and ``config.log``
contains compiler messages which may help in debugging ``configure`` errors.

Upon successful completion of ``configure`` the file ``config/Makefile.config``
is created from its template ``config/Makefile.config.in`` and hypre is ready to
be built.

Executing ``make``, with or without targets being specified, in the ``src``
directory initiates compiling of all of the source code and building of the
hypre library.  If any errors occur while compiling, the user can edit the file
``config/Makefile.config`` directly then run ``make`` again; without having to
re-run configure.

When building hypre without the install target, the libraries and include files
will be copied into the default directories, ``src/hypre/lib`` and
``src/hypre/include``, respectively.

When building hypre using the install target, the libraries and include files
will be copied into the directories that the user specified in the options to
``configure``, e.g. ``--prefix=/usr/apps``.  If none were specified the default
directories, ``src/hypre/lib`` and ``src/hypre/include``, are used.

.. _config_options:

Configure Options
------------------------------------------------------------------------------

There are many options to ``configure`` to allow the user to override and refine
the defaults for any system. The best way to find out what options are available
is to display the help package, by executing ``./configure --help``, which also
includes the usage information.  The user can mix and match the configure
options and variable settings to meet their needs.

Some of the commonly used options include:

.. code-block:: none

   --enable-debug                 Sets compiler flags to generate information
                                  needed for debugging.
   --enable-shared                Build shared libraries.
                                  NOTE: in order to use the resulting shared
                                        libraries the user MUST have the path to
                                        the libraries defined in the environment
                                        variable LD_LIBRARY_PATH.
   --with-print-errors            Print HYPRE errors
   --with-openmp                  Use OpenMP. This may affect which compiler is
                                  chosen.
   --enable-bigint                Use long long int for HYPRE_Int (default is NO).
   --enable-mixedint              Use long long int for HYPRE_BigInt and int for
                                  HYPRE_Int.
                                  NOTE: This option disables Euclid, ParaSails,
                                        pilut and CGC coarsening.

The user can mix and match the configure options and variable settings to meet
their needs.  It should be noted that hypre can be configured with external BLAS
and LAPACK libraries, which can be combined with any other option.  This is done
as follows (currently, both libraries must be configured as external together):

.. code-block:: bash

   ./configure  --with-blas-lib="blas-lib-name" \
                --with-blas-lib-dirs="path-to-blas-lib" \
                --with-lapack-lib="lapack-lib-name" \
                --with-lapack-lib-dirs="path-to-lapack-lib"

The output from ``configure`` is several pages long.  It reports the system type
being used for building and executing, compilers being used, libraries being
searched, option flags being set, etc.


Make Targets
------------------------------------------------------------------------------

The make step in building hypre is where the compiling, loading and creation of
libraries occurs.  Make has several options that are called targets.  These
include:

.. code-block:: none

   help         prints the details of each target

   all          default target in all directories
                compile the entire library
                does NOT rebuild documentation

   clean        deletes all files from the current directory that are
                   created by building the library

   distclean    deletes all files from the current directory that are created
                   by configuring or building the library

   install      compile the source code, build the library and copy executables,
                    libraries, etc to the appropriate directories for user access

   uninstall    deletes all files that the install target created

   tags         runs etags to create a tags table
                file is named TAGS and is saved in the top-level directory

   test         depends on the all target to be completed
                removes existing temporary installation directories
                creates temporary installation directories
                copies all libHYPRE* and *.h files to the temporary locations
                builds the test drivers; linking to the temporary locations to
                   simulate how application codes will link to HYPRE

GPU build
------------------------------------------------------------------------------

Hypre can support NVIDIA GPUs with CUDA and OpenMP (:math:`{\ge}` 4.5). The related ``configure`` options are

.. code-block:: none

  --with-cuda             Use CUDA. Require cuda-8.0 or higher (default is
                          NO).

  --with-device-openmp    Use OpenMP 4.5 Device Directives. This may affect
                          which compiler is chosen.

The related environment variables

.. code-block:: none

   HYPRE_CUDA_SM          (default 70)

   CUDA_HOME              the CUDA home directory

need to be set properly, which can be also set by

.. code-block:: none

   --with-gpu-arch=ARG    (e.g., --with-gpu-arch='60 70')

   --with-cuda-home=DIR

When configured with ``--with-cuda`` or ``--with-device-openmp``, the memory allocated on the GPUs, by default, is the GPU device memory, which is not accessible from the CPUs.
Hypre's structured solvers can work fine with device memory,
whereas only selected unstructured solvers can run with device memory. See 
Chapter :ref:`ch-boomeramg-gpu` for details.
In general, BoomerAMG and the SStruct
require  unified (CUDA managed) memory, for which
the following option should be added

.. code-block:: none

  --enable-unified-memory Use unified memory for allocating the memory
                          (default is NO).

Hypre's Struct solvers can also choose RAJA and Kokkos as the backend.
The ``configure`` options are

.. code-block:: none

  --with-raja             Use RAJA. Require RAJA package to be compiled
                          properly (default is NO).

  --with-kokkos           Use Kokkos. Require kokkos package to be compiled
                          properly(default is NO).

To run on the GPUs with RAJA and Kokkos, the options ``--with-cuda`` and ``--with-device-openmp`` are also needed,
and the RAJA and Kokkos libraries should be built with CUDA or OpenMP 4.5 correspondingly.

The other NVIDIA GPU related options include:

* ``--enable-gpu-profiling``  Use NVTX on CUDA, rocTX on HIP (default is NO)
* ``--enable-cusparse``       Use cuSPARSE for GPU sparse kernels (default is YES)
* ``--enable-cublas``         Use cuBLAS for GPU dense kernels (default is NO)
* ``--enable-curand``         Use random numbers generators on GPUs (default is YES)

Allocations and deallocations of GPU memory are expensive. Memory pooling is a common approach to reduce such overhead and improve performance.
hypre provides caching allocators for GPU device memory and unified memory, 
enabled by

.. code-block:: none

  --enable-device-memory-pool  Enable the caching GPU memory allocator in hypre
                               (default is NO)


hypre also supports Umpire [Umpire]_. To enable Umpire pool, include the following options:

.. code-block:: none

  --with-umpire                Use Umpire Allocator for device and unified memory
                               (default is NO)
  --with-umpire-include=/path-of-umpire-install/include
  --with-umpire-lib-dirs=/path-of-umpire-install/lib
  --with-umpire-libs=umpire

For running on AMD GPUs, configure with

.. code-block:: none

  --with-hip              Use HIP for AMD GPUs. (default is NO)
  --with-gpu-arch=ARG     Use appropriate AMD GPU architecture

Currently, only BoomerAMG is supported with HIP. The other AMD GPU related options include:

* ``--enable-gpu-profiling``  Use NVTX on CUDA, rocTX on HIP (default is NO)
* ``--enable-rocsparse``      Use rocSPARSE (default is YES)
* ``--enable-rocblas``        Use rocBLAS (default is NO)
* ``--enable-rocrand``        Use rocRAND (default is YES)

Testing the Library
==============================================================================

The ``examples`` subdirectory contains several codes that can be used to test
the newly created hypre library.  To create the executable versions, move into
the ``examples`` subdirectory, enter ``make`` then execute the codes as
described in the initial comments section of each source code.


Linking to the Library
==============================================================================

An application code linking with hypre must be compiled with
``-I$PREFIX/include`` and linked with ``-L$PREFIX/lib -lHYPRE``, where
``$PREFIX`` is the directory where hypre is installed, default is ``hypre``, or
as defined by the configure option ``--prefix=PREFIX``. As noted above, if hypre
was built as a shared library the user MUST have its location defined in the
environment variable ``LD_LIBRARY_PATH``.

As an example of linking with hypre, a user may refer to the ``Makefile`` in the
``examples`` sub-directory.  It is designed to build codes similar to user
applications that link with and call hypre.  All include and linking flags are
defined in the ``Makefile.config`` file by ``configure``.


Error Flags
==============================================================================

Every hypre function returns an integer, which is used to indicate errors
during execution.  Note that the error flag returned by a given function
reflects the errors from {\em all} previous calls to hypre functions.  In
particular, a value of zero means that all hypre functions up to (and
including) the current one have completed successfully.  This new error flag
system is being implemented throughout the library, but currently there are
still functions that do not support it.  The error flag value is a combination
of one or a few of the following error codes:

#. ``HYPRE_ERROR_GENERIC`` -- describes a generic error
#. ``HYPRE_ERROR_MEMORY`` -- hypre was unable to allocate memory
#. ``HYPRE_ERROR_ARG`` -- error in one of the arguments of a hypre function
#. ``HYPRE_ERROR_CONV`` -- a hypre solver did not converge as expected

One can use the ``HYPRE_CheckError`` function to determine exactly which errors
have occurred:

.. code-block:: c

   /* call some HYPRE functions */
   int  hypre_ierr;
   hypre_ierr = HYPRE_Function();

   /* check if the previously called hypre functions returned error(s) */
   if (hypre_ierr)
      /* check if the error with code HYPRE_ERROR_CODE has occurred */
      if (HYPRE_CheckError(hypre_ierr,HYPRE_ERROR_CODE))

The corresponding FORTRAN code is

.. code-block:: fortran

   ! header file with hypre error codes
   include 'HYPRE_error_f.h'

   ! call some HYPRE functions
   integer  hypre_ierr
   call HYPRE_Function(hypre_ierr)

   ! check if the previously called hypre functions returned error(s)
   if (hypre_ierr .ne. 0) then
      ! check if the error with code HYPRE_ERROR_CODE has occurred
      call HYPRE_CheckError(hypre_ierr, HYPRE_ERROR_CODE, check)
      if (check .ne. 0) then

The global error flag can also be obtained directly, between calls to other
hypre functions, by calling ``HYPRE_GetError()``.  If an argument error
(``HYPRE_ERROR_ARG``) has occurred, the argument index (counting from 1) can be
obtained from ``HYPRE_GetErrorArg()``.  To get a character string with a
description of all errors in a given error flag, use

.. code-block:: c

   HYPRE_DescribeError(int hypre_ierr, char *descr);

The global error flag can be cleared manually by calling
``HYPRE_ClearAllErrors()``, which will essentially ignore all previous hypre
errors. To only clear a specific error code, the user can call
``HYPRE_ClearError(HYPRE_ERROR_CODE)``.  Finally, if hypre was configured with
``--with-print-errors``, additional error information will be printed to the
standard error during execution.


Bug Reporting and General Support
==============================================================================

Simply create an issue at ``https://github.com/hypre-space/hypre/issues`` to
report bugs, request features, or ask general usage questions.

Users should include as much relevant information as possible in their issue
report, including at a minimum, the hypre version number being used.  For
compile and runtime problems, please also include the machine type, operating
system, MPI implementation, compiler, and any error messages produced.


.. _LSI_install:

Using HYPRE in External FEI Implementations
==============================================================================

To set up hypre for use in external, e.g. Sandia's, FEI implementations one
needs to follow the following steps:

#. obtain the hypre and Sandia's FEI source codes,
#. compile Sandia's FEI (fei-2.5.0) to create the ``fei_base`` library.
#. compile hypre

   * unpack the archive and go into the ``src`` directory
   * do a ``configure`` with the ``--with-fei-inc-dir`` option set to the FEI
     include directory plus other compile options
   * compile with ``make install`` to create the ``HYPRE_LSI`` library in
     ``hypre/lib``.

#. call the FEI functions in your application code (as shown in Chapters
   :ref:`ch-FEI` and :ref:`ch-Solvers`)

   * include ``cfei-hypre.h`` in your file
   * include ``FEI_Implementation.h`` in your file

#. Modify your ``Makefile``

   * include hypre's ``include`` and ``lib`` directories in the search paths.
   * Link with ``-lfei_base -lHYPRE_LSI``.  Note that the order in which the
     libraries are listed may be important.

Building an application executable often requires linking with many different
software packages, and many software packages use some LAPACK and/or BLAS
functions.  In order to alleviate the problem of multiply defined functions at
link time, it is recommended that all software libraries are stripped of all
LAPACK and BLAS function definitions.  These LAPACK and BLAS functions should
then be resolved at link time by linking with the system LAPACK and BLAS
libraries (e.g. dxml on DEC cluster).  Both hypre and SuperLU were built with
this in mind.  However, some other software library files needed may have the
BLAS functions defined in them.  To avoid the problem of multiply defined
functions, it is recommended that the offending library files be stripped of the
BLAS functions.


Calling HYPRE from Other Languages
==============================================================================

The hypre library currently supports two languages: C (native) and Fortran (in
version 2.10.1 and earlier, additional language interfaces were also provided
through a tool called Babel).  The Fortran interface is manually supported to
mirror the "native" C interface used throughout most of this manual.  We
describe this interface next.

Typically, the Fortran subroutine name is the same as the C name, unless it is
longer than 31 characters.  In these situations, the name is condensed to 31
characters, usually by simple truncation.  For now, users should look at the
Fortran test drivers (``*.f`` codes) in the ``test`` directory for the correct
condensed names.  In the future, this aspect of the interface conversion will be
made consistent and straightforward.

The Fortran subroutine argument list is always the same as the corresponding C
routine, except that the error return code ``ierr`` is always last.  Conversion
from C parameter types to Fortran argument type is summarized in following
table:

   ======================  =============================
   C parameter             Fortran argument
   ======================  =============================
   ``int i``               ``integer i``
   ``double d``            ``double precision d``
   ``int *array``          ``integer array(*)``
   ``double *array``       ``double precision array(*)``
   ``char *string``        ``character string(*)``
   ``HYPRE_Type object``   ``integer*8 object``
   ``HYPRE_Type *object``  ``integer*8 object``
   ======================  =============================

Array arguments in hypre are always of type ``(int *)`` or ``(double *)``, and
the corresponding Fortran types are simply ``integer`` or ``double precision``
arrays.  Note that the Fortran arrays may be indexed in any manner.  For
example, an integer array of length ``N`` may be declared in fortran as either
of the following:

.. code-block:: fortran

   integer  array(N)
   integer  array(0:N-1)

hypre objects can usually be declared as in the table because ``integer*8``
usually corresponds to the length of a pointer.  However, there may be some
machines where this is not the case.  On such machines, the Fortran type for a
hypre object should be an ``integer`` of the appropriate length.

This simple example illustrates the above information:

C prototype:

.. code-block:: c

   int HYPRE_IJMatrixSetValues(HYPRE_IJMatrix  matrix,
                               int  nrows, int  *ncols,
                               const int *rows, const int  *cols,
                               const double  *values);

The corresponding Fortran code for calling this routine is as follows:

.. code-block:: fortran

   integer*8         matrix
   integer           nrows, ncols(MAX_NCOLS)
   integer           rows(MAX_ROWS), cols(MAX_COLS)
   double precision  values(MAX_COLS)
   integer           ierr

   call HYPRE_IJMatrixSetValues(matrix, nrows, ncols, rows, cols, values, ierr)

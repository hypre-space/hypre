.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


.. _ch-General:

******************************************************************************
General Information
******************************************************************************

In this and the following sections, we discuss how to obtain and build hypre,
interpret error flags, report bugs, and call hypre from different programming languages.
We provide instructions for two build systems: autotools (configure & make) and CMake.
While autotools is traditionally used on Unix-like systems (Linux, macOS, etc.),
CMake provides cross-platform support and is required for Windows builds.
Both systems are actively maintained and supported.

.. _getting-source:

Getting the Source Code
==============================================================================

There are two ways to obtain hypre:

1. **Clone from GitHub (recommended)**::

      git clone https://github.com/hypre-space/hypre.git
      cd hypre/src

2. **Download a Release**

   Download the latest release from our `GitHub releases page <https://github.com/hypre-space/hypre/releases>`_.
   After extracting the archive, you'll find the source code in the ``src`` directory::

      tar -xvf hypre-x.y.z.tar.gz
      cd hypre-x.y.z/src

   where ``x.y.z`` represents the version number (e.g., 2.29.0).

Building the Library
==============================================================================

After obtaining the source code (see :ref:`getting-source`), there are three main ways to build hypre:

1. Using autotools (Configure & Make)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest method is to use the traditional configure and make:

.. code-block:: bash

   cd ${HYPRE_HOME}/src   # Move to the source directory
   ./configure            # Configure the build system
   make -j 4              # Use threads for a faster parallel build
   make install           # (Optional) Install hypre on a user-specified path via --prefix=<path>

This will build and install hypre in the default locations:

   - Libraries: ``${HYPRE_HOME}/src/hypre/lib``
   - Headers: ``${HYPRE_HOME}/src/hypre/include``

There are many options to ``configure`` and ``make`` to customize such things as
installation directories, compilers used, compile and load flags, etc. For more
information on the configure options, see :ref:`build_options`.

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
re-run configure. When building hypre without the install target, the libraries
and include files will be copied into the default directories, ``src/hypre/lib`` and
``src/hypre/include``, respectively. When building hypre using the install target,
the libraries and include files will be copied into the directories that the user
specified in the options to ``configure``, e.g. ``--prefix=/usr/apps``. If none were
specified the default directories, ``src/hypre/lib`` and ``src/hypre/include``, are used.

2. Using CMake (Windows, macOS, Linux, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CMake provides a modern, platform-independent build system. When using CMake to build hypre,
several files and directories are created during the build process:

* ``CMakeCache.txt`` - Stores configuration options and settings
* ``CMakeFiles/`` - Contains intermediate build files and dependency information
* ``cmake_install.cmake`` - Instructions for installing the built library
* ``Makefile`` - Generated build instructions (on Unix-like systems)

The build process has three main steps:

1. **Configure**: CMake reads the CMakeLists.txt files and generates the build system
2. **Build**: The native build tool (make, Visual Studio, etc.) compiles the code
3. **Install**: Built libraries and headers are copied to their final location

The simplest way to build hypre using CMake is:

.. code-block:: bash

   cd ${HYPRE_HOME}/build      # Use a separate build directory to keep source clean
   cmake ../src                # Generate build system
   make -j                     # Build the library in parallel
   make install                # (Optional) Install to specified location via -DCMAKE_INSTALL_PREFIX=<path>

During the configure step, CMake will detect your compiler and build tools,
it will find required dependencies, set up platform-specific build flags, and
generate native build files. If any errors occur during configuration, check
``CMakeCache.txt`` for current settings and ``CMakeFiles/CMakeError.log`` for
detailed error messages. The build step will create:

   - Static library: ``libHYPRE.a`` (Unix/macOS) or ``HYPRE.lib`` (Windows)
   - Shared library: ``libHYPRE.so`` (Linux), ``libHYPRE.dylib`` (macOS),
     or ``HYPRE.dll`` (Windows) if enabled
   - Object files in ``CMakeFiles/`` subdirectories

By default, ``make`` will place the library file in ``${HYPRE_HOME}/src/hypre/lib`` and
the header files in ``${HYPRE_HOME}/src/hypre/include``. As with the autotools method,
hypre's CMake build provides several options. For more information, see :ref:`build_options`.

.. note::

   CMake GUI (``ccmake`` or ``cmake-gui``) provides an interactive way to change build options:

   - **Unix**: From the ``${HYPRE_HOME}/build`` directory:

     1. Run ``ccmake ../src``
     2. Change options:
        - Press Enter to modify a variable
        - Boolean options (ON/OFF) toggle with Enter
        - String/file options allow text editing
     3. Press 'c' to configure
     4. Repeat until satisfied
     5. Press 'g' to generate

   - **Windows**: Using Visual Studio:

     1. Change desired options
     2. Click "Configure"
     3. Click "Generate"

3. Using Spack (Recommended for HPC environments)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Spack <https://spack.io/>`_  is a package manager designed for supercomputers, Linux, and macOS.
It makes installing scientific software easy and handles dependencies automatically. To build hypre using Spack:

.. code-block:: bash

   # Install Spack if you haven't already
   git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git
   . spack/share/spack/setup-env.sh

   # Install hypre with default options
   spack install hypre

   # Or install with specific options (e.g., with CUDA support)
   spack install hypre+cuda

Common Spack variants for hypre include:

* ``+mpi`` / ``~mpi`` - Enable/disable MPI support (default: +mpi)
* ``+cuda`` / ``~cuda`` - Enable/disable CUDA support (default: ~cuda)
* ``+openmp`` / ``~openmp`` - Enable/disable OpenMP support (default: ~openmp)
* ``+shared`` / ``~shared`` - Build shared libraries (default: ~shared)
* ``+debug`` / ``~debug`` - Build with debug flags (default: ~debug)

To see all available build options:

.. code-block:: bash

   spack info hypre

.. note::

   Spack will automatically handle dependencies and choose appropriate versions based on
   your system and requirements. It's particularly useful in HPC environments where you
   need to manage multiple versions or build configurations of hypre and its dependencies.

.. _build_options:

Build System Options
==============================================================

The table below lists the most commonly used build options for both autotools and CMake build systems.
Each option is shown with its default value (if applicable) and any relevant platform restrictions.
For GPU-specific options, see the :ref:`gpu_build` section below.

.. list-table:: Build Configuration Options
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - Autotools (configure)
     - CMake
   * - Install Path
     - ``--prefix=<path>``
     - ``-DCMAKE_INSTALL_PREFIX=<path>``
   * - | Debug Build
       | (default is off)
     - ``--enable-debug``
     - ``-DCMAKE_BUILD_TYPE=Debug``
   * - | Print Errors
       | (default is off)
     - ``--with-print-errors``
     - ``-DHYPRE_ENABLE_PRINT_ERRORS=ON``
   * - | Shared Library
       | (default is off)
     - ``--enable-shared``
     - ``-DBUILD_SHARED_LIBS=ON``
   * - | 64-bit integers
       | (default is off,
       | no GPU support)
     - ``--enable-bigint``
     - ``-DHYPRE_ENABLE_BIGINT=ON``
   * - | Mixed 32/64-bit integers
       | (default is off)
     - ``--enable-mixedint``
     - ``-DHYPRE_ENABLE_MIXEDINT=ON``
   * - | Single FP precision
       | (default is off)
     - ``--enable-single``
     - ``-DHYPRE_ENABLE_SINGLE=ON``
   * - | Long double precision
       | (default is off,
       | no GPU support)
     - ``--enable-long-double``
     - ``-DHYPRE_ENABLE_LONG_DOUBLE=ON``
   * - | Link-time optimization
       | (default is off)
     - N/A
     - ``-DHYPRE_ENABLE_LTO=ON``
   * - | MPI Support
       | (default is on)
     - ``--enable-mpi``
     - ``-DHYPRE_ENABLE_MPI=ON``
   * - | MPI Persistent
       | (default is off)
     - ``--enable-persistent``
     - ``-DHYPRE_ENABLE_PERSISTENT_COMM=ON``
   * - | OpenMP Support
       | (default is off)
     - ``--with-openmp``
     - ``-DHYPRE_ENABLE_OPENMP=ON``
   * - | Hopscotch hashing
       | (requires OpenMP)
       | (default is off)
     - ``--enable-hopscotch``
     - ``-DHYPRE_ENABLE_HOPSCOTCH=ON``
   * - | Fortran Support
       | (default is on)
     - ``--enable-fortran``
     - ``-DHYPRE_ENABLE_FORTRAN=ON``
   * - | Fortran mangling
       | (default is 0)
       | (values are 0...5)
     - ``--with-fmangle``
     - ``-DHYPRE_ENABLE_FMANGLE=0``
   * - | Fortran BLAS mangling
       | (default is 0)
       | (values are 0...5)
     - ``--with-fmangle-blas``
     - ``-DHYPRE_ENABLE_FMANGLE_BLAS=0``
   * - | Fortran LAPACK mangling
       | (default is 0)
       | (values are 0...5)
     - ``--with-fmangle-lapack``
     - ``-DHYPRE_ENABLE_FMANGLE_LAPACK=0``
   * - | External BLAS
       | (default is off)
     - | ``--with-blas-lib=<lib>``
       | ``--with-blas-lib-dirs=<path>``
     - ``-DHYPRE_ENABLE_HYPRE_BLAS=OFF``
   * - | External LAPACK
       | (default is off)
     - | ``--with-lapack-lib=<lib>``
       | ``--with-lapack-lib-dirs=<path>``
     - ``-DHYPRE_ENABLE_HYPRE_LAPACK=OFF``
   * - | SuperLU_DIST Support
       | (default is off)
     - ``--with-dsuperlu``
     - ``-DHYPRE_ENABLE_DSUPERLU=ON``
   * - | MAGMA Support
       | (default is off)
     - ``--with-magma``
     - ``-DHYPRE_ENABLE_MAGMA=ON``
   * - | Caliper Support
       | (default is off)
     - ``--with-caliper``
     - ``-DHYPRE_ENABLE_CALIPER=ON``
   * - Build Examples
     - N/A
     - ``-DHYPRE_BUILD_EXAMPLES=ON``
   * - Build Tests
     - N/A
     - ``-DHYPRE_BUILD_TESTS=ON``

.. note::

   * CMake options are case-sensitive
   * Boolean CMake options accept ``ON``/``OFF`` values
   * Executables located under ``src/test`` and ``src/examples``
     are built separately when using the autotools build system
   * For a complete list of options:

     * **Autotools**: Run ``./configure --help``
     * **CMake**: See ``CMakeLists.txt`` or run ``cmake -LAH``
   * For third-party libraries (TPLs), hypre supports two methods:

     1. **CMake Package Config (recommended)**:
        Use ``-DPackage_ROOT=/path/to/package`` to help CMake find package
        configuration files

     2. **Manual specification**:

        a. **Autotools**:

           .. code-block:: bash

              --with-pkg-include=/path/to/pkg-include
              --with-pkg-lib=/path/to/pkg-lib

        b. **CMake**:

           .. code-block:: bash

              -DTPL_PACKAGE_INCLUDE_DIRS=/path/to/pkg-include
              -DTPL_PACKAGE_LIBRARIES=/path/to/pkg-lib/libpackage.so

.. _gpu_build:

GPU Build Options
==============================================================

The hypre library provides support for multiple GPU architectures through different
programming models: CUDA (for NVIDIA GPUs), HIP (for AMD GPUs), and SYCL (for Intel
GPUs). Each model has its own set of build options and requirements. Some solvers and
features may have different levels of support across these platforms. Key considerations
when building for GPUs are:

1. Only one GPU backend can be enabled at a time (CUDA, HIP, or SYCL)
2. Some features like full support for 64-bit integers (`BigInt`) are not available
3. Memory management options (device vs unified memory) affect solver availability

The table below lists the available GPU-specific build options for both autotools and CMake
build systems.

.. list-table:: GPU Configuration Options
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - Autotools (configure)
     - CMake
   * - | CUDA Support
       | (default is off)
     - ``--with-cuda``
     - ``-DHYPRE_ENABLE_CUDA=ON``
   * - | HIP Support
       | (default is off)
     - ``--with-hip``
     - ``-DHYPRE_ENABLE_HIP=ON``
   * - | SYCL Support
       | (default is off)
     - ``--with-sycl``
     - ``-DHYPRE_ENABLE_SYCL=ON``
   * - | SYCL Target
       | (default is empty,
       | **SYCL** only)
     - ``--with-sycl-target=ARG``
     - ``-DHYPRE_SYCL_TARGET=ARG``
   * - | SYCL Target Backend
       | (default is empty,
       | **SYCL** only)
     - ``--with-sycl-target-backend=ARG``
     - ``-DHYPRE_SYCL_TARGET_BACKEND=ARG``
   * - | GPU architecture
       | (determined automatically)
     - ``--with-gpu-arch=ARG``
     - | ``-DCMAKE_CUDA_ARCHITECTURES=ARG``
       | ``-DCMAKE_HIP_ARCHITECTURES=ARG``
   * - | GPU Profiling
       | (default is off)
     - ``--enable-gpu-profiling``
     - ``-DHYPRE_ENABLE_GPU_PROFILING=ON``
   * - | GPU-aware MPI
       | (default is off)
     - ``--enable-gpu-aware-mpi``
     - ``-DHYPRE_ENABLE_GPU_AWARE_MPI=ON``
   * - | Unified Memory
       | (default is off)
     - ``--enable-unified-memory``
     - ``-DHYPRE_ENABLE_UNIFIED_MEMORY=ON``
   * - | Device async malloc
       | (default is off)
     - ``--enable-device-malloc-async``
     - ``-DHYPRE_ENABLE_DEVICE_MALLOC_ASYNC=ON``
   * - | Thrust async execution
       | (default is off)
     - ``--enable-thrust-async``
     - ``-DHYPRE_ENABLE_THRUST_ASYNC=ON``
   * - | cuSPARSE Support
       | (default is on, **CUDA** only)
     - ``--enable-cusparse``
     - ``-DHYPRE_ENABLE_CUSPARSE=ON``
   * - | cuSOLVER Support
       | (default is on, **CUDA** only)
     - ``--enable-cusolver``
     - ``-DHYPRE_ENABLE_CUSOLVER=ON``
   * - | cuBLAS Support
       | (default is on, **CUDA** only)
     - ``--enable-cublas``
     - ``-DHYPRE_ENABLE_CUBLAS=ON``
   * - | cuRAND Support
       | (default is on, **CUDA** only)
     - ``--enable-curand``
     - ``-DHYPRE_ENABLE_CURAND=ON``
   * - | rocSPARSE Support
       | (default is on, **HIP** only)
     - ``--enable-rocsparse``
     - ``-DHYPRE_ENABLE_ROCSOLVER=ON``
   * - | rocSOLVER Support
       | (default is on, **HIP** only)
     - ``--enable-rocsolver``
     - ``-DHYPRE_ENABLE_ROCSOLVER=ON``
   * - | rocBLAS Support
       | (default is on, **HIP** only)
     - ``--enable-rocblas``
     - ``-DHYPRE_ENABLE_ROCBLAS=ON``
   * - | rocRAND Support
       | (default is on, **HIP** only)
     - ``--enable-rocrand``
     - ``-DHYPRE_ENABLE_ROCRAND=ON``
   * - | oneMKLSparse Support
       | (default is on, **SYCL** only)
     - ``--enable-onemklsparse``
     - ``-DHYPRE_ENABLE_ONEMKLSPARSE=ON``
   * - | oneMKLBLAS Support
       | (default is on, **SYCL** only)
     - ``--enable-onemklblas``
     - ``-DHYPRE_ENABLE_ONEMKLBLAS=ON``
   * - | oneMKLRAND Support
       | (default is on, **SYCL** only)
     - ``--enable-onemklrand``
     - ``-DHYPRE_ENABLE_ONEMKLRAND=ON``
   * - | Umpire Support
       | (default is off)
     - ``--with-umpire``
     - ``-DHYPRE_ENABLE_UMPIRE=ON``
   * - | Umpire Unified Memory
       | (default is off)
     - ``--with-umpire-um``
     - ``-DHYPRE_ENABLE_UMPIRE_UM=ON``
   * - | Umpire Device Memory
       | (default is off)
     - ``--with-umpire-device``
     - ``-DHYPRE_ENABLE_UMPIRE_DEVICE=ON``

.. warning::

   Allocations and deallocations of GPU memory can be slow. Memory pooling is a
   common approach to reduce such overhead and improve performance. We recommend using
   [Umpire]_ for memory management, which provides robust pooling capabilities for both
   device and unified memory. For Umpire support, the Umpire library must be installed
   and properly configured. See the note in the previous section for more details on
   how to specify the installation path for dependency libraries.

.. note::

   When hypre is configured with device support, but without unified memory, the
   memory allocated on the GPUs, by default, is the GPU device memory, which is
   not accessible from the CPUs. Hypre's structured solvers can run with device
   memory, whereas only selected unstructured solvers can run with device memory.
   See :ref:`ch-boomeramg-gpu` for details. Some solver options for BoomerAMG
   require unified (managed) memory.

Make Targets
=====================

The make step in building hypre is where the compiling, loading and creation of
libraries occurs. Make has several options that are called targets. These
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

Using the Library
=================

The ``examples`` subdirectory contains several codes that demonstrate hypre's features
and can be used to test the library. These examples can be built in two ways:

1. **Using CMake**:
   Enable the ``HYPRE_BUILD_EXAMPLES`` option during configuration:

   .. code-block:: bash

      cmake -DHYPRE_BUILD_EXAMPLES=ON ..
      make

2. **Using Makefiles**:
   Navigate to the ``examples`` subdirectory and build directly:

   .. code-block:: bash

      cd examples
      make

Each example contains detailed comments at the beginning of its source file explaining
its purpose and how to run it. The examples demonstrate various interfaces, solvers,
and problem types. For a categorized list of examples and their features, see the
HTML documentation in the ``examples/docs`` directory.

.. note::

   The examples are designed to mimic real application codes and can serve as
   templates for your own implementations.

Testing the Library
===================

hypre provides several approaches to test the library, in increasing order of comprehensiveness:

1. **Basic Tests** (Recommended first step):
   Quick tests to check library functionality (CMake requires ``-DBUILD_TESTING=ON``):

   .. code-block:: bash

      # Single test for each linear system interface
      make check

      # Test IJ, Struct and SStruct linear solvers in parallel
      make checkpar

2. **Comprehensive Tests** (CMake only):
   Test linear solvers for all linear system interfaces (linear-algebraic, Struct and SStruct):

   .. code-block:: bash

      cmake -DBUILD_TESTING=ON ..
      make -j
      make test # or ctest

3. **Automated Testing** (Developers only):
   For thorough testing across different configurations and machines including regression
   tests, and performance benchmarks, with support for both CPU and GPU executions. Test
   results are automatically compared against saved baseline outputs, with the ability to
   update these baselines when legitimate changes occur. The automated testing
   infrastructure is particularly focused on ensuring consistency across different build
   configurations and execution environments. For more information, see the `README
   <https://github.com/hypre-space/hypre/blob/master/AUTOTEST/README.txt>`_ file.

.. note::

   * Test tolerance can be adjusted using ``-DHYPRE_CHECK_TOL=<value>`` during CMake configuration. Default tolerance is 1.0e-6
   * Test output files with ``.err`` extension contain error messages and diagnostics
   * AUTOTEST configurations can be customized by modifying machine-specific files in the AUTOTEST directory

For detailed test results and logs:

* Make check results: ``build/test/*.err`` (CMake) or ``src/test/TEST_(ij|struct|ssstruct)/*.err`` (Autotools)
* CTest results: ``build/Testing/Temporary/LastTest.log``
* AUTOTEST results: ``src/AUTOTEST/machine_name.dir/machine_name.err``

Linking to the Library
==============================================================================

There are two main approaches to link your application with hypre:

Using CMake
^^^^^^^^^^^

The hypre library provides CMake configuration files that enable easy integration. Create a
``CMakeLists.txt`` with:

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.21)
   project(MyApp LANGUAGES C)

   find_package(HYPRE REQUIRED)

   add_executable(myapp main.c)
   target_link_libraries(myapp PUBLIC HYPRE::HYPRE lm)

If hypre is not in a standard location, specify its path:

.. code-block:: bash

   cmake -DHYPRE_ROOT=/path/to/hypre-install-directory ..

Using Autotools
^^^^^^^^^^^^^^^

For non-CMake builds, manually specify compilation and linking flags:

.. code-block:: bash

   # Compilation
   -I${HYPRE_INSTALL_DIR}/include

   # Linking
   -L${HYPRE_INSTALL_DIR}/lib -lHYPRE -lm

Where ``${HYPRE_INSTALL_DIR}`` is your hypre installation directory (default is ``${HYPRE_HOME}/src/hypre``,
or as specified by ``--prefix=PREFIX`` during configuration).

Shared Library Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If hypre was built as a shared library, you have several options:

1. **Environment Variables**:
   Add hypre's library location to your system's library path:

   .. code-block:: bash

      # Linux/Unix
      export LD_LIBRARY_PATH=${HYPRE_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

      # macOS
      export DYLD_LIBRARY_PATH=${HYPRE_INSTALL_DIR}/lib:${DYLD_LIBRARY_PATH}

      # Windows
      set PATH=%HYPRE_INSTALL_DIR%\lib;%PATH%

2. **RPATH/RUNPATH**:
   Set the runtime search path during linking. With CMake:

   .. code-block:: cmake

      # Use RPATH (searched before LD_LIBRARY_PATH)
      set(CMAKE_INSTALL_RPATH "${HYPRE_INSTALL_DIR}/lib")
      set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

      # Or use RUNPATH (searched after LD_LIBRARY_PATH)
      set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--enable-new-dtags")
      set(CMAKE_INSTALL_RPATH "${HYPRE_INSTALL_DIR}/lib")
      set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

   Or with manual linking:

   .. code-block:: bash

      # RPATH
      -Wl,-rpath,${HYPRE_INSTALL_DIR}/lib

      # RUNPATH
      -Wl,--enable-new-dtags,-rpath,${HYPRE_INSTALL_DIR}/lib

   ``RPATH`` is searched before ``LD_LIBRARY_PATH`` while ``RUNPATH`` is searched
   after, giving you flexibility in controlling library resolution precedence.

.. note::

   For examples of linking applications with hypre, refer to the ``examples`` subdirectory.

Error Flags
==============================================================================

Every hypre function returns an integer, which is used to indicate errors
during execution.  Note that the error flag returned by a given function
reflects the errors from *all* previous calls to hypre functions.  In
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
``--with-print-errors`` or ``-DHYPRE_ENABLE_PRINT_ERRORS=ON``, additional error
information will be printed to the standard error during execution.


Bug Reporting and General Support
==============================================================================

For bug reports, feature requests, and general usage questions, please create an issue on
`GitHub issues <https://github.com/hypre-space/hypre/issues>`_. You can also browse existing
issues to see if your question has already been addressed. To help us address your issue
effectively, please include:

**Required Information:**

- hypre version number
- Description of the problem or feature request
- Minimal example demonstrating the issue (if applicable)

**For Build Issues:**

- Build system used (CMake or autotools)
- Build configuration options
- Complete build output showing the error
- Operating system and version
- Compiler and version
- MPI implementation and version

**For Runtime Issues:**

- Command line arguments used
- Problem size and configuration
- Number of processes/threads
- Complete error messages or stack traces
- Information about the computing environment:

  * GPU type and driver version (for GPU builds)
  * Relevant environment variables
  * System architecture (CPU type, memory)

**For Performance Issues:**

- Performance measurements or profiling data
- Comparison with previous versions (if applicable)
- Problem size and scaling information
- Hardware configuration details

.. _LSI_install:

Using HYPRE in External FEI Implementations
==============================================================================

.. warning::
   FEI is not actively supported by the hypre development team. For similar
   functionality, we recommend using :ref:`sec-Block-Structured-Grids-FEM`, which
   allows the representation of block-structured grid problems via hypre's
   SStruct interface.

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

.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


.. _ch-MPrecision:

******************************************************************************
Mixed Precision
******************************************************************************

The hypre library has provided compile-time multi-precision support for many
years.  For example, the autotools option ``--enable-single`` or the CMake
option ``-DHYPRE_SINGLE=ON`` will produce a single precision library.

Starting in hypre version 3.0, multiple precision and mixed precision support
are provided at runtime.  To turn this on, use

- ``--enable-mixed-precision`` (autotools)
- ``-DHYPRE_ENABLE_MIXED_PRECISION=ON`` (CMake)

With the above, users can compile, link, and run as before without changes to
their code.  To access runtime precision, there are several levels of support
that can be used, outlined in the following sections.  For clarity, a generic
name ``Foo`` is used to represent a function in hypre, e.g., ``HYPRE_PCGSolve``.


.. _sec-MP-Fixed:

Calling functions with fixed precision
==============================================================================

For every function ``Foo`` in hypre, the following fixed-precision versions are
also available,

- ``Foo_flt``
- ``Foo_dbl``
- ``Foo_long_dbl``

where the precision of each function is determined by the C compiler and the
respective C types ``float``, ``double``, and ``long double``.  The prototypes
for these functions are exactly the same as for ``Foo``, but with real-valued
arguments like ``HYPRE_Real`` mapping to the specific C types (and precisions)
indicated above.


.. _sec-MP-Multiple:

Calling functions with multiple precisions
==============================================================================

Every user-API function ``Foo`` in hypre (any function beginning with the upper
case ``HYPRE_`` prefix) is also available in the mixed-precision configuration
of the library, but its precision is determined by a global runtime precision
that can be set by calling

.. code-block:: c

   HYPRE_Int HYPRE_SetGlobalPrecision(HYPRE_Precision precision)

where ``precision`` is either ``HYPRE_REAL_SINGLE``, ``HYPRE_REAL_DOUBLE``, or
``HYPRE_REAL_LONGDOUBLE``.  Real-valued arguments for ``Foo`` have different
types from the functions described in Section :ref:`sec-MP-Fixed` because they
have to support all three precisions, but calling ``Foo`` in practice is much
the same.  Specifically, real arrays such as ``HYPRE_Real *`` become ``void *``,
and real values such as ``HYPRE_Real`` become ``long double``.  This prototyping
enables multiple-precision functionality, although strong type checking at
compile time is lost.


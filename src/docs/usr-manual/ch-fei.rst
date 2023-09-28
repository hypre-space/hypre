.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


.. _ch-FEI:

******************************************************************************
Finite Element Interface
******************************************************************************

.. warning::
   FEI is not actively supported by the hypre development team. For similar
   functionality, we recommend using :ref:`sec-Block-Structured-Grids-FEM`, which
   allows the representation of block-structured grid problems via hypre's
   SStruct interface.

Introduction
==============================================================================

Many application codes use unstructured finite element meshes.  This section
describes an interface for finite element problems, called the FEI, which is
supported in hypre.

.. figure:: figSquareHole.*
   :align: center

   Example of an unstructured mesh.

FEI refers to a specific interface for black-box finite element solvers,
originally developed in Sandia National Lab, see [ClEA1999]_.  It differs from
the rest of the conceptual interfaces in hypre in two important aspects: it is
written in C++, and it does not separate the construction of the linear system
matrix from the solution process.  A complete description of Sandia's FEI
implementation can be obtained by contacting Alan Williams at Sandia
(william@sandia.gov).  A simplified version of the FEI has been implemented at
LLNL and is included in hypre.  More details about this implementation can be
found in the header files of the ``FEI_mv/fei-base`` and ``FEI_mv/fei-hypre``
directories.


A Brief Description of the Finite Element Interface
==============================================================================

Typically, finite element codes contain data structures storing element
connectivities, element stiffness matrices, element loads, boundary conditions,
nodal coordinates, etc.  One of the purposes of the FEI is to assemble the
global linear system in parallel based on such local element data.  We
illustrate this in the rest of the section and refer to example 10 (in the
``examples`` directory) for more implementation details.

In hypre, one creates an instance of the FEI as follows:

.. code-block:: c++

   LLNL_FEI_Impl *feiPtr = new LLNL_FEI_Impl(mpiComm);

Here ``mpiComm`` is an MPI communicator (e.g. ``MPI\_COMM\_WORLD``).  If
Sandia's FEI package is to be used, one needs to define a hypre solver object
first:

.. code-block:: c++

   LinearSystemCore   *solver = HYPRE_base_create(mpiComm);
   FEI_Implementation *feiPtr = FEI_Implementation(solver,mpiComm,rank);

where ``rank`` is the number of the master processor (used only to identify
which processor will produce the screen outputs).  The ``LinearSystemCore``
class is the part of the FEI that interfaces with the linear solver library. It
will be discussed later in Sections :ref:`LSI_solvers` and :ref:`LSI_install`.

Local finite element information is passed to the FEI using several methods of
the ``feiPtr`` object.  The first entity to be submitted is the *field*
information.  A *field* has an identifier called ``fieldID`` and a rank or
``fieldSize`` (number of degree of freedom). For example, a discretization of
the Navier Stokes equations in 3D can consist of velocity vector having
:math:`3` degrees of freedom in every node (vertex) of the mesh and a scalar
pressure variable, which is constant over each element. If these are the only
variables, and if we assign ``fieldID`` :math:`7` and :math:`8` to them,
respectively, then the finite element field information can be set up by

.. code-block:: c++

   nFields   = 2;                 /* number of unknown fields */
   fieldID   = new int[nFields];  /* field identifiers */
   fieldSize = new int[nFields];  /* vector dimension of each field */

   /* velocity (a 3D vector) */
   fieldID[0]   = 7;
   fieldSize[0] = 3;

   /* pressure (a scalar function) */
   fieldID[1]   = 8;
   fieldSize[1] = 1;

   feiPtr -> initFields(nFields, fieldSize, fieldID);

Once the field information has been established, we are ready to initialize an
element block. An element block is characterized by the block identifier, the
number of elements, the number of nodes per element, the nodal fields and the
element fields (fields that have been defined previously). Suppose we use
:math:`1000` hexahedral elements in the element block :math:`0`, the setup
consists of

.. code-block:: c++

   elemBlkID  = 0;     /* identifier for a block of elements */
   nElems     = 1000;  /* number of elements in the block */
   elemNNodes = 8;     /* number of nodes per element */

   /* nodal-based field for the velocity */
   nodeNFields     = 1;
   nodeFieldIDs    = new[nodeNFields];
   nodeFieldIDs[0] = fieldID[0];

   /* element-based field for the pressure */
   elemNFields     = 1;
   elemFieldIDs    = new[elemNFields];
   elemFieldIDs[0] = fieldID[1];

   feiPtr -> initElemBlock(elemBlkID, nElems, elemNNodes, nodeNFields,
                           nodeFieldIDs, elemNFields, elemFieldIDs, 0);

The last argument above specifies how the dependent variables are arranged in
the element matrices. A value of :math:`0` indicates that each variable is to be
arranged in a separate block (as opposed to interleaving).

In a parallel environment, each processor has one or more element blocks.
Unless the element blocks are all disjoint, some of them share a common set of
nodes on the subdomain boundaries. To facilitate setting up interprocessor
communications, shared nodes between subdomains on different processors are to
be identified and sent to the FEI.  Hence, each node in the whole domain is
assigned a unique global identifier. The shared node list on each processor
contains a subset of the global node list corresponding to the local nodes that
are shared with the other processors.  The syntax for setting up the shared
nodes is

.. code-block:: c++

   feiPtr -> initSharedNodes(nShared, sharedIDs, sharedLengs, sharedProcs);

This completes the initialization phase, and a completion signal is sent to the
FEI via

.. code-block:: c++

   feiPtr -> initComplete();

Next, we begin the *load* phase. The first entity for loading is the nodal
boundary conditions. Here we need to specify the number of boundary equations
and the boundary values given by ``alpha``, ``beta``, and ``gamma``.  Depending
on whether the boundary conditions are Dirichlet, Neumann, or mixed, the three
values should be passed into the FEI accordingly.

.. code-block:: c++

   feiPtr -> loadNodeBCs(nBCs, BCEqn, fieldID, alpha, beta, gamma);

The element stiffness matrices are to be loaded in the next step. We need to
specify the element number :math:`i`, the element block to which element
:math:`i` belongs, the element connectivity information, the element load, and
the element matrix format. The element connectivity specifies a set of :math:`8`
node global IDs (for hexahedral elements), and the element load is the load or
force for each degree of freedom.  The element format specifies how the
equations are arranged (similar to the interleaving scheme mentioned above).
The calling sequence for loading element stiffness matrices is

.. code-block:: c++

   for (i = 0; i < nElems; i++)
      feiPtr -> sumInElem(elemBlkID, elemID, elemConn[i], elemStiff[i],
                          elemLoads[i], elemFormat);

To complete the assembling of the global stiffness matrix and the corresponding
right hand side, a signal is sent to the FEI via

.. code-block:: c++

   feiPtr -> loadComplete();

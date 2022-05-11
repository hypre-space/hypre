.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


.. _ch-SStruct:

******************************************************************************
Semi-Structured-Grid System Interface (SStruct)
******************************************************************************

The ``SStruct`` interface is appropriate for applications with grids that are
mostly---but not entirely---structured, e.g. block-structured grids (see Figure
:ref:`fig-sstruct-example`), composite grids in structured adaptive mesh
refinement (AMR) applications (see Figure :ref:`fig-sstruct-samr-grid`), and
overset grids.  In addition, it supports more general PDEs than the ``Struct``
interface by allowing multiple variables (system PDEs) and multiple variable
types (e.g. cell-centered, face-centered, etc.).  The interface provides access
to data structures and linear solvers in hypre that are designed for
semi-structured grid problems, but also to the most general data structures and
solvers.

The ``SStruct`` grid is composed out of a number of structured grid *parts*,
where the physical inter-relationship between the parts is arbitrary.  Each part
is constructed out of two basic components: boxes (see Figure
:ref:`fig-struct-boxes`) and *variables*.  Variables represent the actual
unknown quantities in the grid, and are associated with the box indices in a
variety of ways, depending on their types.  In hypre, variables may be
cell-centered, node-centered, face-centered, or edge-centered.  Face-centered
variables are split into x-face, y-face, and z-face, and edge-centered variables
are split into x-edge, y-edge, and z-edge.  See Figure :ref:`fig-gridvars` for
an illustration in 2D.

.. _fig-gridvars:

.. figure:: figSStructGridVars.*
   :align: center

   Figure 5

   Grid variables in hypre are referenced by the abstract cell-centered index
   to the left and down in 2D (analogously in 3D).  In the figure, index :math:`(i,j)`
   is used to reference the variables in black.  The variables in grey---although
   contained in the pictured cell---are not referenced by the :math:`(i,j)` index.

The ``SStruct`` interface uses a *graph* to allow nearly arbitrary relationships
between part data.  The graph is constructed from stencils or finite element
stiffness matrices plus some additional data-coupling information set by the
``GraphAddEntries()`` routine.  Two other methods for relating part data are the
``GridSetNeighborPart()`` and ``GridSetSharedPart()`` routines, which are
particularly well suited for block-structured grid problems.  The latter is
useful for finite element codes.

There are five basic steps involved in setting up the linear system to be
solved:

#. set up the grid,
#. set up the stencils (if needed),
#. set up the graph,
#. set up the matrix,
#. set up the right-hand-side vector.


.. _sec-Block-Structured-Grids:

Block-Structured Grids with Stencils
==============================================================================

In this section, we describe how to use the ``SStruct`` interface to define
block-structured grid problems.  We do this primarily by example, paying
particular attention to the construction of stencils and the use of the
``GridSetNeighborPart()`` interface routine.

Consider the solution of the diffusion equation

.. math::
   :label: eqn-block-diffusion

   - \nabla \cdot (D \nabla u) + \sigma u = f

on the block-structured grid in Figure :ref:`fig-sstruct-example`, where
:math:`D` is a scalar diffusion coefficient, and :math:`\sigma \geq 0`.  The
discretization [MoRS1998]_ introduces three different types of variables:
cell-centered, :math:`x`-face, and :math:`y`-face.  The three discretization
stencils that couple these variables are also given in the figure.  The
information in this figure is essentially all that is needed to describe the
nonzero structure of the linear system we wish to solve.

.. _fig-sstruct-example:

.. figure:: figSStructExample1a.*
   :align: center

   Figure 6a

   Example of a block-structured grid with five logically-rectangular blocks and
   three variables types: cell-centered, :math:`x`-face, and :math:`y`-face.
   Discretization stencils for the cell-centered (left), :math:`x`-face
   (middle), and :math:`y`-face (right) variables are also pictured.

.. figure:: figSStructExample1b.*
   :align: center

   Figure 6b

   Need to combine this with 6a.

.. _fig-sstruct-example-parts:

.. figure:: figSStructExample1c.*
   :align: center

   Figure 7

   One possible labeling of the grid in Figure :ref:`fig-sstruct-example`.

The grid in Figure :ref:`fig-sstruct-example` is defined in terms of five
separate logically-rectangular parts as shown in Figure
:ref:`fig-sstruct-example-parts`, and each part is given a unique label between
0 and 4.  Each part consists of a single box with lower index :math:`(1,1)` and
upper index :math:`(4,4)` (see Section :ref:`sec-Struct-Grid`), and the grid
data is distributed on five processes such that data associated with part
:math:`p` lives on process :math:`p`.  Note that in general, parts may be
composed out of arbitrary unions of boxes, and indices may consist of
non-positive integers (see Figure :ref:`fig-struct-boxes`).  Also note that the
``SStruct`` interface expects a domain-based data distribution by boxes, but the
actual distribution is determined by the user and simply described (in parallel)
through the interface.

.. |figSStructGrid1| image:: figSStructGrid1.*
   :width: 100%
.. |figSStructGrid2| image:: figSStructGrid2.*
   :width: 100%
.. |figSStructGrid3| image:: figSStructGrid3.*
   :width: 100%
.. |figSStructGrid4| image:: figSStructGrid4.*
   :width: 100%
.. |figSStructGrid5| image:: figSStructGrid5.*
   :width: 100%
.. |figSStructGrid6| image:: figSStructGrid6.*
   :width: 100%

.. _fig-sstruct-grid:
    
+----------------------+----------------------+----------------------+
| 1: |figSStructGrid1| | 2: |figSStructGrid2| | 3: |figSStructGrid3| |
+----------------------+----------------------+----------------------+ 
| 4: |figSStructGrid4| | 5: |figSStructGrid5| | 6: |figSStructGrid6| |
+----------------------+----------------------+----------------------+

.. code-block:: c
   
       HYPRE_SStructGrid grid;
       int ndim = 2, nparts = 5, nvars = 3, part = 3;
       int extents[][2] = {{1,1}, {4,4}};
       int vartypes[]   = {HYPRE_SSTRUCT_VARIABLE_CELL,
                           HYPRE_SSTRUCT_VARIABLE_XFACE,
                           HYPRE_SSTRUCT_VARIABLE_YFACE};
       int nb2_n_part      = 2,              nb4_n_part      = 4;
       int nb2_exts[][2]   = {{1,0}, {4,0}}, nb4_exts[][2]   = {{0,1}, {0,4}};
       int nb2_n_exts[][2] = {{1,1}, {1,4}}, nb4_n_exts[][2] = {{4,1}, {4,4}};
       int nb2_map[2]      = {1,0},          nb4_map[2]      = {0,1};
       int nb2_dir[2]      = {1,-1},         nb4_dir[2]      = {1,1};
   
   1:  HYPRE_SStructGridCreate(MPI_COMM_WORLD, ndim, nparts, &grid);
       
       /* Set grid extents and grid variables for part 3 */
   2:  HYPRE_SStructGridSetExtents(grid, part, extents[0], extents[1]);
   3:  HYPRE_SStructGridSetVariables(grid, part, nvars, vartypes);
       
       /* Set spatial relationship between parts 3 and 2, then parts 3 and 4 */
   4:  HYPRE_SStructGridSetNeighborPart(grid, part, nb2_exts[0], nb2_exts[1],
          nb2_n_part, nb2_n_exts[0], nb2_n_exts[1], nb2_map, nb2_dir);
   5:  HYPRE_SStructGridSetNeighborPart(grid, part, nb4_exts[0], nb4_exts[1],
          nb4_n_part, nb4_n_exts[0], nb4_n_exts[1], nb4_map, nb4_dir);
       
   6:  HYPRE_SStructGridAssemble(grid);
    
Code on process 3 for setting up the grid in Figure :ref:`fig-sstruct-example}.`

As with the ``Struct`` interface, each process describes that portion of the
grid that it "owns", one box at a time.  Figure :ref:`fig-sstruct-grid` shows
the code for setting up the grid on process 3 (the code for the other processes
is similar).  The "icons" at the top of the figure illustrate the result of the
numbered lines of code.  Process 3 needs to describe the data pictured in the
bottom-right of the figure.  That is, it needs to describe part 3 plus some
additional neighbor information that ties part 3 together with the rest of the
grid.  The ``Create()`` routine creates an empty 2D grid object with five parts
that lives on the ``MPI_COMM_WORLD`` communicator.  The ``SetExtents()`` routine
adds a new box to the grid.  The ``SetVariables()`` routine associates three
variables of type cell-centered, :math:`x`-face, and :math:`y`-face with part 3.

At this stage, the description of the data on part 3 is complete.  However, the
spatial relationship between this data and the data on neighboring parts is not
yet defined.  To do this, we need to relate the index space for part 3 with the
index spaces of parts 2 and 4.  More specifically, we need to tell the interface
that the two grey boxes neighboring part 3 in the bottom-right of
Figure :ref:`fig-sstruct-grid` also correspond to boxes on parts 2 and 4.  This
is done through the two calls to the ``SetNeighborPart()`` routine.  We
discuss only the first call, which describes the grey box on the right of the
figure.  Note that this grey box lives outside of the box extents for the grid
on part 3, but it can still be described using the index-space for part 3
(recall Figure :ref:`fig-struct-boxes`).  That is, the grey box has extents
:math:`(1,0)` and :math:`(4,0)` on part 3's index-space, which is outside of part 3's grid.
The arguments for the ``SetNeighborPart()`` call are simply the lower and
upper indices on part 3 and the corresponding indices on part 2.  The final two
arguments to the routine indicate that the positive :math:`x`-direction on part 3
(i.e., the :math:`i` component of the tuple :math:`(i,j)`) corresponds to the positive
:math:`y`-direction on part 2 and that the positive :math:`y`-direction on part 3
corresponds to the positive :math:`x`-direction on part 2.

The ``Assemble()`` routine is a collective call (i.e., must be called on all
processes from a common synchronization point), and finalizes the grid assembly,
making the grid "ready to use".

With the neighbor information, it is now possible to determine where off-part
stencil entries couple.  Take, for example, any shared part boundary such as the
boundary between parts 2 and 3.  Along these boundaries, some stencil entries
reach outside of the part.  If no neighbor information is given, these entries
are effectively zeroed out, i.e., they don't participate in the discretization.
However, with the additional neighbor information, when a stencil entry reaches
into a neighbor box it is then coupled to the part described by that neighbor
box information.

Another important consequence of the use of the ``SetNeighborPart()`` routine is
that it can declare variables on different parts as being the same.  For
example, the face variables on the boundary of parts 2 and 3 are recognized as
being shared by both parts (prior to the ``SetNeighborPart()`` call, there were
two distinct sets of variables).  Note also that these variables are of
different types on the two parts; on part 2 they are :math:`x`-face variables,
but on part 3 they are :math:`y`-face variables.

For brevity, we consider only the description of the :math:`y`-face stencil in
Figure :ref:`fig-sstruct-example`, i.e. the third stencil in the figure.  To do
this, the stencil entries are assigned unique labels between 0 and 8 and their
"offsets" are described relative to the "center" of the stencil.  This process
is illustrated in Figure :ref:`fig-sstruct-stencil`.  Nine calls are made to the
routine ``HYPRE_SStructStencilSetEntry()``.  As an example, the call that
describes stencil entry 5 in the figure is given the entry number 5, the offset
:math:`(-1,0)`, and the identifier for the :math:`x`-face variable (the variable
to which this entry couples).  Recall from Figure :ref:`fig-gridvars` the
convention used for referencing variables of different types.  The geometry
description uses the same convention, but with indices numbered relative to the
referencing index :math:`(0,0)` for the stencil's center.  Figure
:ref:`fig-sstruct-graph` shows the code for setting up the graph .

.. _fig-sstruct-stencil:

.. figure:: figSStructStenc0.*
   :align: center

   Figure 7a

   Assignment of labels and geometries to the :math:`y`-face stencil in Figure
   :ref:`fig-sstruct-example}.`

.. figure:: figSStructStenc1.*
   :align: center

   Figure 7b

   Need to combine this with 7a.

.. |figSStructGraph1| image:: figSStructGraph1.*
   :width: 100%
.. |figSStructGraph2| image:: figSStructGraph2.*
   :width: 100%
.. |figSStructGraph5| image:: figSStructGraph5.*
   :width: 100%

.. _fig-sstruct-graph:

+-----------------------+-----------------------+-----------------------+
| 1: |figSStructGraph1| | 2: |figSStructGraph2| | 3: |figSStructGraph5| |
+-----------------------+-----------------------+-----------------------+

.. code-block:: c
   
       HYPRE_SStructGraph graph;
       HYPRE_SStructStencil c_stencil, x_stencil, y_stencil;
       int c_var = 0, x_var = 1, y_var = 2;
       int part;
       
   1:  HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);
       
       /* Set the cell-centered, x-face, and y-face stencils for each part */
       for (part = 0; part < 5; part++)
       {
   2:     HYPRE_SStructGraphSetStencil(graph, part, c_var, c_stencil);
          HYPRE_SStructGraphSetStencil(graph, part, x_var, x_stencil);
          HYPRE_SStructGraphSetStencil(graph, part, y_var, y_stencil);
       }
       
   3:  HYPRE_SStructGraphAssemble(graph);

Code on process 3 for setting up the graph for Figure :ref:`fig-sstruct-example}`.

With the above, we now have a complete description of the nonzero structure for
the matrix.  The matrix coefficients are then easily set in a manner similar to
what is described in Section :ref:`sec-Struct-Matrix` using routines
``MatrixSetValues()`` and ``MatrixSetBoxValues()`` in the ``SStruct`` interface.
As before, there are also ``AddTo`` variants of these routines.  Likewise,
setting up the right-hand-side is similar to what is described in Section
:ref:`sec-Struct-RHS`.  See the hypre reference manual for details.

An alternative approach for describing the above problem through the interface
is to use the ``GraphAddEntries()`` routine instead of the
``GridSetNeighborPart()`` routine.  In this approach, the five parts would be
explicitly "sewn" together by adding non-stencil couplings to the matrix graph.
The main downside to this approach for block-structured grid problems is that
variables along block boundaries are no longer considered to be the same
variables on the corresponding parts that share these boundaries.  For example,
any face variable along the boundary between parts 2 and 3 in Figure
:ref:`fig-sstruct-example` would represent two different variables that live on
different parts.  To "sew" the parts together correctly, we would need to
explicitly select one of these variables as the representative that participates
in the discretization, and make the other variable a dummy variable that is
decoupled from the discretization by zeroing out appropriate entries in the
matrix.  All of these complications are avoided by using the
``GridSetNeighborPart()`` for this example.


.. _sec-Block-Structured-Grids-FEM:

Block-Structured Grids with Finite Elements
==============================================================================

In this section, we describe how to use the ``SStruct`` interface to define
block-structured grid problems with finite elements.  We again do this by
example, paying particular attention to the use of the ``FEM`` interface
routines and the ``GridSetSharedPart()`` routine.  See example code ``ex14.c``
for a complete implementation.

Consider a nodal finite element (FEM) discretization of the Laplace equation on
the star-shaped grid in Figure :ref:`fig-sstruct-fem-example`.  The local FEM
stiffness matrix in the figure describes the coupling between the grid
variables.  Although we could still describe this problem using stencils as in
Section :ref:`sec-Block-Structured-Grids`, an FEM-based approach (available in
hypre version ``2.6.0b`` and later) is a more natural alternative.

.. _fig-sstruct-fem-example:

.. figure:: figSStructExample3a.*
   :align: center

   Figure 8a

   Example of a star-shaped grid with six logically-rectangular blocks and one
   nodal variable.  Each block has an angle at the origin given by
   :math:`\gamma=\pi/3`.  The finite element stiffness matrix (right) is given
   in terms of the pictured variable ordering (left).

.. figure:: figSStructExample3b.*
   :align: center

   Figure 8b

   Need to combine this with 8a.

The grid in Figure :ref:`fig-sstruct-fem-example` is defined in terms of six
separate logically-rectangular parts, and each part is given a unique label
between 0 and 5.  Each part consists of a single box with lower index
:math:`(1,1)` and upper index :math:`(9,9)`, and the grid data is distributed on
six processes such that data associated with part :math:`p` lives on process
:math:`p`.

.. |figSStructGridFEM1| image:: figSStructGridFEM1.*
   :width: 100%
.. |figSStructGridFEM2| image:: figSStructGridFEM2.*
   :width: 100%
.. |figSStructGridFEM3| image:: figSStructGridFEM3.*
   :width: 100%
.. |figSStructGridFEM4| image:: figSStructGridFEM4.*
   :width: 100%
.. |figSStructGridFEM5| image:: figSStructGridFEM5.*
   :width: 100%
.. |figSStructGridFEM6| image:: figSStructGridFEM6.*
   :width: 100%

.. _fig-sstruct-fem-grid:

+-------------------------+-------------------------+-------------------------+
| 1: |figSStructGridFEM1| | 2: |figSStructGridFEM2| | 3: |figSStructGridFEM3| |
+-------------------------+-------------------------+-------------------------+
| 4: |figSStructGridFEM4| | 5: |figSStructGridFEM5| | 6: |figSStructGridFEM6| |
+-------------------------+-------------------------+-------------------------+

.. code-block:: c
   
       HYPRE_SStructGrid grid;
       int ndim = 2, nparts = 6, nvars = 1, part = 0;
       int ilower[2]    = {1,1}, iupper[2] = {9,9};
       int vartypes[]   = {HYPRE_SSTRUCT_VARIABLE_NODE};
       int ordering[12] = {0,-1,-1,  0,+1,-1,  0,+1,+1,  0,-1,+1};
   
       int s_part   = 2;
       int ilo[2]   = {1,1}, iup[2]   = {1,9}, offset[2]   = {-1,0};
       int s_ilo[2] = {1,1}, s_iup[2] = {9,1}, s_offset[2] = {0,-1};
       int map[2]   = {1,0};
       int dir[2]   = {-1,1};
   
   1:  HYPRE_SStructGridCreate(MPI_COMM_WORLD, ndim, nparts, &grid);
       
       /* Set grid extents, grid variables, and FEM ordering for part 0 */
   2:  HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
   3:  HYPRE_SStructGridSetVariables(grid, part, nvars, vartypes);
   4:  HYPRE_SStructGridSetFEMOrdering(grid, part, ordering);
   
       /* Set shared variables for parts 0 and 1 (0 and 2/3/4/5 not shown) */
   5:  HYPRE_SStructGridSetSharedPart(grid, part, ilo, iup, offset,
          s_part, s_ilo, s_iup, s_offset, map, dir);
   
   6:  HYPRE_SStructGridAssemble(grid);
    
Code on process 0 for setting up the grid in Figure :ref:`fig-sstruct-fem-example`.

As in Section :ref:`sec-Block-Structured-Grids`, each process describes that
portion of the grid that it "owns", one box at a time.  Figure
:ref:`fig-sstruct-fem-grid` shows the code for setting up the grid on process 0
(the code for the other processes is similar).  The "icons" at the top of the
figure illustrate the result of the numbered lines of code.  Process 0 needs to
describe the data pictured in the bottom-right of the figure.  That is, it needs
to describe part 0 plus some additional information about shared data with other
parts on the grid.  The ``SetFEMOrdering()`` routine sets the ordering of the
unknowns in an element (an element is always a grid cell in hypre).  This
determines the ordering of the data passed into the routines
``MatrixAddFEMValues()`` and ``VectorAddFEMValues()`` discussed later.

At this point, the layout of the data on part 0 is complete, but there is no
relationship to the rest of the grid.  To couple the parts, we need to tell
hypre that some of the boundary variables on part 0 are shared with other parts,
i.e., they are the same as some of the variables on other parts.  This is done
through five calls to the ``SetSharedPart()`` routine.  Only the first call is
shown in the figure; the other four calls are similar.  The arguments to this
routine are the same as ``SetNeighborPart()`` with the addition of two new
offset arguments, named ``offset`` and ``s_offset`` in the figure.  Each offset
represents a pointer from the cell center to one of the following: all variables
in the cell (no nonzeros in offset); all variables on a face (only 1 nonzero);
all variables on an edge (2 nonzeros); all variables at a point (3 nonzeros).
The two offsets must be consistent with each other.

The graph is set up similarly to Figure :ref:`fig-sstruct-graph`, except that
the stencil calls are replaced by calls to ``GraphSetFEM()``.  The nonzero
pattern of the stiffness matrix can also be set by calling the optional routine
``GraphSetFEMSparsity()``.

Matrix and vector values are set one element at a time.  For the example in this
section, calls on part 0 would have the following form:

.. code-block:: c
   
   int part = 0;
   int index[2] = {i,j};
   double m_values[16] = {...};
   double v_values[4]  = {...};
   
   HYPRE_SStructMatrixAddFEMValues(A, part, index, m_values);
   HYPRE_SStructVectorAddFEMValues(v, part, index, v_values);

Here, ``m_values`` contains local stiffness matrix values and ``v_values``
contains local variable values.  The global matrix and vector are assembled
internally by hypre, using the shared variables to couple the parts.


.. _sec-Structured-Adaptive-Mesh-Refinement:

Structured Adaptive Mesh Refinement
==============================================================================

We now briefly discuss how to use the ``SStruct`` interface in a structured AMR
application.  Consider Poisson's equation on the simple cell-centered example
grid illustrated in Figure :ref:`fig-sstruct-samr-grid`.  For structured AMR
applications, each refinement level should be defined as a unique part.  There
are two parts in this example: part 0 is the global coarse grid and part 1 is
the single refinement patch.  Note that the coarse unknowns underneath the
refinement patch (gray dots in Figure :ref:`fig-sstruct-samr-grid`) are not real
physical unknowns; the solution in this region is given by the values on the
refinement patch.  In setting up the composite grid matrix [McCo1989]_ for hypre
the equations for these "dummy" unknowns should be uncoupled from the other
unknowns (this can easily be done by setting all off-diagonal couplings to zero
in this region).

.. _fig-sstruct-samr-grid:

.. figure:: figSStructExample2a.*
   :align: center

   Figure 9

   Structured AMR grid example. Shaded regions correspond to process 0, unshaded
   to process 1.  The grey dots are dummy variables.


In the example, parts are distributed across the same two processes with process
0 having the "left" half of both parts.  The composite grid is then set up
part-by-part by making calls to ``GridSetExtents()`` just as was done in Section
:ref:`sec-Block-Structured-Grids` and Figure :ref:`fig-sstruct-grid` (no
``SetNeighborPart`` calls are made in this example).  Note that in the interface
there is no required rule relating the indexing on the refinement patch to that
on the global coarse grid; they are separate parts and thus each has its own
index space.  In this example, we have chosen the indexing such that refinement
cell :math:`(2i,2j)` lies in the lower left quadrant of coarse cell
:math:`(i,j)`.  Then the stencil is set up.  In this example we are using a
finite volume approach resulting in the standard 5-point stencil in Section
:ref:`sec-Struct-Grid` in both parts.

The grid and stencil are used to define all intra-part coupling in the graph,
the non-zero pattern of the composite grid matrix.  The inter-part coupling at
the coarse-fine interface is described by ``GraphAddEntries()`` calls.  This
coupling in the composite grid matrix is typically the composition of an
interpolation rule and a discretization formula.  In this example, we use a
simple piecewise constant interpolation, i.e. the solution value in a coarse
cell is equal to the solution value at the cell center.  Then the flux across a
portion of the coarse-fine interface is approximated by a difference of the
solution values on each side.  As an example, consider approximating the flux
across the left interface of cell :math:`(6,6)` in Figure
:ref:`fig-sstruct-samr-stencil`.  Let :math:`h` be the coarse grid mesh size,
and consider a local coordinate system with the origin at the center of cell
:math:`(6,6)`.  We approximate the flux as follows

.. math::

   \int_{-h/4}^{h/4}{u_x(-h/4,s)} ds
      & \approx \frac{h}{2} u_x(-h/4,0)
        \approx \frac{h}{2} \frac{u(0,0)-u(-3h/4,0)}{3h/4} \\
      & \approx \frac{2}{3} (u_{6,6}-u_{2,3}) .

The first approximation uses the midpoint rule for the edge integral, the second
uses a finite difference formula for the derivative, and the third the piecewise
constant interpolation to the solution in the coarse cell.  This means that the
equation for the variable at cell :math:`(6,6)` involves not only the stencil
couplings to :math:`(6,7)` and :math:`(7,6)` on part 1 but also non-stencil
couplings to :math:`(2,3)` and :math:`(3,2)` on part 0.  These non-stencil
couplings are described by ``GraphAddEntries()`` calls.  The syntax for this
call is simply the part and index for both the variable whose equation is being
defined and the variable to which it couples.  After these calls, the non-zero
pattern of the matrix (and the graph) is complete.  Note that the "west" and
"south" stencil couplings simply "drop off" the part, and are effectively zeroed
out (currently, this is only supported for the ``HYPRE_PARCSR`` object type, and
these values must be manually zeroed out for other object types; see
``MatrixSetObjectType()`` in the reference manual).

.. _fig-sstruct-samr-stencil:

.. figure:: figSStructExample2b.*
   :align: center

   Figure 2

   Coupling for equation at corner of refinement patch. Black lines (solid and
   broken) are stencil couplings. Gray line are non-stencil couplings.

The remaining step is to define the actual numerical values for the composite
grid matrix.  This can be done by either ``MatrixSetValues()`` calls to set
entries in a single equation, or by ``MatrixSetBoxValues()`` calls to set
entries for a box of equations in a single call.  The syntax for the
``MatrixSetValues()`` call is a part and index for the variable whose equation
is being set and an array of entry numbers identifying which entries in that
equation are being set.  The entry numbers may correspond to stencil entries or
non-stencil entries.


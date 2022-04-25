.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


.. _ADS:

ADS
==============================================================================

The Auxiliary-space Divergence Solver (ADS) is a parallel unstructured solver
similar to AMS, but targeting :math:`H(div)` instead of :math:`H(curl)`
problems. Its usage and options are very similar to those of AMS, and in general
the relationship between ADS and AMS is analogous to that between AMS and AMG.

Specifically ADS was designed for the scalable solution of linear systems
arising in the finite element discretization of the variational problem

.. math::
   :label: ads-hdiv

   \mbox{Find } {\mathbf u} \in {\mathbf W}_h \>:\qquad
   (\alpha\, \nabla \cdot {\mathbf u},  \nabla \cdot {\mathbf v}) +
   (\beta\, {\mathbf u},  {\mathbf v}) = ({\mathbf f},  {\mathbf v})\,,
   \qquad \mbox{for all } {\mathbf v} \in {\mathbf W}_h \,,

where :math:`{\mathbf W}_h` is the lowest order Raviart-Thomas (face) finite
element space, and :math:`\alpha>0` and :math:`\beta>0` are scalar, or SPD
matrix variable coefficients.  It is based on the auxiliary space methods for
:math:`H(div)` problems proposed in [HiXu2006]_.


Overview
------------------------------------------------------------------------------

Let :math:`{\mathbf A}` and :math:`{\mathbf b}` be the stiffness matrix and the
load vector corresponding to :eq:`ads-hdiv`. Then the resulting linear system of
interest reads,

.. math::
   :label: ads-hdiv-ls

   {\mathbf A}\, {\mathbf x} = {\mathbf b} \,.

The coefficients :math:`\alpha` and :math:`\beta` are naturally associated with
the "stiffness" and "mass" terms of :math:`{\mathbf A}`.  Besides
:math:`{\mathbf A}` and :math:`{\mathbf b}`, ADS requires the following
additional user input:

#. The discrete curl matrix :math:`C` representing the faces of the mesh in
   terms of its edges. :math:`C` has as many rows as the number of faces in the
   mesh, with each row having nonzero entries :math:`+1` and :math:`-1` in the
   columns corresponding to the edges composing the face. The sign is determined
   based on the orientation of the edges relative to the face.  We require that
   :math:`C` includes all (interior and boundary) faces and edges.

#. The discrete gradient matrix :math:`G` representing the edges of the mesh in
   terms of its vertices. :math:`G` has as many rows as the number of edges in
   the mesh, with each row having two nonzero entries: :math:`+1` and :math:`-1`
   in the columns corresponding to the vertices composing the edge. The sign is
   determined based on the orientation of the edge.  We require that :math:`G`
   includes all (interior and boundary) edges and vertices.

#. Vectors :math:`x`, :math:`y`, and :math:`z` representing the coordinates of
   the vertices of the mesh.

Internally, ADS proceeds with the construction of the following additional objects:

* :math:`A_C` -- the curl-curl matrix :math:`C^{\,T} {\mathbf A} C`.
* :math:`{\mathbf \Pi}` -- the matrix representation of the interpolation
  operator from vector linear to face finite elements.
* :math:`{\mathbf A}_{{\mathbf \Pi}}` -- the vector nodal matrix :math:`{\mathbf
  \Pi}^{\,T} {\mathbf A} {\mathbf \Pi}`.
* :math:`B_C` and :math:`{\mathbf B}_{{\mathbf \Pi}}` -- efficient (AMS/AMG)
  solvers for :math:`A_C` and :math:`{\mathbf A}_{{\mathbf \Pi}}`.

The solution procedure then is a three-level method using smoothing in the
original face space and subspace corrections based on :math:`B_C` and
:math:`{\mathbf B}_{{\mathbf \Pi}}`.  We can employ a number of options here
utilizing various combinations of the smoother and solvers in additive or
multiplicative fashion.


Sample Usage
------------------------------------------------------------------------------

ADS can be used either as a solver or as a preconditioner.  Below we list the
sequence of hypre calls needed to create and use it as a solver. We start with
the allocation of the ``HYPRE_Solver`` object:

.. code-block:: c
   
   HYPRE_Solver solver;
   HYPRE_ADSCreate(&solver);

Next, we set a number of solver parameters. Some of them are optional, while
others are necessary in order to perform the solver setup.

The user is required to provide the discrete curl and gradient matrices
:math:`C` and :math:`G`.  ADS expects a matrix defined on the whole mesh with no
boundary faces, edges or nodes excluded. It is essential to **not** impose any
boundary conditions on :math:`C` or :math:`G`.  Regardless of which hypre
conceptual interface was used to construct the matrices, one can always obtain a
ParCSR version of them. This is the expected format in the following functions.

.. code-block:: c
   
   HYPRE_ADSSetDiscreteCurl(solver, C);
   HYPRE_ADSSetDiscreteGradient(solver, G);

Next, ADS requires the coordinates of the vertices in the mesh as three hypre
parallel vectors.  The corresponding function call reads:

.. code-block:: c
   
   HYPRE_ADSSetCoordinateVectors(solver, x, y, z);

The remaining solver parameters are optional.  For example, the user can choose
a different cycle type by calling

.. code-block:: c
   
   HYPRE_ADSSetCycleType(solver, cycle_type); /* default value: 1 */

The available cycle types in ADS are:

* ``cycle_type=1``: multiplicative solver :math:`(01210)`
* ``cycle_type=2``: additive solver :math:`(0+1+2)`
* ``cycle_type=3``: multiplicative solver :math:`(02120)`
* ``cycle_type=4``: additive solver :math:`(010+2)`
* ``cycle_type=5``: multiplicative solver :math:`(0102010)`
* ``cycle_type=6``: additive solver :math:`(1+020)`
* ``cycle_type=7``: multiplicative solver :math:`(0201020)`
* ``cycle_type=8``: additive solver :math:`(0(1+2)0)`
* ``cycle_type=11``: multiplicative solver :math:`(013454310)`
* ``cycle_type=12``: additive solver :math:`(0+1+3+4+5)`
* ``cycle_type=13``: multiplicative solver :math:`(034515430)`
* ``cycle_type=14``: additive solver :math:`(01(3+4+5)10)`

Here we use the following convention for the three subspace correction methods:
:math:`0` refers to smoothing, :math:`1` stands for AMS based on :math:`B_C`,
and :math:`2` refers to a call to BoomerAMG for :math:`{\mathbf B}_{{\mathbf
\Pi}}`.  The values :math:`3`, :math:`4` and :math:`5` refer to the scalar
subspaces corresponding to the :math:`x`, :math:`y` and :math:`z` components of
:math:`\mathbf \Pi`.

The abbreviation :math:`xyyz` for :math:`x,y,z \in \{0,1,2,3,4,5\}` refers to a
multiplicative subspace correction based on solvers :math:`x`, :math:`y`,
:math:`y`, and :math:`z` (in that order).  The abbreviation :math:`x+y+z` stands
for an additive subspace correction method based on :math:`x`, :math:`y` and
:math:`z` solvers.  The additive cycles are meant to be used only when ADS is
called as a preconditioner.  In our experience the choices
``cycle_type=1,5,8,11,13`` often produced fastest solution times, while
``cycle_type=7`` resulted in smallest number of iterations.

Additional solver parameters, such as the maximum number of iterations, the
convergence tolerance and the output level, can be set with

.. code-block:: c
   
   HYPRE_ADSSetMaxIter(solver, maxit);     /* default value: 20 */
   HYPRE_ADSSetTol(solver, tol);           /* default value: 1e-6 */
   HYPRE_ADSSetPrintLevel(solver, print);  /* default value: 1 */

More advanced parameters, affecting the smoothing and the internal AMS and AMG
solvers, can be set with the following three functions:

.. code-block:: c
   
   HYPRE_ADSSetSmoothingOptions(solver, 2, 1, 1.0, 1.0);
   HYPRE_ADSSetAMSOptions(solver, 11, 10, 1, 3, 0.25, 0, 0);
   HYPRE_ADSSetAMGOptions(solver, 10, 1, 3, 0.25, 0, 0);

We note that the AMS cycle type, which is the second parameter of
``HYPRE_ADSSetAMSOptions`` should be greater than 10, unless the high-order
interface of ``HYPRE_ADSSetInterpolations`` described in the next subsection is
being used.

After the above calls, the solver is ready to be constructed.  The user has to
provide the stiffness matrix :math:`{\mathbf A}` (in ParCSR format) and the
hypre parallel vectors :math:`{\mathbf b}` and :math:`{\mathbf x}`. (The vectors
are actually not used in the current ADS setup.) The setup call reads,

.. code-block:: c
   
   HYPRE_ADSSetup(solver, A, b, x);

It is important to note the order of the calling sequence. For example, do
**not** call ``HYPRE_ADSSetup`` before calling each of the functions
``HYPRE_ADSSetDiscreteCurl``, ``HYPRE_ADSSetDiscreteGradient`` and
``HYPRE_ADSSetCoordinateVectors``.

Once the setup has completed, we can solve the linear system by calling

.. code-block:: c
   
   HYPRE_ADSSolve(solver, A, b, x);

Finally, the solver can be destroyed with

.. code-block:: c
   
   HYPRE_ADSDestroy(&solver);

More details can be found in the files ``ads.h`` and ``ads.c`` located in the
``parcsr_ls`` directory.


High-order Discretizations
------------------------------------------------------------------------------

Similarly to AMS, ADS also provides support for (arbitrary) high-order
:math:`H(div)` discretizations. Since the robustness of ADS depends on the
performance of AMS and BoomerAMG on the associated (high-order) auxiliary
subspace problems, we note that the convergence may not be optimal for large
polynomial degrees :math:`k \geq 1`.

In the high-order ADS interface, the user does not need to provide the
coordinates of the vertices, but instead should construct and pass the
Raviart-Thomas and Nedelec interpolation matrices :math:`{\mathbf \Pi}_{RT}` and
:math:`{\mathbf \Pi}_{ND}` which map (high-order) vector nodal finite elements
into the (high-order) Raviart-Thomas and Nedelec space. In other words, these
are the (parallel) matrix representation of the interpolation mappings from
:math:`\mathrm{P}_k^3 / \mathrm{Q}_k^3` into :math:`\mathrm{RT}_{k-1}` and
:math:`\mathrm{ND}_k`, see [HiXu2006]_, [KoVa2009]_.  We require these matrices
as inputs, since in the high-order case their entries very much depend on the
particular choice of the basis functions in the finite element spaces, as well
as on the geometry of the mesh elements. The columns of the :math:`{\mathbf
\Pi}` matrices should use a node-based numbering, where the
:math:`x`/:math:`y`/:math:`z` components of the first node (vertex or high-order
degree of freedom) should be listed first, followed by the
:math:`x`/:math:`y`/:math:`z` components of the second node and so on (see the
documentation of ``HYPRE_BoomerAMGSetDofFunc``). Furthermore, each interpolation
matrix can be split into :math:`x`, :math:`y` and :math:`z` components by
defining :math:`{\mathbf \Pi}^x \varphi = {\mathbf \Pi} (\varphi,0,0)`, and
similarly for :math:`{\mathbf \Pi}^y` and :math:`{\mathbf \Pi}^z`.

The discrete gradient and curl matrices :math:`G` and :math:`C` should
correspond to the mappings :math:`\varphi \in \mathrm{P}_k^3 / \mathrm{Q}_k^3
\mapsto \nabla \varphi \in \mathrm{ND}_k` and :math:`{\mathbf u} \in
\mathrm{ND}_k \mapsto \nabla \times {\mathbf u} \in \mathrm{RT}_{k-1}`, so even
though their values are still independent of the mesh coordinates, they will not
be :math:`\pm 1`, but will be determined by the particular form of the
high-order basis functions and degrees of freedom.

With these matrices, the high-order setup procedure is simply

.. code-block:: c
   
   HYPRE_ADSSetDiscreteCurl(solver, C);
   HYPRE_ADSSetDiscreteGradient(solver, G);
   HYPRE_ADSSetInterpolations(solver, RT_Pi, NULL, NULL, NULL,
                                      ND_Pi, NULL, NULL, NULL);

We remark that the above interface calls can also be used in the lowest-order
case (or even other types of discretizations), but we recommend calling the
previously described ``HYPRE_ADSSetCoordinateVectors`` instead, since this
allows ADS to handle the construction and use of the interpolations internally.


Specifying the monolithic :math:`{\mathbf \Pi}_{RT}` limits the ADS cycle type
options to those less than 10. Alternatively one can separately specify the
:math:`x`, :math:`y` and :math:`z` components of :math:`{\mathbf \Pi}_{RT}`.

.. code-block:: c
   
   HYPRE_ADSSetInterpolations(solver, NULL, RT_Pix, RT_Piy, RT_Piz,
                                      ND_Pi, NULL, NULL, NULL);

which enables the use of ADS cycle types with index greater than 10. The same
holds for :math:`{\mathbf \Pi}_{ND}` and its components, e.g. to enable the
subspace AMS cycle type greater then 10 we need to call

.. code-block:: c
   
   HYPRE_ADSSetInterpolations(solver, NULL, RT_Pix, RT_Piy, RT_Piz,
                                      NULL, ND_Pix, ND_Piy, ND_Piz);

Finally, both :math:`{\mathbf \Pi}` and their components can be passed to the solver:

.. code-block:: c
   
   HYPRE_ADSSetInterpolations(solver, RT_Pi, RT_Pix, RT_Piy, RT_Piz
                                      ND_Pi, ND_Pix, ND_Piy, ND_Piz);

which will duplicate some memory, but allows for experimentation with all
available ADS and AMS cycle types.


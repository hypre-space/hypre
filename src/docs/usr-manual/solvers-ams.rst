.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


.. _AMS:

AMS
==============================================================================

AMS (the Auxiliary-space Maxwell Solver) is a parallel unstructured Maxwell
solver for edge finite element discretizations of the variational problem

.. math::
   :label: ams-maxwell

   \mbox{Find } {\mathbf u} \in {\mathbf V}_h \>:\qquad
   (\alpha\, \nabla \times {\mathbf u},  \nabla \times {\mathbf v}) +
   (\beta\, {\mathbf u},  {\mathbf v}) = ({\mathbf f},  {\mathbf v})\,,
   \qquad \mbox{for all } {\mathbf v} \in {\mathbf V}_h \,.

Here :math:`{\mathbf V}_h` is the lowest order Nedelec (edge) finite element
space, and :math:`\alpha>0` and :math:`\beta \ge 0` are scalar, or SPD matrix
coefficients.  AMS was designed to be scalable on problems with variable
coefficients, and allows for :math:`\beta` to be zero in part or the whole
domain.  In either case the resulting problem is only semidefinite, and for
solvability the right-hand side should be chosen to satisfy compatibility
conditions.

AMS is based on the auxiliary space methods for definite Maxwell problems
proposed in [HiXu2006]_.  For more details, see [KoVa2009]_.


Overview
------------------------------------------------------------------------------

Let :math:`{\mathbf A}` and :math:`{\mathbf b}` be the stiffness matrix and the
load vector corresponding to :eq:`ams-maxwell`. Then the resulting linear system
of interest reads,

.. math::
   :label: ams-maxwell-ls

   {\mathbf A}\, {\mathbf x} = {\mathbf b} \,.

The coefficients :math:`\alpha` and :math:`\beta` are naturally associated with
the "stiffness" and "mass" terms of :math:`{\mathbf A}`.  Besides
:math:`{\mathbf A}` and :math:`{\mathbf b}`, AMS requires the following
additional user input:

#. The discrete gradient matrix :math:`G` representing the edges of the mesh in
   terms of its vertices. :math:`G` has as many rows as the number of edges in
   the mesh, with each row having two nonzero entries: :math:`+1` and :math:`-1`
   in the columns corresponding to the vertices composing the edge. The sign is
   determined based on the orientation of the edge.  We require that :math:`G`
   includes all (interior and boundary) edges and vertices.

#. The representations of the constant vector fields :math:`(1,0,0)`,
   :math:`(0,1,0)`, and :math:`(0,0,1)` in the :math:`{\mathbf V}_h` basis, given
   as three vectors: :math:`G_x`, :math:`G_y`, and :math:`G_z`.  Note that since no
   boundary conditions are imposed on :math:`G`, the above vectors can be computed
   as :math:`G_x = G x`, :math:`G_y = G y` and :math:`G_z = G z`, where :math:`x`,
   :math:`y`, and :math:`z` are vectors representing the coordinates of the
   vertices of the mesh.

In addition to the above quantities, AMS can utilize the following (optional)
information:

* The Poisson matrices :math:`A_\alpha` and :math:`A_\beta`, corresponding to
  assembling of the forms :math:`(\alpha\, \nabla u, \nabla v)+(\beta\, u, v)`
  and :math:`(\beta\, \nabla u, \nabla v)` using standard linear finite elements
  on the same mesh.

Internally, AMS proceeds with the construction of the following additional objects:

* :math:`A_G` -- a matrix associated with the mass term which is either
  :math:`G^T {\mathbf A} G` or the Poisson matrix :math:`A_\beta` (if given).

* :math:`{\mathbf \Pi}` -- the matrix representation of the interpolation
  operator from vector linear to edge finite elements.

* :math:`{\mathbf A}_{{\mathbf \Pi}}` -- a matrix associated with the stiffness
  term which is either :math:`{\mathbf \Pi}^{\,T} {\mathbf A} {\mathbf \Pi}` or
  a block-diagonal matrix with diagonal blocks :math:`A_\alpha` (if given).

* :math:`B_G` and :math:`{\mathbf B}_{{\mathbf \Pi}}` -- efficient (AMG) solvers
  for :math:`A_G` and :math:`{\mathbf A}_{{\mathbf \Pi}}`.

The solution procedure then is a three-level method using smoothing in the
original edge space and subspace corrections based on :math:`B_G` and
:math:`{\mathbf B}_{{\mathbf \Pi}}`.  We can employ a number of options here
utilizing various combinations of the smoother and solvers in additive or
multiplicative fashion.  If :math:`\beta` is identically zero one can skip the
subspace correction associated with :math:`G`, in which case the solver is a
two-level method.


Sample Usage
------------------------------------------------------------------------------

AMS can be used either as a solver or as a preconditioner.  Below we list the
sequence of hypre calls needed to create and use it as a solver.  See example
code ``ex15.c`` for a complete implementation.  We start with the allocation of
the ``HYPRE_Solver`` object:

.. code-block:: c

   HYPRE_Solver solver;
   HYPRE_AMSCreate(&solver);

Next, we set a number of solver parameters. Some of them are optional, while
others are necessary in order to perform the solver setup.

AMS offers the option to set the space dimension.  By default we consider the
dimension to be :math:`3`. The only other option is :math:`2`, and it can be set
with the function given below.  We note that a 3D solver will still work for a
2D problem, but it will be slower and will require more memory than necessary.

.. code-block:: c

   HYPRE_AMSSetDimension(solver, dim);

The user is required to provide the discrete gradient matrix :math:`G`.  AMS
expects a matrix defined on the whole mesh with no boundary edges/nodes
excluded. It is essential to **not** impose any boundary conditions on
:math:`G`.  Regardless of which hypre conceptual interface was used to construct
:math:`G`, one can obtain a ParCSR version of it. This is the expected format in
the following function.

.. code-block:: c

   HYPRE_AMSSetDiscreteGradient(solver, G);

In addition to :math:`G`, we need one additional piece of information in order
to construct the solver.  The user has the option to choose either the
coordinates of the vertices in the mesh or the representations of the constant
vector fields in the edge element basis.  In both cases three hypre parallel
vectors should be provided.  For 2D problems, the user can set the third vector
to NULL.  The corresponding function calls read:

.. code-block:: c

   HYPRE_AMSSetCoordinateVectors(solver,x,y,z);

or

.. code-block:: c

   HYPRE_AMSSetEdgeConstantVectors(solver, one_zero_zero, zero_one_zero, zero_zero_one);

The vectors ``one_zero_zero``, ``zero_one_zero`` and ``zero_zero_one`` above
correspond to the constant vector fields :math:`(1,0,0)`, :math:`(0,1,0)` and
:math:`(0,0,1)`.

The remaining solver parameters are optional.  For example, the user can choose
a different cycle type by calling

.. code-block:: c

   HYPRE_AMSSetCycleType(solver, cycle_type); /* default value: 1 */

The available cycle types in AMS are:

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
:math:`0` refers to smoothing, :math:`1` stands for BoomerAMG based on
:math:`B_G`, and :math:`2` refers to a call to BoomerAMG for :math:`{\mathbf
B}_{{\mathbf \Pi}}`.  The values :math:`3`, :math:`4` and :math:`5` refer to the
scalar subspaces corresponding to the :math:`x`, :math:`y` and :math:`z`
components of :math:`\mathbf \Pi`.

The abbreviation :math:`xyyz` for :math:`x,y,z \in \{0,1,2,3,4,5\}` refers to a
multiplicative subspace correction based on solvers :math:`x`, :math:`y`,
:math:`y`, and :math:`z` (in that order).  The abbreviation :math:`x+y+z` stands
for an additive subspace correction method based on :math:`x`, :math:`y` and
:math:`z` solvers.  The additive cycles are meant to be used only when AMS is
called as a preconditioner.  In our experience the choices
``cycle_type=1,5,8,11,13`` often produced fastest solution times, while
``cycle_type=7`` resulted in smallest number of iterations.

Additional solver parameters, such as the maximum number of iterations, the
convergence tolerance and the output level, can be set with

.. code-block:: c

   HYPRE_AMSSetMaxIter(solver, maxit);     /* default value: 20 */
   HYPRE_AMSSetTol(solver, tol);           /* default value: 1e-6 */
   HYPRE_AMSSetPrintLevel(solver, print);  /* default value: 1 */

More advanced parameters, affecting the smoothing and the internal AMG solvers,
can be set with the following three functions:

.. code-block:: c

   HYPRE_AMSSetSmoothingOptions(solver, 2, 1, 1.0, 1.0);
   HYPRE_AMSSetAlphaAMGOptions(solver, 10, 1, 3, 0.25, 0, 0);
   HYPRE_AMSSetBetaAMGOptions(solver, 10, 1, 3, 0.25, 0, 0);

For (singular) problems where :math:`\beta = 0` in the whole domain, different
(in fact simpler) version of the AMS solver is offered.  To allow for this
simplification, use the following hypre call

.. code-block:: c

   HYPRE_AMSSetBetaPoissonMatrix(solver, NULL);

If :math:`\beta` is zero only in parts of the domain, the problem is still
singular, but the AMS solver will try to detect this and construct a
non-singular preconditioner. Though this often works well in practice, AMS also
provides a more robust version for solving such singular problems to very low
convergence tolerances. This version takes advantage of additional information:
the list of nodes which are interior to a zero-conductivity region provided by
the function

.. code-block:: c

   HYPRE_AMSSetInteriorNodes(solver, HYPRE_ParVector interior_nodes);

A node is interior, if its entry in the ``interior_nodes`` array is :math:`1.0`.
Based on this array, a restricted discrete gradient operator :math:`G_0` is
constructed, and AMS is then defined based on the matrix :math:`{\mathbf
A}+\delta G_0^TG_0` which is non-singular, and a small :math:`\delta>0`
perturbation of :math:`{\mathbf A}`. When iterating with this preconditioner, it
is advantageous to project on the compatible subspace :math:`Ker(G_0^T)`. This
can be done periodically, or manually through the functions

.. code-block:: c

   HYPRE_AMSSetProjectionFrequency(solver, int projection_frequency);
   HYPRE_AMSProjectOutGradients(solver, HYPRE_ParVector x);

Two additional matrices are constructed in the setup of the AMS method---one
corresponding to the coefficient :math:`\alpha` and another corresponding to
:math:`\beta`.  This may lead to prohibitively high memory requirements, and the
next two function calls may help to save some memory.  For example, if the
Poisson matrix with coefficient :math:`\beta` (denoted by ``Abeta``) is
available then one can avoid one matrix construction by calling

.. code-block:: c

   HYPRE_AMSSetBetaPoissonMatrix(solver, Abeta);

Similarly, if the Poisson matrix with coefficient :math:`\alpha` is available
(denoted by ``Aalpha``) the second matrix construction can also be avoided by
calling

.. code-block:: c

   HYPRE_AMSSetAlphaPoissonMatrix(solver, Aalpha);

Note the following regarding these functions:

* Both of them change their input. More specifically, the diagonal entries of
  the input matrix corresponding to eliminated degrees of freedom (due to
  essential boundary conditions) are penalized.
* It is assumed that their essential boundary conditions of :math:`{\mathbf A}`,
  ``Abeta`` and ``Aalpha`` are on the same part of the boundary.
* ``HYPRE_AMSSetAlphaPoissonMatrix`` forces the AMS method to use a simpler, but
  weaker (in terms of convergence) method.  With this option, the multiplicative
  AMS cycle is not guaranteed to converge with the default parameters. The
  reason for this is the fact the solver is not variationally obtained from the
  original matrix (it utilizes the auxiliary Poisson--like matrices ``Abeta``
  and ``Aalpha``).  Therefore, it is recommended in this case to use AMS as
  preconditioner only.

After the above calls, the solver is ready to be constructed.  The user has to
provide the stiffness matrix :math:`{\mathbf A}` (in ParCSR format) and the
hypre parallel vectors :math:`{\mathbf b}` and :math:`{\mathbf x}`. (The vectors
are actually not used in the current AMS setup.) The setup call reads,

.. code-block:: c

   HYPRE_AMSSetup(solver, A, b, x);

It is important to note the order of the calling sequence. For example, do
**not** call ``HYPRE_AMSSetup`` before calling ``HYPRE_AMSSetDiscreteGradient``
and one of the functions ``HYPRE_AMSSetCoordinateVectors`` or
``HYPRE_AMSSetEdgeConstantVectors``.

Once the setup has completed, we can solve the linear system by calling

.. code-block:: c

   HYPRE_AMSSolve(solver, A, b, x);

Finally, the solver can be destroyed with

.. code-block:: c

   HYPRE_AMSDestroy(&solver);

More details can be found in the files ``ams.h`` and ``ams.c`` located in the
``parcsr_ls`` directory.


High-order Discretizations
------------------------------------------------------------------------------

In addition to the interface for the lowest-order Nedelec elements described in
the previous subsections, AMS also provides support for (arbitrary) high-order
Nedelec element discretizations. Since the robustness of AMS depends on the
performance of BoomerAMG on the associated (high-order) auxiliary subspace
problems, we note that the convergence may not be optimal for large polynomial
degrees :math:`k \geq 1`.

In the high-order AMS interface, the user does not need to provide the
coordinates of the vertices (or the representations of the constant vector
fields in the edge basis), but instead should construct and pass the Nedelec
interpolation matrix :math:`{\mathbf \Pi}` which maps (high-order) vector nodal
finite elements into the (high-order) Nedelec space. In other words,
:math:`{\mathbf \Pi}` is the (parallel) matrix representation of the
interpolation mapping from :math:`\mathrm{P}_k^3`/:math:`\mathrm{Q}_k^3` into
:math:`\mathrm{ND}_k`, see [HiXu2006]_, [KoVa2009]_.  We require this matrix as
an input, since in the high-order case its entries very much depend on the
particular choice of the basis functions in the edge and nodal spaces, as well
as on the geometry of the mesh elements. The columns of :math:`{\mathbf \Pi}`
should use a node-based numbering, where the :math:`x`/:math:`y`/:math:`z`
components of the first node (vertex or high-order degree of freedom) should be
listed first, followed by the :math:`x`/:math:`y`/:math:`z` components of the
second node and so on (see the documentation of ``HYPRE_BoomerAMGSetDofFunc``).

Similarly to the Nedelec interpolation, the discrete gradient matrix :math:`G`
should correspond to the mapping :math:`\varphi \in \mathrm{P}_k^3 /
\mathrm{Q}_k^3 \mapsto \nabla \varphi \in \mathrm{ND}_k`, so even though its
values are still independent of the mesh coordinates, they will not be
:math:`\pm 1`, but will be determined by the particular form of the high-order
basis functions and degrees of freedom.

With these matrices, the high-order setup procedure is simply

.. code-block:: c

   HYPRE_AMSSetDimension(solver, dim);
   HYPRE_AMSSetDiscreteGradient(solver, G);
   HYPRE_AMSSetInterpolations(solver, Pi, NULL, NULL, NULL);

We remark that the above interface calls can also be used in the lowest-order
case (or even other types of discretizations such as those based on the second
family of Nedelec elements), but we recommend calling the previously described
``HYPRE_AMSSetCoordinateVectors`` instead, since this allows AMS to handle the
construction and use of :math:`{\mathbf \Pi}` internally.

Specifying the monolithic :math:`{\mathbf \Pi}` limits the AMS cycle type
options to those less than 10. Alternatively one can separately specify the
:math:`x`, :math:`y` and :math:`z` components of :math:`\mathbf \Pi`:

.. code-block:: c

   HYPRE_AMSSetInterpolations(solver, NULL, Pix, Piy, Piz);

which enables the use of AMS cycle types with index greater than 10. By
definition, :math:`{\mathbf \Pi}^x \varphi = {\mathbf \Pi} (\varphi,0,0)`, and
similarly for :math:`{\mathbf \Pi}^y` and :math:`{\mathbf \Pi}^z`. Each of these
matrices has the same sparsity pattern as :math:`G`, but their entries depend on
the coordinates of the mesh vertices.

Finally, both :math:`{\mathbf \Pi}` and its components can be passed to the solver:

.. code-block:: c

   HYPRE_AMSSetInterpolations(solver, Pi, Pix, Piy, Piz);

which will duplicate some memory, but allows for experimentation with all
available AMS cycle types.


Non-conforming AMR Grids
------------------------------------------------------------------------------

AMS could also be applied to problems with adaptive mesh refinement (AMR) posed
on non-conforming quadrilateral/hexahedral meshes, see [GrKo2015]_ for more
details.

On non-conforming grids (assuming also arbitrarily high-order elements), each
finite element space has two versions: a conforming one,
e.g. :math:`\mathrm{Q}_k^{c} / \mathrm{ND}_k^c`, where the *hanging* degrees of
freedom are constrained by the conforming (*real*) degrees of freedom, and a
non-conforming one, e.g. :math:`\mathrm{Q}_k^{nc} / \mathrm{ND}_k^{nc}` where
the non-conforming degrees of freedom (hanging and real) are unconstrained.
These spaces are related with the conforming prolongation and the pure
restriction operators :math:`P` and :math:`R`, as well as the conforming and
non-conforming version of the discrete gradient operator as follows:

.. math::

   \begin{array}{ccc}
   \mathrm{Q}_k^{c}   &  \xrightarrow{G_{c}}   &  \mathrm{ND}_k^{c}  \\
   {\scriptstyle P_{\mathrm{Q}}}  \Bigg\downarrow \Bigg\uparrow {\scriptstyle R_{\mathrm{Q}}}  &&
   {\scriptstyle P_{\mathrm{ND}}} \Bigg\downarrow \Bigg\uparrow {\scriptstyle R_{\mathrm{ND}}} \\
   \mathrm{Q}_k^{nc}  &  \xrightarrow{G_{nc}}  &  \mathrm{ND}_k^{nc} \\
   \end{array}

..
   \xymatrix{
   \mathrm{Q}_k^{c} \ar[r]^{G_{c}} \ar@<-2pt>[d]_{P_{\mathrm{Q}}} & \mathrm{ND}_k^c \ar@<-2pt>[d]_{P_{\mathrm{ND}}} \\
   \mathrm{Q}_k^{nc} \ar[r]^{G_{nc}} \ar@<-2pt>[u]_{R_{\mathrm{Q}}} & \mathrm{ND}_k^{nc} \ar@<-2pt>[u]_{R_{\mathrm{ND}}}
   }

Since the linear system is posed on :math:`\mathrm{ND}_k^c`, the user needs to
provide the conforming discrete gradient matrix :math:`G_c` to AMS, using
``HYPRE_AMSSetDiscreteGradient``.  This matrix is defined by the requirement
that the above diagram commutes from :math:`\mathrm{Q}_k^{c}` to
:math:`\mathrm{ND}_k^{nc}`, corresponding to the definition

.. math::

   G_{c} = R_{\mathrm{ND}}\, G_{nc}\, P_{\mathrm{Q}} \,,

i.e. the conforming gradient is computed by starting with a conforming nodal
:math:`\mathrm{Q}_k` function, interpolating it in the hanging nodes, computing
the gradient locally and representing it in the Nedelec space on each element
(the non-conforming discrete gradient :math:`G_{nc}` in the above formula), and
disregarding the values in the hanging :math:`\mathrm{ND}_k` degrees of freedom.

Similar considerations imply that the conforming Nedelec interpolation matrix
:math:`{\mathbf \Pi}_{c}` should be defined as

.. math::

   {\mathbf \Pi}_{c} = R_{\mathrm{ND}}\, {\mathbf \Pi}_{nc}\, P_{\mathrm{Q}^3} \,,

with :math:`{\mathbf \Pi}_{nc}` computed element-wise as in the previous
subsection. Note that in the low-order case, :math:`{\mathbf \Pi}_{c}` can be
computed internally in AMS based only :math:`G_c` and the conforming coordinates
of the vertices :math:`x_c`/:math:`y_c`/:math:`z_c`, see [GrKo2015]_.

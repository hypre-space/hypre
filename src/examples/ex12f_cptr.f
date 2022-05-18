!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!
!     Example 12
!
!     Interface:    Semi-Structured interface (SStruct)
!
!     Compile with: make ex12f (may need to edit HYPRE_DIR in Makefile)
!
!     Sample runs:  mpirun -np 2 ex12f
!
!     Description: The grid layout is the same as ex1, but with nodal
!     unknowns. The solver is PCG preconditioned with either PFMG or
!     BoomerAMG, set with 'precond_id' below.
!
!     We recommend viewing the Struct examples before viewing this and
!     the other SStruct examples.  This is one of the simplest SStruct
!     examples, used primarily to demonstrate how to set up
!     non-cell-centered problems, and to demonstrate how easy it is to
!     switch between structured solvers (PFMG) and solvers designed for
!     more general settings (AMG).
!

      program ex12f

      use, intrinsic :: iso_c_binding
      use, intrinsic :: iso_fortran_env, only: int64
      use cudaf

      implicit none

      include 'mpif.h'
      include 'HYPREf.h'

      integer    ierr
      integer    i, j, myid, num_procs

      integer*8  grid
      integer*8  graph
      integer*8  stencil
      integer*8  A
      integer*8  b
      integer*8  x

      integer    nparts
      integer    nvars
      integer    part
      integer    var

      integer    precond_id, object_type

      integer    ilower(2), iupper(2)
      integer    vartypes(1)
      integer    offsets(2,5)
      integer    ent
      integer    nentries, nvalues, stencil_indices(5)

      double precision tol

      double precision, pointer :: values(:)
      type(c_ptr) :: p_values

      integer :: stat

!     This comes from 'sstruct_mv/HYPRE_sstruct_mv.h'
      integer    HYPRE_SSTRUCT_VARIABLE_NODE
      parameter( HYPRE_SSTRUCT_VARIABLE_NODE = 1 )

      integer*8  sA
      integer*8  sb
      integer*8  sx
      integer*8  parA
      integer*8  parb
      integer*8  parx
      integer*8  solver
      integer*8  precond

      character*32  matfile

      stat = device_malloc_managed(int(100 * 8, int64), p_values)

      call c_f_pointer(p_values, values, [100])

!     We only have one part and one variable
      nparts = 1
      nvars  = 1
      part   = 0
      var    = 0

!     Initialize MPI
      call MPI_Init(ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD, myid, ierr)
      call MPI_Comm_size(MPI_COMM_WORLD, num_procs, ierr)

      call HYPRE_Init(ierr)

      if (num_procs .ne. 2) then
         if (myid .eq. 0) then
            print *, "Must run with 2 processors!"
            stop
         endif
      endif

!     Set preconditioner id (PFMG = 1, BoomerAMG = 2)
      precond_id = 1

      if (precond_id .eq. 1) then
         object_type = HYPRE_STRUCT
      else if (precond_id .eq. 2) then
         object_type = HYPRE_PARCSR
      else
         if (myid .eq. 0) then
            print *, "Invalid solver!"
            stop
         endif
      endif

!-----------------------------------------------------------------------
!     1. Set up the grid.  Here we use only one part.  Each processor
!     describes the piece of the grid that it owns.
!-----------------------------------------------------------------------

!     Create an empty 2D grid object
      call HYPRE_SStructGridCreate(MPI_COMM_WORLD, 2, nparts, grid,
     +     ierr)

!     Add boxes to the grid
      if (myid .eq. 0) then
         ilower(1) = -3
         ilower(2) =  1
         iupper(1) = -1
         iupper(2) =  2
         call HYPRE_SStructGridSetExtents(grid, part, ilower, iupper,
     +        ierr)
      else if (myid .eq. 1) then
         ilower(1) =  0
         ilower(2) =  1
         iupper(1) =  2
         iupper(2) =  4
         call HYPRE_SStructGridSetExtents(grid, part, ilower, iupper,
     +        ierr)
      endif

!     Set the variable type and number of variables on each part
      vartypes(1) = HYPRE_SSTRUCT_VARIABLE_NODE
      call HYPRE_SStructGridSetVariables(grid, part, nvars, vartypes,
     +     ierr)

!     This is a collective call finalizing the grid assembly
      call HYPRE_SStructGridAssemble(grid, ierr)

!-----------------------------------------------------------------------
!     2. Define the discretization stencil
!-----------------------------------------------------------------------

!     Create an empty 2D, 5-pt stencil object
      call HYPRE_SStructStencilCreate(2, 5, stencil, ierr)

!     Define the geometry of the stencil.  Each represents a relative
!     offset (in the index space).
      offsets(1,1) =  0
      offsets(2,1) =  0
      offsets(1,2) = -1
      offsets(2,2) =  0
      offsets(1,3) =  1
      offsets(2,3) =  0
      offsets(1,4) =  0
      offsets(2,4) = -1
      offsets(1,5) =  0
      offsets(2,5) =  1

!     Assign numerical values to the offsets so that we can easily refer
!     to them - the last argument indicates the variable for which we
!     are assigning this stencil
      do ent = 1, 5
         call HYPRE_SStructStencilSetEntry(stencil,
     +        ent-1, offsets(1,ent), var, ierr)
      enddo

!-----------------------------------------------------------------------
!     3. Set up the Graph - this determines the non-zero structure of
!     the matrix and allows non-stencil relationships between the parts
!-----------------------------------------------------------------------

!     Create the graph object
      call HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, graph, ierr)

!     See MatrixSetObjectType below
      call HYPRE_SStructGraphSetObjectType(graph, object_type, ierr)

!     Now we need to tell the graph which stencil to use for each
!     variable on each part (we only have one variable and one part)
      call HYPRE_SStructGraphSetStencil(graph, part, var, stencil, ierr)

!     Here we could establish connections between parts if we had more
!     than one part using the graph. For example, we could use
!     HYPRE_GraphAddEntries() routine or HYPRE_GridSetNeighborPart()

!     Assemble the graph
      call HYPRE_SStructGraphAssemble(graph, ierr)

!-----------------------------------------------------------------------
!     4. Set up a SStruct Matrix
!-----------------------------------------------------------------------

!     Create an empty matrix object
      call HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, A, ierr)

!     Set the object type (by default HYPRE_SSTRUCT). This determines
!     the data structure used to store the matrix.  For PFMG we use
!     HYPRE_STRUCT, and for BoomerAMG we use HYPRE_PARCSR (set above).
      call HYPRE_SStructMatrixSetObjectTyp(A, object_type, ierr)

!     Get ready to set values
      call HYPRE_SStructMatrixInitialize(A, ierr)

!     Set the matrix coefficients.  Each processor assigns coefficients
!     for the boxes in the grid that it owns.  Note that the
!     coefficients associated with each stencil entry may vary from grid
!     point to grid point if desired.  Here, we first set the same
!     stencil entries for each grid point.  Then we make modifications
!     to grid points near the boundary.  Note that the ilower values are
!     different from those used in ex1 because of the way nodal
!     variables are referenced.  Also note that some of the stencil
!     values are set on both processor 0 and processor 1.  See the User
!     and Reference manuals for more details.

!     Stencil entry labels correspond to the offsets defined above
      do i = 1, 5
         stencil_indices(i) = i-1
      enddo
      nentries = 5

      if (myid .eq. 0) then
         ilower(1) = -4
         ilower(2) =  0
         iupper(1) = -1
         iupper(2) =  2
!        12 grid points, each with 5 stencil entries
         nvalues = 60
      else if (myid .eq. 1) then
         ilower(1) = -1
         ilower(2) =  0
         iupper(1) =  2
         iupper(2) =  4
!        12 grid points, each with 5 stencil entries
         nvalues = 100
      endif

      do i = 1, nvalues, nentries
         values(i) = 4.0
         do j = 1, nentries-1
            values(i+j) = -1.0
         enddo
      enddo

      call HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
     +     var, nentries, stencil_indices, values, ierr)

!     Set the coefficients reaching outside of the boundary to 0.  Note
!     that both ilower *and* iupper may be different from those in ex1.

      do i = 1, 5
         values(i) = 0.0
      enddo

      if (myid .eq. 0) then

!        values below our box
         ilower(1) = -4
         ilower(2) =  0
         iupper(1) = -1
         iupper(2) =  0
         stencil_indices(1) = 3
         call HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
     +        var, 1, stencil_indices, values, ierr)
!        values to the left of our box
         ilower(1) = -4
         ilower(2) =  0
         iupper(1) = -4
         iupper(2) =  2
         stencil_indices(1) = 1
         call HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
     +        var, 1, stencil_indices, values, ierr)
!        values above our box
         ilower(1) = -4
         ilower(2) =  2
         iupper(1) = -2
         iupper(2) =  2
         stencil_indices(1) = 4
         call HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
     +        var, 1, stencil_indices, values, ierr)

      else if (myid .eq. 1) then

!        values below our box
         ilower(1) = -1
         ilower(2) =  0
         iupper(1) =  2
         iupper(2) =  0
         stencil_indices(1) = 3
         call HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
     +        var, 1, stencil_indices, values, ierr)
!        values to the right of our box
         ilower(1) =  2
         ilower(2) =  0
         iupper(1) =  2
         iupper(2) =  4
         stencil_indices(1) = 2
         call HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
     +        var, 1, stencil_indices, values, ierr)
!        values above our box
         ilower(1) = -1
         ilower(2) =  4
         iupper(1) =  2
         iupper(2) =  4
         stencil_indices(1) = 4
         call HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
     +        var, 1, stencil_indices, values, ierr)
!        values to the left of our box
!        (that do not border the other box on proc. 0)
         ilower(1) = -1
         ilower(2) =  3
         iupper(1) = -1
         iupper(2) =  4
         stencil_indices(1) = 1
         call HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
     +        var, 1, stencil_indices, values, ierr)

      endif

!     This is a collective call finalizing the matrix assembly
      call HYPRE_SStructMatrixAssemble(A, ierr)

!      matfile = 'ex12f.out'
!      matfile(10:10) = char(0)
!      call HYPRE_SStructMatrixPrint(matfile, A, 0, ierr)

!     Create an empty vector object
      call HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, b, ierr)
      call HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, x, ierr)

!     As with the matrix, set the appropriate object type for the vectors
      call HYPRE_SStructVectorSetObjectTyp(b, object_type, ierr)
      call HYPRE_SStructVectorSetObjectTyp(x, object_type, ierr)

!     Indicate that the vector coefficients are ready to be set
      call HYPRE_SStructVectorInitialize(b, ierr)
      call HYPRE_SStructVectorInitialize(x, ierr)

!     Set the vector coefficients.  Again, note that the ilower values
!     are different from those used in ex1, and some of the values are
!     set on both processors.

      if (myid .eq. 0) then

         ilower(1) = -4
         ilower(2) =  0
         iupper(1) = -1
         iupper(2) =  2

         do i = 1, 12
            values(i) = 1.0
         enddo
         call HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper,
     +        var, values, ierr)
         do i = 1, 12
            values(i) = 0.0
         enddo
         call HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper,
     +        var, values, ierr)

      else if (myid .eq. 1) then

         ilower(1) =  0
         ilower(2) =  1
         iupper(1) =  2
         iupper(2) =  4

         do i = 1, 20
            values(i) = 1.0
         enddo
         call HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper,
     +        var, values, ierr)
         do i = 1, 20
            values(i) = 0.0
         enddo
         call HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper,
     +        var, values, ierr)

      endif

!     This is a collective call finalizing the vector assembly
      call HYPRE_SStructVectorAssemble(b, ierr)
      call HYPRE_SStructVectorAssemble(x, ierr)

!-----------------------------------------------------------------------
!     6. Set up and use a solver (See the Reference Manual for
!     descriptions of all of the options.)
!-----------------------------------------------------------------------

      tol = 1.0E-6

      if (precond_id .eq. 1) then

!        PFMG

!        Because we are using a struct solver, we need to get the object
!        of the matrix and vectors to pass in to the struct solvers
         call HYPRE_SStructMatrixGetObject(A, sA, ierr)
         call HYPRE_SStructVectorGetObject(b, sb, ierr)
         call HYPRE_SStructVectorGetObject(x, sx, ierr)

!        Create an empty PCG Struct solver
         call HYPRE_StructPCGCreate(MPI_COMM_WORLD, solver, ierr)
!        Set PCG parameters
         call HYPRE_StructPCGSetTol(solver, tol, ierr)
         call HYPRE_StructPCGSetPrintLevel(solver, 2, ierr)
         call HYPRE_StructPCGSetMaxIter(solver, 50, ierr)

!        Create the Struct PFMG solver for use as a preconditioner
         call HYPRE_StructPFMGCreate(MPI_COMM_WORLD, precond, ierr)
!        Set PFMG parameters
         call HYPRE_StructPFMGSetMaxIter(precond, 1, ierr)
         call HYPRE_StructPFMGSetTol(precond, 0.0d0, ierr)
         call HYPRE_StructPFMGSetZeroGuess(precond, ierr)
         call HYPRE_StructPFMGSetNumPreRelax(precond, 2, ierr)
         call HYPRE_StructPFMGSetNumPostRelax(precond, 2, ierr)
!        Non-Galerkin coarse grid (more efficient for this problem)
         call HYPRE_StructPFMGSetRAPType(precond, 1, ierr)
!        R/B Gauss-Seidel
         call HYPRE_StructPFMGSetRelaxType(precond, 2, ierr)
!        Skip relaxation on some levels (more efficient for this problem)
         call HYPRE_StructPFMGSetSkipRelax(precond, 1, ierr)
!        Set preconditioner (PFMG = 1) and solve
         call HYPRE_StructPCGSetPrecond(solver, 1, precond, ierr)
         call HYPRE_StructPCGSetup(solver, sA, sb, sx, ierr)
         call HYPRE_StructPCGSolve(solver, sA, sb, sx, ierr)

!        Free memory
         call HYPRE_StructPCGDestroy(solver, ierr)
         call HYPRE_StructPFMGDestroy(precond, ierr)

      else if (precond_id .eq. 2) then

!        BoomerAMG

!        Because we are using a struct solver, we need to get the object
!        of the matrix and vectors to pass in to the struct solvers
         call HYPRE_SStructMatrixGetObject(A, parA, ierr)
         call HYPRE_SStructVectorGetObject(b, parb, ierr)
         call HYPRE_SStructVectorGetObject(x, parx, ierr)

!        Create an empty PCG Struct solver
         call HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, solver, ierr)
!        Set PCG parameters
         call HYPRE_ParCSRPCGSetTol(solver, tol, ierr)
         call HYPRE_ParCSRPCGSetPrintLevel(solver, 2, ierr)
         call HYPRE_ParCSRPCGSetMaxIter(solver, 50, ierr)

!        Create the BoomerAMG solver for use as a preconditioner
         call HYPRE_BoomerAMGCreate(precond, ierr)
!        Set BoomerAMG parameters
         call HYPRE_BoomerAMGSetMaxIter(precond, 1, ierr)
         call HYPRE_BoomerAMGSetTol(precond, 0.0, ierr)
!        Print amg solution info
         call HYPRE_BoomerAMGSetPrintLevel(precond, 1, ierr)
         call HYPRE_BoomerAMGSetCoarsenType(precond, 6, ierr)
         call HYPRE_BoomerAMGSetOldDefault(precond, ierr)
!        Sym G.S./Jacobi hybrid
         call HYPRE_BoomerAMGSetRelaxType(precond, 6, ierr)
         call HYPRE_BoomerAMGSetNumSweeps(precond, 1, ierr)
!        Set preconditioner (BoomerAMG = 2) and solve
         call HYPRE_ParCSRPCGSetPrecond(solver, 2, precond, ierr)
         call HYPRE_ParCSRPCGSetup(solver, parA, parb, parx, ierr)
         call HYPRE_ParCSRPCGSolve(solver, parA, parb, parx, ierr)

!        Free memory
         call HYPRE_ParCSRPCGDestroy(solver, ierr)
         call HYPRE_BoomerAMGDestroy(precond, ierr)

      endif

!     Free memory
      call HYPRE_SStructGridDestroy(grid, ierr)
      call HYPRE_SStructStencilDestroy(stencil, ierr)
      call HYPRE_SStructGraphDestroy(graph, ierr)
      call HYPRE_SStructMatrixDestroy(A, ierr)
      call HYPRE_SStructVectorDestroy(b, ierr)
      call HYPRE_SStructVectorDestroy(x, ierr)

      call HYPRE_Finalize(ierr)

!     Finalize MPI
      call MPI_Finalize(ierr)

      stat = device_free(p_values)

      stop
      end


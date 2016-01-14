c
c   Example 6
c
c   Interface:    Semi-Structured interface (SStruct).  Fortran - Babel version.
c
c   Compile with: make ex6
c
c   Sample run:   mpirun -np 2 ex6
c
c   Description:  This is a two processor example and is the same problem
c                 as is solved with the structured interface in Example 2.
c                 (The grid boxes are exactly those in the example
c                 diagram in the struct interface chapter of the User's Manual.
c                 Processor 0 owns two boxes and processor 1 owns one box.)
c
c                 This is the simplest sstruct example, and it demonstrates how
c                 the semi-structured interface can be used for structured problems.
c                 There is one part and one variable.  The solver is PCG with SMG
c                 preconditioner. We use a structured solver for this example.


      program ex6b77

      implicit none

      include 'mpif.h'

      include "HYPREf.h"
c     ...If HYPREf.h doesn't exist in your version of HYPRE, just use
c     these two lines from it:
c      integer HYPRE_STRUCT
c      parameter( HYPRE_STRUCT =  1111 )

      include 'bHYPRE_SStructVariable.inc'
c     ... This file is generated into babel/bHYPREClient-F.  A future version
c     of the makefiles will copy it to hypre/include.

      integer    MAX_LOCAL_SIZE
      parameter  (MAX_LOCAL_SIZE=123000)
      integer    MAXBLKS
      parameter  (MAXBLKS=32)
      integer    MAX_STENCIL_SIZE
      parameter  (MAX_STENCIL_SIZE=27)
      integer    MAXDIM
      parameter  (MAXDIM=3)

      integer myid, num_procs

      integer*8 grid
      integer*8 graph
      integer*8 stencil
      integer*8 A
      integer*8 b
      integer*8 x
      integer*8  bHYPRE_mpicomm
      integer*8 except
      integer*8 mpi_comm
      integer*8 sA
      integer*8 vb
      integer*8 vx
      integer*8 dummy
      integer*8 opA


c     We are using struct solvers for this example
      integer*8 PCGsolver
      integer*8 SMGprecond
      integer*8 precond

      integer object_type, ndim, ierr, ierrtmp
      integer ilower(MAXDIM)
      integer iupper(MAXDIM)
      integer i, j, nparts, part, nvars, var
      integer vartypes(1)
      integer entry, nentries, nvalues, maxnvalues
      double precision tol
      double precision values(MAX_STENCIL_SIZE*MAX_LOCAL_SIZE)
      integer stencil_indices(MAX_STENCIL_SIZE)
      integer offsets(2,5)

      ndim = 2
      ierr = 0
      ierrtmp = 0

c     Initialize MPI
      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)
      mpi_comm = MPI_COMM_WORLD
      call bHYPRE_MPICommunicator_CreateF_f( mpi_comm, bHYPRE_mpicomm,
     1      except )

      if ( num_procs .ne. 2 ) then
         print *, 'Must run with 2 processors!'
         call MPI_Finalize(ierrtmp)
         stop
      endif



c    1. Set up the 2D grid.  This gives the index space in each part.
c       Here we only use one part and one variable. (So the part id is 0
c       and the variable id is 0)

      nparts = 1
      part = 0

c     Create an empty 2D grid object
      call bHYPRE_SStructGrid_Create_f( bHYPRE_mpicomm, ndim, nparts,
     1     grid, except );

c     Set the extents of the grid - each processor sets its grid
c       boxes.  Each part has its own relative index space numbering,
c       but in this example all boxes belong to the same part.

c     Processor 0 owns two boxes in the grid.
      if ( myid .eq. 0 ) then
c        Add a new box to the grid
         ilower(1) = -3
         ilower(2) = 1
         iupper(1) = -1
         iupper(2) = 2
         call bHYPRE_SStructGrid_SetExtents_f( grid, part,
     1        ilower, iupper, ndim, ierr, except )

c        Add a new box to the grid */
         ilower(1) = 0
         ilower(2) = 1
         iupper(1) = 2
         iupper(2) = 4
         call bHYPRE_SStructGrid_SetExtents_f( grid, part,
     1        ilower, iupper, ndim, ierr, except )

c     Processor 1 owns one box in the grid.
      elseif ( myid .eq. 1 ) then
c        Add a new box to the grid */
         ilower(1) = 3
         ilower(2) = 1
         iupper(1) = 6
         iupper(2) = 4
         call bHYPRE_SStructGrid_SetExtents_f( grid, part,
     1        ilower, iupper, ndim, ierr, except )
      endif

c     Set the variable type and number of variables on each part.
      nvars = 1
      var = 0
      vartypes(1) = CELL

      do i = 0, nparts-1
         call bHYPRE_SStructGrid_SetVariable_f( grid, i, var, nvars,
     1        vartypes(1), ierr, except )
      enddo
c     ... if nvars>1 we would need a number of AddVariable calls

c     Now the grid is ready to use
      call bHYPRE_SStructGrid_Assemble_f( grid, ierr, except )

c   2. Define the discretization stencil(s)
c      Create an empty 2D, 5-pt stencil object */
      call bHYPRE_SStructStencil_Create_f( 2, 5, stencil, except )

c     Define the geometry of the stencil. Each represents a
c     relative offset (in the index space).
      offsets(1,1) = 0
      offsets(2,1) = 0
      offsets(1,2) = -1
      offsets(2,2) = 0
      offsets(1,3) = 1
      offsets(2,3) = 0
      offsets(1,4) = 0
      offsets(2,4) = -1
      offsets(1,5) = 0
      offsets(2,5) = 1
      var = 0

c     Assign numerical values to the offsets so that we can
c     easily refer to them  - the last argument indicates the
c     variable for which we are assigning this stencil - we are
c     just using one variable in this example so it is the first one (0)
      do entry = 1, 5
         call bHYPRE_SStructStencil_SetEntry_f( stencil, entry-1,
     1        offsets(1,entry), ndim, var, ierr, except )
      enddo

c     3. Set up the Graph  - this determines the non-zero structure
c     of the matrix and allows non-stencil relationships between the parts
      var = 0
      part = 0

c     Create the graph object
      call bHYPRE_SStructGraph_Create_f( bHYPRE_mpicomm, grid, graph,
     1     except )

c     Now we need to tell the graph which stencil to use for each
c     variable on each part (we only have one variable and one part)
      call bHYPRE_SStructGraph_SetStencil_f( graph, part, var, stencil,
     1     ierr, except )

c     Here we could establish connections between parts if we
c     had more than one part using the graph. For example, we could
c     use HYPRE_GraphAddEntries() routine or HYPRE_GridSetNeighborBox()

c     Assemble the graph
      call bHYPRE_SStructGraph_Assemble_f( graph, ierr, except )

c     4. Set up a SStruct Matrix
      part = 0
      var = 0

c     Create the empty matrix object
      call bHYPRE_SStructMatrix_Create_f( bHYPRE_mpicomm, graph, A,
     1     except )

c     Set the object type (by default HYPRE_SSTRUCT). This determines the
c     data structure used to store the matrix.  If you want to use unstructured
c     solvers, e.g. BoomerAMG, the object type should be HYPRE_PARCSR.
c     If the problem is purely structured (with one part), you may want to use
c     HYPRE_STRUCT to access the structured solvers. Here we have a purely
c     structured example.
      object_type = HYPRE_STRUCT
      call bHYPRE_SStructMatrix_SetObjectType_f( A, object_type,
     1     ierr, except )

c     Get ready to set values
      call bHYPRE_SStructMatrix_Initialize_f( A, ierr, except )

c     Each processor must set the stencil values for their boxes on each part.
c     In this example, we only set stencil entries and therefore use
c     HYPRE_SStructMatrixSetBoxValues.  If we need to set non-stencil entries,
c     we have to use HYPRE_SStructMatrixSetValues (shown in a later example).

      if ( myid .eq. 0 ) then
c     Set the matrix coefficients for some set of stencil entries
c     over all the gridpoints in my first box (account for boundary
c     grid points later)
         ilower(1) = -3
         ilower(2) = 1
         iupper(1) = -1
         iupper(2) = 2
         nentries = 5
         nvalues  = 30
c        ... nvalues=30 from 6 grid points, each with 5 stencil entries

         do j = 1, nentries
c           label the stencil indices - these correspond to the offsets
c           defined above ...
            stencil_indices(j) = j-1
         enddo

         do i = 1, nvalues, nentries
            values(i) = 4.0
            do j = 1, nentries-1
               values(i+j) = -1.0
               enddo
            enddo

            call bHYPRE_SStructMatrix_SetBoxValues_f( A, part,
     1           ilower, iupper, ndim, var, nentries, stencil_indices,
     2           values, nvalues, ierr, except )

c     Set the matrix coefficients for some set of stencil entries
c     over the gridpoints in my second box
            ilower(1) = 0
            ilower(2) = 1
            iupper(1) = 2
            iupper(2) = 4
            nentries = 5
            nvalues  = 60
c           ... nvalues=60 from 12 grid points, each with 5 stencil entries

            do j = 1, nentries
               stencil_indices(j) = j-1
            enddo

            do i = 1, nvalues, nentries
               values(i) = 4.0
               do j = 1, nentries-1
                  values(i+j) = -1.0;
               enddo
            enddo

            call bHYPRE_SStructMatrix_SetBoxValues_f( A, part,
     1           ilower, iupper, ndim, var, nentries, stencil_indices,
     2           values, nvalues, ierr, except )

      elseif ( myid .eq. 1 ) then
c     Set the matrix coefficients for some set of stencil entries
c     over the gridpoints in my box
         ilower(1) = 3
         ilower(2) = 1
         iupper(1) = 6
         iupper(2) = 4
         nentries = 5
         nvalues  = 80
c        ... nentries=80 from 16 grid points, each with 5 stencil entries

         do j = 1, nentries
            stencil_indices(j) = j-1
         enddo

         do i = 1, nvalues, nentries
            values(i) = 4.0
            do j = 1, nentries-1
               values(i+j) = -1.0
            enddo
         enddo

         call bHYPRE_SStructMatrix_SetBoxValues_f( A, part,
     1        ilower, iupper, ndim, var, nentries, stencil_indices,
     2        values, nvalues, ierr, except )

      endif

c     For each box, set any coefficients that reach ouside of the
c     boundary to 0
      if ( myid .eq. 0 )then
         maxnvalues = 6;

         do i = 1, maxnvalues
            values(i) = 0.0
         enddo

c     Values below our first AND second box
         ilower(1) = -3
         ilower(2) = 1
         iupper(1) = 2
         iupper(2) = 1
         nvalues = 6
         stencil_indices(1) = 3

         call bHYPRE_SStructMatrix_SetBoxValues_f( A, part,
     1        ilower, iupper, ndim, var, 1, stencil_indices,
     2        values, nvalues, ierr, except )

c     Values to the left of our first box
         ilower(1) = -3
         ilower(2) = 1
         iupper(1) = -3
         iupper(2) = 2
         nvalues = 2
         stencil_indices(1) = 1

         call bHYPRE_SStructMatrix_SetBoxValues_f( A, part,
     1        ilower, iupper, ndim, var, 1, stencil_indices,
     2        values, nvalues, ierr, except )

c     Values above our first box
         ilower(1) = -3
         ilower(2) = 2
         iupper(1) = -1
         iupper(2) = 2
         nvalues = 3
         stencil_indices(1) = 4

         call bHYPRE_SStructMatrix_SetBoxValues_f( A, part,
     1        ilower, iupper, ndim, var, 1, stencil_indices,
     2        values, nvalues, ierr, except )

c     Values to the left of our second box (that do not border the
c     first box).
         ilower(1) = 0
         ilower(2) = 3
         iupper(1) = 0
         iupper(2) = 4
         nvalues = 2
         stencil_indices(1) = 1

         call bHYPRE_SStructMatrix_SetBoxValues_f( A, part,
     1        ilower, iupper, ndim, var, 1, stencil_indices,
     2        values, nvalues, ierr, except )

c     Values above our second box
         ilower(1) = 0
         ilower(2) = 4
         iupper(1) = 2
         iupper(2) = 4
         nvalues = 3
         stencil_indices(1) = 4

         call bHYPRE_SStructMatrix_SetBoxValues_f( A, part,
     1        ilower, iupper, ndim, var, 1, stencil_indices,
     2        values, nvalues, ierr, except )

      elseif ( myid .eq. 1 ) then
         maxnvalues = 4;

         do i = 1, maxnvalues
            values(i) = 0.0
         enddo

c     Values below our box
         ilower(1) = 3
         ilower(2) = 1
         iupper(1) = 6
         iupper(2) = 1
         nvalues = 4
         stencil_indices(1) = 3

         call bHYPRE_SStructMatrix_SetBoxValues_f( A, part,
     1        ilower, iupper, ndim, var, 1, stencil_indices,
     2        values, nvalues, ierr, except )

c     Values to the right of our box 
         ilower(1) = 6
         ilower(2) = 1
         iupper(1) = 6
         iupper(2) = 4
         nvalues = 4
         stencil_indices(1) = 2

         call bHYPRE_SStructMatrix_SetBoxValues_f( A, part,
     1        ilower, iupper, ndim, var, 1, stencil_indices,
     2        values, nvalues, ierr, except )

c     Values above our box
         ilower(1) = 3
         ilower(2) = 4
         iupper(1) = 6
         iupper(2) = 4
         nvalues = 4
         stencil_indices(1) = 4

         call bHYPRE_SStructMatrix_SetBoxValues_f( A, part,
     1        ilower, iupper, ndim, var, 1, stencil_indices,
     2        values, nvalues, ierr, except )

         endif

c     This is a collective call finalizing the matrix assembly.
c     The matrix is now ``ready to be used''
         call bHYPRE_SStructMatrix_Assemble_f( A, ierr, except )



c     5. Set up SStruct Vectors for b and x

c     We have one part and one variable.
         part = 0
         var = 0

c     Create an empty vector object
         call bHYPRE_SStructVector_Create_f( bHYPRE_mpicomm, grid, b,
     1        except )
         call bHYPRE_SStructVector_Create_f( bHYPRE_mpicomm, grid, x,
     1        except )

c     As with the matrix,  set the object type for the vectors
c     to be the struct type
      object_type = HYPRE_STRUCT;
      call bHYPRE_SStructVector_SetObjectType_f( b, object_type,
     1     ierr, except)
      call bHYPRE_SStructVector_SetObjectType_f( x, object_type,
     1     ierr, except)

c     Indicate that the vector coefficients are ready to be set
      call bHYPRE_SStructVector_Initialize_f( b, ierr, except )
      call bHYPRE_SStructVector_Initialize_f( x, ierr, except )

      if ( myid .eq. 0 ) then
c           Set the vector coefficients over the gridpoints in my first box
         ilower(1) = -3
         ilower(2) = 1
         iupper(1) = -1
         iupper(2) = 2
         nvalues = 6
c        ...  6 grid points

         do i = 1, nvalues
            values(i) = 1.0
         enddo
         call bHYPRE_SStructVector_SetBoxValues_f( b, part,
     1        ilower, iupper, ndim, var, values, nvalues, ierr, except )

         do i = 1, nvalues
            values(i) = 0.0
         enddo
         call bHYPRE_SStructVector_SetBoxValues_f( x, part,
     1        ilower, iupper, ndim, var, values, nvalues, ierr, except )

c     Set the vector coefficients over the gridpoints in my second box
         ilower(1) = 0
         ilower(2) = 1
         iupper(1) = 2
         iupper(2) = 4
         nvalues = 12
c        ... 12 grid points

         do i = 1, nvalues
            values(i) = 1.0
         enddo
         call bHYPRE_SStructVector_SetBoxValues_f( b, part,
     1        ilower, iupper, ndim, var, values, nvalues, ierr, except )

         do i = 1, nvalues
            values(i) = 0.0
         enddo
         call bHYPRE_SStructVector_SetBoxValues_f( x, part,
     1        ilower, iupper, ndim, var, values, nvalues, ierr, except )

      elseif ( myid .eq. 1 ) then
c     Set the vector coefficients over the gridpoints in my box
         ilower(1) = 3
         ilower(2) = 1
         iupper(1) = 6
         iupper(2) = 4
         nvalues = 16
c        ... 16 grid points

         do i = 1, nvalues
            values(i) = 1.0
         enddo
         call bHYPRE_SStructVector_SetBoxValues_f( b, part,
     1        ilower, iupper, ndim, var, values, nvalues, ierr, except )

         do i = 1, nvalues
            values(i) = 0.0
         enddo
         call bHYPRE_SStructVector_SetBoxValues_f( x, part,
     1        ilower, iupper, ndim, var, values, nvalues, ierr, except )

      endif

c     This is a collective call finalizing the vector assembly.
c     The vectors are now ``ready to be used''
      call bHYPRE_SStructVector_Assemble_f( b, ierr, except )
      call bHYPRE_SStructVector_Assemble_f( x, ierr, except )



c     6. Set up and use a solver (See the Reference Manual for descriptions
c     of all of the options.)

c        Because we are using a struct solver, we need to get the
c     object of the matrix and vectors to pass in to the struct solvers
      call bHYPRE_SStructMatrix_GetObject_f( A, dummy, ierr, except )
      call bHYPRE_StructMatrix__cast_f( dummy, sA,except )
      call sidl_BaseInterface_deleteRef_f( dummy, except )
      call bHYPRE_SStructVector_GetObject_f( b, dummy, ierr, except )
      call bHYPRE_Vector__cast_f( dummy, vb, except )
      call sidl_BaseInterface_deleteRef_f( dummy, except )
      call bHYPRE_SStructVector_GetObject_f( x, dummy, ierr, except )
      call bHYPRE_Vector__cast_f( dummy, vx, except )
      call sidl_BaseInterface_deleteRef_f( dummy, except )

c     Create an empty PCG Struct solver
      call bHYPRE_Operator__cast_f( sA, opA, except )
      call bHYPRE_PCG_Create_f( bHYPRE_mpicomm, opA, PCGsolver,
     1     except )

c     Set PCG parameters
c       Note that tol must be passed as a variable - putting 1.0e-6 directly
c       in the argument list won't work.
      tol = 1.0e-6
      call bHYPRE_PCG_SetDoubleParameter_f( PCGsolver, "Tolerance",
     1        tol, ierr, except )
      call bHYPRE_PCG_SetIntParameter_f( PCGsolver, "PrintLevel",
     1     2, ierr, except )
      call bHYPRE_PCG_SetIntParameter_f( PCGsolver, "MaxIter",
     1     50, ierr, except )

c     Create the Struct SMG solver for use as a preconditioner
      call bHYPRE_StructSMG_Create_f( bHYPRE_mpicomm, sA, SMGprecond,
     1     except )

c     Set SMG parameters
      call bHYPRE_StructSMG_SetIntParameter_f( SMGprecond,
     1     "MaxIter", 1, ierr, except )
      tol = 0.0
      call bHYPRE_StructSMG_SetDoubleParameter_f( SMGprecond,
     1     "Tolerance", tol, ierr, except )
      call bHYPRE_StructSMG_SetIntParameter_f( SMGprecond,
     1     "ZeroGuess", 1, ierr, except )
      call bHYPRE_StructSMG_SetIntParameter_f( SMGprecond,
     1     "NumPreRelax", 1, ierr, except )
      call bHYPRE_StructSMG_SetIntParameter_f( SMGprecond,
     1     "NumPostRelax", 1, ierr, except )

c     Set preconditioner and solve
      call bHYPRE_Solver__cast_f( SMGprecond, precond, except )
      call bHYPRE_PCG_SetPreconditioner_f( PCGsolver, precond,
     1     ierr, except )

      call bHYPRE_PCG_Setup_f( PCGsolver, vb, vx, ierr, except )
      call bHYPRE_PCG_Apply_f( PCGsolver, vb, vx, ierr, except )

      call bHYPRE_Operator_deleteRef_f( opA, except )
      call bHYPRE_Vector_deleteRef_f( vx, except )
      call bHYPRE_Vector_deleteRef_f( vb, except )
      call bHYPRE_StructMatrix_deleteRef_f( sA, except )


c     Free memory
      call bHYPRE_Solver_deleteRef_f( precond, except )
      call bHYPRE_StructSMG_deleteRef_f( SMGprecond, except )
      call bHYPRE_PCG_deleteRef_f( PCGsolver, except )
      call bHYPRE_SStructVector_deleteRef_f( x, except )
      call bHYPRE_SStructVector_deleteRef_f( b, except )
      call bHYPRE_SStructMatrix_deleteRef_f( A, except )
      call bHYPRE_SStructGraph_deleteRef_f( graph, except )
      call bHYPRE_SStructStencil_deleteRef_f( stencil, except )
      call bHYPRE_SStructGrid_deleteRef_f( grid, except )
      call bHYPRE_MPICommunicator_deleteRef_f( bHYPRE_mpicomm, except )

c     Finalize MPI
      call MPI_Finalize(ierrtmp)

      return
      end

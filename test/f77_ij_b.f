c-----------------------------------------------------------------------
c     Test driver for unstructured matrix interface (structured storage)
c-----------------------------------------------------------------------
c     jfp: Babel code added.  Inputs hardwired in.  Code using other values
c     of the inputs deleted.  The idea is to narrow this down for simplicity.
      
c-----------------------------------------------------------------------
c     Standard 7-point laplacian in 3D with grid and anisotropy determined
c     as user settings.
c-----------------------------------------------------------------------

      program test

      implicit none

      include 'mpif.h'

      integer MAXZONS, MAXBLKS, MAXDIM, MAXLEVELS
      integer HYPRE_PARCSR

      parameter (MAXZONS=4194304)
      parameter (MAXBLKS=32)
      parameter (MAXDIM=3)
      parameter (MAXLEVELS=25)
      parameter (HYPRE_PARCSR=5555)

      integer             num_procs, myid

      integer             dim
      integer             nx, ny, nz
      integer             Px, Py, Pz
      integer             bx, by, bz
      double precision    cx, cy, cz
      integer             n_pre, n_post
      integer             solver_id
      integer             precond_id

      integer             setup_type
      integer             debug_flag, ioutdat, k_dim
      integer             nlevels

      integer             zero, one
      parameter           (zero = 0, one = 1)
      integer             maxiter, num_iterations
      integer             generate_matrix, generate_rhs
      character           matfile(32), vecfile(32)
      character*31        matfile_str, vecfile_str

      double precision    tol, pc_tol, convtol
      parameter           (pc_tol = 0.0)
      double precision    final_res_norm
      
c     parameters for BoomerAMG
      integer             hybrid, coarsen_type, measure_type
      integer             cycle_type
      integer             smooth_num_sweep
c     The next two lines may become the only place so far where I have substantively changed the
c     original Fortran code. num_grid_sweeps and grid_relax_type are actually pointers
c     set in the Fortran-called C code.  One C function sets the pointer.  Then the Fortran
c     passes it on to another C function.  But the Babel-based code actually wants to get
c     data out of them.  I'm still working on getting this to work...
      integer*8           num_grid_sweeps
      integer*8           grid_relax_type
      integer*8           grid_relax_points
      integer*8           relax_weights
      double precision    strong_threshold, trunc_factor, drop_tol
      double precision    max_row_sum
      data                max_row_sum /1.0/

c     parameters for ParaSails
      double precision    sai_threshold
      double precision    sai_filter

      integer*8           A, A_storage
      integer*8           b, b_storage
      integer*8           x, x_storage

      integer*8           solver
      integer*8           precond
      integer*8           precond_gotten
      integer*8           row_starts

      double precision    values(4)

      integer             p, q, r

      integer             ierr

      integer             i
      integer             first_local_row, last_local_row
      integer             first_local_col, last_local_col
      integer             indices(MAXZONS)
      double precision    vals(MAXZONS)

c     Babel-interface variables
      integer*8 A_parcsr
      integer*8 Hypre_parcsr_A
      integer*8 Hypre_ij_A
      integer*8 Hypre_parcsr_x
      integer*8 Hypre_ij_x
      integer*8 Hypre_parcsr_b
      integer*8 Hypre_ij_b
      integer*8 Hypre_object
      integer*8 Hypre_object_tmp
      integer ierrtmp
      integer*8 Hypre_values
      integer*8 Hypre_indices
      integer local_num_rows, local_num_cols
      integer dimsl(2), dimsu(2)
      integer lower(1), upper(1)
      integer*8 Hypre_row_sizes
      integer size
      integer*8 Hypre_ncols
      integer stride(3)
      integer col_inds(1)
      integer*8 Hypre_col_inds
      integer*8 Hypre_AMG
      integer*8 Hypre_Vector_x
      integer*8 Hypre_Vector_b
      integer*8 Hypre_op_A
      integer*8 Hypre_num_grid_sweeps
      integer*8 Hypre_grid_relax_type
      integer*8 Hypre_relax_weight
      integer max_levels
c original:      data   max_levels /25/
      data   max_levels /1/
      double precision relax_weight(25)
      double precision double_zero
      data   double_zero /0.0/
      integer num_grid_sweeps_f(1)
      integer grid_relax_type_f(1)
      integer*8 linop;

c     equivalence ( num_grid_sweeps_f(1), num_grid_sweeps )
c     equivalence ( grid_relax_type_f(1), grid_relax_type )


c-----------------------------------------------------------------------
c     Initialize MPI
c-----------------------------------------------------------------------

      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)

c-----------------------------------------------------------------------
c     Set defaults
c-----------------------------------------------------------------------

      dim = 3

      nx = 10
      ny = 10
      nz = 10

      Px  = num_procs
      Py  = 1
      Pz  = 1

      bx = 1
      by = 1
      bz = 1

      cx = 1.0
      cy = 1.0
      cz = 1.0

      n_pre  = 1
      n_post = 1

c-----------------------------------------------------------------------
c     Read options
c-----------------------------------------------------------------------
      
      generate_matrix = 1
      generate_rhs = 1
      solver_id = 0
      tol = 1.0e-6

c-----------------------------------------------------------------------
c     Check a few things
c-----------------------------------------------------------------------

      if ((Px*Py*Pz) .ne. num_procs) then
         print *, 'Error: Invalid number of processors or topology'
         stop
      endif

      if ((dim .lt. 1) .or. (dim .gt. 3)) then
         print *, 'Error: Invalid problem dimension'
         stop
      endif

      if ((nx*ny*nz) .gt. MAXZONS) then
         print *, 'Error: Invalid number of zones'
         stop
      endif

      if ((bx*by*bz) .gt. MAXBLKS) then
         print *, 'Error: Invalid number of blocks'
         stop
      endif

c-----------------------------------------------------------------------
c     Print driver parameters
c-----------------------------------------------------------------------

      if (myid .eq. 0) then
         print *, 'Running with these driver parameters:'
         print *, '  (nx, ny, nz)    = (', nx, ',', ny, ',', nz, ')'
         print *, '  (Px, Py, Pz)    = (',  Px, ',',  Py, ',',  Pz, ')'
         print *, '  (bx, by, bz)    = (', bx, ',', by, ',', bz, ')'
         print *, '  (cx, cy, cz)    = (', cx, ',', cy, ',', cz, ')'
         print *, '  (n_pre, n_post) = (', n_pre, ',', n_post, ')'
         print *, '  dim             = ', dim
      endif

c-----------------------------------------------------------------------
c     Compute some grid and processor information
c-----------------------------------------------------------------------

      if (dim .eq. 1) then

c     compute p from Px and myid
         p = mod(myid,Px)

      elseif (dim .eq. 2) then

c     compute p,q from Px, Py and myid
         p = mod(myid,Px)
         q = mod(((myid - p)/Px),Py)

      elseif (dim .eq. 3) then

c     compute p,q,r from Px,Py,Pz and myid
         p = mod(myid,Px)
         q = mod((( myid - p)/Px),Py)
         r = (myid - (p + Px*q))/(Px*Py)

      endif

c----------------------------------------------------------------------
c     Set up the matrix
c-----------------------------------------------------------------------

      values(2) = -cx
      values(3) = -cy
      values(4) = -cz

      values(1) = 0.0
      if (nx .gt. 1) values(1) = values(1) + 2d0*cx
      if (ny .gt. 1) values(1) = values(1) + 2d0*cy
      if (nz .gt. 1) values(1) = values(1) + 2d0*cz

c     Generate a Dirichlet Laplacian

      call HYPRE_GenerateLaplacian(MPI_COMM_WORLD, nx, ny, nz,
     &     Px, Py, Pz, p, q, r, values,
     &     A_storage, ierr)

      call HYPRE_ParCSRMatrixGetLocalRange(A_storage,
     &     first_local_row, last_local_row,
     &     first_local_col, last_local_col, ierr)

      call HYPRE_IJMatrixCreate(MPI_COMM_WORLD,
     &     first_local_row, last_local_row,
     &     first_local_col, last_local_col, A, ierr)

      call HYPRE_IJMatrixSetObject(A, A_storage, ierr)

      call HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR, ierr)

      matfile(1)  = 'd'
      matfile(2)  = 'r'
      matfile(3)  = 'i'
      matfile(4)  = 'v'
      matfile(5)  = 'e'
      matfile(6)  = 'r'
      matfile(7)  = '.'
      matfile(8)  = 'o'
      matfile(9)  = 'u'
      matfile(10) = 't'
      matfile(11) = '.'
      matfile(12) = 'A'
      matfile(13) = char(0)
      
      call HYPRE_IJMatrixPrint(A, matfile, ierr)

      local_num_rows = last_local_row - first_local_row + 1
      local_num_cols = last_local_col - first_local_col + 1
      call hypre_ParCSRMatrixRowStarts(A_storage, row_starts, ierr)


      call Hypre_ParCSRMatrix__create_f( Hypre_parcsr_A )
      print *, 'finished _create_f'

      call Hypre_ParCSRMatrix__cast_f
     1     ( Hypre_parcsr_A, "Hypre.IJBuildMatrix", Hypre_ij_A )
      print *, 'finished cast'
      if ( Hypre_ij_A .eq. 0 ) then
         write(6,*) 'Cast failed'
         stop
      endif

c     The following will cancel each other out, but it is good practice
c     to perform them.
      call Hypre_IJBuildMatrix_addReference_f( Hypre_ij_A )
      call Hypre_ParCSRMatrix_deleteReference_f( Hypre_parcsr_A )

      call Hypre_IJBuildMatrix_SetCommunicator_f( Hypre_ij_A,
     1     MPI_COMM_WORLD, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJBuildMatrix_Create_f( Hypre_ij_A,
     1     first_local_row, last_local_row,
     1     first_local_col, last_local_col, ierrtmp )
      ierr = ierr + ierrtmp
      write(6,*) 'finished Create'

      call SIDL_int__array_create1d_f( local_num_rows, Hypre_row_sizes )
      size = 7

      do i = 1, local_num_rows
         call SIDL_int__array_set1_f( Hypre_row_sizes, i, size )
      enddo
      call Hypre_IJBuildMatrix_SetRowSizes_f(
     1     Hypre_ij_A, Hypre_row_sizes, ierrtmp )
      ierr = ierr + ierrtmp
      call SIDL_int__array_deleteReference_f( Hypre_row_sizes)

      call Hypre_IJBuildMatrix_Initialize_f( Hypre_ij_A, ierrtmp )
      ierr = ierr + ierrtmp

      call SIDL_int__array_create1d_f( 1, Hypre_row_sizes )
      call SIDL_int__array_create1d_f( 1, Hypre_ncols )

c     Loop through all locally stored rows and insert them into ij_matrix
      call HYPRE_IJMatrixGetObject( A, A_parcsr, ierrtmp)
      ierr = ierr + ierrtmp

      do i = first_local_row, last_local_row
         call HYPRE_ParCSRMatrixGetRow(
     1        A_parcsr, i, size, col_inds, values, ierrtmp )
         ierr = ierr + ierrtmp
         call SIDL_int__array_set1_f( Hypre_row_sizes, 0, size )
         call SIDL_int__array_set1_f( Hypre_ncols, 0, i )
         upper(1) = size - 1
         call SIDL_int__array_borrow_f(
     1        col_inds, 1, lower, upper, stride, Hypre_col_inds )
         call SIDL_double__array_borrow_f(
     1        col_inds, 1, lower, upper, stride, Hypre_values )
         call Hypre_IJBuildMatrix_SetValues_f(
     1        Hypre_ij_A, 1, Hypre_row_sizes, Hypre_ncols,
     1        Hypre_col_inds, Hypre_values, ierrtmp )
         ierr = ierr + ierrtmp
         call HYPRE_ParCSRMatrixRestoreRow(
     1        A_parcsr, i, size, col_inds, values, ierrtmp )
         ierr = ierr + ierrtmp
      enddo
      call SIDL_int__array_deleteReference_f( Hypre_row_sizes )
      call SIDL_int__array_deleteReference_f( Hypre_ncols )

      call Hypre_IJBuildMatrix_Assemble_f( Hypre_ij_A, ierrtmp )
      ierr = ierr + ierrtmp

      call SIDL_int__array_deleteReference_f( Hypre_col_inds )
      call SIDL_int__array_deleteReference_f( Hypre_values )

c     
c     Fetch the resulting underlying matrix out
c     
c     At this point we could destroy A_parcsr if this were a pure Babel-interface
c     code.  But as long as the direct HYPRE_ interface is in use, we have to keep
c     it around.
c     call HYPRE_ParCSRMatrixDestroy( A_parcsr, ierrtmp )
c     ierr = ierr + ierrtmp
      call Hypre_IJBuildMatrix_GetObject_f(
     1     Hypre_ij_A, Hypre_object, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_IJBuildMatrix_deleteReference_f( Hypre_ij_A )
c     
c     The QueryInterface below checks to see if the returned object can
c     return a Hypre.ParCSRMatrix. The "cast" is necessary because of the
c     restrictions of the C language, and is merely to please the compiler.
c     It is the QueryInterface that actually has semantic meaning.
c     >>> keep the cast for now, take it out once this works.  This isn't C>>>
      call SIDL_BaseInterface_queryInterface_f(
     1     Hypre_object, "Hypre.ParCSRMatrix", Hypre_object_tmp )
      call SIDL_BaseInterface__cast_f(
     1     Hypre_object_tmp, "Hypre.ParCSRMatrix", Hypre_parcsr_A )
      if ( Hypre_parcsr_A .eq. 0 ) then
         write (6,*) 'Matrix cast/QI failed\n'
         stop
      endif

c     
c     {

c     /* Break encapsulation so that the rest of the driver stays the same */
c     struct Hypre_ParCSRMatrix__data * temp_data;
c     jfp This isn't very doable in Fortran.
c     temp_data = Hypre_ParCSRMatrix__get_data( Hypre_parcsr_A );
c     
c     ij_A = temp_data ->ij_A ;
c     
c     ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
c     parcsr_A = (HYPRE_ParCSRMatrix) object;
c     
c     }



c-----------------------------------------------------------------------
c     Set up the rhs and initial guess
c-----------------------------------------------------------------------

      call HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col,
     &     last_local_col, b, ierr)
      call HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR, ierr)
      call HYPRE_IJVectorInitialize(b, ierr)

c     Set up a Dirichlet 0 problem
      do i = 1, last_local_col - first_local_col + 1
         indices(i) = first_local_col - 1 + i
         vals(i) = 1.
      enddo

      call HYPRE_IJVectorSetValues(b,
     &     last_local_col - first_local_col + 1, indices, vals, ierr)

      call HYPRE_IJVectorGetObject(b, b_storage, ierr)

      vecfile(1)  = 'd'
      vecfile(2)  = 'r'
      vecfile(3)  = 'i'
      vecfile(4)  = 'v'
      vecfile(5)  = 'e'
      vecfile(6)  = 'r'
      vecfile(7)  = '.'
      vecfile(8)  = 'o'
      vecfile(9)  = 'u'
      vecfile(10) = 't'
      vecfile(11) = '.'
      vecfile(12) = 'b'
      vecfile(13) = char(0)
      
      call HYPRE_IJVectorPrint(b, vecfile, ierr)

      call HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col,
     &     last_local_col, x, ierr)
      call HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR, ierr)
      call HYPRE_IJVectorInitialize(x, ierr)
      do i = 1, last_local_col - first_local_col + 1
         indices(i) = first_local_col - 1 + i
         vals(i) = 0.
      enddo
      call HYPRE_IJVectorSetValues(x,
     &     last_local_col - first_local_col + 1, indices, vals, ierr)

      call Hypre_ParCSRVector__create_f( Hypre_parcsr_b )
      call Hypre_ParCSRVector__cast_f
     1     ( Hypre_parcsr_b, "Hypre.IJBuildVector", Hypre_ij_b )
      print *, 'finished cast'
      if ( Hypre_ij_b .eq. 0 ) then
         write(6,*) 'Cast failed'
         stop
      endif
      call Hypre_IJBuildVector_addReference_f( Hypre_ij_b )
      call Hypre_ParCSRVector_deleteReference_f( Hypre_parcsr_b )
      call Hypre_IJBuildVector_SetCommunicator_f( Hypre_ij_b,
     1     MPI_COMM_WORLD, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJBuildVector_Create_f( Hypre_ij_b, MPI_COMM_WORLD,
     1     first_local_row, last_local_row, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJBuildVector_Initialize_f( Hypre_ij_b, ierrtmp )
      ierr = ierr + ierrtmp

      call SIDL_int__array_create1d_f( local_num_cols, Hypre_indices )
      call SIDL_double__array_create1d_f( local_num_cols, Hypre_values )
      do i=0, local_num_cols-1
         call SIDL_int__array_set1_f( Hypre_indices, i,
     1        first_local_col + i )
         call SIDL_double__array_set1_f( Hypre_values, i, double_zero )
      enddo
      call Hypre_IJBuildVector_SetValues_f( Hypre_ij_b,
     1     local_num_cols, Hypre_indices, Hypre_values, ierrtmp )
      ierr = ierr + ierrtmp

      call SIDL_int__array_deleteReference_f( Hypre_indices)
      call SIDL_double__array_deleteReference_f( Hypre_values )

      call Hypre_IJBuildVector_Assemble_f( Hypre_ij_b, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_IJBuildVector_GetObject_f(
     1     Hypre_ij_b, Hypre_object, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJBuildVector_deleteReference_f( Hypre_ij_b )
      call SIDL_BaseInterface_queryInterface_f(
     1     Hypre_object, "Hypre.ParCSRVector", Hypre_object_tmp )
      call SIDL_BaseInterface__cast_f(
     1     Hypre_object_tmp, "Hypre.ParCSRVector", Hypre_parcsr_b )
      if ( Hypre_parcsr_b .eq. 0 ) then
         write (6,*) 'Cast/QI failed\n'
         stop
      endif


      call Hypre_ParCSRVector__create_f( Hypre_parcsr_x )
      call Hypre_ParCSRVector__cast_f
     1     ( Hypre_parcsr_x, "Hypre.IJBuildVector", Hypre_ij_x )
      print *, 'finished cast'
      if ( Hypre_ij_x .eq. 0 ) then
         write(6,*) 'Cast failed'
         stop
      endif
      call Hypre_IJBuildVector_addReference_f( Hypre_ij_x )
      call Hypre_ParCSRVector_deleteReference_f( Hypre_parcsr_x )
      call Hypre_IJBuildVector_SetCommunicator_f( Hypre_ij_x,
     1     MPI_COMM_WORLD, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJBuildVector_Create_f( Hypre_ij_x, MPI_COMM_WORLD,
     1     first_local_row, last_local_row, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJBuildVector_Initialize_f( Hypre_ij_x, ierrtmp )
      ierr = ierr + ierrtmp

      call SIDL_int__array_create1d_f( local_num_cols, Hypre_indices )
      call SIDL_double__array_create1d_f( local_num_cols, Hypre_values )
      do i=0, local_num_cols-1
         call SIDL_int__array_set1_f( Hypre_indices, i,
     1        first_local_col + i )
         call SIDL_double__array_set1_f( Hypre_values, i, double_zero )
      enddo
      call Hypre_IJBuildVector_SetValues_f( Hypre_ij_x,
     1     local_num_cols, Hypre_indices, Hypre_values, ierrtmp )
      ierr = ierr + ierrtmp

      call SIDL_int__array_deleteReference_f( Hypre_indices)
      call SIDL_double__array_deleteReference_f( Hypre_values )

      call Hypre_IJBuildVector_Assemble_f( Hypre_ij_x, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_IJBuildVector_GetObject_f(
     1     Hypre_ij_x, Hypre_object, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJBuildVector_deleteReference_f( Hypre_ij_x )
      call SIDL_BaseInterface_queryInterface_f(
     1     Hypre_object, "Hypre.ParCSRVector", Hypre_object_tmp )
      call SIDL_BaseInterface__cast_f(
     1     Hypre_object_tmp, "Hypre.ParCSRVector", Hypre_parcsr_x )
      if ( Hypre_parcsr_x .eq. 0 ) then
         write (6,*) 'Cast/QI failed\n'
         stop
      endif


c     The C program did this:
c     /* Break encapsulation so that the rest of the driver stays the same */
c     That's not so easily done in Fortran.


c     Choose a nonzero initial guess
      call HYPRE_IJVectorGetObject(x, x_storage, ierr)

      vecfile(1)  = 'd'
      vecfile(2)  = 'r'
      vecfile(3)  = 'i'
      vecfile(4)  = 'v'
      vecfile(5)  = 'e'
      vecfile(6)  = 'r'
      vecfile(7)  = '.'
      vecfile(8)  = 'o'
      vecfile(9)  = 'u'
      vecfile(10) = 't'
      vecfile(11) = '.'
      vecfile(12) = 'x'
      vecfile(13) = '0'
      vecfile(14) = char(0)
      
      call HYPRE_IJVectorPrint(x, vecfile, ierr)

c-----------------------------------------------------------------------
c     Solve the linear system
c-----------------------------------------------------------------------

      write (6,*) "solver_id", solver_id
c     General solver parameters, passing hard coded constants
c     will break the interface.

      maxiter = 100
      convtol = 0.9
      debug_flag = 0
      ioutdat = 1

c     Set defaults for BoomerAMG
      maxiter = 500
      coarsen_type = 6
      hybrid = 1
      measure_type = 0
      strong_threshold = 0.25
      trunc_factor = 0.0
      cycle_type = 1
      smooth_num_sweep = 1

      print *, 'Solver: AMG'

c     The direct HYPRE interfaces and the Babel interface will have to live together
c     until the Babel one is completely working.  (jfp)  Here are the
c     HYPRE interface calls.

cold      call HYPRE_BoomerAMGCreate(solver, ierr)
cold      call HYPRE_BoomerAMGSetCoarsenType(solver,
cold     &     (hybrid*coarsen_type), ierr)
cold      call HYPRE_BoomerAMGSetMeasureType(solver, measure_type, ierr)
cold      call HYPRE_BoomerAMGSetTol(solver, tol, ierr)
cold      call HYPRE_BoomerAMGSetStrongThrshld(solver,
cold     &     strong_threshold, ierr)
cold      call HYPRE_BoomerAMGSetTruncFactor(solver, trunc_factor, ierr)
cold      call HYPRE_BoomerAMGSetPrintLevel(solver, ioutdat,ierr)
coldc     the old Fortran interface isn't handling the string right:
coldc     call HYPRE_BoomerAMGSetPrintFileName(solver,"test.out.log",ierr)
cold      call HYPRE_BoomerAMGSetMaxIter(solver, maxiter, ierr)
cold      call HYPRE_BoomerAMGSetCycleType(solver, cycle_type, ierr)
cold      call HYPRE_BoomerAMGInitGridRelaxatn(num_grid_sweeps,
cold     &     grid_relax_type,
cold     &     grid_relax_points,
cold     &     coarsen_type,
cold     &     relax_weights,
cold     &     MAXLEVELS,ierr)
cold      call HYPRE_BoomerAMGSetNumGridSweeps(solver,
cold     &     num_grid_sweeps, ierr)
cold      call HYPRE_BoomerAMGSetGridRelaxType(solver,
cold     &     grid_relax_type, ierr)
cold      call HYPRE_BoomerAMGSetRelaxWeight(solver,
cold     &     relax_weights, ierr)
coldc     call HYPRE_BoomerAMGSetSmoothOption(solver, smooth_option,
coldc     &                                      ierr)
coldc     call HYPRE_BoomerAMGSetSmoothNumSwp(solver, smooth_num_sweep,
coldc     &                                      ierr)
cold      call HYPRE_BoomerAMGSetGridRelaxPnts(solver,
cold     &     grid_relax_points,
cold     &     ierr)
cold      call HYPRE_BoomerAMGSetMaxLevels(solver, MAXLEVELS, ierr)
cold      call HYPRE_BoomerAMGSetMaxRowSum(solver, max_row_sum,
cold     &     ierr)
cold      call HYPRE_BoomerAMGSetDebugFlag(solver, debug_flag, ierr)
cold      call HYPRE_BoomerAMGSetup(solver, A_storage, b_storage,
cold     &     x_storage, ierr)
cold      call HYPRE_BoomerAMGSolve(solver, A_storage, b_storage,
cold     &     x_storage, ierr)
cold      call HYPRE_BoomerAMGGetNumIterations(solver, num_iterations, 
cold     &     ierr)
cold      call HYPRE_BoomerAMGGetFinalReltvRes(solver, final_res_norm,
cold     &     ierr)
cold      call HYPRE_BoomerAMGDestroy(solver, ierr)


c     and here are of the Babel interface calls, adapted from the C/Babel code:
      call Hypre_ParAMG__create_f( Hypre_AMG )
      call Hypre_ParCSRVector__cast_f
     1     ( Hypre_parcsr_b, "Hypre.Vector", Hypre_Vector_b )
      call Hypre_ParCSRVector__cast_f
     1     ( Hypre_parcsr_x, "Hypre.Vector", Hypre_Vector_x )
      call Hypre_ParCSRVector__cast_f
     1     ( Hypre_parcsr_A, "Hypre.Operator", Hypre_op_A )
      call Hypre_ParAMG_SetCommunicator_f(
     1     Hypre_AMG, MPI_COMM_WORLD, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_ParAMG_SetOperator_f( Hypre_AMG, Hypre_op_A )
c     write (6,*) "**** before calling Hypre_ParAMGSet*Parameter"
      call Hypre_ParAMG_SetIntParameter_f(
     1     Hypre_AMG, "CoarsenType", hybrid*coarsen_type, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_ParAMG_SetIntParameter_f(
     1     Hypre_AMG, "MeasureType", measure_type, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_ParAMG_SetDoubleParameter_f(
     1     Hypre_AMG, "StrongThreshold", strong_threshold, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_ParAMG_SetDoubleParameter_f(
     1     Hypre_AMG, "TruncFactor", trunc_factor, ierrtmp )
      ierr = ierr + ierrtmp
c     /* note: log output not specified ... */
      call Hypre_ParAMG_SetIntParameter_f(
     1     Hypre_AMG, "PrintLevel", ioutdat, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_ParAMG_SetIntParameter_f(
     1     Hypre_AMG, "CycleType", cycle_type, ierrtmp )
      ierr = ierr + ierrtmp
c     dimsl[0] = 0;   dimsu[0] = 4;
c     Hypre_num_grid_sweeps = SIDL_int__array_create( 1, dimsl, dimsu );
c     for ( i=0; i<4; ++i )
c     SIDL_int__array_set1( Hypre_num_grid_sweeps, i, num_grid_sweeps[i] );
c     Hypre_ParAMG_SetIntArrayParameter( Hypre_AMG, "NumGridSweeps", Hypre_num_grid_sweeps );
c     dimsl[0] = 0;   dimsu[0] = 4;
c     Hypre_grid_relax_type = SIDL_int__array_create( 1, dimsl, dimsu );
c     for ( i=0; i<4; ++i )
c     SIDL_int__array_set1( Hypre_grid_relax_type, i, grid_relax_type[i] );
c     Hypre_ParAMG_SetIntArrayParameter( Hypre_AMG, "GridRelaxType", Hypre_grid_relax_type );
cdbug      dimsl(1) = 1
cdbug      dimsu(1) = 4
cdbug      call SIDL_int__array_create1d_f(
cdbug     1     4, Hypre_num_grid_sweeps )
cdbug      do i = 1, 4
cdbug         call SIDL_int__array_set1_f(
cdbug     1        Hypre_num_grid_sweeps, i-1, num_grid_sweeps_f(i) )
cdbug      enddo
cdbug      call Hypre_ParAMG_SetIntArrayParameter_f( Hypre_AMG,
cdbug     1     "NumGridSweeps", Hypre_num_grid_sweeps, ierrtmp )
cdbug      ierr = ierr + ierrtmp
cdbug      dimsl(1) = 1
cdbug      dimsu(1) = 4
cdbug      call SIDL_int__array_create1d_f(
cdbug     1     4, Hypre_grid_relax_type )
cdbug      do i = 1, 4
cdbug         call SIDL_int__array_set1_f(
cdbug     1        Hypre_grid_relax_type, i-1, grid_relax_type_f(i) )
cdbug      enddo
cdbug      call Hypre_ParAMG_SetIntArrayParameter_f( Hypre_AMG,
cdbug     1     "GridRelaxType", Hypre_grid_relax_type, ierrtmp )
cdbug      ierr = ierr + ierrtmp

c     dimsl[0] = 0;   dimsu[0] = max_levels;
c     Hypre_relax_weight = SIDL_double__array_create( 1, dimsl, dimsu );
c     for ( i=0; i<max_levels; ++i )
c     SIDL_double__array_set1( Hypre_relax_weight, i, relax_weight[i] );
c     Hypre_ParAMG_SetDoubleArrayParameter( Hypre_AMG, "RelaxWeight", Hypre_relax_weight );
      dimsl(1) = 0
      dimsu(1) = max_levels
      call SIDL_double__array_create1d_f(
     1     max_levels+1, Hypre_relax_weight )
      do i=1, max_levels
c     relax_weight(i)=1.0: simple to set, fine for testing:
         relax_weight(i) = 1.0
      enddo
      do i=1, max_levels
         call SIDL_double__array_set1_f(
     1        Hypre_relax_weight, i, relax_weight(i) )
      enddo
      call Hypre_ParAMG_SetDoubleArrayParameter_f(
     1     Hypre_AMG, "RelaxWeight", Hypre_relax_weight, ierrtmp )
      ierr = ierr + ierrtmp

c left at default: GridRelaxPoints
      call Hypre_ParAMG_SetIntParameter_f(
     1     Hypre_AMG, "MaxLevels", max_levels, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_ParAMG_SetIntParameter_f(
     1     Hypre_AMG, "DebugFlag", debug_flag, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_ParAMG_SetDoubleParameter_f(
     1     Hypre_AMG, "MaxRowSum", max_row_sum, ierrtmp )
      ierr = ierr + ierrtmp

c      linop = (Hypre_LinearOperator) Hypre_ParCSRMatrix_castTo(
c         ij_matrix_Hypre, "Hypre.LinearOperator" );
c      ierr += Hypre_ParAMG_Setup( AMG_Solver, linop, b_HypreV, x_HypreV );
c      ierr += Hypre_ParAMG_Apply( AMG_Solver, b_HypreV, &x_HypreV );
      call Hypre_ParCSRMatrix__cast_f(
     1     Hypre_ParCSR_A, "Hypre.LinearOperator", linop )
      call Hypre_ParAMG_Setup_f(
     1     Hypre_AMG, linop, Hypre_Vector_b, Hypre_Vector_x, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_ParAMG_Apply_f(
     1     Hypre_AMG, Hypre_Vector_b, Hypre_Vector_x, ierrtmp )
      ierr = ierr + ierrtmp



c-----------------------------------------------------------------------
c     Print the solution and other info
c-----------------------------------------------------------------------

      vecfile(1)  = 'd'
      vecfile(2)  = 'r'
      vecfile(3)  = 'i'
      vecfile(4)  = 'v'
      vecfile(5)  = 'e'
      vecfile(6)  = 'r'
      vecfile(7)  = '.'
      vecfile(8)  = 'o'
      vecfile(9)  = 'u'
      vecfile(10) = 't'
      vecfile(11) = '.'
      vecfile(12) = 'x'
      vecfile(13) = char(0)
      
      call HYPRE_IJVectorPrint(x, vecfile, ierr)

      if (myid .eq. 0) then
         print *, 'Iterations = ', num_iterations
         print *, 'Final Residual Norm = ', final_res_norm
      endif

c-----------------------------------------------------------------------
c     Finalize things
c-----------------------------------------------------------------------

      call HYPRE_ParCSRMatrixDestroy(A_storage, ierr)
      call HYPRE_IJVectorDestroy(b, ierr)
      call HYPRE_IJVectorDestroy(x, ierr)

c     Finalize MPI

      call MPI_FINALIZE(ierr)

      stop
      end

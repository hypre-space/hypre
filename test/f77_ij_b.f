c-----------------------------------------------------------------------
c     Test driver for unstructured matrix interface (structured storage)
c-----------------------------------------------------------------------
c     jfp: Babel code added.  Inputs hardwired in.  Code using other values
c     of the inputs deleted.  The idea is to narrow this down for simplicity.
      
c-----------------------------------------------------------------------
c     Standard 7-point laplacian in 3D with grid and anisotropy determined
c     as user settings (now hardwired early in the code).
c     The problem is first set up with some calls of special-purpose
c     functions in the older hypre interface, which don't exist in the
c     Babel-based interface.  Then the data is converted to a form suitable
c     for the Babel-based interface, which is used to call the solver.
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
      data   max_levels /25/
      double precision relax_weight(25)
      double precision double_zero
      data   double_zero /0.0/
      double precision double_one
      data   double_one /1.0/
      integer*8 linop;


c-----------------------------------------------------------------------
c     Initialize MPI
c-----------------------------------------------------------------------

      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)

c-----------------------------------------------------------------------
c     Set the former input parameters
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


      call Hypre_IJParCSRMatrix__create_f( Hypre_parcsr_A )

      call Hypre_IJParCSRMatrix__cast_f
     1     ( Hypre_parcsr_A, "Hypre.IJBuildMatrix", Hypre_ij_A )
      if ( Hypre_ij_A .eq. 0 ) then
         write(6,*) 'Cast failed'
         stop
      endif

c     The following will cancel each other out, but it is good practice
c     to perform them.
      call Hypre_IJBuildMatrix_addref_f( Hypre_ij_A )
      call Hypre_IJParCSRMatrix_deleteref_f( Hypre_parcsr_A )

      call Hypre_IJBuildMatrix_SetCommunicator_f( Hypre_ij_A,
     1     MPI_COMM_WORLD, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJBuildMatrix_SetLocalRange_f( Hypre_ij_A,
     1     first_local_row, last_local_row,
     1     first_local_col, last_local_col, ierrtmp )
      ierr = ierr + ierrtmp

      call SIDL_int__array_create1d_f( local_num_rows, Hypre_row_sizes )
      size = 7

      do i = 1, local_num_rows
         call SIDL_int__array_set1_f( Hypre_row_sizes, i, size )
      enddo
      call Hypre_IJBuildMatrix_SetRowSizes_f(
     1     Hypre_ij_A, Hypre_row_sizes, ierrtmp )
      ierr = ierr + ierrtmp
      call SIDL_int__array_deleteref_f( Hypre_row_sizes)

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
         call SIDL_int__array_borrow_deref_f(
     1        col_inds, 1, lower, upper, stride, Hypre_col_inds )
         call SIDL_double__array_borrow_deref_f(
     1        values, 1, lower, upper, stride, Hypre_values )
         call Hypre_IJBuildMatrix_SetValues_f(
     1        Hypre_ij_A, 1, Hypre_row_sizes, Hypre_ncols,
     1        Hypre_col_inds, Hypre_values, ierrtmp )
         ierr = ierr + ierrtmp
         call HYPRE_ParCSRMatrixRestoreRow(
     1        A_parcsr, i, size, col_inds, values, ierrtmp )
         ierr = ierr + ierrtmp
      enddo
      call SIDL_int__array_deleteref_f( Hypre_row_sizes )
      call SIDL_int__array_deleteref_f( Hypre_ncols )

      call Hypre_IJBuildMatrix_Assemble_f( Hypre_ij_A, ierrtmp )
      ierr = ierr + ierrtmp

      call SIDL_int__array_deleteref_f( Hypre_col_inds )
      call SIDL_int__array_deleteref_f( Hypre_values )

c     
c     Fetch the resulting underlying matrix out
c     
      call Hypre_IJBuildMatrix_GetObject_f(
     1     Hypre_ij_A, Hypre_object, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_IJBuildMatrix_deleteref_f( Hypre_ij_A )
c     
c     The Queryint below checks to see if the returned object can
c     return a Hypre.ParCSRMatrix. The "cast" is necessary because of the
c     restrictions of the C language, and is merely to please the compiler.
c     It is the Queryint that actually has semantic meaning.
c     ( cast removed for Fortran )
      call SIDL_BaseInterface_queryint_f(
     1     Hypre_object, "Hypre.ParCSRMatrix", Hypre_parcsr_A )
      if ( Hypre_parcsr_A .eq. 0 ) then
         write (6,*) 'Matrix cast/QI failed\n'
         stop
      endif


c-----------------------------------------------------------------------
c     Set up the rhs and initial guess
c-----------------------------------------------------------------------

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

      call Hypre_IJParCSRVector__create_f( Hypre_parcsr_b )
      call Hypre_IJParCSRVector__cast_f
     1     ( Hypre_parcsr_b, "Hypre.IJBuildVector", Hypre_ij_b )
      if ( Hypre_ij_b .eq. 0 ) then
         write(6,*) 'Cast failed'
         stop
      endif
      call Hypre_IJBuildVector_addref_f( Hypre_ij_b )
      call Hypre_IJParCSRVector_deleteref_f( Hypre_parcsr_b )
      call Hypre_IJBuildVector_SetCommunicator_f( Hypre_ij_b,
     1     MPI_COMM_WORLD, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJBuildVector_SetLocalRange_f( Hypre_ij_b,
     1      first_local_row, last_local_row, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJBuildVector_Initialize_f( Hypre_ij_b, ierrtmp )
      ierr = ierr + ierrtmp

      call SIDL_int__array_create1d_f( local_num_cols, Hypre_indices )
      call SIDL_double__array_create1d_f( local_num_cols, Hypre_values )
      do i=0, local_num_cols-1
         call SIDL_int__array_set1_f( Hypre_indices, i,
     1        first_local_col + i )
         call SIDL_double__array_set1_f( Hypre_values, i, double_one )
      enddo
      call Hypre_IJBuildVector_SetValues_f( Hypre_ij_b,
     1     local_num_cols, Hypre_indices, Hypre_values, ierrtmp )
      ierr = ierr + ierrtmp

      call SIDL_int__array_deleteref_f( Hypre_indices)
      call SIDL_double__array_deleteref_f( Hypre_values )

      call Hypre_IJBuildVector_Assemble_f( Hypre_ij_b, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_IJBuildVector_GetObject_f(
     1     Hypre_ij_b, Hypre_object, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJBuildVector_deleteref_f( Hypre_ij_b )
      call SIDL_BaseInterface_queryint_f(
     1     Hypre_object, "Hypre.ParCSRVector", Hypre_object_tmp )
      call SIDL_BaseInterface__cast_f(
     1     Hypre_object_tmp, "Hypre.ParCSRVector", Hypre_parcsr_b )
      if ( Hypre_parcsr_b .eq. 0 ) then
         write (6,*) 'Cast/QI failed\n'
         stop
      endif


      call Hypre_IJParCSRVector_print_f(
     1     Hypre_parcsr_b, "driver.out.b", ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJParCSRVector__create_f( Hypre_parcsr_x )
      call Hypre_IJParCSRVector__cast_f
     1     ( Hypre_parcsr_x, "Hypre.IJBuildVector", Hypre_ij_x )
      if ( Hypre_ij_x .eq. 0 ) then
         write(6,*) 'Cast failed'
         stop
      endif
      call Hypre_IJBuildVector_addref_f( Hypre_ij_x )
      call Hypre_IJBuildVector_SetCommunicator_f( Hypre_ij_x,
     1     MPI_COMM_WORLD, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJBuildVector_SetLocalRange_f( Hypre_ij_x,
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

      call SIDL_int__array_deleteref_f( Hypre_indices)
      call SIDL_double__array_deleteref_f( Hypre_values )

      call Hypre_IJBuildVector_Assemble_f( Hypre_ij_x, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_IJBuildVector_GetObject_f(
     1     Hypre_ij_x, Hypre_object, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJBuildVector_deleteref_f( Hypre_ij_x )
      call SIDL_BaseInterface_queryint_f(
     1     Hypre_object, "Hypre.ParCSRVector", Hypre_object_tmp )
      call SIDL_BaseInterface__cast_f(
     1     Hypre_object_tmp, "Hypre.ParCSRVector", Hypre_parcsr_x )
      if ( Hypre_parcsr_x .eq. 0 ) then
         write (6,*) 'Cast/QI failed\n'
         stop
      endif

      call Hypre_IJParCSRVector_print_f(
     1     Hypre_parcsr_x, "driver.out.x0", ierrtmp )
      ierr = ierr + ierrtmp

c-----------------------------------------------------------------------
c     Solve the linear system
c-----------------------------------------------------------------------

c      write (6,*) "solver_id", solver_id
c     General solver parameters, passing hard coded constants
c     will break the interface.

      maxiter = 100
      convtol = 0.9
      debug_flag = 0
c      ioutdat = 1
      ioutdat = 3

c     Set defaults for BoomerAMG
      maxiter = 500
      coarsen_type = 6
      hybrid = 1
      measure_type = 0
      strong_threshold = 0.25
      trunc_factor = 0.0
      cycle_type = 1
      smooth_num_sweep = 1

c      print *, 'Solver: AMG'

      call HYPRE_BoomerAMGInitGridRelaxatn(num_grid_sweeps,
     &     grid_relax_type,
     &     grid_relax_points,
     &     coarsen_type,
     &     relax_weights,
     &     MAXLEVELS,ierr)

      call Hypre_BoomerAMG__create_f( Hypre_AMG )
      call Hypre_IJParCSRVector__cast_f
     1     ( Hypre_parcsr_b, "Hypre.Vector", Hypre_Vector_b )
      call Hypre_IJParCSRVector__cast_f
     1     ( Hypre_parcsr_x, "Hypre.Vector", Hypre_Vector_x )
      call Hypre_IJParCSRVector__cast_f
     1     ( Hypre_parcsr_A, "Hypre.Operator", Hypre_op_A )
      call Hypre_BoomerAMG_SetCommunicator_f(
     1     Hypre_AMG, MPI_COMM_WORLD, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_BoomerAMG_SetOperator_f( Hypre_AMG, Hypre_op_A )
      call Hypre_BoomerAMG_SetIntParameter_f(
     1     Hypre_AMG, "CoarsenType", hybrid*coarsen_type, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_BoomerAMG_SetIntParameter_f(
     1     Hypre_AMG, "MeasureType", measure_type, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_BoomerAMG_SetDoubleParameter_f(
     1     Hypre_AMG, "StrongThreshold", strong_threshold, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_BoomerAMG_SetDoubleParameter_f(
     1     Hypre_AMG, "TruncFactor", trunc_factor, ierrtmp )
      ierr = ierr + ierrtmp
c     /* note: log output not specified ... */
      call Hypre_BoomerAMG_SetIntParameter_f(
     1     Hypre_AMG, "PrintLevel", ioutdat, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_BoomerAMG_SetIntParameter_f(
     1     Hypre_AMG, "CycleType", cycle_type, ierrtmp )
      ierr = ierr + ierrtmp
      dimsl(1) = 1
      dimsu(1) = 4
      call SIDL_int__array_create1d_f(
     1     4, Hypre_num_grid_sweeps )
      do i = 1, 4
         call SIDL_int__array_set1_deref_f(
     1        Hypre_num_grid_sweeps, i-1, num_grid_sweeps, i-1 )
      enddo
      call Hypre_BoomerAMG_SetIntArrayParameter_f( Hypre_AMG,
     1     "NumGridSweeps", Hypre_num_grid_sweeps, ierrtmp )
      ierr = ierr + ierrtmp
      dimsl(1) = 1
      dimsu(1) = 4
      call SIDL_int__array_create1d_f(
     1     4, Hypre_grid_relax_type )
      do i = 1, 4
        call SIDL_int__array_set1_deref_f(
     1        Hypre_grid_relax_type, i-1, grid_relax_type, i-1 )
      enddo
      call Hypre_BoomerAMG_SetIntArrayParameter_f( Hypre_AMG,
     1     "GridRelaxType", Hypre_grid_relax_type, ierrtmp )
      ierr = ierr + ierrtmp

      dimsl(1) = 0
      dimsu(1) = max_levels
      call SIDL_double__array_create1d_f(
     1     max_levels+1, Hypre_relax_weight )
      do i=1, max_levels
c        relax_weight(i)=1.0: simple to set, fine for testing:
         relax_weight(i) = 1.0
      enddo
      do i=1, max_levels
         call SIDL_double__array_set1_f(
     1        Hypre_relax_weight, i-1, relax_weight(i) )
      enddo
      call Hypre_BoomerAMG_SetDoubleArrayParameter_f(
     1     Hypre_AMG, "RelaxWeight", Hypre_relax_weight, ierrtmp )
      ierr = ierr + ierrtmp

c left at default: GridRelaxPoints
      call Hypre_BoomerAMG_SetIntParameter_f(
     1     Hypre_AMG, "MaxLevels", max_levels, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_BoomerAMG_SetIntParameter_f(
     1     Hypre_AMG, "DebugFlag", debug_flag, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_BoomerAMG_SetDoubleParameter_f(
     1     Hypre_AMG, "MaxRowSum", max_row_sum, ierrtmp )
      ierr = ierr + ierrtmp

      call Hypre_IJParCSRMatrix__cast_f(
     1     Hypre_ParCSR_A, "Hypre.LinearOperator", linop )
      call Hypre_BoomerAMG_Setup_f(
     1     Hypre_AMG, Hypre_Vector_b, Hypre_Vector_x, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_IJParCSRVector_print_f(
     1     Hypre_parcsr_x, "driver.out.x2", ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_BoomerAMG_Apply_f(
     1     Hypre_AMG, Hypre_Vector_b, Hypre_Vector_x, ierrtmp )
      ierr = ierr + ierrtmp



c-----------------------------------------------------------------------
c     Print the solution and other info
c-----------------------------------------------------------------------

      call Hypre_IJParCSRVector_print_f(
     1     Hypre_parcsr_x, "driver.out.x", ierrtmp )
      ierr = ierr + ierrtmp

      if (myid .eq. 0) then
         call Hypre_BoomerAMG_GetIntValue_f(
     1        Hypre_AMG, "Iterations", num_iterations, ierrtmp )
         ierr = ierr + ierrtmp
         call Hypre_BoomerAMG_GetDoubleValue_f(
     1        Hypre_AMG, "Final Relative Residual Norm", final_res_norm,
     1        ierrtmp )
         ierr = ierr + ierrtmp
         print *, 'Iterations = ', num_iterations
         print *, 'Final Residual Norm = ', final_res_norm
         print *, 'Error Flag = ', ierr
      endif

c-----------------------------------------------------------------------
c     Finalize things
c-----------------------------------------------------------------------

      call Hypre_IJParCSRVector_deleteref_f( Hypre_parcsr_x )
      call HYPRE_ParCSRMatrixDestroy(A_storage, ierr)
c      call HYPRE_IJVectorDestroy(b, ierr)
      call HYPRE_IJVectorDestroy(x, ierr)

c     Finalize MPI

      call MPI_FINALIZE(ierr)

      stop
      end

c
c
c-----------------------------------------------------------------------
c     Test driver for unstructured matrix interface (structured storage)
c-----------------------------------------------------------------------
c     This differs from the C test drivers.  Inputs are hardwired in.
c     Code using other values of the inputs is deleted.
c     The idea is to narrow this down for simplicity, just test the Fortran.
      
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
      integer HYPRE_PARCSR, NNX, NNY, NNZ, VECLEN

      parameter (MAXZONS=4194304)
      parameter (MAXBLKS=32)
      parameter (MAXDIM=3)
      parameter (MAXLEVELS=25)
      parameter (HYPRE_PARCSR=5555)
      parameter ( NNX=10 )
      parameter ( NNY=10 )
      parameter ( NNZ=10 )
      parameter (VECLEN=NNX*NNY*NNZ)

      integer             num_procs, myid

      integer             dim
      integer             nx, ny, nz
      integer             Px, Py, Pz
      integer             bx, by, bz
      double precision    cx, cy, cz
      integer             n_pre, n_post
      integer             solver_id
      integer             debug_flag, ioutdat

      integer             zero, one
      parameter           (zero = 0, one = 1)
      integer             maxiter, num_iterations
      integer             generate_matrix, generate_rhs

      double precision    tol, pc_tol, convtol
      parameter           (pc_tol = 0.0)
      double precision    final_res_norm
      
c     parameters for BoomerAMG
      integer             hybrid, coarsen_type, measure_type
      integer             cycle_type
      integer             smooth_num_sweep
      integer             num_grid_sweeps(4)
      integer             grid_relax_type(1)
      integer             lower_ngs(1), lower_grt(1)
      integer             lower_rw(1)
      integer             upper_ngs(1), upper_grt(1)
      integer             upper_rw(1)
      integer             stride_ngs(1), stride_grt(1)
      integer             stride_rw(1)
      integer             refindex_ngs(1), refindex_grt(1)
      integer             refindex_rw(1)
      double precision    strong_threshold, trunc_factor
      double precision    max_row_sum
      data                max_row_sum /1.0/

      double precision    values(4)

      integer             p, q, r

      integer             ierr

      integer             i
      integer             first_local_row, last_local_row
      integer             first_local_col, last_local_col
      integer             indices(MAXZONS)
      double precision    vals0(MAXZONS)
      double precision    vals1(MAXZONS)

c     Babel-interface variables
      integer*8 bHYPRE_mpicomm
      integer*8 bHYPRE_parcsr_A
      integer*8 bHYPRE_parcsr_x
      integer*8 bHYPRE_parcsr_b
      integer ierrtmp
      integer local_num_rows, local_num_cols
      integer*8 bHYPRE_AMG
      integer*8 bHYPRE_Vector_x
      integer*8 bHYPRE_Vector_b
      integer*8 bHYPRE_op_A
      integer*8 bHYPRE_num_grid_sweeps
      integer*8 bHYPRE_grid_relax_type
      integer*8 bHYPRE_grid_relax_points
      integer*8 bHYPRE_relax_weight
      integer max_levels
      data   max_levels /25/
      double precision relax_weight(MAXLEVELS)
      integer*8 mpi_comm


c-----------------------------------------------------------------------
c     Initialize MPI
c-----------------------------------------------------------------------

      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)
      mpi_comm = MPI_COMM_WORLD
c     MPI_COMM_WORLD cannot be directly passed through the Babel interface
c     because its byte length is unspecified.
      call bHYPRE_MPICommunicator_CreateF_f( mpi_comm, bHYPRE_mpicomm )

c-----------------------------------------------------------------------
c     Set the former input parameters
c-----------------------------------------------------------------------

      dim = 3

      nx = NNX
      ny = NNX
      nz = NNX

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
      tol = 1.0e-8

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

c     Disadvantage of using this GenerateLaplacian: it does several bHYPRE calls
c     in C which we'd like to test in Fortran.  But we want to use the underlying
c     C function GenerateLaplacian, which returns a HYPRE-level matrix. It's
c     more C function calls to get the data out of it, as double* etc.

      call bHYPRE_IJParCSRMatrix_GenerateLaplacian_f(
     1     bHYPRE_mpicomm, nx, ny, nz, Px, Py, Pz,
     2     p, q, r, values, 4, 7, bHYPRE_parcsr_A )

      call bHYPRE_IJParCSRMatrix_GetLocalRange_f( bHYPRE_parcsr_A,
     &     first_local_row, last_local_row,
     &     first_local_col, last_local_col, ierr)
      local_num_rows = last_local_row - first_local_row + 1
      local_num_cols = last_local_col - first_local_col + 1

      call bHYPRE_IJParCSRMatrix_Print_f(
     1     bHYPRE_parcsr_A, "driver.out", ierrtmp )
      ierr = ierr + ierrtmp

      call bHYPRE_IJParCSRMatrix_Assemble_f( bHYPRE_parcsr_A, ierrtmp )
      ierr = ierr + ierrtmp


c-----------------------------------------------------------------------
c     Set up the rhs and initial guess
c-----------------------------------------------------------------------

      do i = 1, last_local_col - first_local_col + 1
         indices(i) = first_local_col - 1 + i
         vals0(i) = 0.
         vals1(i) = 1.
      enddo

      call bHYPRE_IJParCSRVector_Create_f(
     1     bHYPRE_mpicomm, first_local_col, last_local_col,
     2     bHYPRE_parcsr_x )

      call bHYPRE_IJParCSRVector__create_f( bHYPRE_parcsr_b )
      call bHYPRE_IJParCSRVector_SetCommunicator_f( bHYPRE_parcsr_b,
     1     bHYPRE_mpicomm, ierrtmp )
      ierr = ierr + ierrtmp

      call bHYPRE_IJParCSRVector_SetLocalRange_f( bHYPRE_parcsr_b,
     1      first_local_row, last_local_row, ierrtmp )
      ierr = ierr + ierrtmp

      call bHYPRE_IJParCSRVector_Initialize_f( bHYPRE_parcsr_b,
     1     ierrtmp )
      ierr = ierr + ierrtmp

      call bHYPRE_IJParCSRVector_SetValues_f( bHYPRE_parcsr_b,
     1     local_num_cols, indices, vals1, ierrtmp )
      ierr = ierr + ierrtmp

      call bHYPRE_IJParCSRVector_Assemble_f( bHYPRE_parcsr_b, ierrtmp )
      ierr = ierr + ierrtmp


      call bHYPRE_IJParCSRVector_print_f(
     1     bHYPRE_parcsr_b, "driver.out.b", ierrtmp )
      ierr = ierr + ierrtmp

      call bHYPRE_IJParCSRVector_Initialize_f( bHYPRE_parcsr_x,
     1     ierrtmp )
      ierr = ierr + ierrtmp

      call bHYPRE_IJParCSRVector_SetValues_f( bHYPRE_parcsr_x,
     1     local_num_cols, indices, vals0, ierrtmp )
      ierr = ierr + ierrtmp

      call bHYPRE_IJParCSRVector_Assemble_f( bHYPRE_parcsr_x, ierrtmp )
      ierr = ierr + ierrtmp

      call bHYPRE_IJParCSRVector_print_f(
     1     bHYPRE_parcsr_x, "driver.out.x0", ierrtmp )
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
      ierr = 0

c      print *, 'Solver: AMG'

      call bHYPRE_BoomerAMG__create_f( bHYPRE_AMG )
      call bHYPRE_IJParCSRVector__cast2_f
     1     ( bHYPRE_parcsr_b, "bHYPRE.Vector", bHYPRE_Vector_b )
      call bHYPRE_IJParCSRVector__cast2_f
     1     ( bHYPRE_parcsr_x, "bHYPRE.Vector", bHYPRE_Vector_x )
      call bHYPRE_IJParCSRVector__cast2_f
     1     ( bHYPRE_parcsr_A, "bHYPRE.Operator", bHYPRE_op_A )
      call bHYPRE_BoomerAMG_SetCommunicator_f(
     1     bHYPRE_AMG, bHYPRE_mpicomm, ierrtmp )
      ierr = ierr + ierrtmp
      call bHYPRE_BoomerAMG_SetOperator_f( bHYPRE_AMG, bHYPRE_op_A,
     1     ierrtmp )
      ierr = ierr + ierrtmp
      call bHYPRE_BoomerAMG_SetIntParameter_f(
     1     bHYPRE_AMG, "CoarsenType", hybrid*coarsen_type, ierrtmp )
      ierr = ierr + ierrtmp
      call bHYPRE_BoomerAMG_SetIntParameter_f(
     1     bHYPRE_AMG, "MeasureType", measure_type, ierrtmp )
      ierr = ierr + ierrtmp
      call bHYPRE_BoomerAMG_SetDoubleParameter_f(
     1     bHYPRE_AMG, "StrongThreshold", strong_threshold, ierrtmp )
      ierr = ierr + ierrtmp
      call bHYPRE_BoomerAMG_SetDoubleParameter_f(
     1     bHYPRE_AMG, "TruncFactor", trunc_factor, ierrtmp )
      ierr = ierr + ierrtmp
c     /* note: log output not specified ... */
      call bHYPRE_BoomerAMG_SetPrintLevel_f(
     1     bHYPRE_AMG, ioutdat, ierrtmp )
      ierr = ierr + ierrtmp
      call bHYPRE_BoomerAMG_SetIntParameter_f(
     1     bHYPRE_AMG, "CycleType", cycle_type, ierrtmp )
      ierr = ierr + ierrtmp
      call bHYPRE_BoomerAMG_SetDoubleParameter_f(
     1     bHYPRE_AMG, "Tol", tol, ierrtmp )
      ierr = ierr + ierrtmp

      call bHYPRE_BoomerAMG_InitGridRelaxation_f( bHYPRE_AMG,
     &     bHYPRE_num_grid_sweeps, bHYPRE_grid_relax_type,
     &     bHYPRE_grid_relax_points, coarsen_type, bHYPRE_relax_weight,
     &     MAXLEVELS, ierrtmp )
      ierr = ierr + ierrtmp
      call sidl_int__array_access_f(
     &     bHYPRE_num_grid_sweeps, num_grid_sweeps, lower_ngs,
     &     upper_ngs, stride_ngs, refindex_ngs )
      call sidl_int__array_access_f(
     &     bHYPRE_grid_relax_type, grid_relax_type, lower_grt,
     &     upper_grt, stride_grt, refindex_grt )
      call sidl_double__array_access_f(
     &     bHYPRE_relax_weight, relax_weight, lower_rw,
     &     upper_rw, stride_rw, refindex_rw )

      call bHYPRE_BoomerAMG_SetIntArray1Parameter_f( bHYPRE_AMG,
     1     "NumGridSweeps", num_grid_sweeps(refindex_ngs(1)),
     2     upper_ngs(1)-lower_ngs(1), ierrtmp )
      ierr = ierr + ierrtmp
      call bHYPRE_BoomerAMG_SetIntArray1Parameter_f( bHYPRE_AMG,
     1     "GridRelaxType", grid_relax_type(refindex_grt(1)),
     2     upper_grt(1)-lower_grt(1), ierrtmp )
      ierr = ierr + ierrtmp
      call bHYPRE_BoomerAMG_SetDoubleArray1Parameter_f(
     1     bHYPRE_AMG, "RelaxWeight", relax_weight(refindex_rw(1)),
     2     upper_rw(1)-lower_rw(1), ierrtmp )
      ierr = ierr + ierrtmp
      call bHYPRE_BoomerAMG_SetIntArray2Parameter_f(
     1     bHYPRE_AMG, "GridRelaxPoints", bHYPRE_grid_relax_points,
     2     ierrtmp )
      ierr = ierr + ierrtmp

      call bHYPRE_BoomerAMG_SetIntParameter_f(
     1     bHYPRE_AMG, "MaxLevels", max_levels, ierrtmp )
      ierr = ierr + ierrtmp
      call bHYPRE_BoomerAMG_SetIntParameter_f(
     1     bHYPRE_AMG, "DebugFlag", debug_flag, ierrtmp )
      ierr = ierr + ierrtmp
      call bHYPRE_BoomerAMG_SetDoubleParameter_f(
     1     bHYPRE_AMG, "MaxRowSum", max_row_sum, ierrtmp )
      ierr = ierr + ierrtmp

      call bHYPRE_BoomerAMG_Setup_f(
     1     bHYPRE_AMG, bHYPRE_Vector_b, bHYPRE_Vector_x, ierrtmp )
      ierr = ierr + ierrtmp

      call bHYPRE_BoomerAMG_Apply_f(
     1     bHYPRE_AMG, bHYPRE_Vector_b, bHYPRE_Vector_x, ierrtmp )
      ierr = ierr + ierrtmp



c-----------------------------------------------------------------------
c     Print the solution and other info
c-----------------------------------------------------------------------

      call bHYPRE_IJParCSRVector_print_f(
     1     bHYPRE_parcsr_x, "driver.out.x", ierrtmp )
      ierr = ierr + ierrtmp

      if (myid .eq. 0) then
         call bHYPRE_BoomerAMG_GetNumIterations_f(
     1        bHYPRE_AMG, num_iterations, ierrtmp )
         ierr = ierr + ierrtmp
         call bHYPRE_BoomerAMG_GetRelResidualNorm_f(
     1        bHYPRE_AMG,
     1        final_res_norm, ierrtmp )
         ierr = ierr + ierrtmp
         print *, 'Iterations = ', num_iterations
         print *, 'Final Residual Norm = ', final_res_norm
         print *, 'Error Flag = ', ierr
      endif

c-----------------------------------------------------------------------
c     Finalize things
c-----------------------------------------------------------------------

      call bHYPRE_BoomerAMG_deleteref_f( bHYPRE_AMG );
      call bHYPRE_IJParCSRVector_deleteref_f( bHYPRE_parcsr_x )
      call bHYPRE_IJParCSRVector_deleteref_f( bHYPRE_parcsr_b )
      call bHYPRE_IJParCSRMatrix_deleteref_f( bHYPRE_parcsr_A )
c      call HYPRE_ParCSRMatrixDestroy(A_storage, ierr)
c      call HYPRE_IJVectorDestroy(b, ierr)
c      call HYPRE_IJVectorDestroy(x, ierr)

c     Finalize MPI

      call MPI_FINALIZE(ierr)

      stop
      end

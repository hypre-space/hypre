C-----------------------------------------------------------------------
c Test driver for unstructured matrix interface (structured storage)
c-----------------------------------------------------------------------
 
c-----------------------------------------------------------------------
c Standard 7-point laplacian in 3D with grid and anisotropy determined
c as user settings.
c-----------------------------------------------------------------------

      program test

      implicit none

      include 'mpif.h'

      integer maxzons, maxblks, maxdim

      parameter (MAXZONS=4194304)
      parameter (MAXBLKS=32)
      parameter (MAXDIM=3)

      integer             num_procs, myid

      integer             dim
      integer             nx, ny, nz
      integer             Px, Py, Pz
      integer             bx, by, bz
      double precision    cx, cy, cz
      integer             n_pre, n_post
      integer             solver_id
      integer             precond_id

      integer             hybrid, coarsen_type, measure_type
      integer             ioutdat, cycle_type, num_grid_sweeps
      integer             grid_relax_type, max_levels
      integer             num_rows

      integer             zero, one
      integer             maxiter, dscgmaxiter, pcgmaxiter
      double precision    tol, convtol

      integer             num_iterations
      double precision    final_res_norm, relax_weight
      double precision    strong_threshold, trunc_factor, drop_tol
                     
c     HYPRE_ParCSRMatrix  A
c     HYPRE_ParVector     b
c     HYPRE_ParVector     x

      integer*8           A
      integer*8           b
      integer*8           x

c     HYPRE_Solver        solver
c     HYPRE_Solver        precond

      integer*8           solver
      integer*8           precond
      integer*8           grid_relax_points
      integer*8           row_starts

      double precision    values(4)

      integer             p, q, r
      integer             ierr

      data zero          / 0 /
      data one           / 1 /

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

      solver_id = 3
      max_levels = 25

c-----------------------------------------------------------------------
c     Read options
c-----------------------------------------------------------------------
 
c     open( 5, file='parcsr_linear_solver.in', status='old')
c
c     read( 5, *) dim
c
c     read( 5, *) nx
c     read( 5, *) ny
c     read( 5, *) nz
c
c     read( 5, *) Px
c     read( 5, *) Py
c     read( 5, *) Pz
c
c     read( 5, *) bx
c     read( 5, *) by
c     read( 5, *) bz
c
c     read( 5, *) cx
c     read( 5, *) cy
c     read( 5, *) cz
c
c     read( 5, *) n_pre
c     read( 5, *) n_post
c
c     read( 5, *) solver_id
c
c     close( 5 )

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

c        compute p from Px and myid
         p = mod(myid,Px)

      elseif (dim .eq. 2) then

c        compute p,q from Px, Py and myid
         p = mod(myid,Px)
         q = mod(((myid - p)/Px),Py)

      elseif (dim .eq. 3) then

c        compute p,q,r from Px,Py,Pz and myid
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

c Generate a Dirichlet Laplacian
      call hypre_GenerateLaplacian(MPI_COMM_WORLD, nx, ny, nz,
     &                             Px, Py, Pz, p, q, r, values, A, ierr)

      call HYPRE_PrintParCSRMatrix(A, "driver.out.A", ierr)

c     call HYPRE_NewParCSRMatrix(MPI_COMM_WORLD, gnrows, gncols,
c    &   rstarts, cstarts, ncoloffdg, nonzsdg, nonzsoffdg, A, ierr)

c     call HYPRE_InitializeParCSRMatrix(A, ierr)

      call hypre_ParCSRMatrixGlobalNumRows(A, num_rows, ierr)
      call hypre_ParCSRMatrixRowStarts(A, row_starts, ierr)

c-----------------------------------------------------------------------
c     Set up the rhs and initial guess
c-----------------------------------------------------------------------

      call HYPRE_NewParVector(MPI_COMM_WORLD, num_rows, row_starts,
     &                        b, ierr)
      call hypre_SetParVectorPartitioningO(b, 0, ierr)
      call HYPRE_InitializeParVector(b, ierr)
c Set up a Dirichlet 0 problem
      call hypre_SetParVectorConstantValue(b, 0d0, ierr)
      call HYPRE_PrintParVector(b, "driver.out.b", ierr)

      call HYPRE_NewParVector(MPI_COMM_WORLD, num_rows, row_starts,
     &                        x, ierr)
      call hypre_SetParVectorPartitioningO(x, 0, ierr)
      call HYPRE_InitializeParVector(x, ierr)
c Choose a nonzero initial guess
      call hypre_SetParVectorConstantValue(x, 1d0, ierr)
      call HYPRE_PrintParVector(x, "driver.out.x0", ierr)

c-----------------------------------------------------------------------
c     Solve the linear system
c-----------------------------------------------------------------------

c     General solver parameters, passing hard coded constants
c     will break the interface.
      maxiter = 50
      dscgmaxiter = 100
      pcgmaxiter = 50
      tol = 0.000001
      convtol = 0.9

      if (solver_id .eq. 0 .or. solver_id .eq. 3 .or.
     &    solver_id .eq. 4 .or. solver_id .eq. 7) then
c       Solve the system using preconditioned GMRES

        call HYPRE_ParCSRGMRESInitialize(MPI_COMM_WORLD, solver, ierr)
        call HYPRE_ParCSRGMRESSetMaxIter(solver, maxiter, ierr)
        call HYPRE_ParCSRGMRESSetTol(solver, tol, ierr)
        call HYPRE_ParCSRGMRESSetLogging(solver, solver, one, ierr)

        if (solver_id .eq. 0) then

          print *, 'GMRES'

        else if (solver_id .eq. 3) then

c Set defaults for BoomerAMG
          coarsen_type = 0
          hybrid = 1
          measure_type = 0
          strong_threshold = 0.25
          trunc_factor = 0.0
          cycle_type = 1

          print *, 'AMG preconditioned GMRES'

          precond_id = 2

          call HYPRE_ParAMGInitialize(precond, ierr)

          call HYPRE_ParAMGSetCoarsenType(precond,
     &                                    (hybrid*coarsen_type), ierr)

          call HYPRE_ParAMGSetMeasureType(precond, measure_type, ierr)

          call HYPRE_ParAMGSetStrongThreshold(precond,
     &                                        strong_threshold, ierr)

          call HYPRE_ParAMGSetTruncFactor(precond, trunc_factor, ierr)

          call HYPRE_ParAMGSetLogging(precond, ioutdat,
     &                                "test.out.log", ierr)

          call HYPRE_ParAMGSetMaxIter(precond, 1, ierr)

          call HYPRE_ParAMGSetCycleType(precond, cycle_type, ierr)

          call HYPRE_ParAMGSetNumGridSweeps(precond,
     &                                      num_grid_sweeps, ierr)

          call HYPRE_ParAMGSetGridRelaxType(precond,
     &                                      grid_relax_type, ierr)

          call HYPRE_ParAMGSetRelaxWeight(precond,
     &                                    relax_weight, ierr)

          call HYPRE_ParAMGSetGridRelaxPoints(precond,
     &                                        grid_relax_points, ierr)

          call HYPRE_ParAMGSetMaxLevels(precond,
     &                                  max_levels, ierr)

          call HYPRE_ParCSRGMRESSetPrecond(solver, precond_id,
     &                                     precond, ierr)

        else if (solver_id .eq. 4) then

          print *, 'diagonally scaled GMRES'

          precond_id = 8

          call HYPRE_ParCSRGMRESSetPrecond(solver, precond_id,
     &                                     precond, ierr)

        else if (solver_id .eq. 7) then

          print *, 'PILUT preconditioned GMRES'

          precond_id = 7

          call HYPRE_ParCSRPilutInitialize(MPI_COMM_WORLD,
     &                                     precond, ierr) 

          if (ierr .ne. 0) write(6,*) 'ParCSRPilutInitialize error'

          drop_tol = -1.

          call HYPRE_ParCSRGMRESSetPrecond(solver, precond_id,
     &                                     precond, ierr)

           if (drop_tol .ge. 0.)
     &         call HYPRE_ParCSRPilutSetDropToleran(precond,
     &                                              drop_tol, ierr)

        endif

        call HYPRE_ParCSRGMRESSetup(solver, A, b, x, ierr)

        call HYPRE_ParCSRGMRESSolve(solver, A, b, x, ierr)

        call HYPRE_ParCSRGMRESGetNumIteratio(solver,
     &                                       num_iterations, ierr)

        call HYPRE_ParCSRGMRESGetFinalRelati(solver,
     &                                       final_res_norm, ierr)

        call HYPRE_ParCSRGMRESFinalize(solver, ierr)

      endif

c-----------------------------------------------------------------------
c     Print the solution and other info
c-----------------------------------------------------------------------

      call HYPRE_PrintParVector(x, "driver.out.x", ierr)

      if (myid .eq. 0) then
         print *, 'Iterations = ', num_iterations
         print *, 'Final Residual Norm = ', final_res_norm
      endif

c-----------------------------------------------------------------------
c     Finalize things
c-----------------------------------------------------------------------

      call HYPRE_DestroyParCSRMatrix(A, ierr)

      call HYPRE_DestroyParVector(b, ierr)

      call HYPRE_DestroyParVector(x, ierr)

c     Finalize MPI

      call MPI_FINALIZE(ierr)

      stop
      end

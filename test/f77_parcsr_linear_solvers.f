C-----------------------------------------------------------------------
c Test driver for unstructured matrix interface (structured storage)
c-----------------------------------------------------------------------
 
c-----------------------------------------------------------------------
c Standard 7-point laplacian in 3D with grid and anisotropy determined
c as user settings.
c-----------------------------------------------------------------------

      program test

      include 'mpif.h'

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

      integer             zero, one
      integer             maxiter, dscgmaxiter, pcgmaxiter
      double precision    tol, convtol

      integer             num_iterations
      double precision    final_res_norm
                     
c     HYPRE_ParCSRMatrix  A
c     HYPRE_ParVector  b
c     HYPRE_ParVector  x

      integer*8           A
      integer*8           b
      integer*8           x

c     HYPRE_Solver        solver
c     HYPRE_Solver        precond

      integer*8           solver
      integer*8           precond

      integer             A_num_ghost(6)

      integer             iupper(3,MAXBLKS), ilower(3,MAXBLKS)

      integer             istart(3)

      integer             offsets(3,MAXDIM+1)

      integer             stencil_indices(MAXDIM+1)
      double precision    values((MAXDIM+1)*MAXZONS)

      integer             p, q, r
      integer             nblocks, volume

      integer             i, s, d
      integer             ix, iy, iz, ib
      integer             ierr

      data A_num_ghost   / 0, 0, 0, 0, 0, 0 /
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

      solver_id = 0

      istart(1) = -3
      istart(2) = -3
      istart(3) = -3

c-----------------------------------------------------------------------
c     Read options
c-----------------------------------------------------------------------
 
c     open( 5, file='parcsr_linear_solver.in', status='old')
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
         print *, '  solver ID       = ', solver_id
      endif

c-----------------------------------------------------------------------
c     Compute some grid and processor information
c-----------------------------------------------------------------------

      if (dim .eq. 1) then
         volume  = nx
         nblocks = bx

c        compute p from Px and myid
         p = mod(myid,Px)
      elseif (dim .eq. 2) then
         volume  = nx*ny
         nblocks = bx*by

c        compute p,q from Px, Py and myid
         p = mod(myid,Px)
         q = mod(((myid - p)/Px),Py)
      elseif (dim .eq. 3) then
         volume  = nx*ny*nz
         nblocks = bx*by*bz

c        compute p,q,r from Px,Py,Pz and myid
         p = mod(myid,Px)
         q = mod((( myid - p)/Px),Py)
         r = (myid - (p + Px*q))/(Px*Py)
      endif

c----------------------------------------------------------------------
c    Compute ilower and iupper from (p,q,r), (bx,by,bz), and (nx,ny,nz)
c    and set up the grid structure.
c----------------------------------------------------------------------

      ib = 1
      if (dim .eq. 1) then
         do ix=0,bx-1
            ilower(1,ib) = istart(1) + nx*(bx*p+ix)
            iupper(1,ib) = istart(1) + nx*(bx*p+ix+1)-1
            ib = ib + 1
         enddo
      elseif (dim .eq. 2) then
         do iy=0,by-1
            do ix=0,bx-1
               ilower(1,ib) = istart(1) + nx*(bx*p+ix)
               iupper(1,ib) = istart(1) + nx*(bx*p+ix+1)-1
               ilower(2,ib) = istart(2) + ny*(by*q+iy)
               iupper(2,ib) = istart(2) + ny*(by*q+iy+1)-1
               ib = ib + 1
            enddo
         enddo
      elseif (dim .eq. 3) then
         do iz=0,bz-1
            do iy=0,by-1
               do ix=0,bx-1
                  ilower(1,ib) = istart(1) + nx*(bx*p+ix)
                  iupper(1,ib) = istart(1) + nx*(bx*p+ix+1)-1
                  ilower(2,ib) = istart(2) + ny*(by*q+iy)
                  iupper(2,ib) = istart(2) + ny*(by*q+iy+1)-1
                  ilower(3,ib) = istart(3) + nz*(bz*r+iz)
                  iupper(3,ib) = istart(3) + nz*(bz*r+iz+1)-1
                  ib = ib + 1
               enddo
            enddo
         enddo
      endif 

c----------------------------------------------------------------------
c     Compute the offsets and set up the stencil structure.
c----------------------------------------------------------------------

      if (dim .eq. 1) then
         offsets(1,1) = -1
         offsets(1,2) =  0
      elseif (dim .eq. 2) then
         offsets(1,1) = -1
         offsets(2,1) =  0 
         offsets(1,2) =  0
         offsets(2,2) = -1 
         offsets(1,3) =  0
         offsets(2,3) =  0
      elseif (dim .eq. 3) then
         offsets(1,1) = -1
         offsets(2,1) =  0
         offsets(3,1) =  0
         offsets(1,2) =  0
         offsets(2,2) = -1
         offsets(3,2) =  0 
         offsets(1,3) =  0
         offsets(2,3) =  0
         offsets(3,3) = -1
         offsets(1,4) =  0
         offsets(2,4) =  0
         offsets(3,4) =  0
      endif
 
c-----------------------------------------------------------------------
c     Set up the matrix structure
c-----------------------------------------------------------------------

      call HYPRE_CreateParCSRMatrix(MPI_COMM_WORLD, gnrows, gncols,
     &   rstarts, cstarts, ncoloffdg, nonzsdg, nonzsoffdg)
      call HYPRE_InitializeParCSRMatrix(MPI_COMM_WORLD, gnrows, gncols,
     &   starts, cstarts, ncoloffdg, nonzsdg, nonzsoffdg)

c-----------------------------------------------------------------------
c     Set up the rhs and initial guess
c-----------------------------------------------------------------------

#if 0
      call HYPRE_NewStructVector(MPI_COMM_WORLD, grid, stencil, b, ierr)
      call HYPRE_InitializeStructVector(b, ierr)
      do i=1,volume
         values(i) = 1.0
      enddo
      do ib=1,nblocks
         call HYPRE_SetStructVectorBoxValues(b, ilower(1,ib),
     & iupper(1,ib), values, ierr)
      enddo
      call HYPRE_AssembleStructVector(b, ierr)
c     call HYPRE_PrintStructVector("driver.out.b", b, zero, ierr)

      call HYPRE_NewStructVector(MPI_COMM_WORLD, grid, stencil, x, ierr)
      call HYPRE_InitializeStructVector(x, ierr)
      do i=1,volume
         values(i) = 0.0
      enddo
      do ib=1,nblocks
         call HYPRE_SetStructVectorBoxValues(x, ilower(1,ib),
     & iupper(1,ib), values, ierr)
      enddo
      call HYPRE_AssembleStructVector(x, ierr)
c     call HYPRE_PrintStructVector("driver.out.x0", x, zero, ierr)
#endif
 
      call HYPRE_NewParVector(MPI_COMM_WORLD, gnrows, partitioning, b, ierr)
      call HYPRE_InitializeParVector(b, ierr)
c     call HYPRE_PrintParVector("driver.out.b", b, zero, ierr)

      call HYPRE_NewParVector(MPI_COMM_WORLD, gnrows, partitioning, x, ierr)
      call HYPRE_InitializeParVector(x, ierr)
c     call HYPRE_PrintParVector("driver.out.x0", x, zero, ierr)

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

      if (solver_id .eq. 0 .or. solver_id .eq. 3
     &                     .or. solver_id .eq. 7) then
c       Solve the system using preconditioned GMRES

        call HYPRE_ParCSRGMRESInitialize(MPI_COMM_WORLD, solver, ierr)
        call HYPRE_ParCSRGMRESSetMaxIter(solver, maxiter, ierr)
        call HYPRE_ParCSRGMRESSetTol(solver, tol, ierr)
        call HYPRE_ParCSRGMRESSetRelChange(solver, zero, ierr)
        call HYPRE_ParCSRGMRESSetLogging(solver, solver, one, ierr)

        if (solver_id .eq. 3) then
c         Use BoomerAMG as preconditioner

          call HYPRE_ParAMGInitialize(gmres_precond, ierr)

          call HYPRE_ParAMGSetCoarsenType(gmres_precond,
     &                                    (hybrid*coarsen_type), ierr)

          call HYPRE_ParAMGSetMeasureType(gmres_precond,
     &                                    measure_type, ierr)

          call HYPRE_ParAMGSetStrongThreshold(amg_precond,
     &                                        strong_threshold, ierr)

          call HYPRE_ParAMGSetLogging(gmres_precond,
     &                                ioutdat,
     &                                "test.out.log", ierr)

          call HYPRE_ParAMGSetMaxIter(gmres_precond, 1, ierr)

          call HYPRE_ParAMGSetCycleType(gmres_precond,
     &                                  cycle_type, ierr)

          call HYPRE_ParAMGSetNumGridSweeps(gmres_precond,
     &                                      num_grid_sweeps, ierr)

          call HYPRE_ParAMGSetGridRelaxType(gmres_precond,
     &                                      grid_relax_type, ierr)

          call HYPRE_ParAMGSetRelaxWeight(gmres_precond,
     &                                    relax_weight, ierr)

          call HYPRE_ParAMGSetGridRelaxPoints(gmres_precond,
     &                                        grid_relax_points, ierr)

          call HYPRE_ParAMGSetMaxLevels(gmres_precond,
     &                                  max_levels, ierr)

          call HYPRE_ParCSRPCGSetPrecond(solver,
     &                                   HYPRE_ParAMGSolve,
     &                                   HYPRE_ParAMGSetup,
     &                                   gmres_precond, ierr)

        else if (solver_id .eq. 4)
          pcg_precond = 0

          call HYPRE_ParCSRGMRESSetPrecond(solver,
                                           HYPRE_ParCSRDiagScale,
                                           HYPRE_ParCSRDiagScaleSetup,
                                           gmres_precond, ierr);

        else if (solver_id .eq. 7)

          call HYPRE_ParCSRPilutInitialize(MPI_COMM_WORLD,
     &                                     gmres_precond, ierr) 

            if (ierr .neq. 0)
               write(6,*) 'Error in ParCSRPilutInitialize'

            call  HYPRE_ParCSRGMRESSetPrecond(solver,
                                              HYPRE_ParCSRPilutSolve,
                                              HYPRE_ParCSRPilutSetup,
                                              gmres_precond, ierr);

            if (drop_tol .ge. 0.)
               call HYPRE_ParCSRPilutSetDropTolerance(gmres_precond,
     &                                                drop_tol, ierr)
         endif

         call HYPRE_ParCSRGMRESSetup(solver, A, b, x, ierr)

         call HYPRE_ParCSRGMRESSolve(solver, A, b, x, ierr)

         call HYPRE_ParCSRGMRESGetNumIterations(solver,
     &                                          num_iterations, ierr)

         call HYPRE_ParCSRGMRESGetFinalRelative(solver,
     &                                          final_res_norm, ierr)

         call HYPRE_ParCSRGMRESFinalize(solver, ierr)

      endif

c-----------------------------------------------------------------------
c     Print the solution and other info
c-----------------------------------------------------------------------

c  call HYPRE_PrintParCSRVector("driver.out.x", x, zero, ierr)

      if (myid .eq. 0) then
         print *, 'Iterations = ', num_iterations
         print *, 'Final Relative Residual Norm = ', final_res_norm
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

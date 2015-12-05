cBHEADER**********************************************************************
c Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
c Produced at the Lawrence Livermore National Laboratory.
c This file is part of HYPRE.  See file COPYRIGHT for details.
c
c HYPRE is free software; you can redistribute it and/or modify it under the
c terms of the GNU Lesser General Public License (as published by the Free
c Software Foundation) version 2.1 dated February 1999.
c
c $Revision: 1.5 $
cEHEADER**********************************************************************
c-----------------------------------------------------------------------
c Test driver for unstructured matrix interface (structured storage)
c-----------------------------------------------------------------------
 
c-----------------------------------------------------------------------
c Standard 7-point laplacian in 3D with grid and anisotropy determined
c as user settings.
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
                     
c parameters for BoomerAMG
      integer             hybrid, coarsen_type, measure_type
      integer             cycle_type
      integer             smooth_num_sweep
      integer*8           num_grid_sweeps
      integer*8           num_grid_sweeps2(4)
      integer*8           grid_relax_type
      integer*8           grid_relax_points
      integer*8           relax_weights
      double precision    strong_threshold, trunc_factor, drop_tol
      double precision    max_row_sum
      data                max_row_sum /1.0/

c parameters for ParaSails
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

      integer             dof_func(1000), j

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
c     write(6,*) 'Generate matrix? !0 yes, 0 no (from file)'
      read(5,*) generate_matrix

      if (generate_matrix .eq. 0) then
c       write(6,*) 'What file to use for matrix (<= 31 chars)?'
        read(5,*) matfile_str
        i = 1
  100   if (matfile_str(i:i) .ne. ' ') then
          matfile(i) = matfile_str(i:i)
        else
          goto 200
        endif
        i = i + 1
        goto 100
  200   matfile(i) = char(0)
      endif

c     write(6,*) 'Generate right-hand side? !0 yes, 0 no (from file)'
      read(5,*) generate_rhs

      if (generate_rhs .eq. 0) then
c       write(6,*)
c    &    'What file to use for right-hand side (<= 31 chars)?'
        read(5,*) vecfile_str
        i = 1
  300   if (vecfile_str(i:i) .ne. ' ') then
          vecfile(i) = vecfile_str(i:i)
        else
          goto 400
        endif
        i = i + 1
        goto 300
  400   vecfile(i) = char(0)
      endif

c     write(6,*) 'What solver_id?'
c     write(6,*) '0 AMG, 1 AMG-PCG, 2 DS-PCG, 3 AMG-GMRES, 4 DS-GMRES,'
c     write(6,*) '5 AMG-CGNR, 6 DS-CGNR, 7 PILUT-GMRES, 8 ParaSails-GMRES,'
c     write(6,*) '9 AMG-BiCGSTAB, 10 DS-BiCGSTAB'
      read(5,*) solver_id

      if (solver_id .eq. 7) then
c       write(6,*) 'What drop tolerance?  <0 do not drop'
        read(5,*) drop_tol
      endif
 
c     write(6,*) 'What relative residual norm tolerance?'
      read(5,*) tol

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
      if (generate_matrix .eq. 0) then

        call HYPRE_IJMatrixRead(matfile, MPI_COMM_WORLD,
     &                          HYPRE_PARCSR, A, ierr)

        call HYPRE_IJMatrixGetObject(A, A_storage, ierr)

        call HYPRE_ParCSRMatrixGetLocalRange(A_storage,
     &            first_local_row, last_local_row,
     &            first_local_col, last_local_col, ierr)

      else

        call HYPRE_GenerateLaplacian(MPI_COMM_WORLD, nx, ny, nz,
     &                               Px, Py, Pz, p, q, r, values,
     &                               A_storage, ierr)

        call HYPRE_ParCSRMatrixGetLocalRange(A_storage,
     &            first_local_row, last_local_row,
     &            first_local_col, last_local_col, ierr)

        call HYPRE_IJMatrixCreate(MPI_COMM_WORLD,
     &            first_local_row, last_local_row,
     &            first_local_col, last_local_col, A, ierr)

        call HYPRE_IJMatrixSetObject(A, A_storage, ierr)

        call HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR, ierr)

      endif

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

      call hypre_ParCSRMatrixRowStarts(A_storage, row_starts, ierr)

c-----------------------------------------------------------------------
c     Set up the rhs and initial guess
c-----------------------------------------------------------------------

      if (generate_rhs .eq. 0) then

        call HYPRE_IJVectorRead(vecfile, MPI_COMM_WORLD,
     &                          HYPRE_PARCSR, b, ierr)

        call HYPRE_IJVectorGetObject(b, b_storage, ierr)

      else

        call HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col,
     &                            last_local_col, b, ierr)
        call HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR, ierr)
        call HYPRE_IJVectorInitialize(b, ierr)

c Set up a Dirichlet 0 problem
        do i = 1, last_local_col - first_local_col + 1
          indices(i) = first_local_col - 1 + i
          vals(i) = 1.
        enddo

        call HYPRE_IJVectorSetValues(b,
     &    last_local_col - first_local_col + 1, indices, vals, ierr)

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

      endif

      call HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col,
     &                          last_local_col, x, ierr)
      call HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR, ierr)
      call HYPRE_IJVectorInitialize(x, ierr)
      do i = 1, last_local_col - first_local_col + 1
          indices(i) = first_local_col - 1 + i
          vals(i) = 0.
      enddo
      call HYPRE_IJVectorSetValues(x,
     &  last_local_col - first_local_col + 1, indices, vals, ierr)

c Choose a nonzero initial guess
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

c     General solver parameters, passing hard coded constants
c     will break the interface.

      maxiter = 100
      convtol = 0.9
      debug_flag = 0
      ioutdat = 1

      if (solver_id .eq. 0) then

c Set defaults for BoomerAMG
        maxiter = 500
        coarsen_type = 6
        hybrid = 1
        measure_type = 0
        strong_threshold = 0.25
        trunc_factor = 0.0
        cycle_type = 1
        smooth_num_sweep = 1

        print *, 'Solver: AMG'

        call HYPRE_BoomerAMGCreate(solver, ierr)
        call HYPRE_BoomerAMGSetCoarsenType(solver,
     &                                  (hybrid*coarsen_type), ierr)
        call HYPRE_BoomerAMGSetMeasureType(solver, measure_type, ierr)
        call HYPRE_BoomerAMGSetTol(solver, tol, ierr)
        call HYPRE_BoomerAMGSetStrongThrshld(solver,
     &                                      strong_threshold, ierr)
        call HYPRE_BoomerAMGSetTruncFactor(solver, trunc_factor, ierr)
        call HYPRE_BoomerAMGSetPrintLevel(solver, ioutdat,ierr)
        call HYPRE_BoomerAMGSetPrintFileName(solver,"test.out.log",ierr)
        call HYPRE_BoomerAMGSetMaxIter(solver, maxiter, ierr)
        call HYPRE_BoomerAMGSetCycleType(solver, cycle_type, ierr)

c RDF: Used this to test the fortran interface for SetDofFunc
c        do i = 1, 1000/2
c           j = 2*i-1
c           dof_func(j) = 0
c           j = j + 1
c           dof_func(j) = 1
c        enddo
c        call HYPRE_BoomerAMGSetNumFunctions(solver, 2, ierr)
c        call HYPRE_BoomerAMGSetDofFunc(solver, dof_func, ierr)

c        call HYPRE_BoomerAMGInitGridRelaxatn(num_grid_sweeps,
c     &                                      grid_relax_type,
c     &                                      grid_relax_points,
c     &                                      coarsen_type,
c     &                                      relax_weights,
c     &                                      MAXLEVELS,ierr)
c        num_grid_sweeps2(1) = 1
c        num_grid_sweeps2(2) = 1
c        num_grid_sweeps2(3) = 1
c        num_grid_sweeps2(4) = 1
c        call HYPRE_BoomerAMGSetNumGridSweeps(solver,
c     &                                       num_grid_sweeps2, ierr)
c        call HYPRE_BoomerAMGSetGridRelaxType(solver,
c     &                                       grid_relax_type, ierr)
c        call HYPRE_BoomerAMGSetRelaxWeight(solver,
c     &                                     relax_weights, ierr)
c       call HYPRE_BoomerAMGSetSmoothOption(solver, smooth_option,
c    &                                      ierr)
c       call HYPRE_BoomerAMGSetSmoothNumSwp(solver, smooth_num_sweep,
c    &                                      ierr)
c        call HYPRE_BoomerAMGSetGridRelaxPnts(solver,
c     &                                       grid_relax_points,
c     &                                       ierr)
        call HYPRE_BoomerAMGSetMaxLevels(solver, MAXLEVELS, ierr)
        call HYPRE_BoomerAMGSetMaxRowSum(solver, max_row_sum,
     &                                   ierr)
        call HYPRE_BoomerAMGSetDebugFlag(solver, debug_flag, ierr)
        call HYPRE_BoomerAMGSetup(solver, A_storage, b_storage,
     &                         x_storage, ierr)
        call HYPRE_BoomerAMGSolve(solver, A_storage, b_storage,
     &                         x_storage, ierr)
        call HYPRE_BoomerAMGGetNumIterations(solver, num_iterations, 
     &						ierr)
        call HYPRE_BoomerAMGGetFinalReltvRes(solver, final_res_norm,
     &                                       ierr)
        call HYPRE_BoomerAMGDestroy(solver, ierr)

      endif

      if (solver_id .eq. 3 .or. solver_id .eq. 4 .or.
     &    solver_id .eq. 7 .or. solver_id .eq. 8) then

        maxiter = 100
        k_dim = 5

c       Solve the system using preconditioned GMRES

        call HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, solver, ierr)
        call HYPRE_ParCSRGMRESSetKDim(solver, k_dim, ierr)
        call HYPRE_ParCSRGMRESSetMaxIter(solver, maxiter, ierr)
        call HYPRE_ParCSRGMRESSetTol(solver, tol, ierr)
        call HYPRE_ParCSRGMRESSetLogging(solver, one, ierr)

        if (solver_id .eq. 4) then

          print *, 'Solver: DS-GMRES'

          precond_id = 1
          precond = 0

          call HYPRE_ParCSRGMRESSetPrecond(solver, precond_id,
     &                                     precond, ierr)

        else if (solver_id .eq. 3) then

          print *, 'Solver: AMG-GMRES'

          precond_id = 2

c Set defaults for BoomerAMG
          maxiter = 1
          coarsen_type = 6
          hybrid = 1
          measure_type = 0
          setup_type = 1
          strong_threshold = 0.25
          trunc_factor = 0.0
          cycle_type = 1
          smooth_num_sweep = 1

          call HYPRE_BoomerAMGCreate(precond, ierr)
          call HYPRE_BoomerAMGSetTol(precond, pc_tol, ierr)
          call HYPRE_BoomerAMGSetCoarsenType(precond,
     &                                    (hybrid*coarsen_type), ierr)
          call HYPRE_BoomerAMGSetMeasureType(precond, measure_type, 
     &						ierr)
          call HYPRE_BoomerAMGSetStrongThrshld(precond,
     &                                        strong_threshold, ierr)
          call HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor,
     &                                       ierr)
          call HYPRE_BoomerAMGSetPrintLevel(precond, ioutdat, ierr)
          call HYPRE_BoomerAMGSetPrintFileName(precond, "test.out.log",
     &                                         ierr)
          call HYPRE_BoomerAMGSetMaxIter(precond, maxiter, ierr)
          call HYPRE_BoomerAMGSetCycleType(precond, cycle_type, ierr)
          call HYPRE_BoomerAMGInitGridRelaxatn(num_grid_sweeps,
     &                                        grid_relax_type,
     &                                        grid_relax_points,
     &                                        coarsen_type,
     &                                        relax_weights,
     &                                        MAXLEVELS,ierr)
          call HYPRE_BoomerAMGSetNumGridSweeps(precond,
     &                                         num_grid_sweeps, ierr)
          call HYPRE_BoomerAMGSetGridRelaxType(precond,
     &                                         grid_relax_type, ierr)
          call HYPRE_BoomerAMGSetRelaxWeight(precond,
     &                                       relax_weights, ierr)
c         call HYPRE_BoomerAMGSetSmoothOption(precond, smooth_option,
c    &                                        ierr)
c         call HYPRE_BoomerAMGSetSmoothNumSwp(precond, smooth_num_sweep,
c    &                                        ierr)
          call HYPRE_BoomerAMGSetGridRelaxPnts(precond,
     &                                        grid_relax_points, ierr)
          call HYPRE_BoomerAMGSetMaxLevels(precond,
     &                                  MAXLEVELS, ierr)
          call HYPRE_BoomerAMGSetMaxRowSum(precond, max_row_sum,
     &                                     ierr)
          call HYPRE_ParCSRGMRESSetPrecond(solver, precond_id,
     &                                     precond, ierr)

          call HYPRE_BoomerAMGSetSetupType(precond,setup_type,ierr)
          
        else if (solver_id .eq. 7) then

          print *, 'Solver: Pilut-GMRES'

          precond_id = 3

          call HYPRE_ParCSRPilutCreate(MPI_COMM_WORLD,
     &                                 precond, ierr) 

          if (ierr .ne. 0) write(6,*) 'ParCSRPilutCreate error'

          call HYPRE_ParCSRGMRESSetPrecond(solver, precond_id,
     &                                     precond, ierr)

          if (drop_tol .ge. 0.)
     &        call HYPRE_ParCSRPilutSetDropToleran(precond,
     &                                              drop_tol, ierr)

        else if (solver_id .eq. 8) then

          print *, 'Solver: ParaSails-GMRES'

          precond_id = 4

          call HYPRE_ParaSailsCreate(MPI_COMM_WORLD, precond,
     &                               ierr)
          call HYPRE_ParCSRGMRESSetPrecond(solver, precond_id,
     &                                     precond, ierr)

          sai_threshold = 0.1
          nlevels       = 1
          sai_filter    = 0.1

          call HYPRE_ParaSailsSetParams(precond, sai_threshold,
     &                                  nlevels, ierr)
          call HYPRE_ParaSailsSetFilter(precond, sai_filter, ierr)
          call HYPRE_ParaSailsSetLogging(precond, ioutdat, ierr)

        endif

        call HYPRE_ParCSRGMRESGetPrecond(solver, precond_gotten,
     &                                   ierr)

        if (precond_gotten .ne. precond) then
          print *, 'HYPRE_ParCSRGMRESGetPrecond got bad precond'
          stop
        else
          print *, 'HYPRE_ParCSRGMRESGetPrecond got good precond'
        endif

        call HYPRE_ParCSRGMRESSetup(solver, A_storage, b_storage,
     &                              x_storage, ierr)
        call HYPRE_ParCSRGMRESSolve(solver, A_storage, b_storage,
     &                              x_storage, ierr)
        call HYPRE_ParCSRGMRESGetNumIteratio(solver,
     &                                       num_iterations, ierr)
        call HYPRE_ParCSRGMRESGetFinalRelati(solver,
     &                                       final_res_norm, ierr)

        if (solver_id .eq. 3) then
           call HYPRE_BoomerAMGDestroy(precond, ierr)
        else if (solver_id .eq. 7) then
           call HYPRE_ParCSRPilutDestroy(precond, ierr)
        else if (solver_id .eq. 8) then
           call HYPRE_ParaSailsDestroy(precond, ierr)
        endif

        call HYPRE_ParCSRGMRESDestroy(solver, ierr)

      endif

      if (solver_id .eq. 1 .or. solver_id .eq. 2) then

        maxiter = 500

        call HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, solver, ierr)
        call HYPRE_ParCSRPCGSetMaxIter(solver, maxiter, ierr)
        call HYPRE_ParCSRPCGSetTol(solver, tol, ierr)
        call HYPRE_ParCSRPCGSetTwoNorm(solver, one, ierr)
        call HYPRE_ParCSRPCGSetRelChange(solver, zero, ierr)
        call HYPRE_ParCSRPCGSetPrintLevel(solver, one, ierr)
  
        if (solver_id .eq. 2) then

          print *, 'Solver: DS-PCG'

          precond_id = 1
          precond = 0

          call HYPRE_ParCSRPCGSetPrecond(solver, precond_id,
     &                                   precond, ierr)

        else if (solver_id .eq. 1) then

          print *, 'Solver: AMG-PCG'

          precond_id = 2

c Set defaults for BoomerAMG
          maxiter = 1
          coarsen_type = 6
          hybrid = 1
          measure_type = 0
          setup_type = 1
          strong_threshold = 0.25
          trunc_factor = 0.0
          cycle_type = 1
          smooth_num_sweep = 1

          call HYPRE_BoomerAMGCreate(precond, ierr)
          call HYPRE_BoomerAMGSetTol(precond, pc_tol, ierr)
          call HYPRE_BoomerAMGSetCoarsenType(precond,
     &                                       (hybrid*coarsen_type),
     &                                       ierr)
          call HYPRE_BoomerAMGSetMeasureType(precond, measure_type, 
     &                                       ierr)
          call HYPRE_BoomerAMGSetStrongThrshld(precond,
     &                                         strong_threshold,
     &                                         ierr)
          call HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor,
     &                                       ierr)
          call HYPRE_BoomerAMGSetPrintLevel(precond, ioutdat,ierr)
          call HYPRE_BoomerAMGSetPrintFileName(precond, "test.out.log",
     &                                         ierr)
          call HYPRE_BoomerAMGSetMaxIter(precond, maxiter, ierr)
          call HYPRE_BoomerAMGSetCycleType(precond, cycle_type, ierr)
          call HYPRE_BoomerAMGInitGridRelaxatn(num_grid_sweeps,
     &                                         grid_relax_type,
     &                                         grid_relax_points,
     &                                         coarsen_type,
     &                                         relax_weights,
     &                                         MAXLEVELS, ierr)
          call HYPRE_BoomerAMGSetNumGridSweeps(precond,
     &                                         num_grid_sweeps, ierr)
          call HYPRE_BoomerAMGSetGridRelaxType(precond,
     &                                         grid_relax_type, ierr)
          call HYPRE_BoomerAMGSetRelaxWeight(precond,
     &                                       relax_weights, ierr)
c         call HYPRE_BoomerAMGSetSmoothOption(precond, smooth_option,
c    &                                        ierr)
c         call HYPRE_BoomerAMGSetSmoothNumSwp(precond,
c    &                                        smooth_num_sweep,
c    &                                        ierr)
          call HYPRE_BoomerAMGSetGridRelaxPnts(precond,
     &                                         grid_relax_points, ierr)
          call HYPRE_BoomerAMGSetMaxLevels(precond, MAXLEVELS, ierr)
          call HYPRE_BoomerAMGSetMaxRowSum(precond, max_row_sum,
     &                                     ierr)

          call HYPRE_ParCSRPCGSetPrecond(solver, precond_id,
     &                                   precond, ierr)

        endif

        call HYPRE_ParCSRPCGGetPrecond(solver,precond_gotten,ierr)

        if (precond_gotten .ne. precond) then
          print *, 'HYPRE_ParCSRPCGGetPrecond got bad precond'
          stop
        else
          print *, 'HYPRE_ParCSRPCGGetPrecond got good precond'
        endif

        call HYPRE_ParCSRPCGSetup(solver, A_storage, b_storage,
     &                            x_storage, ierr)
        call HYPRE_ParCSRPCGSolve(solver, A_storage, b_storage,
     &                            x_storage, ierr)
        call HYPRE_ParCSRPCGGetNumIterations(solver, num_iterations,
     &                                       ierr)
        call HYPRE_ParCSRPCGGetFinalRelative(solver, final_res_norm,
     &                                       ierr)

        if (solver_id .eq. 1) then
          call HYPRE_BoomerAMGDestroy(precond, ierr)
        endif

        call HYPRE_ParCSRPCGDestroy(solver, ierr)

      endif

      if (solver_id .eq. 5 .or. solver_id .eq. 6) then

        maxiter = 1000

        call HYPRE_ParCSRCGNRCreate(MPI_COMM_WORLD, solver, ierr)
        call HYPRE_ParCSRCGNRSetMaxIter(solver, maxiter, ierr)
        call HYPRE_ParCSRCGNRSetTol(solver, tol, ierr)
        call HYPRE_ParCSRCGNRSetLogging(solver, one, ierr)

        if (solver_id .eq. 6) then

          print *, 'Solver: DS-CGNR'

          precond_id = 1
          precond = 0

          call HYPRE_ParCSRCGNRSetPrecond(solver, precond_id,
     &                                    precond, ierr)

        else if (solver_id .eq. 5) then 

          print *, 'Solver: AMG-CGNR'

          precond_id = 2

c Set defaults for BoomerAMG
          maxiter = 1
          coarsen_type = 6
          hybrid = 1
          measure_type = 0
          setup_type = 1
          strong_threshold = 0.25
          trunc_factor = 0.0
          cycle_type = 1
          smooth_num_sweep = 1

          call HYPRE_BoomerAMGCreate(precond, ierr)
          call HYPRE_BoomerAMGSetTol(precond, pc_tol, ierr)
          call HYPRE_BoomerAMGSetCoarsenType(precond,
     &                                       (hybrid*coarsen_type),
     &                                       ierr)
          call HYPRE_BoomerAMGSetMeasureType(precond, measure_type, 
     &                                       ierr)
          call HYPRE_BoomerAMGSetStrongThrshld(precond,
     &                                         strong_threshold, ierr)
          call HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor,
     &                                       ierr)
          call HYPRE_BoomerAMGSetPrintLevel(precond, ioutdat,ierr)
          call HYPRE_BoomerAMGSetPrintFileName(precond, "test.out.log",
     &                                         ierr)
          call HYPRE_BoomerAMGSetMaxIter(precond, maxiter, ierr)
          call HYPRE_BoomerAMGSetCycleType(precond, cycle_type, ierr)
          call HYPRE_BoomerAMGInitGridRelaxatn(num_grid_sweeps,
     &                                         grid_relax_type,
     &                                         grid_relax_points,
     &                                         coarsen_type,
     &                                         relax_weights,
     &                                         MAXLEVELS,ierr)
          call HYPRE_BoomerAMGSetNumGridSweeps(precond,
     &                                         num_grid_sweeps, ierr)
          call HYPRE_BoomerAMGSetGridRelaxType(precond,
     &                                         grid_relax_type, ierr)
          call HYPRE_BoomerAMGSetRelaxWeight(precond,
     &                                       relax_weights, ierr)
c         call HYPRE_BoomerAMGSetSmoothOption(precond, smooth_option,
c    &                                        ierr)
c         call HYPRE_BoomerAMGSetSmoothNumSwp(precond, smooth_num_sweep,
c    &                                        ierr)
          call HYPRE_BoomerAMGSetGridRelaxPnts(precond,
     &                                         grid_relax_points,
     &                                         ierr)
          call HYPRE_BoomerAMGSetMaxLevels(precond, MAXLEVELS, ierr)
          call HYPRE_BoomerAMGSetMaxRowSum(precond, max_row_sum,
     &                                     ierr)

          call HYPRE_ParCSRCGNRSetPrecond(solver, precond_id, precond,
     &                                    ierr)
        endif

        call HYPRE_ParCSRCGNRGetPrecond(solver,precond_gotten,ierr)

        if (precond_gotten .ne. precond) then
          print *, 'HYPRE_ParCSRCGNRGetPrecond got bad precond'
          stop
        else
          print *, 'HYPRE_ParCSRCGNRGetPrecond got good precond'
        endif

        call HYPRE_ParCSRCGNRSetup(solver, A_storage, b_storage,
     &                             x_storage, ierr)
        call HYPRE_ParCSRCGNRSolve(solver, A_storage, b_storage,
     &                             x_storage, ierr)
        call HYPRE_ParCSRCGNRGetNumIteration(solver, num_iterations,
     &                                      ierr)
        call HYPRE_ParCSRCGNRGetFinalRelativ(solver, final_res_norm,
     &                                       ierr)

        if (solver_id .eq. 5) then
          call HYPRE_BoomerAMGDestroy(precond, ierr)
        endif 

        call HYPRE_ParCSRCGNRDestroy(solver, ierr)

      endif

      if (solver_id .eq. 9 .or. solver_id .eq. 10) then

        maxiter = 1000

        call HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, solver, ierr)
        call HYPRE_ParCSRBiCGSTABSetMaxIter(solver, maxiter, ierr)
        call HYPRE_ParCSRBiCGSTABSetTol(solver, tol, ierr)
        call HYPRE_ParCSRBiCGSTABSetLogging(solver, one, ierr)

        if (solver_id .eq. 10) then

          print *, 'Solver: DS-BiCGSTAB'

          precond_id = 1
          precond = 0

          call HYPRE_ParCSRBiCGSTABSetPrecond(solver, precond_id,
     &                                        precond, ierr)

        else if (solver_id .eq. 9) then

          print *, 'Solver: AMG-BiCGSTAB'

          precond_id = 2

c Set defaults for BoomerAMG
          maxiter = 1
          coarsen_type = 6
          hybrid = 1
          measure_type = 0
          setup_type = 1
          strong_threshold = 0.25
          trunc_factor = 0.0
          cycle_type = 1
          smooth_num_sweep = 1

          call HYPRE_BoomerAMGCreate(precond, ierr)
          call HYPRE_BoomerAMGSetTol(precond, pc_tol, ierr)
          call HYPRE_BoomerAMGSetCoarsenType(precond,
     &                                       (hybrid*coarsen_type),
     &                                       ierr)
          call HYPRE_BoomerAMGSetMeasureType(precond, measure_type, 
     &                                       ierr)
          call HYPRE_BoomerAMGSetStrongThrshld(precond,
     &                                         strong_threshold,
     &                                         ierr)
          call HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor,
     &                                       ierr)
          call HYPRE_BoomerAMGSetPrintLevel(precond, ioutdat,ierr)
          call HYPRE_BoomerAMGSetPrintFileName(precond, "test.out.log",
     &                                         ierr)
          call HYPRE_BoomerAMGSetMaxIter(precond, maxiter, ierr)
          call HYPRE_BoomerAMGSetCycleType(precond, cycle_type, ierr)
          call HYPRE_BoomerAMGInitGridRelaxatn(num_grid_sweeps,
     &                                         grid_relax_type,
     &                                         grid_relax_points,
     &                                         coarsen_type,
     &                                         relax_weights,
     &                                         MAXLEVELS, ierr)
          call HYPRE_BoomerAMGSetNumGridSweeps(precond,
     &                                         num_grid_sweeps, ierr)
          call HYPRE_BoomerAMGSetGridRelaxType(precond,
     &                                         grid_relax_type, ierr)
          call HYPRE_BoomerAMGSetRelaxWeight(precond,
     &                                       relax_weights, ierr)
c         call HYPRE_BoomerAMGSetSmoothOption(precond, smooth_option,
c    &                                        ierr)
c         call HYPRE_BoomerAMGSetSmoothNumSwp(precond, smooth_num_sweep,
c    &                                        ierr)
          call HYPRE_BoomerAMGSetGridRelaxPnts(precond,
     &                                         grid_relax_points, ierr)
          call HYPRE_BoomerAMGSetMaxLevels(precond, MAXLEVELS, ierr)
          call HYPRE_BoomerAMGSetMaxRowSum(precond, max_row_sum,
     &                                     ierr)

          call HYPRE_ParCSRBiCGSTABSetPrecond(solver, precond_id,
     &                                        precond,
     &                                        ierr)

        endif

        call HYPRE_ParCSRBiCGSTABGetPrecond(solver,precond_gotten,ierr)

        if (precond_gotten .ne. precond) then
          print *, 'HYPRE_ParCSRBiCGSTABGetPrecond got bad precond'
          stop
        else
          print *, 'HYPRE_ParCSRBiCGSTABGetPrecond got good precond'
        endif

        call HYPRE_ParCSRBiCGSTABSetup(solver, A_storage, b_storage,
     &                                 x_storage, ierr)
        call HYPRE_ParCSRBiCGSTABSolve(solver, A_storage, b_storage,
     &                                 x_storage, ierr)
        call HYPRE_ParCSRBiCGSTABGetNumIter(solver,
     &                                      num_iterations,
     &                                      ierr)
        call HYPRE_ParCSRBiCGSTABGetFinalRel(solver,
     &                                       final_res_norm,
     &                                       ierr)

        if (solver_id .eq. 9) then
          call HYPRE_BoomerAMGDestroy(precond, ierr)
        endif

        call HYPRE_ParCSRBiCGSTABDestroy(solver, ierr)

      endif

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

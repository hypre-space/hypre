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
c Test driver for structured matrix interface (structured storage)
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
                     
c     HYPRE_StructMatrix  A
c     HYPRE_StructVector  b
c     HYPRE_StructVector  x

      integer*8           A
      integer*8           b
      integer*8           x

c     HYPRE_StructSolver  solver
c     HYPRE_StructSolver  precond

      integer*8           solver
      integer*8           precond

c     HYPRE_StructGrid    grid
c     HYPRE_StructStencil stencil

      integer*8           grid
      integer*8           stencil

      double precision    dxyz(3)

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
 
c     open( 5, file='struct_linear_solver.in', status='old')
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
         print *, '  solver ID       = ', solver_id
      endif

c-----------------------------------------------------------------------
c     Set up dxyz for PFMG solver
c-----------------------------------------------------------------------

      dxyz(1) = dsqrt(1.d0 / cx)
      dxyz(2) = dsqrt(1.d0 / cy)
      dxyz(3) = dsqrt(1.d0 / cz)

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

      call HYPRE_StructGridCreate(MPI_COMM_WORLD, dim, grid, ierr)
      do ib=1,nblocks
         call HYPRE_StructGridSetExtents(grid, ilower(1,ib),
     & iupper(1,ib), ierr)
      enddo
      call HYPRE_StructGridAssemble(grid, ierr)

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
 
      call HYPRE_StructStencilCreate(dim, (dim+1), stencil, ierr) 
      do s=1,dim+1
         call HYPRE_StructStencilSetElement(stencil, (s - 1),
     & offsets(1,s), ierr)
      enddo

c-----------------------------------------------------------------------
c     Set up the matrix structure
c-----------------------------------------------------------------------

      do i=1,dim
         A_num_ghost(2*i - 1) = 1
         A_num_ghost(2*i) = 1
      enddo
 
      call HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil,
     & A, ierr)
      call HYPRE_StructMatrixSetSymmetric(A, 1, ierr)
      call HYPRE_StructMatrixSetNumGhost(A, A_num_ghost, ierr)
      call HYPRE_StructMatrixInitialize(A, ierr)

c-----------------------------------------------------------------------
c     Set the coefficients for the grid
c-----------------------------------------------------------------------

      do s=1,(dim + 1)
         stencil_indices(s) = s - 1
      enddo

      do i=1,(dim + 1)*volume,(dim + 1)
         if (dim .eq. 1) then
            values(i  ) = -cx
            values(i+1) = 2.0*(cx)
         elseif (dim .eq. 2) then
            values(i  ) = -cx
            values(i+1) = -cy
            values(i+2) = 2.0*(cx+cy)
         elseif (dim .eq. 3) then
            values(i  ) = -cx
            values(i+1) = -cy
            values(i+2) = -cz
            values(i+3) = 2.0*(cx+cy+cz)
         endif
      enddo

      do ib=1,nblocks
         call HYPRE_StructMatrixSetBoxValues(A, ilower(1,ib),
     & iupper(1,ib), (dim+1), stencil_indices, values, ierr)
      enddo

c-----------------------------------------------------------------------
c     Zero out stencils reaching to real boundary
c-----------------------------------------------------------------------

      do i=1,volume
         values(i) = 0.0
      enddo
      do d=1,dim
         do ib=1,nblocks
            if( ilower(d,ib) .eq. istart(d) ) then
               i = iupper(d,ib)
               iupper(d,ib) = istart(d)
               stencil_indices(1) = d - 1
               call HYPRE_StructMatrixSetBoxValues(A, ilower(1,ib),
     & iupper(1,ib), 1, stencil_indices, values, ierr)
               iupper(d,ib) = i
            endif
         enddo
      enddo

      call HYPRE_StructMatrixAssemble(A, ierr)
c     call HYPRE_StructMatrixPrint(A, zero, ierr)
c     call HYPRE_StructMatrixPrint("driver.out.A", A, zero, ierr)

c-----------------------------------------------------------------------
c     Set up the rhs and initial guess
c-----------------------------------------------------------------------

      call HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, b, ierr)
      call HYPRE_StructVectorInitialize(b, ierr)
      do i=1,volume
         values(i) = 1.0
      enddo
      do ib=1,nblocks
         call HYPRE_StructVectorSetBoxValues(b, ilower(1,ib),
     & iupper(1,ib), values, ierr)
      enddo
      call HYPRE_StructVectorAssemble(b, ierr)
c     call HYPRE_StructVectorPrint("driver.out.b", b, zero, ierr)

      call HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, x, ierr)
      call HYPRE_StructVectorInitialize(x, ierr)
      do i=1,volume
         values(i) = 0.0
      enddo
      do ib=1,nblocks
         call HYPRE_StructVectorSetBoxValues(x, ilower(1,ib),
     & iupper(1,ib), values, ierr)
      enddo
      call HYPRE_StructVectorAssemble(x, ierr)
c     call HYPRE_StructVectorPrint(x, zero, ierr)
c     call HYPRE_StructVectorPrint("driver.out.x0", x, zero, ierr)
 
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

      if (solver_id .eq. 0) then
c        Solve the system using SMG

         call HYPRE_StructSMGCreate(MPI_COMM_WORLD, solver, ierr)
         call HYPRE_StructSMGSetMemoryUse(solver, zero, ierr)
         call HYPRE_StructSMGSetMaxIter(solver, maxiter, ierr)
         call HYPRE_StructSMGSetTol(solver, tol, ierr)
         call HYPRE_StructSMGSetRelChange(solver, zero, ierr)
         call HYPRE_StructSMGSetNumPreRelax(solver, n_pre, ierr)
         call HYPRE_StructSMGSetNumPostRelax(solver, n_post, ierr)
         call HYPRE_StructSMGSetLogging(solver, one, ierr)
         call HYPRE_StructSMGSetup(solver, A, b, x, ierr)


         call HYPRE_StructSMGSolve(solver, A, b, x, ierr)

         call HYPRE_StructSMGGetNumIterations(solver, num_iterations,
     & ierr)
         call HYPRE_StructSMGGetFinalRelative(solver, final_res_norm,
     & ierr)
         call HYPRE_StructSMGDestroy(solver, ierr)
      elseif (solver_id .eq. 1) then
c        Solve the system using PFMG

         call HYPRE_StructPFMGCreate(MPI_COMM_WORLD, solver, ierr)
         call HYPRE_StructPFMGSetMaxIter(solver, maxiter, ierr)
         call HYPRE_StructPFMGSetTol(solver, tol, ierr)
         call HYPRE_StructPFMGSetRelChange(solver, zero, ierr)
c        weighted Jacobi = 1; red-black GS = 2
         call HYPRE_StructPFMGSetRelaxType(solver, one, ierr)
         call HYPRE_StructPFMGSetNumPreRelax(solver, n_pre, ierr)
         call HYPRE_StructPFMGSetNumPostRelax(solver, n_post, ierr)
c        call HYPRE_StructPFMGSetDxyz(solver, dxyz, ierr)
         call HYPRE_StructPFMGSetLogging(solver, one, ierr)
         call HYPRE_StructPFMGSetup(solver, A, b, x, ierr)

         call HYPRE_StructPFMGSolve(solver, A, b, x, ierr)

         call HYPRE_StructPFMGGetNumIteration(solver, num_iterations,
     & ierr)
         call HYPRE_StructPFMGGetFinalRelativ(solver, final_res_norm,
     & ierr)
         call HYPRE_StructPFMGDestroy(solver, ierr)
      elseif ((solver_id .gt. 9) .and. (solver_id .lt. 20)) then
c        Solve the system using CG

         precond_id = -1
         call HYPRE_StructPCGCreate(MPI_COMM_WORLD, solver, ierr)
         call HYPRE_StructPCGSetMaxIter(solver, maxiter, ierr)
         call HYPRE_StructPCGSetTol(solver, tol, ierr)
         call HYPRE_StructPCGSetTwoNorm(solver, one, ierr)
         call HYPRE_StructPCGSetRelChange(solver, zero, ierr)
         call HYPRE_StructPCGSetLogging(solver, one, ierr)

         if (solver_id .eq. 10) then
c           use symmetric SMG as preconditioner
            precond_id = 0
            maxiter = 1
            tol = 0.0

            call HYPRE_StructSMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call HYPRE_StructSMGSetMemoryUse(precond, zero, ierr)
            call HYPRE_StructSMGSetMaxIter(precond, maxiter, ierr)
            call HYPRE_StructSMGSetTol(precond, tol, ierr)
            call HYPRE_StructSMGSetNumPreRelax(precond, n_pre, ierr)
            call HYPRE_StructSMGSetNumPostRelax(precond, n_post, ierr)
            call HYPRE_StructSMGSetLogging(precond, zero, ierr)

            call HYPRE_StructPCGSetPrecond(solver, precond_id, precond,
     & ierr)
         elseif (solver_id .eq. 11) then
c           use symmetric PFMG as preconditioner
            precond_id = 1
            maxiter = 1
            tol = 0.0

            call HYPRE_StructPFMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call HYPRE_StructPFMGSetMaxIter(precond, maxiter, ierr)
            call HYPRE_StructPFMGSetTol(precond, tol, ierr)
c           weighted Jacobi = 1; red-black GS = 2
            call HYPRE_StructPFMGSetRelaxType(precond, one, ierr)
            call HYPRE_StructPFMGSetNumPreRelax(precond, n_pre, ierr)
            call HYPRE_StructPFMGSetNumPostRelax(precond, n_post, ierr)
c           call HYPRE_StructPFMGSetDxyz(precond, dxyz, ierr)
            call HYPRE_StructPFMGSetLogging(precond, zero, ierr)

            call HYPRE_StructPCGSetPrecond(solver, precond_id, precond,
     & ierr)
         elseif (solver_id .eq. 18) then
c           use diagonal scaling as preconditioner
            precond_id = 8
            precond = zero

            call HYPRE_StructPCGSetPrecond(solver, precond_id, precond,
     & ierr)
         elseif (solver_id .eq. 19) then
c           use diagonal scaling as preconditioner
            precond_id = 9

            call HYPRE_StructPCGSetPrecond(solver, precond_id, precond,
     & ierr)
         endif

         call HYPRE_StructPCGSetup(solver, A, b, x, ierr)

         call HYPRE_StructPCGSolve(solver, A, b, x, ierr)

         call HYPRE_StructPCGGetNumIterations(solver, num_iterations,
     & ierr)
         call HYPRE_StructPCGGetFinalRelative(solver, final_res_norm,
     & ierr)
         call HYPRE_StructPCGDestroy(solver, ierr)

         if (solver_id .eq. 10) then
            call HYPRE_StructSMGDestroy(precond, ierr)
         elseif (solver_id .eq. 11) then
            call HYPRE_StructPFMGDestroy(precond, ierr)
         endif
      elseif ((solver_id .gt. 19) .and. (solver_id .le. 30)) then
c        Solve the system using Hybrid

         precond_id = -1
         call HYPRE_StructHybridCreate(MPI_COMM_WORLD, solver, ierr)
         call HYPRE_StructHybridSetDSCGMaxIte(solver, dscgmaxiter, ierr)
         call HYPRE_StructHybridSetPCGMaxIter(solver, pcgmaxiter, ierr)
         call HYPRE_StructHybridSetTol(solver, tol, ierr)
         call HYPRE_StructHybridSetConvergenc(solver, convtol, ierr)
         call HYPRE_StructHybridSetTwoNorm(solver, one, ierr)
         call HYPRE_StructHybridSetRelChange(solver, zero, ierr)
         call HYPRE_StructHybridSetLogging(solver, one, ierr)

         if (solver_id .eq. 20) then
c           use symmetric SMG as preconditioner
            precond_id = 0
            maxiter = 1
            tol = 0.0

            call HYPRE_StructSMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call HYPRE_StructSMGSetMemoryUse(precond, zero, ierr)
            call HYPRE_StructSMGSetMaxIter(precond, maxiter, ierr)
            call HYPRE_StructSMGSetTol(precond, tol, ierr)
            call HYPRE_StructSMGSetNumPreRelax(precond, n_pre, ierr)
            call HYPRE_StructSMGSetNumPostRelax(precond, n_post, ierr)
            call HYPRE_StructSMGSetLogging(precond, zero, ierr)
         elseif (solver_id .eq. 21) then
c           use symmetric PFMG as preconditioner
            precond_id = 1
            maxiter = 1
            tol = 0.0

            call HYPRE_StructPFMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call HYPRE_StructPFMGSetMaxIter(precond, maxiter, ierr)
            call HYPRE_StructPFMGSetTol(precond, tol, ierr)
c           weighted Jacobi = 1; red-black GS = 2
            call HYPRE_StructPFMGSetRelaxType(precond, one, ierr)
            call HYPRE_StructPFMGSetNumPreRelax(precond, n_pre, ierr)
            call HYPRE_StructPFMGSetNumPostRelax(precond, n_post, ierr)
c           call HYPRE_StructPFMGSetDxyz(precond, dxyz, ierr)
            call HYPRE_StructPFMGSetLogging(precond, zero, ierr)
         endif

         call HYPRE_StructHybridSetPrecond(solver, precond_id, precond,
     & ierr)

         call HYPRE_StructHybridSetup(solver, A, b, x, ierr)

         call HYPRE_StructHybridSolve(solver, A, b, x, ierr)

         call HYPRE_StructHybridGetNumIterati(solver, num_iterations,
     & ierr)
         call HYPRE_StructHybridGetFinalRelat(solver, final_res_norm,
     & ierr)
         call HYPRE_StructHybridDestroy(solver, ierr)

         if (solver_id .eq. 20) then
            call HYPRE_StructSMGDestroy(precond, ierr)
         elseif (solver_id .eq. 21) then
            call HYPRE_StructPFMGDestroy(precond, ierr)
         endif
      endif

c-----------------------------------------------------------------------
c     Print the solution and other info
c-----------------------------------------------------------------------

c  call HYPRE_StructVectorPrint("driver.out.x", x, zero, ierr)

      if (myid .eq. 0) then
         print *, 'Iterations = ', num_iterations
         print *, 'Final Relative Residual Norm = ', final_res_norm
      endif

c-----------------------------------------------------------------------
c     Finalize things
c-----------------------------------------------------------------------

      call HYPRE_StructGridDestroy(grid, ierr)
      call HYPRE_StructStencilDestroy(stencil, ierr)
      call HYPRE_StructMatrixDestroy(A, ierr)
      call HYPRE_StructVectorDestroy(b, ierr)
      call HYPRE_StructVectorDestroy(x, ierr)

c     Finalize MPI

      call MPI_FINALIZE(ierr)

      stop
      end

c-----------------------------------------------------------------------
c Test driver for structured matrix interface (structured storage)
c-----------------------------------------------------------------------
 
c-----------------------------------------------------------------------
c Standard 7-point laplacian in 3D with grid and anisotropy determined
c as user settings.
c-----------------------------------------------------------------------

      program test

      include 'mpif.h'

      integer  Hypre_Box_Constructor
      integer  Hypre_StructGrid_Constructor
      integer  Hypre_StructStencil_Constructor
      integer  Hypre_StructMatrixBuilder_Constructor
      integer  Hypre_StructVectorBuilder_Constructor
      integer  Hypre_MPI_Com_Constructor
      integer  Hypre_StructSMG_Constructor

      integer  Hypre_Box_Setup
      integer  Hypre_StructGrid_SetGridExtents
      integer  Hypre_StructGrid_SetParameterIntArray
      integer  Hypre_StructGrid_Setup
      integer  Hypre_StructStencil_SetElement
      integer  Hypre_StructMatrixBuilder_Start
      integer  Hypre_StructMatrixBuilder_SetBoxValues
      integer  Hypre_StructMatrixBuilder_Setup
      integer  Hypre_StructMatrixBuilder_GetConstructedObject
      integer  Hypre_StructSMG_Apply
      integer  Hypre_StructSMG_GetConvergenceInfo
      integer  Hypre_StructSMG_Setup
      integer  Hypre_StructSMG_SetParameterdouble
      integer  Hypre_StructSMG_SetParameterInt
      integer  Hypre_StructVectorBuilder_Start
      integer  Hypre_StructVectorBuilder_Setup
      integer  Hypre_StructVectorBuilder_SetBoxValues
      integer  Hypre_StructVectorBuilder_GetConstructedObject
      integer  Hypre_Vector_castTo
      integer  Hypre_LinearOperator_castTo


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

c$$$      integer*8           A
c$$$      integer*8           b
c$$$      integer*8           x

c      Hypre_StructMatrix  A
c      Hypre_StructVector  b
c      Hypre_StructVector  x
      integer  A
      integer  b
      integer  x

c     HYPRE_StructSolver  solver
c     HYPRE_StructSolver  precond

c$$$      integer*8           solver
c$$$      integer*8           precond

c      Hypre_StructSolver  solver
c      Hypre_StructSolver  precond
c      Hypre_StructJacobi  solver_SJ
c      Hypre_StructSMG     solver_SMG
c      Hypre_PCG           solver_PCG
      integer  solver
      integer  precond
      integer  solver_SMG

c     HYPRE_StructGrid    grid
c     HYPRE_StructStencil stencil

c$$$      integer*8           grid
c$$$      integer*8           stencil

c      Hypre_StructuredGrid  grid
c      Hypre_StructStencil   stencil
      integer  grid
      integer  stencil

c      Hypre_StructMatrixBuilder MatBuilder
c      Hypre_StructVectorBuilder VecBuilder
c      Hypre_LinearOperator lo_test
c      Hypre_StructMatrix  A_SM
c      Hypre_LinearOperator A_LO
c      Hypre_StructVector  b_SV
c      Hypre_Vector  b_V
c      Hypre_StructVector  x_SV
c      Hypre_Vector  x_V
c      Hypre_MPI_Com       comm
c      Hypre_Box           box(MAXBLKS)
c      Hypre_Box           bbox
      integer MatBuilder
      integer VecBuilder
      integer  A_SM
      integer A_LO
      integer  b_SV
      integer  b_V
      integer  x_SV
      integer  x_V
      integer  comm
      integer  box(MAXBLKS)
      integer  bbox
      integer             symmetric

c      array1int arroffsets
c      array1int intvals
c      array1int intvals_lo
c      array1int intvals_hi
c      array1double doubvals
c      array1int num_ghost
c      array1int periodic_arr

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

      double precision    dtemp
      integer             itemp2
      integer             itemp3

      integer             periodic(3)

      data A_num_ghost   / 0, 0, 0, 0, 0, 0 /
      data zero          / 0 /
      data one           / 1 /
c      data ierr          / 0 /
      ierr = 0

c-----------------------------------------------------------------------
c     Initialize MPI
c-----------------------------------------------------------------------

      call MPI_INIT(ierr)

      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)

      comm = Hypre_MPI_Com_Constructor( MPI_COMM_WORLD )

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
            box(ib) = Hypre_Box_Constructor( ilower(1,ib), 1, dim,
     1           iupper(1,ib), 1, dim, dim )
            ierr = ierr + Hypre_Box_Setup( box(ib) )
            ib = ib + 1
         enddo
      elseif (dim .eq. 2) then
         do iy=0,by-1
            do ix=0,bx-1
               ilower(1,ib) = istart(1) + nx*(bx*p+ix)
               iupper(1,ib) = istart(1) + nx*(bx*p+ix+1)-1
               ilower(2,ib) = istart(2) + ny*(by*q+iy)
               iupper(2,ib) = istart(2) + ny*(by*q+iy+1)-1
               box(ib) = Hypre_Box_Constructor( ilower(1,ib), 1, dim,
     1              iupper(1,ib), 1, dim, dim )
               ierr = ierr + Hypre_Box_Setup( box(ib) )
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
                  box(ib) = Hypre_Box_Constructor( ilower(1,ib), 1,
     1                 dim, iupper(1,ib), 1, dim, dim )
                  ierr = ierr + Hypre_Box_Setup( box(ib) )
                  ib = ib + 1
               enddo
            enddo
         enddo
      endif 

      grid = Hypre_StructGrid_Constructor( comm, dim )
c      call HYPRE_StructGridCreate(MPI_COMM_WORLD, dim, grid, ierr)
      do ib=1,nblocks
         ierr = ierr + Hypre_StructGrid_SetGridExtents( grid, box(ib) )
c         call HYPRE_StructGridSetExtents(grid, ilower(1,ib),
c     & iupper(1,ib), ierr)
      enddo

      periodic(1) = 0
      periodic(2) = 0
      periodic(3) = 0
      ierr = ierr + Hypre_StructGrid_SetParameterIntArray( grid,
     1   "periodic", periodic, 0, 3, 8 )
      ierr = ierr + Hypre_StructGrid_Setup( grid )

c      call HYPRE_StructGridAssemble(grid, ierr)

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
 
      stencil = Hypre_StructStencil_Constructor( dim, (dim+1) )
c      call HYPRE_StructStencilCreate(dim, (dim+1), stencil, ierr) 
      do s=1,dim+1
         ierr = ierr +  Hypre_StructStencil_SetElement( stencil, (s-1),
     1        offsets(1,s) )
c         call HYPRE_StructStencilSetElement(stencil, (s - 1),
c     & offsets(1,s), ierr)
      enddo

c-----------------------------------------------------------------------
c     Set up the matrix structure
c-----------------------------------------------------------------------

      do i=1,dim
         A_num_ghost(2*i - 1) = 1
         A_num_ghost(2*i) = 1
      enddo

      symmetric = 1
      itemp2 = 2*dim
      MatBuilder = Hypre_StructMatrixBuilder_Constructor( grid,
     1     stencil, symmetric, A_num_ghost, zero, itemp2 )
      ierr = ierr + Hypre_StructMatrixBuilder_Start( MatBuilder, grid,
     1     stencil, symmetric, A_num_ghost, zero, itemp2 )
c      call HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil,
c     & A, ierr)
c      call HYPRE_StructMatrixSetSymmetric(A, 1, ierr)
c      call HYPRE_StructMatrixSetNumGhost(A, A_num_ghost, ierr)
c      call HYPRE_StructMatrixInitialize(A, ierr)

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

      itemp2 = dim+1
      itemp3 = (dim+1)*volume
      do ib=1,nblocks
         ierr = ierr +  Hypre_StructMatrixBuilder_SetBoxValues(
     1        MatBuilder, box(ib), stencil_indices, zero, itemp2,
     2        values, zero, itemp3 )
c         call HYPRE_StructMatrixSetBoxValues(A, ilower(1,ib),
c     & iupper(1,ib), (dim+1), stencil_indices, values, ierr)
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
               isave = iupper(d,ib)
               iupper(d,ib) = istart(d)
               bbox = Hypre_Box_Constructor( ilower(1,ib), 1, dim,
     1              iupper(1,ib), 1, dim, dim )
               stencil_indices(1) = d - 1
               ierr = ierr + Hypre_StructMatrixBuilder_SetBoxValues(
     1              MatBuilder, bbox, stencil_indices, zero, one,
     2              values, zero, volume )
c               call HYPRE_StructMatrixSetBoxValues(A, ilower(1,ib),
c     & iupper(1,ib), 1, stencil_indices, values, ierr)
               iupper(d,ib) = isave
               call Hypre_Box_deletereference( bbox )
            endif
         enddo
      enddo

      ierr = ierr + Hypre_StructMatrixBuilder_Setup( MatBuilder )
      ierr = ierr + Hypre_StructMatrixBuilder_GetConstructedObject(
     1      MatBuilder, A_LO )
      A_SM = Hypre_LinearOperator_castTo( A_LO, "Hypre.StructMatrix" )
c      call HYPRE_StructMatrixAssemble(A, ierr)
c     call HYPRE_StructMatrixPrint("driver.out.A", A, zero, ierr)

c-----------------------------------------------------------------------
c     Set up the rhs and initial guess
c-----------------------------------------------------------------------

      VecBuilder = Hypre_StructVectorBuilder_Constructor( grid )
      ierr = ierr + Hypre_StructVectorBuilder_Start( VecBuilder, grid )
c      call HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, stencil,
c     & b, ierr)
c      call HYPRE_StructVectorInitialize(b, ierr)
      do i=1,volume
         values(i) = 1.0
      enddo
      do ib=1,nblocks
         ierr = ierr + Hypre_StructVectorBuilder_SetBoxValues(
     1        VecBuilder, box(ib), values )
c         call HYPRE_StructVectorSetBoxValues(b, ilower(1,ib),
c     & iupper(1,ib), values, ierr)
      enddo
      ierr = ierr + Hypre_StructVectorBuilder_Setup( VecBuilder )
      ierr = ierr + Hypre_StructVectorBuilder_GetConstructedObject(
     1    VecBuilder, b_V )
      b_SV = Hypre_Vector_castTo( b_V, "Hypre.StructVector" )
c      call HYPRE_StructVectorAssemble(b, ierr)
c     call HYPRE_StructVectorPrint("driver.out.b", b, zero, ierr)

      ierr = ierr + Hypre_StructVectorBuilder_Start( VecBuilder, grid )
c      call HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, stencil,
c     & x, ierr)
c      call HYPRE_StructVectorInitialize(x, ierr)
      do i=1,volume
         values(i) = 0.0
      enddo
      do ib=1,nblocks
         ierr = ierr + Hypre_StructVectorBuilder_SetBoxValues(
     1        VecBuilder, box(ib), values )
c         call HYPRE_StructVectorSetBoxValues(x, ilower(1,ib),
c     & iupper(1,ib), values, ierr)
      enddo
      ierr = ierr + Hypre_StructVectorBuilder_Setup( VecBuilder )
      ierr = ierr + Hypre_StructVectorBuilder_GetConstructedObject(
     1    VecBuilder, x_V )
      x_SV = Hypre_Vector_castTo( x_V, "Hypre.StructVector" )
c      call HYPRE_StructVectorAssemble(x, ierr)
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

         solver_SMG = Hypre_StructSMG_Constructor( comm )

         ierr = ierr + Hypre_StructSMG_SetParameterInt( solver_SMG,
     1       "memory use", 0 )
         ierr = ierr + Hypre_StructSMG_SetParameterInt( solver_SMG,
     1       "max iter", 50 )
         dtemp = 1.0e-6
         ierr = ierr + Hypre_StructSMG_SetParameterdouble( solver_SMG,
     1       "tol", dtemp )
c  The following line doesn't work, passes 7.4228448038894e-51 :
c         ierr = ierr + Hypre_StructSMG_SetParameterdouble( solver_SMG,
c     1       "tol", 1.0e-6 )
         ierr = ierr + Hypre_StructSMG_SetParameterInt( solver_SMG,
     1        "rel change", 0 )
         ierr = ierr + Hypre_StructSMG_SetParameterInt( solver_SMG,
     1        "num prerelax", n_pre )
         ierr = ierr + Hypre_StructSMG_SetParameterInt( solver_SMG,
     1        "num postrelax", n_post )
         ierr = ierr + Hypre_StructSMG_SetParameterInt( solver_SMG,
     1        "logging", 1 )

         ierr = ierr+Hypre_StructSMG_Setup( solver_SMG, A_LO, b_V, x_V )

c         call HYPRE_StructSMGCreate(MPI_COMM_WORLD, solver, ierr)
c         call HYPRE_StructSMGSetMemoryUse(solver, zero, ierr)
c         call HYPRE_StructSMGSetMaxIter(solver, maxiter, ierr)
c         call HYPRE_StructSMGSetTol(solver, tol, ierr)
c         call HYPRE_StructSMGSetRelChange(solver, zero, ierr)
c         call HYPRE_StructSMGSetNumPreRelax(solver, n_pre, ierr)
c         call HYPRE_StructSMGSetNumPostRelax(solver, n_post, ierr)
c         call HYPRE_StructSMGSetLogging(solver, one, ierr)
c         call HYPRE_StructSMGSetup(solver, A, b, x, ierr)

         ierr = ierr + Hypre_StructSMG_Apply( solver_SMG, b_V, x_V )
c         call HYPRE_StructSMGSolve(solver, A, b, x, ierr)

         ierr = ierr + Hypre_StructSMG_GetConvergenceInfo( solver_SMG,
     1        "num iterations", dtemp )
         num_iterations = dtemp
c ... TO DO: dtemp should be rounded !!! <<<<<<<<<<<<<<<<<<<<<<<
         ierr = ierr + Hypre_StructSMG_GetConvergenceInfo( solver_SMG,
     1        "final relative residual norm", final_res_norm )
c         call HYPRE_StructSMGGetNumIterations(solver, num_iterations,
c     & ierr)
c         call HYPRE_StructSMGGetFinalRelative(solver, final_res_norm,
c     & ierr)

c jfp The following line was commented-out so we don't see a bug elsewhere >>>>>
c jfp I should restore the following line and fix the bug!!! >>>>>>>>>>>>>>>>>>>
c         call Hypre_StructSMG_deletereference( solver_SMG )
c         call HYPRE_StructSMGDestroy(solver, ierr)

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
      elseif ((solver_id .gt. 9) .and. (solver_id .le. 20)) then
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

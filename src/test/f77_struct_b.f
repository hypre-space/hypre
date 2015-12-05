cBHEADER**********************************************************************
c Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
c Produced at the Lawrence Livermore National Laboratory.
c This file is part of HYPRE.  See file COPYRIGHT for details.
c
c HYPRE is free software; you can redistribute it and/or modify it under the
c terms of the GNU Lesser General Public License (as published by the Free
c Software Foundation) version 2.1 dated February 1999.
c
c $Revision: 1.4 $
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
                     
      integer*8  A
      integer*8  b, b_V
      integer*8  x, x_V
      integer*8  solver
      integer*8  precond
      integer*8  solver_SMG
      integer*8  grid
      integer*8  stencil
      integer*8 bHYPRE_mpicomm
      integer*8 mpi_comm
      integer             symmetric
      integer*8  except
c     ... except is for Babel exceptions, which we shall ignore

      double precision    dxyz(3)
      integer             A_num_ghost(6)
      integer             iupper(3,MAXBLKS), ilower(3,MAXBLKS)
      integer             istart(3), iend(3)
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
      ierr = 0

c-----------------------------------------------------------------------
c     Initialize MPI
c-----------------------------------------------------------------------

      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)
      mpi_comm = MPI_COMM_WORLD
c     MPI_COMM_WORLD cannot be directly passed through the Babel interface
c     because its byte length is unspecified.
      call bHYPRE_MPICommunicator_CreateF_f( mpi_comm, bHYPRE_mpicomm,
     1     except )

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
      iend(1) = istart(1) + nx - 1 
      iend(2) = istart(2) + ny - 1 
      iend(3) = istart(3) + nz - 1

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

      call bHYPRE_StructGrid_Create_f( bHYPRE_mpicomm, dim, grid,
     1                                 except )
      do ib=1,nblocks
         call bHYPRE_StructGrid_SetExtents_f( grid, ilower(1,ib),
     1        iupper(1,ib), dim, ierr, except )
      enddo

      periodic(1) = 0
      periodic(2) = 0
      periodic(3) = 0
      call bHYPRE_StructGrid_SetPeriodic_f( grid, periodic, dim,
     1     ierr, except )
      call bHYPRE_StructGrid_Assemble_f( grid, ierr, except )

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
 
      call bHYPRE_StructStencil_Create_f( dim, dim+1, stencil, except )

      do s=1,dim+1
         call bHYPRE_StructStencil_SetElement_f( stencil, (s-1),
     1        offsets(1,s), dim, ierr, except )
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
      call bHYPRE_StructMatrix_Create_f( bHYPRE_mpicomm, grid, stencil,
     1     A, except )
      call bHYPRE_StructMatrix_SetSymmetric_f( A, symmetric,
     2     ierr, except )
      call bHYPRE_StructMatrix_SetNumGhost_f( A, A_num_ghost, 2*dim,
     1     ierr, except )

      call bHYPRE_StructMatrix_Initialize_f( A, ierr, except )

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
         call bHYPRE_StructMatrix_SetBoxValues_f( A,
     1        ilower(1,ib), iupper(1,ib), dim, dim+1,
     2        stencil_indices,  values, (dim+1)*volume, ierr, except )
      enddo

c-----------------------------------------------------------------------
c     Zero out stencils reaching to real boundary, then assemble.
c-----------------------------------------------------------------------

      call SetStencilBndry( A, ilower, iupper,
     1     istart, iend, nblocks, dim, periodic, ierr )

      call bHYPRE_StructMatrix_Assemble_f( A, ierr, except )

c-----------------------------------------------------------------------
c     Set up the rhs and initial guess
c-----------------------------------------------------------------------

      call bHYPRE_StructVector_Create_f( bHYPRE_mpicomm, grid,
     1     b, except )
      call bHYPRE_StructVector_Initialize_f( b, ierr, except )
      do i=1,volume
         values(i) = 1.0
      enddo
      do ib=1,nblocks
         call bHYPRE_StructVector_SetBoxValues_f( b,
     1        ilower(1,ib), iupper(1,ib), dim, values, volume,
     2        ierr, except )
      enddo
      call bHYPRE_StructVector_Assemble_f( b, ierr, except )
      call bHYPRE_Vector__cast_f( b, b_V, except )

      call bHYPRE_StructVector_Create_f( bHYPRE_mpicomm, grid,
     1     x, except )
      call bHYPRE_StructVector_Initialize_f( x, ierr, except )
      do i=1,volume
         values(i) = 0.0
      enddo
      do ib=1,nblocks
         call bHYPRE_StructVector_SetBoxValues_f( x,
     1        ilower(1,ib), iupper(1,ib), dim, values, volume,
     2        ierr, except )
      enddo
      call bHYPRE_StructVector_Assemble_f( x, ierr, except )
      call bHYPRE_Vector__cast_f( x, x_V, except )

 
c-----------------------------------------------------------------------
c     Solve the linear system
c-----------------------------------------------------------------------

c     General solver parameters
      maxiter = 50
      dscgmaxiter = 100
      pcgmaxiter = 50
      tol = 0.000001
      convtol = 0.9

      if (solver_id .eq. 0) then
c        Solve the system using SMG

         call bHYPRE_StructSMG_Create_f( bHYPRE_mpicomm, A, solver_SMG,
     1        except )

         call bHYPRE_StructSMG_SetIntParameter_f( solver_SMG,
     1        "MemoryUse", 0, ierr, except )
         call bHYPRE_StructSMG_SetIntParameter_f( solver_SMG,
     1        "MaxIter", 50, ierr, except )
         call bHYPRE_StructSMG_SetDoubleParameter_f( solver_SMG,
     1        "Tol", tol, ierr, except )
         call bHYPRE_StructSMG_SetIntParameter_f( solver_SMG,
     1        "RelChange", 0, ierr, except )
         call bHYPRE_StructSMG_SetIntParameter_f( solver_SMG,
     1        "NumPrerelax", n_pre, ierr, except )
         call bHYPRE_StructSMG_SetIntParameter_f( solver_SMG,
     1        "NumPostrelax", n_post, ierr, except )
         call bHYPRE_StructSMG_SetIntParameter_f( solver_SMG,
     1        "PrintLevel", 1, ierr, except )
         call bHYPRE_StructSMG_SetIntParameter_f( solver_SMG,
     1        "Logging", 1, ierr, except )

         call bHYPRE_StructSMG_Setup_f( solver_SMG, b_V, x_V,
     1        ierr, except )

         call bHYPRE_StructSMG_Apply_f( solver_SMG, b_V, x_V,
     1        ierr, except )

         call bHYPRE_StructSMG_GetIntValue_f( solver_SMG,
     1        "NumIterations", num_iterations, ierr, except )
         call bHYPRE_StructSMG_GetDoubleValue_f( solver_SMG,
     1        "RelResidualNorm", final_res_norm, ierr, except )

      call bHYPRE_StructSMG_deleteRef_f( solver_SMG, except )

      endif

c-----------------------------------------------------------------------
c     Print the solution and other info
c-----------------------------------------------------------------------

      if (myid .eq. 0) then
         print *, 'Iterations = ', num_iterations
         print *, 'Final Relative Residual Norm = ', final_res_norm
      endif

c-----------------------------------------------------------------------
c     Finalize things
c-----------------------------------------------------------------------

      call bHYPRE_Vector_deleteRef_f( b_V, except )
      call bHYPRE_Vector_deleteRef_f( x_V, except )
      call bHYPRE_StructVector_deleteRef_f( b, except )
      call bHYPRE_StructVector_deleteRef_f( x, except )
      call bHYPRE_StructMatrix_deleteRef_f( A, except )
      call bHYPRE_StructStencil_deleteRef_f( stencil, except )
      call bHYPRE_StructGrid_deleteRef_f( grid, except )
      call bHYPRE_MPICommunicator_deleteRef_f( bHYPRE_mpicomm, except )

c     Finalize MPI

      call MPI_FINALIZE(ierr)

      stop
      end

c-----------------------------------------------------------------------
c this function sets to zero the stencil entries that are on the boundary
c-----------------------------------------------------------------------

      subroutine SetStencilBndry( A_b, ilower, iupper,
     1     istart, iend, nblocks, dim, periodic, ierr )
      parameter (MAXZONS=4194304)
      parameter (MAXBLKS=32)
      integer*8 A_b
      integer ilower(3,MAXBLKS)
      integer iupper(3,MAXBLKS)
      integer istart(3)
      integer iend(3)
      integer dim
      integer nblocks
      integer periodic(3)
      integer ierr

      integer i,j,d,ib
      integer vol
      integer stencil_indices
      double precision values(MAXZONS)
      integer*8  except

      ierr = 0

      do d = 1, dim
         do ib = 1, nblocks
            vol = ( iupper(1,ib)-ilower(1,ib)+1 ) *
     1           ( iupper(2,ib)-ilower(2,ib)+1 ) *
     2           ( iupper(3,ib)-ilower(3,ib)+1 )
            do i = 1, vol
               values(i) = 0.0;
            enddo
            if( ilower(d,ib) .eq. istart(d) ) then
               if( periodic(d) .eq. 0 ) then
                  j = iupper(d,ib)
                  iupper(d,ib) = istart(d)
                  stencil_indices = d - 1

                  call bHYPRE_StructMatrix_SetBoxValues_f( A_b,
     1                 ilower(1,ib), iupper(1,ib), dim, 1,
     2                 stencil_indices,  values, vol, ierr,
     3                 except )
                  iupper(d,ib) = j
               endif
            endif

            if( iupper(d,ib) .eq. iend(d) ) then
               if( periodic(d) .eq. 0 ) then
                  j = ilower(d,ib)
                  ilower(d,ib) = iend(d)
                  stencil_indices = dim + d
                  call bHYPRE_StructMatrix_SetBoxValues_f( A_b,
     1                 ilower(1,ib), iupper(1,ib), dim, 1,
     2                 stencil_indices, values, vol, ierr, except)
                  ilower(d,ib) = j
               endif
            endif
         enddo
      enddo
      
      return
      end


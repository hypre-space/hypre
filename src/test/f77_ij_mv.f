!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!-----------------------------------------------------------------------
! Test driver for unstructured matrix-vector interface
!-----------------------------------------------------------------------
 
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
      double precision    cx, cy, cz

      integer             generate_matrix, generate_vec
      character           matfile(32), vecfile(32)
      character*32        matfile_str, vecfile_str

      integer*8           A, A_storage
      integer*8           x, b

      double precision    values(4)

      integer             p, q, r

      integer             ierr

      integer             i
      integer             first_local_row, last_local_row
      integer             first_local_col, last_local_col
      integer             indices(MAXZONS)

      double precision    vals(MAXZONS)
      double precision    bvals(MAXZONS)
      double precision    xvals(MAXZONS)
      double precision    sum

!-----------------------------------------------------------------------
!     Initialize MPI
!-----------------------------------------------------------------------

      call MPI_INIT(ierr)

      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)

!-----------------------------------------------------------------------
!     Set defaults
!-----------------------------------------------------------------------

      dim = 3

      nx = 10
      ny = 10
      nz = 10

      Px  = num_procs
      Py  = 1
      Pz  = 1

      cx = 1.0
      cy = 1.0
      cz = 1.0

!-----------------------------------------------------------------------
!     Read options
!-----------------------------------------------------------------------
 
!     open( 5, file='parcsr_matrix_vector.in', status='old')
!
!     read( 5, *) dim
!
!     read( 5, *) nx
!     read( 5, *) ny
!     read( 5, *) nz
!
!     read( 5, *) Px
!     read( 5, *) Py
!     read( 5, *) Pz
!
!     read( 5, *) cx
!     read( 5, *) cy
!     read( 5, *) cz
!
!     write(6,*) 'Generate matrix? !0 yes, 0 no (from file)'
      read(5,*) generate_matrix

      if (generate_matrix .eq. 0) then
!       write(6,*) 'What file to use for matrix (<= 32 chars)?'
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

!     write(6,*) 'Generate vector? !0 yes, 0 no (from file)'
      read(5,*) generate_vec

      if (generate_vec .eq. 0) then
!       write(6,*)
!    &    'What file to use for vector (<= 32 chars)?'
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

!     close( 5 )

!-----------------------------------------------------------------------
!     Check a few things
!-----------------------------------------------------------------------

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

!-----------------------------------------------------------------------
!     Print driver parameters
!-----------------------------------------------------------------------

      if (myid .eq. 0) then
         print *, 'Matrix built with these parameters:'
         print *, '  (nx, ny, nz) = (', nx, ',', ny, ',', nz, ')'
         print *, '  (Px, Py, Pz) = (',  Px, ',',  Py, ',',  Pz, ')'
         print *, '  (cx, cy, cz) = (', cx, ',', cy, ',', cz, ')'
         print *, '  dim          = ', dim
      endif

!-----------------------------------------------------------------------
!     Compute some grid and processor information
!-----------------------------------------------------------------------

      if (dim .eq. 1) then

!        compute p from Px and myid
         p = mod(myid,Px)

      elseif (dim .eq. 2) then

!        compute p,q from Px, Py and myid
         p = mod(myid,Px)
         q = mod(((myid - p)/Px),Py)

      elseif (dim .eq. 3) then

!        compute p,q,r from Px,Py,Pz and myid
         p = mod(myid,Px)
         q = mod((( myid - p)/Px),Py)
         r = (myid - (p + Px*q))/(Px*Py)

      endif

!----------------------------------------------------------------------
!     Set up the matrix
!-----------------------------------------------------------------------

      values(2) = -cx
      values(3) = -cy
      values(4) = -cz

      values(1) = 0.0
      if (nx .gt. 1) values(1) = values(1) + 2d0*cx
      if (ny .gt. 1) values(1) = values(1) + 2d0*cy
      if (nz .gt. 1) values(1) = values(1) + 2d0*cz

! Generate a Dirichlet Laplacian
      if (generate_matrix .gt. 0) then

!        Standard 7-point laplacian in 3D with grid and anisotropy
!        determined as user settings.

         call HYPRE_GenerateLaplacian(MPI_COMM_WORLD, nx, ny, nz,
     &                                Px, Py, Pz, p, q, r, values,
     &                                A_storage, ierr)

         call HYPRE_ParCSRMatrixGetLocalRange(A_storage,
     &             first_local_row, last_local_row,
     &             first_local_col, last_local_col, ierr)

         call HYPRE_IJMatrixCreate(MPI_COMM_WORLD,
     &             first_local_row, last_local_row,
     &             first_local_col, last_local_col, A, ierr)

         call HYPRE_IJMatrixSetObject(A, A_storage, ierr)

         if (ierr .ne. 0) write(6,*) 'Matrix object set failed'

         call HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR, ierr)

      else

         call HYPRE_IJMatrixRead(matfile, MPI_COMM_WORLD,
     &                           HYPRE_PARCSR, A, ierr)

         if (ierr .ne. 0) write(6,*) 'Matrix read failed'

         call HYPRE_IJMatrixGetObject(A, A_storage, ierr)

         if (ierr .ne. 0)
     &      write(6,*) 'Matrix object retrieval failed'

         call HYPRE_ParCSRMatrixGetLocalRange(A_storage,
     &             first_local_row, last_local_row,
     &             first_local_col, last_local_col, ierr)

         if (ierr .ne. 0)
     &      write(6,*) 'Matrix local range retrieval failed'

      endif

      matfile(1) = 'm'
      matfile(2) = 'v'
      matfile(3) = '.'
      matfile(4) = 'o'
      matfile(5) = 'u'
      matfile(6) = 't'
      matfile(7) = '.'
      matfile(8) = 'A'
      matfile(9) = char(0)
   
      call HYPRE_IJMatrixPrint(A, matfile, ierr)

      if (ierr .ne. 0) write(6,*) 'Matrix print failed'
  
!-----------------------------------------------------------------------
!     "RHS vector" test
!-----------------------------------------------------------------------
      if (generate_vec .gt. 0) then
        call HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_row,
     &                            last_local_row, b, ierr)

        if (ierr .ne. 0) write(6,*) 'RHS vector creation failed'
  
        call HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR, ierr)

        if (ierr .ne. 0) write(6,*) 'RHS vector object set failed'
  
        call HYPRE_IJVectorInitialize(b, ierr)

        if (ierr .ne. 0) write(6,*) 'RHS vector initialization failed'
  
! Set up a Dirichlet 0 problem
        do i = 1, last_local_row - first_local_row + 1
          indices(i) = first_local_row - 1 + i
          vals(i) = 0.
        enddo
        call HYPRE_IJVectorSetValues(b,
     &    last_local_row - first_local_row + 1, indices, vals, ierr)

        vecfile(1) = 'm'
        vecfile(2) = 'v'
        vecfile(3) = '.'
        vecfile(4) = 'o'
        vecfile(5) = 'u'
        vecfile(6) = 't'
        vecfile(7) = '.'
        vecfile(8) = 'b'
        vecfile(9) = char(0)
   
        call HYPRE_IJVectorPrint(b, vecfile, ierr)

        if (ierr .ne. 0) write(6,*) 'RHS vector print failed'

      else

        call HYPRE_IJVectorRead(vecfile, MPI_COMM_WORLD,
     &                          HYPRE_PARCSR, b, ierr)

        if (ierr .ne. 0) write(6,*) 'RHS vector read failed'

      endif

      do i = 1, last_local_row - first_local_row + 1
        indices(i) = first_local_row - 1 + i
      enddo

      call HYPRE_IJVectorGetValues(b,
     &  last_local_row - first_local_row + 1, indices, bvals, ierr)
  
      if (ierr .ne. 0) write(6,*) 'RHS vector value retrieval failed'
  
!     Set about to modify every other component of b, by adding the
!     negative of the component

      do i = 1, last_local_row - first_local_row + 1, 2
        indices(i) = first_local_row - 1 + i
        vals(i)    = -bvals(i)
      enddo

      call HYPRE_IJVectorAddToValues(b,
     &   1 + (last_local_row - first_local_row)/2, indices, vals, ierr)

      if (ierr .ne. 0) write(6,*) 'RHS vector value addition failed'
  
      do i = 1, last_local_row - first_local_row + 1
        indices(i) = first_local_row - 1 + i
      enddo

      call HYPRE_IJVectorGetValues(b,
     &  last_local_row - first_local_row + 1, indices, bvals, ierr)

      if (ierr .ne. 0) write(6,*) 'RHS vector value retrieval failed'
  
      sum = 0.
      do i = 1, last_local_row - first_local_row + 1, 2
        sum = sum + bvals(i)
      enddo
  
      if (sum .ne. 0.) write(6,*) 'RHS vector value addition error'

!-----------------------------------------------------------------------
!     "Solution vector" test
!-----------------------------------------------------------------------
      call HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col,
     &                          last_local_col, x, ierr)

      if (ierr .ne. 0) write(6,*) 'Solution vector creation failed'
  
      call HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR, ierr)

      if (ierr .ne. 0) write(6,*) 'Solution vector object set failed'
  
      call HYPRE_IJVectorInitialize(x, ierr)

      if (ierr .ne. 0) write(6,*) 'Solution vector initialization',
     &                            ' failed'
  
      do i = 1, last_local_col - first_local_col + 1
          indices(i) = first_local_col - 1 + i
          vals(i) = 0.
      enddo

      call HYPRE_IJVectorSetValues(x,
     &  last_local_col - first_local_col + 1, indices, vals, ierr)

      if (ierr .ne. 0) write(6,*) 'Solution vector value set failed'
  
      vecfile(1)  = 'm'
      vecfile(2)  = 'v'
      vecfile(3)  = '.'
      vecfile(4)  = 'o'
      vecfile(5)  = 'u'
      vecfile(6)  = 't'
      vecfile(7)  = '.'
      vecfile(8)  = 'x'
      vecfile(9) = char(0)
   
      call HYPRE_IJVectorPrint(x, vecfile, ierr)

      if (ierr .ne. 0) write(6,*) 'Solution vector print failed'
  
      indices(1) = last_local_col
      indices(2) = first_local_col
      vals(1) = -99.
      vals(2) = -45.

      call HYPRE_IJVectorAddToValues(x, 2, indices, vals, ierr)

      if (ierr .ne. 0) write(6,*) 'Solution vector value addition',
     &                            ' failed'
  
      do i = 1, last_local_col - first_local_col + 1
        indices(i) = first_local_col - 1 + i
      enddo

      call HYPRE_IJVectorGetValues(x,
     &  last_local_col - first_local_col + 1, indices, xvals, ierr)

      if (ierr .ne. 0) write(6,*) 'Solution vector value retrieval',
     &                            ' failed'
  
      if (xvals(1) .ne. -45.)
     &   write(6,*) 'Solution vector value addition error,',
     &              ' first_local_col'

      if (xvals(last_local_col - first_local_col + 1) .ne. -99.)
     &   write(6,*) 'Solution vector value addition error,',
     &              ' last_local_col'

!-----------------------------------------------------------------------
!     Finalize things
!-----------------------------------------------------------------------

      call HYPRE_ParCSRMatrixDestroy(A_storage, ierr)
      call HYPRE_IJVectorDestroy(b, ierr)
      call HYPRE_IJVectorDestroy(x, ierr)

!     Finalize MPI

      call MPI_FINALIZE(ierr)

      stop
      end

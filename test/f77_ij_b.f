c-----------------------------------------------------------------------
c Test driver for unstructured matrix interface (structured storage)
c-----------------------------------------------------------------------
c jfp: Babel code added.  Inputs hardwired in.  Code using other values
c of the inputs deleted (sometimes).
 
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
c The next two lines may become the only place so far where I have substantively changed the
c original Fortran code. num_grid_sweeps and grid_relax_type are actually pointers
c set in the Fortran-called C code.  One C function sets the pointer.  Then the Fortran
c passes it on to another C function.  But the Babel-based code actually wants to get
c data out of them.  I'm still working on getting this to work...
      integer*8           num_grid_sweeps
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

c Babel-interface variables
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
      integer num_grid_sweeps_f(1)
      integer grid_relax_type_f(1)
c      equivalence ( num_grid_sweeps_f(1), num_grid_sweeps )
c      equivalence ( grid_relax_type_f(1), grid_relax_type )


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
cjfp hardwire it       read(5,*) generate_matrix
      generate_matrix = 1


c     write(6,*) 'Generate right-hand side? !0 yes, 0 no (from file)'
cjfp hardwire it       read(5,*) generate_rhs
      generate_rhs = 1

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
cjfp hardwire it       read(5,*) solver_id
      solver_id = 0

      if (solver_id .eq. 7) then
c       write(6,*) 'What drop tolerance?  <0 do not drop'
        read(5,*) drop_tol
      endif
 
c     write(6,*) 'What relative residual norm tolerance?'
cjfp hardwire it     read(5,*) tol
      tol = 1.0e-6

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
c				   
c      printf("Sparsity known = %d\n", sparsity_known);
c >>> The following is C code to copy the HYPRE matrix into the Hypre. 
c >>> Do it in Fortran
c {
c 
c        /* Create the SIDL arrays */
c        /* Set the indexing into the diag_sizes and offdiag_sizes SIDL arrays
c           to match C conventions */
c        lower[0] = 0;
c        upper[0] = local_num_rows - 1;
c        printf("about to create SIDL array\n");
c        Hypre_row_sizes = SIDL_int__array_create( 1, lower, upper );
c        printf("finished to create SIDL array\n");
      call SIDL_int__array_create1d_f( local_num_rows, Hypre_row_sizes )
c 
c 
c        size = 7;
c 
c        for (i=0; i < local_num_rows; i++)
c        {
c 
c           SIDL_int__array_set1( Hypre_row_sizes, i, size );
c        }
      size = 7
      do i = 1, local_num_rows
         call SIDL_int__array_set1_f( Hypre_row_sizes, i, size )
      enddo

c          printf("about to call SIDL set row sizes\n");
c        ierr = Hypre_IJBuildMatrix_SetRowSizes( Hypre_ij_A, Hypre_row_sizes );
      call Hypre_IJBuildMatrix_SetRowSizes_f(
     1     Hypre_ij_A, Hypre_row_sizes, ierrtmp )
      ierr = ierr + ierrtmp
c           printf("finishedij row sizes\n");
c        SIDL_int__array_destroy( Hypre_row_sizes );
      call SIDL_int__array_deleteReference_f( Hypre_row_sizes)
c 
c        ierr = Hypre_IJBuildMatrix_Initialize( Hypre_ij_A );
      call Hypre_IJBuildMatrix_Initialize_f( Hypre_ij_A, ierrtmp )
      ierr = ierr + ierrtmp

      lower(1) = 0
      upper(1) = 0
      stride(1) = 1
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
c At this point we could destroy A_parcsr if this were a pure Babel-interface
c code.  But as long as the direct HYPRE_ interface is in use, we have to keep
c it around.
c      call HYPRE_ParCSRMatrixDestroy( A_parcsr, ierrtmp )
c      ierr = ierr + ierrtmp
      call Hypre_IJBuildMatrix_GetObject_f(
     1     Hypre_ij_A, Hypre_object, ierrtmp )
      ierr = ierr + ierrtmp
      call Hypre_IJBuildMatrix_deleteReference_f( Hypre_ij_A )
c 
c    The QueryInterface below checks to see if the returned object can
c    return a Hypre.ParCSRMatrix. The "cast" is necessary because of the
c    restrictions of the C language, and is merely to please the compiler.
c    It is the QueryInterface that actually has semantic meaning.
c >>> keep the cast for now, take it out once this works.  This isn't C>>>
      call SIDL_BaseInterface_queryInterface_f(
     1     Hypre_object, "Hypre.ParCSRMatrix", Hypre_object_tmp )
      call SIDL_BaseInterface__cast_f(
     1     Hypre_object_tmp, "Hypre.ParCSRMatrix", Hypre_parcsr_A )
      if ( Hypre_parcsr_A .eq. 0 ) then
         write (6,*) 'Matrix cast/QI failed\n'
         stop
      endif

c 
c    {

c    /* Break encapsulation so that the rest of the driver stays the same */
c    struct Hypre_ParCSRMatrix__data * temp_data;
c jfp This isn't very doable in Fortran.
c    temp_data = Hypre_ParCSRMatrix__get_data( Hypre_parcsr_A );
c    
c    ij_A = temp_data ->ij_A ;
c 
c    ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
c    parcsr_A = (HYPRE_ParCSRMatrix) object;
c    
c    }



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
     1                                first_local_col + i )
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
     1      Hypre_ij_b, Hypre_object, ierrtmp )
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
     1                                first_local_col + i )
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
     1      Hypre_ij_x, Hypre_object, ierrtmp )
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


c  The C program did this:
c      /* Break encapsulation so that the rest of the driver stays the same */
c  That's not so easily done in Fortran.


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

      write (6,*) "solver_id", solver_id
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

c The direct HYPRE interfaces and the Babel interface will have to live together
c until the Babel one is completely working.  (jfp)  Here are the
c HYPRE interface calls.

        call HYPRE_BoomerAMGCreate(solver, ierr)
        call HYPRE_BoomerAMGSetCoarsenType(solver,
     &                                  (hybrid*coarsen_type), ierr)
        call HYPRE_BoomerAMGSetMeasureType(solver, measure_type, ierr)
        call HYPRE_BoomerAMGSetTol(solver, tol, ierr)
        call HYPRE_BoomerAMGSetStrongThrshld(solver,
     &                                      strong_threshold, ierr)
        call HYPRE_BoomerAMGSetTruncFactor(solver, trunc_factor, ierr)
        call HYPRE_BoomerAMGSetPrintLevel(solver, ioutdat,ierr)
c the old Fortran interface isn't handling the string right:
c              call HYPRE_BoomerAMGSetPrintFileName(solver,"test.out.log",ierr)
        call HYPRE_BoomerAMGSetMaxIter(solver, maxiter, ierr)
        call HYPRE_BoomerAMGSetCycleType(solver, cycle_type, ierr)
        call HYPRE_BoomerAMGInitGridRelaxatn(num_grid_sweeps,
     &                                      grid_relax_type,
     &                                      grid_relax_points,
     &                                      coarsen_type,
     &                                      relax_weights,
     &                                      MAXLEVELS,ierr)
        call HYPRE_BoomerAMGSetNumGridSweeps(solver,
     &                                       num_grid_sweeps, ierr)
        call HYPRE_BoomerAMGSetGridRelaxType(solver,
     &                                       grid_relax_type, ierr)
        call HYPRE_BoomerAMGSetRelaxWeight(solver,
     &                                     relax_weights, ierr)
c       call HYPRE_BoomerAMGSetSmoothOption(solver, smooth_option,
c    &                                      ierr)
c       call HYPRE_BoomerAMGSetSmoothNumSwp(solver, smooth_num_sweep,
c    &                                      ierr)
        call HYPRE_BoomerAMGSetGridRelaxPnts(solver,
     &                                       grid_relax_points,
     &                                       ierr)
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

c and here are of the Babel interface calls, adapted from the C/Babel code:
          call Hypre_ParAMG__create_f( Hypre_AMG )
          call Hypre_ParCSRVector__cast_f
     1         ( Hypre_parcsr_b, "Hypre.Vector", Hypre_Vector_b )
          call Hypre_ParCSRVector__cast_f
     1         ( Hypre_parcsr_x, "Hypre.Vector", Hypre_Vector_x )
          call Hypre_ParCSRVector__cast_f
     1         ( Hypre_parcsr_A, "Hypre.Operator", Hypre_op_A )
          call Hypre_ParAMG_SetCommunicator_f(
     1         Hypre_AMG, MPI_COMM_WORLD, ierrtmp )
          ierr = ierr + ierrtmp
          call Hypre_ParAMG_SetOperator_f( Hypre_AMG, Hypre_op_A )
c          write (6,*) "**** before calling Hypre_ParAMGSet*Parameter"
          call Hypre_ParAMG_SetIntParameter_f(
     1         Hypre_AMG, "CoarsenType", hybrid*coarsen_type, ierrtmp )
          ierr = ierr + ierrtmp
          call Hypre_ParAMG_SetIntParameter_f(
     1         Hypre_AMG, "MeasureType", measure_type, ierrtmp )
          ierr = ierr + ierrtmp
          call Hypre_ParAMG_SetDoubleParameter_f(
     1         Hypre_AMG, "StrongThreshold", strong_threshold, ierrtmp )
          ierr = ierr + ierrtmp
          call Hypre_ParAMG_SetDoubleParameter_f(
     1         Hypre_AMG, "TruncFactor", trunc_factor, ierrtmp )
          ierr = ierr + ierrtmp
c      /* note: log output not specified ... */
          call Hypre_ParAMG_SetIntParameter_f(
     1         Hypre_AMG, "PrintLevel", ioutdat, ierrtmp )
          ierr = ierr + ierrtmp
          call Hypre_ParAMG_SetIntParameter_f(
     1         Hypre_AMG, "CycleType", cycle_type, ierrtmp )
          ierr = ierr + ierrtmp
c        dimsl[0] = 0;   dimsu[0] = 4;
c        Hypre_num_grid_sweeps = SIDL_int__array_create( 1, dimsl, dimsu );
c          for ( i=0; i<4; ++i )
c             SIDL_int__array_set1( Hypre_num_grid_sweeps, i, num_grid_sweeps[i] );
c      Hypre_ParAMG_SetIntArrayParameter( Hypre_AMG, "NumGridSweeps", Hypre_num_grid_sweeps );
c        dimsl[0] = 0;   dimsu[0] = 4;
c        Hypre_grid_relax_type = SIDL_int__array_create( 1, dimsl, dimsu );
c        for ( i=0; i<4; ++i )
c           SIDL_int__array_set1( Hypre_grid_relax_type, i, grid_relax_type[i] );
c      Hypre_ParAMG_SetIntArrayParameter( Hypre_AMG, "GridRelaxType", Hypre_grid_relax_type );
          dimsl(1) = 1
          dimsu(1) = 4
          call SIDL_int__array_create1d_f(
     1         4, Hypre_num_grid_sweeps )
          do i = 1, 4
             call SIDL_int__array_set1_f(
     1            Hypre_num_grid_sweeps, i-1, num_grid_sweeps_f(i) )
          enddo
          call Hypre_ParAMG_SetIntArrayParameter_f( Hypre_AMG,
     1         "NumGridSweeps", Hypre_num_grid_sweeps, ierrtmp )
          ierr = ierr + ierrtmp
          dimsl(1) = 1
          dimsu(1) = 4
          call SIDL_int__array_create1d_f(
     1         4, Hypre_grid_relax_type )
          do i = 1, 4
             call SIDL_int__array_set1_f(
     1            Hypre_grid_relax_type, i-1, grid_relax_type_f(i) )
          enddo
          call Hypre_ParAMG_SetIntArrayParameter_f( Hypre_AMG,
     1         "GridRelaxType", Hypre_grid_relax_type, ierrtmp )
          ierr = ierr + ierrtmp

c        dimsl[0] = 0;   dimsu[0] = max_levels;
c        Hypre_relax_weight = SIDL_double__array_create( 1, dimsl, dimsu );
c        for ( i=0; i<max_levels; ++i )
c           SIDL_double__array_set1( Hypre_relax_weight, i, relax_weight[i] );
c      Hypre_ParAMG_SetDoubleArrayParameter( Hypre_AMG, "RelaxWeight", Hypre_relax_weight );
          dimsl(1) = 0
          dimsu(1) = max_levels
          call SIDL_double__array_create1d_f(
     1         max_levels+1, Hypre_relax_weight )
          do i=1, max_levels
c            relax_weight(i)=1.0: simple to set, fine for testing:
             relax_weight(i) = 1.0
          enddo
          do i=1, max_levels
             call SIDL_double__array_set1_f(
     1            Hypre_relax_weight, i, relax_weight(i) )
          enddo
          call Hypre_ParAMG_SetDoubleArrayParameter_f(
     1         Hypre_AMG, "RelaxWeight", Hypre_relax_weight, ierrtmp )
          ierr = ierr + ierrtmp

c        dimsl[0] = 0;   dimsu[0] = max_levels;
c        Hypre_smooth_option = SIDL_int__array_create( 1, dimsl, dimsu );
c        for ( i=0; i<max_levels; ++i )
c           SIDL_int__array_set1( Hypre_smooth_option, i, smooth_option[i] );
c      Hypre_ParAMG_SetIntArrayParameter( Hypre_AMG, "SmoothOption", Hypre_smooth_option );
c smooth_option is obsolete
c          call Hypre_ParAMG_SetIntArrayParameter_f(
c     1         Hypre_AMG, "SmoothOption", Hypre_smooth_option, ierrtmp )
c          ierr = ierr + ierrtmp

c      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "SmoothNumSweep", smooth_num_sweep);
cc          call Hypre_ParAMG_SetIntParameter_f(
cc     1         Hypre_AMG, "SmoothNumSweep", smooth_num_sweep, ierrtmp )
cc          ierr = ierr + ierrtmp
c        dimsl[0] = 0;   dimsl[1] = 0;   dimsu[0] = 4;   dimsu[1] = 4;
c        Hypre_grid_relax_points = SIDL_int__array_create( 2, dimsl, dimsu );
c        for ( i=0; i<4; ++i ) for ( j=0; j<4; ++j )
c           SIDL_int__array_set2( Hypre_grid_relax_points, i, j, grid_relax_points[i][j] );
c      Hypre_ParAMG_SetIntArrayParameter( Hypre_AMG, "GridRelaxPoints", Hypre_grid_relax_points );
cc          dimsl(1) = 0
cc          dimsl(2) = 0
cc          dimsu(1) = 4
cc          dimsu(1) = 4
c      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "MaxLevels", max_levels);
c      Hypre_ParAMG_SetDoubleParameter( Hypre_AMG, "MaxRowSum", max_row_sum);
c      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "DebugFlag", debug_flag);
c      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "Variant", variant);
c      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "Overlap", overlap);
c      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "DomainType", domain_type);
c      Hypre_ParAMG_SetDoubleParameter( Hypre_AMG, "SchwarzRlxWeight", schwarz_rlx_weight);
c      Hypre_ParAMG_SetIntParameter( Hypre_AMG, "NumFunctions", num_functions);
c      if (num_functions > 1) {
c          dimsl[0] = 0;   dimsu[0] = num_functions;
c           Hypre_dof_func = SIDL_int__array_create( 1, dimsl, dimsu );
c           for ( i=0; i<num_functions; ++i )
c              SIDL_int__array_set1( Hypre_dof_func, i, dof_func[i] );
c	 Hypre_ParAMG_SetIntArrayParameter( Hypre_AMG, "DofFunc", Hypre_dof_func );
c      }
c      log_level = 3;
c      Hypre_ParAMG_SetLogging( Hypre_AMG, log_level );
c
c      ierr += Hypre_ParAMG_Setup( Hypre_AMG, Hypre_Vector_b, Hypre_Vector_x );
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


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
c  jfp: this is the current setting, 09oct02

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

c jfp commented-out.  maybe HYPRE_ParCSRPCGGetPrecond isn't working...
c        if (precond_gotten .ne. precond) then
c          print *, 'HYPRE_ParCSRGMRESGetPrecond got bad precond',
c     1          precond, precond_gotten
c          stop
c        else
c          print *, 'HYPRE_ParCSRGMRESGetPrecond got good precond'
c        endif

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

c jfp commented-out.  maybe HYPRE_ParCSRPCGGetPrecond isn't working...
c        call HYPRE_ParCSRPCGGetPrecond(solver,precond_gotten,ierr)
c
c        if (precond_gotten .ne. precond) then
c          print *, 'HYPRE_ParCSRPCGGetPrecond got bad precond'
c          stop
c        else
c          print *, 'HYPRE_ParCSRPCGGetPrecond got good precond'
c        endif

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

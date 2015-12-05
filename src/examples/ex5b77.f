c
c   Example 5
c
c   Interface:    Linear-Algebraic (IJ), Babel-based version in Fortran
c
c   Compile with: make ex5b
c
c   Sample run:   mpirun -np 4 ex5b
c
c   Description:  This example solves the 2-D
c                 Laplacian problem with zero boundary conditions
c                 on an nxn grid.  The number of unknowns is N=n^2.
c                 The standard 5-point stencil is used, and we solve
c                 for the interior nodes only.
c
c                 This example solves the same problem as Example 3.
c                 Available solvers are AMG, PCG, and PCG with AMG or
c                 Parasails preconditioners.

      program ex5b77


      implicit none

      include 'mpif.h'

      integer    MAX_LOCAL_SIZE
      parameter  (MAX_LOCAL_SIZE=123000)

      integer    ierr, ierrtmp
      integer    num_procs, myid
      integer    local_size, extra
      integer    n, solver_id, print_solution, ng
      integer    nnz, ilower, iupper, i
      double precision h, h2
      double precision rhs_values(MAX_LOCAL_SIZE)
      double precision x_values(MAX_LOCAL_SIZE)
      integer    rows(MAX_LOCAL_SIZE)
      integer    cols(5)
      double precision values(5)
      integer    num_iterations
      double precision final_res_norm, tol

      integer*8  bHYPRE_mpicomm
      integer*8  mpi_comm
      integer*8  parcsr_A
      integer*8  op_A
      integer*8  par_b
      integer*8  par_x
      integer*8  vec_b
      integer*8  vec_x
      integer*8  amg_solver
      integer*8  except
c     ... except is for Babel exceptions, which we shall ignore

c-----------------------------------------------------------------------
c     Initialize MPI
c-----------------------------------------------------------------------

      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)
      mpi_comm = MPI_COMM_WORLD
      call bHYPRE_MPICommunicator_CreateF_f( mpi_comm, bHYPRE_mpicomm,
     1      except )

c   Default problem parameters
      n = 33
      solver_id = 0
      print_solution  = 0
      tol = 1.0d-7

c   The input section not implemented yet.

c   Preliminaries: want at least one processor per row
      if ( n*n .lt. num_procs) then
         n = int(sqrt(real(num_procs))) + 1
      endif
c     ng = global no. rows, h = mesh size      
      ng = n*n
      h = 1.0d0/(n+1)
      h2 = h*h

c     Each processor knows only of its own rows - the range is denoted by ilower
c     and upper.  Here we partition the rows. We account for the fact that
c     N may not divide evenly by the number of processors.
      local_size = ng/num_procs
      extra = ng - local_size*num_procs

      ilower = local_size*myid
      ilower = ilower + min(myid, extra)

      iupper = local_size*(myid+1)
      iupper = iupper + min(myid+1, extra)
      iupper = iupper - 1

c     How many rows do I have?
      local_size = iupper - ilower + 1

c     Create the matrix.
c     Note that this is a square matrix, so we indicate the row partition
c     size twice (since number of rows = number of cols)
      call bHYPRE_IJParCSRMatrix_Create_f( bHYPRE_mpicomm, ilower,
     1     iupper, ilower, iupper, parcsr_A, except )

c     op_A will be needed later as a function argument
      call bHYPRE_Operator__cast_f( parcsr_A, op_A, except )

c     Choose a parallel csr format storage (see the User's Manual)
c     Note: Here the HYPRE interface requires a SetObjectType call.
c     I am using the bHYPRE interface in a way which does not because
c     the object type is already specified through the class name.

c     Initialize before setting coefficients
      call bHYPRE_IJParCSRMatrix_Initialize_f( parcsr_A, ierrtmp,
     1     except )

c     Now go through my local rows and set the matrix entries.
c     Each row has at most 5 entries. For example, if n=3:
c
c      A = [M -I 0; -I M -I; 0 -I M]
c      M = [4 -1 0; -1 4 -1; 0 -1 4]
c
c     Note that here we are setting one row at a time, though
c     one could set all the rows together (see the User's Manual).

      do i = ilower, iupper
         nnz = 1

c        The left identity block:position i-n
         if ( (i-n) .ge. 0 ) then
	    cols(nnz) = i-n
	    values(nnz) = -1.0d0
	    nnz = nnz + 1
         endif

c         The left -1: position i-1
         if ( mod(i,n).ne.0 ) then
            cols(nnz) = i-1
            values(nnz) = -1.0d0
            nnz = nnz + 1
         endif

c        Set the diagonal: position i
         cols(nnz) = i
         values(nnz) = 4.0d0
         nnz = nnz + 1

c        The right -1: position i+1
         if ( mod((i+1),n) .ne. 0 ) then
            cols(nnz) = i+1
            values(nnz) = -1.0d0
            nnz = nnz + 1
         endif

c        The right identity block:position i+n
         if ((i+n) .lt. ng ) then
            cols(nnz) = i+n
            values(nnz) = -1.0d0
            nnz = nnz + 1
         endif

c        Set the values for row i
         call bHYPRE_IJParCSRMatrix_SetValues_f(
     1        parcsr_A, 1, nnz-1, i, cols, values, 5, ierrtmp, except )

      enddo


c     Assemble after setting the coefficients
      call bHYPRE_IJParCSRMatrix_Assemble_f( parcsr_A, ierrtmp, except )

c     Create the rhs and solution
      call bHYPRE_IJParCSRVector_Create_f( bHYPRE_mpicomm,
     1     ilower, iupper, par_b, except )
c     vec_b will be needed later for function arguments
      call bHYPRE_Vector__cast_f( par_b, vec_b, except )

      call bHYPRE_IJParCSRVector_Initialize_f( par_b, ierrtmp, except )

      call bHYPRE_IJParCSRVector_Create_f( bHYPRE_mpicomm,
     1     ilower, iupper, par_x, except )
c     vec_x will be needed later for function arguments
      call bHYPRE_Vector__cast_f( par_x, vec_x, except )

      call bHYPRE_IJParCSRVector_Initialize_f( par_x, ierrtmp, except )

c     Set the rhs values to h^2 and the solution to zero
      do i = 1, local_size
         rhs_values(i) = h2
         x_values(i) = 0.0
         rows(i) = ilower + i -1
      enddo
      call bHYPRE_IJParCSRVector_SetValues_f(
     1     par_b, local_size, rows, rhs_values, ierrtmp, except )
      call bHYPRE_IJParCSRVector_SetValues_f(
     1     par_x, local_size, rows, x_values, ierrtmp, except )


      call bHYPRE_IJParCSRVector_Assemble_f( par_b, ierrtmp, except )
      call bHYPRE_IJParCSRVector_Assemble_f( par_x, ierrtmp, except )

c     Choose a solver and solve the system

c      AMG
      if ( solver_id == 0 ) then

c        Create solver
         call bHYPRE_BoomerAMG_Create_f(
     1        bHYPRE_mpicomm, parcsr_A, amg_solver, except )

c        Set some parameters (See Reference Manual for more parameters)
c        PrintLevel=3 means print solve info + parameters
c        CoarsenType=6 means Falgout coarsening
c        RelaxType=3 means Gauss-Seidel/Jacobi hybrid relaxation
         call bHYPRE_BoomerAMG_SetIntParameter_f(
     1        amg_solver, "PrintLevel", 3, ierrtmp, except )
         call bHYPRE_BoomerAMG_SetIntParameter_f(
     1        amg_solver, "CoarsenType", 6, ierrtmp, except )
         call bHYPRE_BoomerAMG_SetIntParameter_f(
     1        amg_solver, "RelaxType", 3, ierrtmp, except )
         call bHYPRE_BoomerAMG_SetIntParameter_f(
     1        amg_solver, "NumSweeps", 1, ierrtmp, except )
         call bHYPRE_BoomerAMG_SetIntParameter_f(
     1        amg_solver, "MaxLevels", 20, ierrtmp, except )
         call bHYPRE_BoomerAMG_SetDoubleParameter_f(
     1        amg_solver, "Tolerance", tol, ierrtmp, except )

c        Now setup and solve!
         call bHYPRE_BoomerAMG_Setup_f(
     1        amg_solver, vec_b, vec_x, ierrtmp, except )
         call bHYPRE_BoomerAMG_Apply_f(
     1        amg_solver, vec_b, vec_x, ierrtmp, except )

c        Run info - needed logging turned on 
         call bHYPRE_BoomerAMG_GetIntValue_f(
     1        amg_solver, "NumIterations", num_iterations, ierrtmp,
     2        except )
         ierr = ierr + ierrtmp
         call bHYPRE_BoomerAMG_GetDoubleValue_f(
     1        amg_solver, "RelResidualNorm", final_res_norm, ierrtmp,
     2        except )

      if (myid .eq. 0) then
         print *
         print *, "Iterations = ", num_iterations
         print *, "Final Relative Residual Norm = ", final_res_norm
         print *
      endif

c     Destroy solver
      call bHYPRE_BoomerAMG_deleteRef_f( amg_solver, except )

      endif

c     The calls of other solvers are not implemented yet.


c     Print the solution
      if ( print_solution .ne. 0 ) then
         call bHYPRE_IJParCSRVector_Print_f( par_x, "ij.out.x", except )
      endif

c     Clean up
      call bHYPRE_Operator_deleteRef_f( op_A, except )
      call bHYPRE_Vector_deleteRef_f( vec_b, except )
      call bHYPRE_Vector_deleteRef_f( vec_x, except )
      call bHYPRE_IJParCSRMatrix_deleteRef_f( parcsr_A, except )
      call bHYPRE_IJParCSRVector_deleteRef_f( par_b, except )
      call bHYPRE_IJParCSRVector_deleteRef_f( par_x, except )
      call bHYPRE_MPICommunicator_deleteRef_f( bHYPRE_mpicomm, except )

c     We need a multi-language equivalent of hypre_assert.
      if ( ierr .ne. 0 ) then
         print *
         print *, "***** Bad ierr = ", ierr
         print *
      endif

c     Finalize MPI
      call MPI_Finalize(ierrtmp)

      stop
      end

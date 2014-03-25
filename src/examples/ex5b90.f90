!
!   Example 5
!
!   Interface:    Linear-Algebraic (IJ), Babel-based version in Fortran90/9x
!
!   Compile with: make ex5b90
!
!   Sample run:   mpirun -np 4 ex5b90
!
!   Description:  This example solves the 2-D
!                 Laplacian problem with zero boundary conditions
!                 on an nxn grid.  The number of unknowns is N=n^2.
!                 The standard 5-point stencil is used, and we solve
!                 for the interior nodes only.
!
!                 This example solves the same problem as Example 3.
!                 Available solvers are AMG, PCG, and PCG with AMG or
!                 Parasails preconditioners.

! HOW TO BUILD:
! Fortran 90 is not yet fully supported by the hypre build system, so this
! is a little more complicated than C or Fortran 77.  These instructions have
! only been tried in one environment so far.
! 1. Make sure you have a Fortran 90 compiler!
! 2. Make sure your MPI library supports Fortran 90, true if it supplies an
!    mpi.mod file and mpif90 compiler-wrapper.
! 3. Set the environment variable FC to your mpif90, the MPI wrapper for your
!    Fortran 90 compiler.
! 4. Install Chasm, which Babel requires for handling Fortran arrays.  See
!    the Babel users manual for more information.
! 5. Set the environment variable CHASMPREFIX, same as your --prefix option
!    in configuring Chasm.
! 6. Run hypre's 'configure --with-babel ...' and make.
! 7. cd babel/bHYPREClient-F90; make
! 8. cd examples; make ex5b90


program ex5b90

  use mpi
!  ... reguires mpi.mod, not available (I think) on my machine's (public) old version
! of mpich, but works with my private latest version of OpenMPI.(JfP)
! If a real Fortran 90 mpi.mod isn't available, the only alternative is to include
! the Fortran 77 one, plus corresponding changes thereafter ...
!#include "mpif.h"

  use sidl_SIDLException
  use bHYPRE_BoomerAMG
  use bHYPRE_MPICommunicator
  use bHYPRE_IJParCSRMatrix
  use bHYPRE_IJParCSRVector

  integer, parameter::  MAX_LOCAL_SIZE = 123000

  integer    ierr, ierrtmp
  integer    num_procs, myid
  integer    local_size, extra
  integer    n, solver_id, print_solution, ng
  integer    nnz, ilower, iupper, i
  real(8)    h, h2
  real(8)    rhs_values(MAX_LOCAL_SIZE)
  real(8)    x_values(MAX_LOCAL_SIZE)
  integer    rows(MAX_LOCAL_SIZE)
  integer    cols(5)
  real(8)    values(5)
  integer    num_iterations
  real(8)    final_res_norm, tol
  integer(8) mpi_comm

  type(bHYPRE_MPICommunicator_t) bHYPRE_mpicomm
  type(bHYPRE_IJParCSRMatrix_t)  parcsr_A
  type(bHYPRE_Operator_t)        op_A
  type(bHYPRE_IJParCSRVector_t)  par_b
  type(bHYPRE_IJParCSRVector_t)  par_x
  type(bHYPRE_Vector_t)          vec_b
  type(bHYPRE_Vector_t)          vec_x
  type(bHYPRE_BoomerAMG_t)       amg_solver
  type(sidl_SIDLException_t)     except
  !     ... except is for Babel exceptions, which we shall ignore

!-----------------------------------------------------------------------
!     Initialize MPI
!-----------------------------------------------------------------------

      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)
      mpi_comm = MPI_COMM_WORLD
      call bHYPRE_MPICommunicator_CreateF_f( mpi_comm, bHYPRE_mpicomm, except )

!   Default problem parameters
      n = 33
      solver_id = 0
      print_solution  = 0
      tol = 1.0d-7

!   The input section not implemented yet.

!   Preliminaries: want at least one processor per row
      if ( n*n .lt. num_procs) then
         n = int(sqrt(real(num_procs))) + 1
      endif
!     ng = global no. rows, h = mesh size      
      ng = n*n
      h = 1.0d0/(n+1)
      h2 = h*h

!     Each processor knows only of its own rows - the range is denoted by ilower
!     and upper.  Here we partition the rows. We account for the fact that
!     N may not divide evenly by the number of processors.
      local_size = ng/num_procs
      extra = ng - local_size*num_procs

      ilower = local_size*myid
      ilower = ilower + min(myid, extra)

      iupper = local_size*(myid+1)
      iupper = iupper + min(myid+1, extra)
      iupper = iupper - 1

!     How many rows do I have?
      local_size = iupper - ilower + 1

!     Create the matrix.
!     Note that this is a square matrix, so we indicate the row partition
!     size twice (since number of rows = number of cols)
      call bHYPRE_IJParCSRMatrix_Create_f( bHYPRE_mpicomm, ilower, iupper, ilower, iupper, &
           parcsr_A, except )

!     op_A will be needed later as a function argument
      call bHYPRE_Operator__cast_f( parcsr_A, op_A, except )

!     Choose a parallel csr format storage (see the User's Manual)
!     Note: Here the HYPRE interface requires a SetObjectType call.
!     I am using the bHYPRE interface in a way which does not because
!     the object type is already specified through the class name.

!     Initialize before setting coefficients
      call bHYPRE_IJParCSRMatrix_Initialize_f( parcsr_A, ierrtmp, except )

!     Now go through my local rows and set the matrix entries.
!     Each row has at most 5 entries. For example, if n=3:
!
!      A = [M -I 0; -I M -I; 0 -I M]
!      M = [4 -1 0; -1 4 -1; 0 -1 4]
!
!     Note that here we are setting one row at a time, though
!     one could set all the rows together (see the User's Manual).

      do i = ilower, iupper
         nnz = 1

!        The left identity block:position i-n
         if ( (i-n) .ge. 0 ) then
	    cols(nnz) = i-n
	    values(nnz) = -1.0d0
	    nnz = nnz + 1
         endif

!         The left -1: position i-1
         if ( mod(i,n).ne.0 ) then
            cols(nnz) = i-1
            values(nnz) = -1.0d0
            nnz = nnz + 1
         endif

!        Set the diagonal: position i
         cols(nnz) = i
         values(nnz) = 4.0d0
         nnz = nnz + 1

!        The right -1: position i+1
         if ( mod((i+1),n) .ne. 0 ) then
            cols(nnz) = i+1
            values(nnz) = -1.0d0
            nnz = nnz + 1
         endif

!        The right identity block:position i+n
         if ((i+n) .lt. ng ) then
            cols(nnz) = i+n
            values(nnz) = -1.0d0
            nnz = nnz + 1
         endif

!        Set the values for row i
         call bHYPRE_IJParCSRMatrix_SetValues_f( parcsr_A, 1, nnz-1, i, cols, values, 5, &
              ierrtmp, except )

      enddo

!     Assemble after setting the coefficients
      call bHYPRE_IJParCSRMatrix_Assemble_f( parcsr_A, ierrtmp, except )

!     Create the rhs and solution
      call bHYPRE_IJParCSRVector_Create_f( bHYPRE_mpicomm, ilower, iupper, par_b, except )
!     vec_b will be needed later for function arguments
      call bHYPRE_Vector__cast_f( par_b, vec_b, except )

      call bHYPRE_IJParCSRVector_Initialize_f( par_b, ierrtmp, except )

      call bHYPRE_IJParCSRVector_Create_f( bHYPRE_mpicomm, ilower, iupper, par_x, except )
!     vec_x will be needed later for function arguments
      call bHYPRE_Vector__cast_f( par_x, vec_x, except )

      call bHYPRE_IJParCSRVector_Initialize_f( par_x, ierrtmp, except )

!     Set the rhs values to h^2 and the solution to zero
      do i = 1, local_size
         rhs_values(i) = h2
         x_values(i) = 0.0
         rows(i) = ilower + i -1
      enddo
      call bHYPRE_IJParCSRVector_SetValues_f( par_b, local_size, rows, rhs_values, &
           ierrtmp, except )
      call bHYPRE_IJParCSRVector_SetValues_f( par_x, local_size, rows, x_values, ierrtmp, except )


      call bHYPRE_IJParCSRVector_Assemble_f( par_b, ierrtmp, except )
      call bHYPRE_IJParCSRVector_Assemble_f( par_x, ierrtmp, except )

!     Choose a solver and solve the system

!      AMG
      if ( solver_id == 0 ) then

!        Create solver
         call bHYPRE_BoomerAMG_Create_f( bHYPRE_mpicomm, parcsr_A, amg_solver, except )

!        Set some parameters (See Reference Manual for more parameters)
!        PrintLevel=3 means print solve info + parameters
!        CoarsenType=6 means Falgout coarsening
!        RelaxType=3 means Gauss-Seidel/Jacobi hybrid relaxation
         call bHYPRE_BoomerAMG_SetIntParameter_f( amg_solver, "PrintLevel", 3, ierrtmp, except )
         call bHYPRE_BoomerAMG_SetIntParameter_f( amg_solver, "CoarsenType", 6, ierrtmp, except )
         call bHYPRE_BoomerAMG_SetIntParameter_f( amg_solver, "RelaxType", 3, ierrtmp, except )
         call bHYPRE_BoomerAMG_SetIntParameter_f( amg_solver, "NumSweeps", 1, ierrtmp, except )
         call bHYPRE_BoomerAMG_SetIntParameter_f( amg_solver, "MaxLevels", 20, ierrtmp, except )
         call bHYPRE_BoomerAMG_SetDoubleParameter_f( amg_solver, "Tolerance", tol, ierrtmp, &
              except )

!        Now setup and solve!
         call bHYPRE_BoomerAMG_Setup_f( amg_solver, vec_b, vec_x, ierrtmp, except )
         call bHYPRE_BoomerAMG_Apply_f( amg_solver, vec_b, vec_x, ierrtmp, except )

!        Run info - needed logging turned on 
         call bHYPRE_BoomerAMG_GetIntValue_f( amg_solver, "NumIterations", num_iterations, &
              ierrtmp, except )
         ierr = ierr + ierrtmp
         call bHYPRE_BoomerAMG_GetDoubleValue_f( amg_solver, "RelResidualNorm", final_res_norm, &
              ierrtmp, except )

      if (myid .eq. 0) then
         print *
         print *, "Iterations = ", num_iterations
         print *, "Final Relative Residual Norm = ", final_res_norm
         print *
      endif

!     Destroy solver
      call bHYPRE_BoomerAMG_deleteRef_f( amg_solver, except )

      endif

!     The calls of other solvers are not implemented yet.


!     Print the solution
      if ( print_solution .ne. 0 ) then
         call bHYPRE_IJParCSRVector_Print_f( par_x, "ij.out.x", except )
      endif

!     Clean up
      call bHYPRE_Operator_deleteRef_f( op_A, except )
      call bHYPRE_Vector_deleteRef_f( vec_b, except )
      call bHYPRE_Vector_deleteRef_f( vec_x, except )
      call bHYPRE_IJParCSRMatrix_deleteRef_f( parcsr_A, except )
      call bHYPRE_IJParCSRVector_deleteRef_f( par_b, except )
      call bHYPRE_IJParCSRVector_deleteRef_f( par_x, except )
      call bHYPRE_MPICommunicator_deleteRef_f( bHYPRE_mpicomm, except )

!     We need a multi-language equivalent of hypre_assert.
      if ( ierr .ne. 0 ) then
         print *
         print *, "***** Bad ierr = ", ierr
         print *
      endif

!     Finalize MPI
      call MPI_Finalize(ierrtmp)

      stop
      end

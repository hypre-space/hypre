!     Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!
!   Example 5
!
!   Interface:    Linear-Algebraic (IJ), Fortran (77) version
!
!   Compile with: make ex5f
!
!   Sample run:   mpirun -np 4 ex5f
!
!   Description:  This example solves the 2-D
!                 Laplacian problem with zero boundary conditions
!                 on an nxn grid.  The number of unknowns is N=n^2.
!                 The standard 5-point stencil is used, and we solve
!                 for the interior nodes only.
!
!                 This example solves the same problem as Example 3.
!                 Available solvers are AMG, PCG, and PCG with AMG,
!                 and PCG with ParaSails
!
!
!                 Notes: for PCG, GMRES and BiCGStab, precond_id means:
!                        0 - do not set up a preconditioner
!                        1 - set up a ds preconditioner
!                        2 - set up an amg preconditioner
!                        3 - set up a pilut preconditioner
!                        4 - set up a ParaSails preconditioner
!

      program ex5f


      implicit none

      include 'mpif.h'

      integer    MAX_LOCAL_SIZE
      integer    HYPRE_PARCSR

      parameter  (MAX_LOCAL_SIZE=123000)

!     the following is from HYPRE.c
      parameter  (HYPRE_PARCSR=5555)

      integer    ierr
      integer    num_procs, myid
      integer    local_size, extra
      integer    n, solver_id, print_solution, ng
      integer    nnz, ilower, iupper, i
      integer    precond_id;
      double precision h, h2
      double precision rhs_values(MAX_LOCAL_SIZE)
      double precision x_values(MAX_LOCAL_SIZE)
      integer    rows(MAX_LOCAL_SIZE)
      integer    cols(5)
      double precision values(5)
      integer    num_iterations
      double precision final_res_norm, tol

      integer    mpi_comm

      integer*8  parcsr_A
      integer*8  A
      integer*8  b
      integer*8  x
      integer*8  par_b
      integer*8  par_x
      integer*8  solver
      integer*8  precond

!-----------------------------------------------------------------------
!     Initialize MPI
!-----------------------------------------------------------------------

      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)
      mpi_comm = MPI_COMM_WORLD

      call HYPRE_Init(ierr)

!   Default problem parameters
      n = 33
      solver_id = 0
      print_solution  = 0
      tol = 1.0d-7

!   The input section not implemented yet.

!   Preliminaries: want at least one processor per row
      if ( n*n .lt. num_procs ) then
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
      call HYPRE_IJMatrixCreate(mpi_comm, ilower,
     1     iupper, ilower, iupper, A, ierr)


!     Choose a parallel csr format storage (see the User's Manual)
      call HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR, ierr)

!     Initialize before setting coefficients
      call HYPRE_IJMatrixInitialize(A, ierr)


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
         if ( (i+n) .lt. ng ) then
            cols(nnz) = i+n
            values(nnz) = -1.0d0
            nnz = nnz + 1
         endif

!        Set the values for row i
         call HYPRE_IJMatrixSetValues(
     1        A, 1, nnz-1, i, cols, values, ierr)

      enddo


!     Assemble after setting the coefficients
      call HYPRE_IJMatrixAssemble(A, ierr)

!     Get parcsr matrix object
      call HYPRE_IJMatrixGetObject(A, parcsr_A, ierr)


!     Create the rhs and solution
      call HYPRE_IJVectorCreate(mpi_comm,
     1     ilower, iupper, b, ierr)
      call HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR, ierr)
      call HYPRE_IJVectorInitialize(b, ierr)

      call HYPRE_IJVectorCreate(mpi_comm,
     1     ilower, iupper, x, ierr)
      call HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR, ierr)
      call HYPRE_IJVectorInitialize(x, ierr)


!     Set the rhs values to h^2 and the solution to zero
      do i = 1, local_size
         rhs_values(i) = h2
         x_values(i) = 0.0
         rows(i) = ilower + i -1
      enddo
      call HYPRE_IJVectorSetValues(
     1     b, local_size, rows, rhs_values, ierr)
      call HYPRE_IJVectorSetValues(
     1     x, local_size, rows, x_values, ierr)


      call HYPRE_IJVectorAssemble(b, ierr)
      call HYPRE_IJVectorAssemble(x, ierr)

! get the x and b objects

      call HYPRE_IJVectorGetObject(b, par_b, ierr)
      call HYPRE_IJVectorGetObject(x, par_x, ierr)


!     Choose a solver and solve the system

!     AMG
      if ( solver_id .eq. 0 ) then

!        Create solver
         call HYPRE_BoomerAMGCreate(solver, ierr)


!        Set some parameters (See Reference Manual for more parameters)

!        print solve info + parameters
         call HYPRE_BoomerAMGSetPrintLevel(solver, 3, ierr)
!        old defaults, Falgout coarsening, mod. class. interpolation
         call HYPRE_BoomerAMGSetOldDefault(solver, ierr)
!        G-S/Jacobi hybrid relaxation
         call HYPRE_BoomerAMGSetRelaxType(solver, 3, ierr)
!        C/F relaxation
         call HYPRE_BoomerAMGSetRelaxOrder(solver, 1, ierr)
!        Sweeeps on each level
         call HYPRE_BoomerAMGSetNumSweeps(solver, 1, ierr)
!         maximum number of levels
         call HYPRE_BoomerAMGSetMaxLevels(solver, 20, ierr)
!        conv. tolerance
         call HYPRE_BoomerAMGSetTol(solver, 1.0d-7, ierr)

!        Now setup and solve!
         call HYPRE_BoomerAMGSetup(
     1        solver, parcsr_A, par_b, par_x, ierr)
         call HYPRE_BoomerAMGSolve(
     1        solver, parcsr_A, par_b, par_x, ierr)


!        Run info - needed logging turned on
         call HYPRE_BoomerAMGGetNumIterations(solver, num_iterations,
     1        ierr)
         call HYPRE_BoomerAMGGetFinalReltvRes(solver, final_res_norm,
     1        ierr)


         if ( myid .eq. 0 ) then
            print *
            print '(A,I2)', " Iterations = ", num_iterations
            print '(A,ES16.8)',
     1            " Final Relative Residual Norm = ", final_res_norm
            print *
         endif

!        Destroy solver
         call HYPRE_BoomerAMGDestroy(solver, ierr)

!     PCG (with DS)
      elseif ( solver_id .eq. 50 ) then


!        Create solver
         call HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, solver, ierr)

!        Set some parameters (See Reference Manual for more parameters)
         call HYPRE_ParCSRPCGSetMaxIter(solver, 1000, ierr)
         call HYPRE_ParCSRPCGSetTol(solver, 1.0d-7, ierr)
         call HYPRE_ParCSRPCGSetTwoNorm(solver, 1, ierr)
         call HYPRE_ParCSRPCGSetPrintLevel(solver, 2, ierr)
         call HYPRE_ParCSRPCGSetLogging(solver, 1, ierr)

!        set ds (diagonal scaling) as the pcg preconditioner
         precond_id = 1
         call HYPRE_ParCSRPCGSetPrecond(solver, precond_id,
     1        precond, ierr)



!        Now setup and solve!
         call HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b,
     &                            par_x, ierr)
         call HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b,
     &                            par_x, ierr)


!        Run info - needed logging turned on

        call HYPRE_ParCSRPCGGetNumIterations(solver, num_iterations,
     &                                       ierr)
        call HYPRE_ParCSRPCGGetFinalRelative(solver, final_res_norm,
     &                                       ierr)

       if ( myid .eq. 0 ) then
            print *
            print *, "Iterations = ", num_iterations
            print *, "Final Relative Residual Norm = ", final_res_norm
            print *
         endif

!       Destroy solver
        call HYPRE_ParCSRPCGDestroy(solver, ierr)


!     PCG with AMG preconditioner
      elseif ( solver_id == 1 ) then

!        Create solver
         call HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, solver, ierr)

!        Set some parameters (See Reference Manual for more parameters)
         call HYPRE_ParCSRPCGSetMaxIter(solver, 1000, ierr)
         call HYPRE_ParCSRPCGSetTol(solver, 1.0d-7, ierr)
         call HYPRE_ParCSRPCGSetTwoNorm(solver, 1, ierr)
         call HYPRE_ParCSRPCGSetPrintLevel(solver, 2, ierr)
         call HYPRE_ParCSRPCGSetLogging(solver, 1, ierr)

!        Now set up the AMG preconditioner and specify any parameters

         call HYPRE_BoomerAMGCreate(precond, ierr)


!        Set some parameters (See Reference Manual for more parameters)

!        print less solver info since a preconditioner
         call HYPRE_BoomerAMGSetPrintLevel(precond, 1, ierr);
!        Falgout coarsening
         call HYPRE_BoomerAMGSetCoarsenType(precond, 6, ierr)
!        old defaults
         call HYPRE_BoomerAMGSetOldDefault(precond, ierr)
!        SYMMETRIC G-S/Jacobi hybrid relaxation
         call HYPRE_BoomerAMGSetRelaxType(precond, 6, ierr)
!        Sweeeps on each level
         call HYPRE_BoomerAMGSetNumSweeps(precond, 1, ierr)
!        conv. tolerance
         call HYPRE_BoomerAMGSetTol(precond, 0.0d0, ierr)
!        do only one iteration!
         call HYPRE_BoomerAMGSetMaxIter(precond, 1, ierr)

!        set amg as the pcg preconditioner
         precond_id = 2
         call HYPRE_ParCSRPCGSetPrecond(solver, precond_id,
     1        precond, ierr)


!        Now setup and solve!
         call HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b,
     1                            par_x, ierr)
         call HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b,
     1                            par_x, ierr)


!        Run info - needed logging turned on

        call HYPRE_ParCSRPCGGetNumIterations(solver, num_iterations,
     1                                       ierr)
        call HYPRE_ParCSRPCGGetFinalRelative(solver, final_res_norm,
     1                                       ierr)

       if ( myid .eq. 0 ) then
            print *
            print *, "Iterations = ", num_iterations
            print *, "Final Relative Residual Norm = ", final_res_norm
            print *
         endif

!       Destroy precond and solver

        call HYPRE_BoomerAMGDestroy(precond, ierr)
        call HYPRE_ParCSRPCGDestroy(solver, ierr)

!     PCG with ParaSails
      elseif ( solver_id .eq. 8 ) then

!        Create solver
         call HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, solver, ierr)

!        Set some parameters (See Reference Manual for more parameters)
         call HYPRE_ParCSRPCGSetMaxIter(solver, 1000, ierr)
         call HYPRE_ParCSRPCGSetTol(solver, 1.0d-7, ierr)
         call HYPRE_ParCSRPCGSetTwoNorm(solver, 1, ierr)
         call HYPRE_ParCSRPCGSetPrintLevel(solver, 2, ierr)
         call HYPRE_ParCSRPCGSetLogging(solver, 1, ierr)

!        Now set up the Parasails preconditioner and specify any parameters
         call HYPRE_ParaSailsCreate(MPI_COMM_WORLD, precond,ierr)
         call HYPRE_ParaSailsSetParams(precond, 0.1d0, 1, ierr)
         call HYPRE_ParaSailsSetFilter(precond, 0.05d0, ierr)
         call HYPRE_ParaSailsSetSym(precond, 1, ierr)
         call HYPRE_ParaSailsSetLogging(precond, 3, ierr)

!        set parsails as the pcg preconditioner
         precond_id = 4
         call HYPRE_ParCSRPCGSetPrecond(solver, precond_id,
     1        precond, ierr)


!        Now setup and solve!
         call HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b,
     1                            par_x, ierr)
         call HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b,
     1                            par_x, ierr)


!        Run info - needed logging turned on

        call HYPRE_ParCSRPCGGetNumIterations(solver, num_iterations,
     1                                       ierr)
        call HYPRE_ParCSRPCGGetFinalRelative(solver, final_res_norm,
     1                                       ierr)

       if ( myid .eq. 0 ) then
            print *
            print *, "Iterations = ", num_iterations
            print *, "Final Relative Residual Norm = ", final_res_norm
            print *
         endif

!       Destroy precond and solver

        call HYPRE_ParaSailsDestroy(precond, ierr)
        call HYPRE_ParCSRPCGDestroy(solver, ierr)

      else
         if ( myid .eq. 0 ) then
           print *,'Invalid solver id specified'
           stop
         endif
      endif



!     Print the solution
      if ( print_solution .ne. 0 ) then
         call HYPRE_IJVectorPrint(x, "ij.out.x", ierr)
      endif

!     Clean up

      call HYPRE_IJMatrixDestroy(A, ierr)
      call HYPRE_IJVectorDestroy(b, ierr)
      call HYPRE_IJVectorDestroy(x, ierr)

      call HYPRE_Finalize(ierr)

!     Finalize MPI
      call MPI_Finalize(ierr)

      stop
      end

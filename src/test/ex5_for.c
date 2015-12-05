/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.6 $
 ***********************************************************************EHEADER*/

/*
   Example 5

   Interface:    Linear-Algebraic (IJ)

   Compile with: make ex5

   Sample run:   mpirun -np 4 ex5

   Description:  This example solves the 2-D
                 Laplacian problem with zero boundary conditions
                 on an nxn grid.  The number of unknowns is N=n^2.
                 The standard 5-point stencil is used, and we solve
                 for the interior nodes only.

                 This example solves the same problem as Example 3.
                 Available solvers are AMG, PCG, and PCG with AMG or
                 Parasails preconditioners.
*/

#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

#ifdef HYPRE_FORTRAN
#include "fortran.h"
#include "hypre_ij_fortran_test.h"
#include "hypre_parcsr_fortran_test.h"
#endif

HYPRE_Int main (HYPRE_Int argc, char *argv[])
{
   HYPRE_Int i;
   HYPRE_Int myid, num_procs;
   HYPRE_Int N, n;

   HYPRE_Int ilower, iupper;
   HYPRE_Int local_size, extra;

   HYPRE_Int solver_id;
   HYPRE_Int print_solution;

   double h, h2;

#ifdef HYPRE_FORTRAN
   hypre_F90_Obj A;
   hypre_F90_Obj parcsr_A;
   hypre_F90_Obj b;
   hypre_F90_Obj par_b;
   hypre_F90_Obj x;
   hypre_F90_Obj par_x;

   hypre_F90_Obj solver, precond;

   hypre_F90_Obj long_temp_COMM;
        HYPRE_Int temp_COMM;
        HYPRE_Int precond_id;

        HYPRE_Int one = 1;
        HYPRE_Int two = 2;
        HYPRE_Int three = 3;
        HYPRE_Int six = 6;
        HYPRE_Int twenty = 20;
        HYPRE_Int thousand = 1000;
        HYPRE_Int hypre_type = HYPRE_PARCSR;

     double oo1 = 1.e-3;
     double tol = 1.e-7;
#else
   HYPRE_IJMatrix A;
   HYPRE_ParCSRMatrix parcsr_A;
   HYPRE_IJVector b;
   HYPRE_ParVector par_b;
   HYPRE_IJVector x;
   HYPRE_ParVector par_x;

   HYPRE_Solver solver, precond;
#endif

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   /* Default problem parameters */
   n = 33;
   solver_id = 0;
   print_solution  = 0;

   /* Parse command line */
   {
      HYPRE_Int arg_index = 0;
      HYPRE_Int print_usage = 0;

      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-n") == 0 )
         {
            arg_index++;
            n = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-solver") == 0 )
         {
            arg_index++;
            solver_id = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-print_solution") == 0 )
         {
            arg_index++;
            print_solution = 1;
         }
         else if ( strcmp(argv[arg_index], "-help") == 0 )
         {
            print_usage = 1;
            break;
         }
         else
         {
            arg_index++;
         }
      }

      if ((print_usage) && (myid == 0))
      {
         hypre_printf("\n");
         hypre_printf("Usage: %s [<options>]\n", argv[0]);
         hypre_printf("\n");
         hypre_printf("  -n <n>              : problem size in each direction (default: 33)\n");
         hypre_printf("  -solver <ID>        : solver ID\n");
         hypre_printf("                        0  - AMG (default) \n");
         hypre_printf("                        1  - AMG-PCG\n");
         hypre_printf("                        8  - ParaSails-PCG\n");
         hypre_printf("                        50 - PCG\n");
         hypre_printf("  -print_solution     : print the solution vector\n");
         hypre_printf("\n");
      }

      if (print_usage)
      {
         hypre_MPI_Finalize();
         return (0);
      }
   }

   /* Preliminaries: want at least one processor per row */
   if (n*n < num_procs) n = sqrt(num_procs) + 1;
   N = n*n; /* global number of rows */
   h = 1.0/(n+1); /* mesh size*/
   h2 = h*h;

   /* Each processor knows only of its own rows - the range is denoted by ilower
      and upper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   local_size = N/num_procs;
   extra = N - local_size*num_procs;

   ilower = local_size*myid;
   ilower += hypre_min(myid, extra);

   iupper = local_size*(myid+1);
   iupper += hypre_min(myid+1, extra);
   iupper = iupper - 1;

   /* How many rows do I have? */
   local_size = iupper - ilower + 1;

   /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
#ifdef HYPRE_FORTRAN
   long_temp_COMM = (hypre_F90_Obj) hypre_MPI_COMM_WORLD;
   temp_COMM = (HYPRE_Int) hypre_MPI_COMM_WORLD;
   HYPRE_IJMatrixCreate(&long_temp_COMM, &ilower, &iupper, &ilower, &iupper, &A);
#else
   HYPRE_IJMatrixCreate(hypre_MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
#endif

   /* Choose a parallel csr format storage (see the User's Manual) */
#ifdef HYPRE_FORTRAN
   HYPRE_IJMatrixSetObjectType(&A, &hypre_type);
#else
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
#endif

   /* Initialize before setting coefficients */
#ifdef HYPRE_FORTRAN
   HYPRE_IJMatrixInitialize(&A);
#else
   HYPRE_IJMatrixInitialize(A);
#endif

   /* Now go through my local rows and set the matrix entries.
      Each row has at most 5 entries. For example, if n=3:

      A = [M -I 0; -I M -I; 0 -I M]
      M = [4 -1 0; -1 4 -1; 0 -1 4]

      Note that here we are setting one row at a time, though
      one could set all the rows together (see the User's Manual).
   */
   {
      HYPRE_Int nnz;
      double values[5];
      HYPRE_Int cols[5];

      for (i = ilower; i <= iupper; i++)
      {
         nnz = 0;

         /* The left identity block:position i-n */
         if ((i-n)>=0)
         {
	    cols[nnz] = i-n;
	    values[nnz] = -1.0;
	    nnz++;
         }

         /* The left -1: position i-1 */
         if (i%n)
         {
            cols[nnz] = i-1;
            values[nnz] = -1.0;
            nnz++;
         }

         /* Set the diagonal: position i */
         cols[nnz] = i;
         values[nnz] = 4.0;
         nnz++;

         /* The right -1: position i+1 */
         if ((i+1)%n)
         {
            cols[nnz] = i+1;
            values[nnz] = -1.0;
            nnz++;
         }

         /* The right identity block:position i+n */
         if ((i+n)< N)
         {
            cols[nnz] = i+n;
            values[nnz] = -1.0;
            nnz++;
         }

         /* Set the values for row i */
#ifdef HYPRE_FORTRAN
         HYPRE_IJMatrixSetValues(&A, &one, &nnz, &i, &cols[0], &values[0]);
#else
         HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, cols, values);
#endif
      }
   }

   /* Assemble after setting the coefficients */
#ifdef HYPRE_FORTRAN
   HYPRE_IJMatrixAssemble(&A);
#else
   HYPRE_IJMatrixAssemble(A);
#endif
   /* Get the parcsr matrix object to use */
#ifdef HYPRE_FORTRAN
   HYPRE_IJMatrixGetObject(&A, &parcsr_A);
   HYPRE_IJMatrixGetObject(&A, &parcsr_A);
#else
   HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
   HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
#endif

   /* Create the rhs and solution */
#ifdef HYPRE_FORTRAN
   HYPRE_IJVectorCreate(&temp_COMM, &ilower, &iupper, &b);
   HYPRE_IJVectorSetObjectType(&b, &hypre_type);
   HYPRE_IJVectorInitialize(&b);
#else
   HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, ilower, iupper,&b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);
#endif

#ifdef HYPRE_FORTRAN
   HYPRE_IJVectorCreate(&temp_COMM, &ilower, &iupper, &x);
   HYPRE_IJVectorSetObjectType(&x, &hypre_type);
   HYPRE_IJVectorInitialize(&x);
#else
   HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, ilower, iupper,&x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);
#endif

   /* Set the rhs values to h^2 and the solution to zero */
   {
      double *rhs_values, *x_values;
      HYPRE_Int    *rows;

      rhs_values = calloc(local_size, sizeof(double));
      x_values = calloc(local_size, sizeof(double));
      rows = calloc(local_size, sizeof(HYPRE_Int));

      for (i=0; i<local_size; i++)
      {
         rhs_values[i] = h2;
         x_values[i] = 0.0;
         rows[i] = ilower + i;
      }
#ifdef HYPRE_FORTRAN
      HYPRE_IJVectorSetValues(&b, &local_size, &rows[0], &rhs_values[0]);
      HYPRE_IJVectorSetValues(&x, &local_size, &rows[0], &x_values[0]);
#else
      HYPRE_IJVectorSetValues(b, local_size, rows, rhs_values);
      HYPRE_IJVectorSetValues(x, local_size, rows, x_values);
#endif

      free(x_values);
      free(rhs_values);
      free(rows);
   }

#ifdef HYPRE_FORTRAN
   HYPRE_IJVectorAssemble(&b);
   HYPRE_IJVectorGetObject(&b, &par_b);
#else
   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorGetObject(b, (void **) &par_b);
#endif

#ifdef HYPRE_FORTRAN
   HYPRE_IJVectorAssemble(&x);
   HYPRE_IJVectorGetObject(&x, &par_x);
#else
   HYPRE_IJVectorAssemble(x);
   HYPRE_IJVectorGetObject(x, (void **) &par_x);
#endif

   /* Choose a solver and solve the system */

   /* AMG */
   if (solver_id == 0)
   {
      HYPRE_Int num_iterations;
      double final_res_norm;

      /* Create solver */
#ifdef HYPRE_FORTRAN
      HYPRE_BoomerAMGCreate(&solver);
#else
      HYPRE_BoomerAMGCreate(&solver);
#endif

      /* Set some parameters (See Reference Manual for more parameters) */
#ifdef HYPRE_FORTRAN
      HYPRE_BoomerAMGSetPrintLevel(&solver, &three);  /* print solve info + parameters */
      HYPRE_BoomerAMGSetCoarsenType(&solver, &six); /* Falgout coarsening */
      HYPRE_BoomerAMGSetRelaxType(&solver, &three);   /* G-S/Jacobi hybrid relaxation */
      HYPRE_BoomerAMGSetNumSweeps(&solver, &one);   /* Sweeeps on each level */
      HYPRE_BoomerAMGSetMaxLevels(&solver, &twenty);  /* maximum number of levels */
      HYPRE_BoomerAMGSetTol(&solver, &tol);      /* conv. tolerance */
#else
      HYPRE_BoomerAMGSetPrintLevel(solver, 3);  /* print solve info + parameters */
      HYPRE_BoomerAMGSetCoarsenType(solver, 6); /* Falgout coarsening */
      HYPRE_BoomerAMGSetRelaxType(solver, 3);   /* G-S/Jacobi hybrid relaxation */
      HYPRE_BoomerAMGSetNumSweeps(solver, 1);   /* Sweeeps on each level */
      HYPRE_BoomerAMGSetMaxLevels(solver, 20);  /* maximum number of levels */
      HYPRE_BoomerAMGSetTol(solver, 1e-7);      /* conv. tolerance */
#endif

      /* Now setup and solve! */
#ifdef HYPRE_FORTRAN
      HYPRE_BoomerAMGSetup(&solver, &parcsr_A, &par_b, &par_x);
      HYPRE_BoomerAMGSolve(&solver, &parcsr_A, &par_b, &par_x);
#else
      HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
      HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
#endif

      /* Run info - needed logging turned on */
#ifdef HYPRE_FORTRAN
      HYPRE_BoomerAMGGetNumIterations(&solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(&solver, &final_res_norm);
#else
      HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
#endif
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

      /* Destroy solver */
#ifdef HYPRE_FORTRAN
      HYPRE_BoomerAMGDestroy(&solver);
#else
      HYPRE_BoomerAMGDestroy(solver);
#endif
   }
   /* PCG */
   else if (solver_id == 50)
   {
      HYPRE_Int num_iterations;
      double final_res_norm;

      /* Create solver */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGCreate(&temp_COMM, &solver);
#else
      HYPRE_ParCSRPCGCreate(hypre_MPI_COMM_WORLD, &solver);
#endif

      /* Set some parameters (See Reference Manual for more parameters) */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGSetMaxIter(&solver, &thousand); /* max iterations */
      HYPRE_ParCSRPCGSetTol(&solver, &tol); /* conv. tolerance */
      HYPRE_ParCSRPCGSetTwoNorm(&solver, &one); /* use the two norm as the stopping criteria */
      HYPRE_ParCSRPCGSetPrintLevel(&solver, &two); /* prints out the iteration info */
#else
      HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
      HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, 2); /* prints out the iteration info */
      HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */
#endif

      /* Now setup and solve! */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGSetup(&solver, &parcsr_A, &par_b, &par_x);
      HYPRE_ParCSRPCGSolve(&solver, &parcsr_A, &par_b, &par_x);
#else
      HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
      HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);
#endif

      /* Run info - needed logging turned on */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGGetNumIterations(&solver, &num_iterations);
      HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(&solver, &final_res_norm);
#else
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
#endif
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

      /* Destroy solver */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGDestroy(&solver);
#else
      HYPRE_ParCSRPCGDestroy(solver);
#endif
   }
   /* PCG with AMG preconditioner */
   else if (solver_id == 1)
   {
      HYPRE_Int num_iterations;
      double final_res_norm;

      /* Create solver */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGCreate(&temp_COMM, &solver);
#else
      HYPRE_ParCSRPCGCreate(hypre_MPI_COMM_WORLD, &solver);
#endif

      /* Set some parameters (See Reference Manual for more parameters) */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGSetMaxIter(&solver, &thousand); /* max iterations */
      HYPRE_ParCSRPCGSetTol(&solver, &tol); /* conv. tolerance */
      HYPRE_ParCSRPCGSetTwoNorm(&solver, &one); /* use the two norm as the stopping criteria */
      HYPRE_ParCSRPCGSetPrintLevel(&solver, &two); /* print solve info */
#else
      HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
      HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
      HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */
#endif

      /* Now set up the AMG preconditioner and specify any parameters */
#ifdef HYPRE_FORTRAN
      HYPRE_BoomerAMGCreate(&precond);
      HYPRE_BoomerAMGSetPrintLevel(&precond, &one); /* print amg solution info*/
      HYPRE_BoomerAMGSetCoarsenType(&precond, &six);
      HYPRE_BoomerAMGSetRelaxType(&precond, &three);
      HYPRE_BoomerAMGSetNumSweeps(&precond, &one);
      HYPRE_BoomerAMGSetTol(&precond, &oo1);
#else
      HYPRE_BoomerAMGCreate(&precond);
      HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info*/
      HYPRE_BoomerAMGSetCoarsenType(precond, 6);
      HYPRE_BoomerAMGSetRelaxType(precond, 3);
      HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      HYPRE_BoomerAMGSetTol(precond, 1e-3);
#endif

      /* Set the PCG preconditioner */
#ifdef HYPRE_FORTRAN
      precond_id = 2;
      HYPRE_ParCSRPCGSetPrecond(&solver, &precond_id, &precond);
#else
      HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                          (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
#endif

      /* Now setup and solve! */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGSetup(&solver, &parcsr_A, &par_b, &par_x);
      HYPRE_ParCSRPCGSolve(&solver, &parcsr_A, &par_b, &par_x);
#else
      HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
      HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);
#endif

      /* Run info - needed logging turned on */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGGetNumIterations(&solver, &num_iterations);
      HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(&solver, &final_res_norm);
#else
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
#endif
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

      /* Destroy solver and preconditioner */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGDestroy(&solver);
      HYPRE_BoomerAMGDestroy(&precond);
#else
      HYPRE_ParCSRPCGDestroy(solver);
      HYPRE_BoomerAMGDestroy(precond);
#endif
   }
   /* PCG with Parasails Preconditioner */
   else if (solver_id == 8)
   {
      HYPRE_Int    num_iterations;
      double final_res_norm;

      HYPRE_Int      sai_max_levels = 1;
      double   sai_threshold = 0.1;
      double   sai_filter = 0.05;
      HYPRE_Int      sai_sym = 1;

      /* Create solver */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGCreate(&temp_COMM, &solver);
#else
      HYPRE_ParCSRPCGCreate(hypre_MPI_COMM_WORLD, &solver);
#endif

      /* Set some parameters (See Reference Manual for more parameters) */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGSetMaxIter(&solver, &thousand); /* max iterations */
      HYPRE_ParCSRPCGSetTol(&solver, &tol); /* conv. tolerance */
      HYPRE_ParCSRPCGSetTwoNorm(&solver, &one); /* use the two norm as the stopping criteria */
      HYPRE_ParCSRPCGSetPrintLevel(&solver, &two); /* print solve info */
#else
      HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
      HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
      HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */
#endif

      /* Now set up the ParaSails preconditioner and specify any parameters */
#ifdef HYPRE_FORTRAN
      HYPRE_ParaSailsCreate(&temp_COMM, &precond);
#else
      HYPRE_ParaSailsCreate(hypre_MPI_COMM_WORLD, &precond);
#endif

      /* Set some parameters (See Reference Manual for more parameters) */
#ifdef HYPRE_FORTRAN
      HYPRE_ParaSailsSetParams(&precond, &sai_threshold, &sai_max_levels);
      HYPRE_ParaSailsSetFilter(&precond, &sai_filter);
      HYPRE_ParaSailsSetSym(&precond, &sai_sym);
      HYPRE_ParaSailsSetLogging(&precond, &three);
#else
      HYPRE_ParaSailsSetParams(precond, sai_threshold, sai_max_levels);
      HYPRE_ParaSailsSetFilter(precond, sai_filter);
      HYPRE_ParaSailsSetSym(precond, sai_sym);
      HYPRE_ParaSailsSetLogging(precond, 3);
#endif

      /* Set the PCG preconditioner */
#ifdef HYPRE_FORTRAN
      precond_id = 4;
      HYPRE_ParCSRPCGSetPrecond(&solver, &precond_id, &precond);
#else
      HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
                          (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup, precond);
#endif

      /* Now setup and solve! */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGSetup(&solver, &parcsr_A, &par_b, &par_x);
      HYPRE_ParCSRPCGSolve(&solver, &parcsr_A, &par_b, &par_x);
#else
      HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
      HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);
#endif


      /* Run info - needed logging turned on */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGGetNumIterations(&solver, &num_iterations);
      HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(&solver, &final_res_norm);
#else
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
#endif
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

      /* Destory solver and preconditioner */
#ifdef HYPRE_FORTRAN
      HYPRE_ParCSRPCGDestroy(&solver);
      HYPRE_ParaSailsDestroy(&precond);
#else
      HYPRE_ParCSRPCGDestroy(solver);
      HYPRE_ParaSailsDestroy(precond);
#endif
   }
   else
   {
      if (myid ==0) hypre_printf("Invalid solver id specified.\n");
   }

   /* Print the solution */
#ifdef HYPRE_FORTRAN
   if (print_solution)
      HYPRE_IJVectorPrint(&x, "ij.out.x");
#else
   if (print_solution)
      HYPRE_IJVectorPrint(x, "ij.out.x");
#endif

   /* Clean up */
#ifdef HYPRE_FORTRAN
   HYPRE_IJMatrixDestroy(&A);
   HYPRE_IJVectorDestroy(&b);
   HYPRE_IJVectorDestroy(&x);
#else
   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);
#endif

   /* Finalize MPI*/
   hypre_MPI_Finalize();

   return(0);
}

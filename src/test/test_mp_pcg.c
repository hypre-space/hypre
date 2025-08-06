/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*!
   This file contains a mocked-up example, based on ex5.c in the examples directory.
   The goal is to give an idea of how a user may utilize hypre to assemble matrix data
   and access solvers in a way that would facilitate a mixed-precision solution of the
   linear system. This particular driver demonstrates how the mixed-precision build may
   be used to develop mixed-precision solvers, such as the defect-correction-based solver
   implemented here. Feel free to ask questions, make comments or suggestions
   regarding any of the information below.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_IJ_mv_mup.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv_mup.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_mv_mup.h"
#include "HYPRE_krylov.h"
#include "HYPRE_krylov_mup.h"

#include <time.h>

#define MAXITS 1000

#define my_min(a,b)  (((a)<(b)) ? (a) : (b))

HYPRE_Int HYPRE_DefectCorrectionSolver(HYPRE_ParCSRMatrix A,
                                       HYPRE_ParCSRMatrix B,
                                       HYPRE_ParVector x,
                                       HYPRE_ParVector b,
                                       HYPRE_Solver solver,
                                       HYPRE_PtrToSolverFcn approx_solve,
                                       HYPRE_Int maxits);

int main (int argc, char *argv[])
{
   int           arg_index;
   int i;
   int myid, num_procs;
   int N, n;
   int ilower, iupper;
   int local_size, extra;
   int solver_id;
   float h, h2;
   double dh, dh2;
   //double d_one = 1.0;
   double d_zero = 0.;
   float zero = 0.;
   int num_iterations;
   double dfinal_res_norm;
   float     final_res_norm;
   int          time_index;
   //float   wall_time;
   /*! Matrix and preconditioner declarations. Here, we declare IJMatrices and parcsr matrices
       for the solver (A, parcsr_A) and the preconditioner (B, parcsr_B). I have included two suggestions
       below on how we would utilize both of these matrices.
   */

   //HYPRE_IJMatrix C;
   //HYPRE_ParCSRMatrix parcsr_C;

   HYPRE_IJMatrix A;
   HYPRE_ParCSRMatrix parcsr_A;
   HYPRE_IJVector b;
   HYPRE_ParVector par_b;
   HYPRE_IJVector x;
   HYPRE_ParVector par_x;

   HYPRE_IJMatrix B;
   HYPRE_ParCSRMatrix parcsr_B;
   HYPRE_IJVector bb;
   HYPRE_ParVector par_bb;
   HYPRE_IJVector xb;
   HYPRE_ParVector par_xb;

   HYPRE_IJVector ijres;
   HYPRE_IJVector ijhres;
   HYPRE_IJVector ije;
   HYPRE_IJVector ijxtmp;
   /*! Solver and preconditioner and declarations and solver_precision variable. Internally, HYPRE_SolverPrecision
       is an enum struct containing HYPRE_REAL_float, HYPRE_REAL_SINGLE and HYPRE_REAL_LONG.
   */
   //HYPRE_Solver solver, precond;
   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /*! We set up the linear system following ex5. */
   /* Some default problem parameters */
   n = 25;
   solver_id = 0;

   /*--------------------------
    * Parse command line
    *--------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         n = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /* Preliminaries: want at least one processor per row */
   if (n * n < num_procs) { n = sqrt(num_procs) + 1; }
   N = n * n; /* global number of rows */
   /* double and float variants of mesh spacing */
   h = 1.0 / (float)(n + 1); /* mesh size*/
   dh = 1.0 / (double)(n + 1);
   h2 = h * h;
   dh2 = dh * dh;
   /* partition rows */
   local_size = N / num_procs;
   extra = N - local_size * num_procs;

   ilower = local_size * myid;
   ilower += my_min(myid, extra);

   iupper = local_size * (myid + 1);
   iupper += my_min(myid + 1, extra);
   iupper = iupper - 1;

   local_size = iupper - ilower + 1;

   HYPRE_IJMatrixCreate_flt(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
   HYPRE_IJMatrixSetObjectType_flt(A, HYPRE_PARCSR);

   HYPRE_IJMatrixCreate_dbl(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &B);
   HYPRE_IJMatrixSetObjectType_dbl(B, HYPRE_PARCSR);

   /*! Initialize before setting coefficients */
   HYPRE_IJMatrixInitialize_flt(A);
   HYPRE_IJMatrixInitialize_dbl(B);
   /*! Set matrix entries */
   {
      int nnz;
      /* double and float variants of values */
      float values[5];
      double dvalues[5];
      int cols[5];

      for (i = ilower; i <= iupper; i++)
      {
         nnz = 0;
         /* The left identity block:position i-n */
         if ((i - n) >= 0)
         {
            cols[nnz] = i - n;
            values[nnz] = -1.0;
            dvalues[nnz] = -1.0;
            nnz++;
         }

         /* The left -1: position i-1 */
         if (i % n)
         {
            cols[nnz] = i - 1;
            values[nnz] = -1.0;
            dvalues[nnz] = -1.0;
            nnz++;
         }

         /* Set the diagonal: position i */
         cols[nnz] = i;
         values[nnz] = 4.0;
         dvalues[nnz] = 4.0;
         nnz++;

         /* The right -1: position i+1 */
         if ((i + 1) % n)
         {
            cols[nnz] = i + 1;
            values[nnz] = -1.0;
            dvalues[nnz] = -1.0;
            nnz++;
         }

         /* The right identity block:position i+n */
         if ((i + n) < N)
         {
            cols[nnz] = i + n;
            values[nnz] = -1.0;
            dvalues[nnz] = -1.0;
            nnz++;
         }

         /* Set the values for row i */
         HYPRE_IJMatrixSetValues_flt(A, 1, &nnz, &i, cols, values);
         HYPRE_IJMatrixSetValues_dbl(B, 1, &nnz, &i, cols, dvalues);
      }
   }

   /*! Assemble after setting the coefficients */
   HYPRE_IJMatrixAssemble_flt(A);
   HYPRE_IJMatrixAssemble_dbl(B);
   /*! Get the parcsr matrix object to use */
   HYPRE_IJMatrixGetObject_flt(A, (void**) &parcsr_A);
   HYPRE_IJMatrixGetObject_dbl(B, (void**) &parcsr_B);
   /*! Create the rhs and solution. Here, we only account for the solver precision. Since the preconditioner solve
        is done internally, we can pass the appropriate vector types there.
   */
   {
      HYPRE_IJVectorCreate_flt(MPI_COMM_WORLD, ilower, iupper, &b);
      HYPRE_IJVectorSetObjectType_flt(b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_flt(b);

      HYPRE_IJVectorCreate_flt(MPI_COMM_WORLD, ilower, iupper, &x);
      HYPRE_IJVectorSetObjectType_flt(x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_flt(x);

      HYPRE_IJVectorCreate_dbl(MPI_COMM_WORLD, ilower, iupper, &bb);
      HYPRE_IJVectorSetObjectType_dbl(bb, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_dbl(bb);

      HYPRE_IJVectorCreate_dbl(MPI_COMM_WORLD, ilower, iupper, &xb);
      HYPRE_IJVectorSetObjectType_dbl(xb, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_dbl(xb);

      HYPRE_IJVectorCreate_flt(MPI_COMM_WORLD, ilower, iupper, &ijres);
      HYPRE_IJVectorSetObjectType_flt(ijres, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_flt(ijres);

      HYPRE_IJVectorCreate_flt(MPI_COMM_WORLD, ilower, iupper, &ije);
      HYPRE_IJVectorSetObjectType_flt(ije, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_flt(ije);

      HYPRE_IJVectorCreate_dbl(MPI_COMM_WORLD, ilower, iupper, &ijhres);
      HYPRE_IJVectorSetObjectType_dbl(ijhres, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_dbl(ijhres);

      HYPRE_IJVectorCreate_dbl(MPI_COMM_WORLD, ilower, iupper, &ijxtmp);
      HYPRE_IJVectorSetObjectType_dbl(ijxtmp, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_dbl(ijxtmp);
   }


   /* Initialize rhs and solution */
   {
      float *rhs_values, *x_values;
      double *drhs_values, *dx_values;
      int    *rows;

      rhs_values =  (float*) calloc(local_size, sizeof(float));
      x_values =  (float*) calloc(local_size, sizeof(float));

      drhs_values =  (double*) calloc(local_size, sizeof(double));
      dx_values =  (double*) calloc(local_size, sizeof(double));

      rows = (int*) calloc(local_size, sizeof(int));

      for (i = 0; i < local_size; i++)
      {
         rhs_values[i] = h2;
         x_values[i] = 0.0;

         drhs_values[i] = dh2;
         dx_values[i] = 0.0;

         rows[i] = ilower + i;
      }

      HYPRE_IJVectorSetValues_flt(b, local_size, rows, rhs_values);
      HYPRE_IJVectorSetValues_flt(x, local_size, rows, x_values);

      HYPRE_IJVectorSetValues_dbl(bb, local_size, rows, drhs_values);
      HYPRE_IJVectorSetValues_dbl(xb, local_size, rows, dx_values);

      HYPRE_IJVectorSetValues_flt(ijres, local_size, rows, rhs_values);
      HYPRE_IJVectorSetValues_flt(ije, local_size, rows, x_values);
      HYPRE_IJVectorSetValues_dbl(ijhres, local_size, rows, drhs_values);
      HYPRE_IJVectorSetValues_dbl(ijxtmp, local_size, rows, dx_values);

      free(x_values);
      free(rhs_values);

      free(dx_values);
      free(drhs_values);

      free(rows);
   }

   /* Assemble vector and get parcsr vector object */
   HYPRE_IJVectorAssemble_flt(b);
   HYPRE_IJVectorGetObject_flt(b, (void **) &par_b);
   HYPRE_IJVectorAssemble_flt(x);
   HYPRE_IJVectorGetObject_flt(x, (void **) &par_x);

   HYPRE_IJVectorAssemble_dbl(bb);
   HYPRE_IJVectorGetObject_dbl(bb, (void **) &par_bb);
   HYPRE_IJVectorAssemble_dbl(xb);
   HYPRE_IJVectorGetObject_dbl(xb, (void **) &par_xb);

   /*! Done with linear system setup. Now proceed to solve the system. */
   // Double precision
   {
      /* reset solution vector */
      HYPRE_ParVectorSetConstantValues_dbl(par_xb, d_zero);

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver        pcg_precond_gotten;

      time_index = hypre_InitializeTiming("DBL Setup");
      MPI_Barrier(hypre_MPI_COMM_WORLD);
      hypre_BeginTiming(time_index);
      // Create PCG solver
      HYPRE_ParCSRPCGCreate_dbl(hypre_MPI_COMM_WORLD, &pcg_solver);
      HYPRE_PCGSetMaxIter_dbl(pcg_solver, MAXITS);
      //      HYPRE_PCGSetTol_dbl(pcg_solver, 1.0e-10);
      HYPRE_PCGSetTwoNorm_dbl(pcg_solver, 1);
      //      HYPRE_PCGSetRelChange_dbl(pcg_solver, rel_change);
      HYPRE_PCGSetPrintLevel_dbl(pcg_solver, 3);
      //      HYPRE_PCGSetAbsoluteTo_dbll(pcg_solver, atol);
      //      HYPRE_PCGSetRecomputeResidual_dbl(pcg_solver, recompute_res);


      /* Now set up the AMG preconditioner and specify any parameters */
      if (solver_id == 1)
      {
         if (myid == 0) { hypre_printf("\n\n***** Solver: DOUBLE PRECISION AMG-PCG *****\n"); }

         HYPRE_BoomerAMGCreate_dbl(&amg_solver);
         //         HYPRE_BoomerAMGSetPrintLevel_dbl(amg_solver, 0); /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType_dbl(amg_solver, 8);
         //         HYPRE_BoomerAMGSetOldDefault_dbl(amg_solver);
         HYPRE_BoomerAMGSetRelaxType_dbl(amg_solver, 18); /* Sym G.S./Jacobi hybrid */
         HYPRE_BoomerAMGSetNumSweeps_dbl(amg_solver, 1);
         HYPRE_BoomerAMGSetTol_dbl(amg_solver, 0.0); /* conv. tolerance zero */
         HYPRE_BoomerAMGSetMaxIter_dbl(amg_solver, 1); /* do only one iteration! */

         // Set the preconditioner for PCG
         HYPRE_PCGSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)parcsr_B);

         HYPRE_PCGSetPrecond_dbl(pcg_solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_dbl,
                                 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_dbl,
                                 amg_solver);

         HYPRE_PCGGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf("HYPRE_ParCSRPCGGetPrecond got good precond\n");
         }
      }
      else
      {
         if (myid == 0) { hypre_printf("\n\n***** Solver: DOUBLE PRECISION DS-PCG *****\n"); }
      }
      // Setup PCG solver
      HYPRE_PCGSetup_dbl(pcg_solver, (HYPRE_Matrix)parcsr_B,  (HYPRE_Vector)par_bb, (HYPRE_Vector)par_xb);
      MPI_Barrier(hypre_MPI_COMM_WORLD);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Double precision Setup Time", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      fflush(NULL);

      time_index = hypre_InitializeTiming("DBL Solve");
      MPI_Barrier(hypre_MPI_COMM_WORLD);
      hypre_BeginTiming(time_index);
      //  PCG solve
      HYPRE_PCGSolve_dbl(pcg_solver, (HYPRE_Matrix)parcsr_B,  (HYPRE_Vector)par_bb, (HYPRE_Vector)par_xb);

      MPI_Barrier(hypre_MPI_COMM_WORLD);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Double precision Solve Time", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations_dbl(pcg_solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm_dbl(pcg_solver, &dfinal_res_norm);
      if (myid == 0)
      {
         hypre_printf("final relative residual norm = %e \n", dfinal_res_norm);
         hypre_printf("Iteration count = %d \n", num_iterations);
      }
      fflush(NULL);
      // destroy pcg solver
      HYPRE_ParCSRPCGDestroy_dbl(pcg_solver);
      if (solver_id == 1) { HYPRE_BoomerAMGDestroy_dbl(amg_solver); }
   }
   // Single precision
   {
      /* reset solution vector */
      HYPRE_ParVectorSetConstantValues_flt(par_x, zero);

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver        pcg_precond_gotten;

      time_index = hypre_InitializeTiming("FLT Setup");
      MPI_Barrier(hypre_MPI_COMM_WORLD);
      hypre_BeginTiming(time_index);
      // Create PCG solver
      HYPRE_ParCSRPCGCreate_flt(hypre_MPI_COMM_WORLD, &pcg_solver);
      HYPRE_PCGSetMaxIter_flt(pcg_solver, MAXITS);
      //      HYPRE_PCGSetTol_flt(pcg_solver, 1.0e-10);
      HYPRE_PCGSetTwoNorm_flt(pcg_solver, 1);
      //      HYPRE_PCGSetRelChange_flt(pcg_solver, rel_change);
      HYPRE_PCGSetPrintLevel_flt(pcg_solver, 3);
      //      HYPRE_PCGSetAbsoluteTo_fltl(pcg_solver, atol);
      //      HYPRE_PCGSetRecomputeResidual_flt(pcg_solver, recompute_res);


      /* Now set up the AMG preconditioner and specify any parameters */
      if (solver_id == 1)
      {
         if (myid == 0) { hypre_printf("\n\n***** Solver: SINGLE PRECISION AMG-PCG *****\n"); }
         HYPRE_BoomerAMGCreate_flt(&amg_solver);
         //         HYPRE_BoomerAMGSetPrintLevel_flt(amg_solver, 0); /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType_flt(amg_solver, 8);
         //         HYPRE_BoomerAMGSetOldDefault_flt(amg_solver);
         HYPRE_BoomerAMGSetRelaxType_flt(amg_solver, 18); /* Sym G.S./Jacobi hybrid */
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, 1);
         HYPRE_BoomerAMGSetTol_flt(amg_solver, 0.0); /* conv. tolerance zero */
         HYPRE_BoomerAMGSetMaxIter_flt(amg_solver, 1); /* do only one iteration! */

         // Set the preconditioner for PCG
         HYPRE_PCGSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)parcsr_A);

         HYPRE_PCGSetPrecond_flt(pcg_solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_flt,
                                 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_flt,
                                 amg_solver);

         HYPRE_PCGGetPrecond_flt(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf("HYPRE_ParCSRPCGGetPrecond got good precond\n");
         }
      }
      else
      {
         if (myid == 0) { hypre_printf("\n\n***** Solver: SINGLE PRECISION DS-PCG *****\n"); }
      }
      // Setup PCG solver
      HYPRE_PCGSetup_flt(pcg_solver, (HYPRE_Matrix)parcsr_A,  (HYPRE_Vector)par_b, (HYPRE_Vector)par_x);
      MPI_Barrier(hypre_MPI_COMM_WORLD);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Single precision Setup Time", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      fflush(NULL);

      time_index = hypre_InitializeTiming("FLT Solve");
      MPI_Barrier(hypre_MPI_COMM_WORLD);
      hypre_BeginTiming(time_index);
      //  PCG solve
      HYPRE_PCGSolve_flt(pcg_solver, (HYPRE_Matrix)parcsr_A,  (HYPRE_Vector)par_b, (HYPRE_Vector)par_x);

      MPI_Barrier(hypre_MPI_COMM_WORLD);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Single precision Solve Time", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations_flt(pcg_solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm_flt(pcg_solver, &final_res_norm);
      if (myid == 0)
      {
         hypre_printf("final relative residual norm = %e \n", final_res_norm);
         hypre_printf("Iteration count = %d \n", num_iterations);
      }
      fflush(NULL);
      // destroy pcg solver
      HYPRE_ParCSRPCGDestroy_flt(pcg_solver);
      if (solver_id == 1) { HYPRE_BoomerAMGDestroy_flt(amg_solver); }
   }
   // mixed-precision
   {
      /* reset solution vector */
      HYPRE_ParVectorSetConstantValues_dbl(par_xb, d_zero);

      HYPRE_Solver amg_solver;
      HYPRE_Solver pcg_solver;
      HYPRE_Solver        pcg_precond_gotten;

      time_index = hypre_InitializeTiming("DBL Setup");
      MPI_Barrier(hypre_MPI_COMM_WORLD);
      hypre_BeginTiming(time_index);
      // Create PCG solver
      HYPRE_ParCSRPCGCreate_dbl(hypre_MPI_COMM_WORLD, &pcg_solver);
      HYPRE_PCGSetMaxIter_dbl(pcg_solver, MAXITS);
      //      HYPRE_PCGSetTol_dbl(pcg_solver, 1.0e-10);
      HYPRE_PCGSetTwoNorm_dbl(pcg_solver, 1);
      //      HYPRE_PCGSetRelChange_dbl(pcg_solver, rel_change);
      HYPRE_PCGSetPrintLevel_dbl(pcg_solver, 3);
      //      HYPRE_PCGSetAbsoluteTo_dbll(pcg_solver, atol);
      //      HYPRE_PCGSetRecomputeResidual_dbl(pcg_solver, recompute_res);


      /* Now set up the AMG preconditioner and specify any parameters */
      if (solver_id == 1)
      {
         if (myid == 0) { hypre_printf("\n\n***** Solver: MIXED PRECISION AMG-PCG *****\n"); }
         HYPRE_BoomerAMGCreate_flt(&amg_solver);
         //         HYPRE_BoomerAMGSetPrintLevel_flt(amg_solver, 0); /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType_flt(amg_solver, 8);
         //         HYPRE_BoomerAMGSetOldDefault_flt(amg_solver);
         HYPRE_BoomerAMGSetRelaxType_flt(amg_solver, 18); /* Sym G.S./Jacobi hybrid */
         HYPRE_BoomerAMGSetNumSweeps_flt(amg_solver, 1);
         HYPRE_BoomerAMGSetTol_flt(amg_solver, 0.0); /* conv. tolerance zero */
         HYPRE_BoomerAMGSetMaxIter_flt(amg_solver, 1); /* do only one iteration! */

         // Set the preconditioner for PCG (single precision matrix)
         HYPRE_PCGSetPrecondMatrix_dbl(pcg_solver, (HYPRE_Matrix)parcsr_A);
         // Set the preconditioner for PCG.
         // This actually sets a pointer to a single precision AMG solver.
         // The setup and solve functions just allow us to accept double precision
         // rhs and sol vectors from the PCG solver to do the preconditioner solve.
         HYPRE_PCGSetPrecond_dbl(pcg_solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve_mp,
                                 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup_mp,
                                 amg_solver);

         HYPRE_PCGGetPrecond_dbl(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  amg_solver)
         {
            hypre_printf("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            hypre_printf("HYPRE_ParCSRPCGGetPrecond got good precond\n");
         }
      }
      else
      {
         if (myid == 0) { hypre_printf("\n\n***** Solver: DOUBLE PRECISION DS-PCG *****\n"); }
      }
      // Setup PCG solver (double precision)
      HYPRE_PCGSetup_dbl(pcg_solver, (HYPRE_Matrix)parcsr_B,  (HYPRE_Vector)par_bb, (HYPRE_Vector)par_xb);
      MPI_Barrier(hypre_MPI_COMM_WORLD);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Mixed precision Setup Time", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      fflush(NULL);

      time_index = hypre_InitializeTiming("DBL Solve");
      MPI_Barrier(hypre_MPI_COMM_WORLD);
      hypre_BeginTiming(time_index);
      //  PCG solve (double precision)
      HYPRE_PCGSolve_dbl(pcg_solver, (HYPRE_Matrix)parcsr_B,  (HYPRE_Vector)par_bb, (HYPRE_Vector)par_xb);

      MPI_Barrier(hypre_MPI_COMM_WORLD);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Mixed precision Solve Time", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations_dbl(pcg_solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm_dbl(pcg_solver, &dfinal_res_norm);
      if (myid == 0)
      {
         hypre_printf("final relative residual norm = %e \n", dfinal_res_norm);
         hypre_printf("Iteration count = %d \n", num_iterations);
      }
      fflush(NULL);
      // destroy pcg solver
      HYPRE_ParCSRPCGDestroy_dbl(pcg_solver);
      if (solver_id == 1) { HYPRE_BoomerAMGDestroy_flt(amg_solver); }
   }

   /* Clean up */
   HYPRE_IJMatrixDestroy_flt(A);
   HYPRE_IJVectorDestroy_flt(b);
   HYPRE_IJVectorDestroy_flt(x);
   HYPRE_IJVectorDestroy_flt(ijres);
   HYPRE_IJVectorDestroy_flt(ije);

   HYPRE_IJMatrixDestroy_dbl(B);
   HYPRE_IJVectorDestroy_dbl(bb);
   HYPRE_IJVectorDestroy_dbl(xb);
   HYPRE_IJVectorDestroy_dbl(ijhres);
   HYPRE_IJVectorDestroy_dbl(ijxtmp);

   /* Finalize MPI*/
   MPI_Finalize();

   return (0);
}

HYPRE_Int HYPRE_DefectCorrectionSolver(HYPRE_ParCSRMatrix A,
                                       HYPRE_ParCSRMatrix B,
                                       HYPRE_ParVector x,
                                       HYPRE_ParVector b,
                                       HYPRE_Solver solver,
                                       HYPRE_PtrToSolverFcn approx_solve,
                                       HYPRE_Int maxits)
{



   return 0;

}

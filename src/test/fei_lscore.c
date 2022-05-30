/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (IJ_matrix interface).
 * Do `driver -help' for usage info.
 * This driver started from the driver for parcsr_linear_solvers, and it
 * works by first building a parcsr matrix as before and then "copying"
 * that matrix row-by-row into the IJMatrix interface. AJC 7/99.
 *--------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_LinSysCore.h"

HYPRE_Int BuildParLaplacian27pt (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                 HYPRE_ParCSRMatrix *A_ptr );

#define SECOND_TIME 0

HYPRE_Int main( HYPRE_Int   argc, char *argv[] )
{
   HYPRE_Int                 arg_index;
   HYPRE_Int                 print_usage;
   HYPRE_Int                 build_matrix_arg_index;
   HYPRE_Int                 solver_id;
   HYPRE_Int                 ierr, i, j;
   HYPRE_Int                 num_iterations;

   HYPRE_ParCSRMatrix  parcsr_A;
   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 local_row;
   HYPRE_Int             time_index;
   MPI_Comm            comm;
   HYPRE_Int                 M, N;
   HYPRE_Int                 first_local_row, last_local_row;
   HYPRE_Int                 first_local_col, last_local_col;
   HYPRE_Int                 size, *col_ind;
   HYPRE_Real          *values;

   /* parameters for BoomerAMG */
   HYPRE_Real          strong_threshold;
   HYPRE_Int                 num_grid_sweeps;
   HYPRE_Real          relax_weight;

   /* parameters for GMRES */
   HYPRE_Int                  k_dim;

   char *paramString = new char[100];

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   build_matrix_arg_index = argc;
   solver_id              = 0;
   strong_threshold       = 0.25;
   num_grid_sweeps        = 2;
   relax_weight           = 0.5;
   k_dim                  = 20;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dbg") == 0 )
      {
         arg_index++;
         atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/

   if ( (print_usage) && (myid == 0) )
   {
      hypre_printf("\n");
      hypre_printf("Usage: %s [<options>]\n", argv[0]);
      hypre_printf("\n");
      hypre_printf("  -solver <ID>           : solver ID\n");
      hypre_printf("       0=DS-PCG      1=ParaSails-PCG \n");
      hypre_printf("       2=AMG-PCG     3=DS-GMRES     \n");
      hypre_printf("       4=PILUT-GMRES 5=AMG-GMRES    \n");
      hypre_printf("\n");
      hypre_printf("  -rlx <val>             : relaxation type\n");
      hypre_printf("       0=Weighted Jacobi  \n");
      hypre_printf("       1=Gauss-Seidel (very slow!)  \n");
      hypre_printf("       3=Hybrid Jacobi/Gauss-Seidel  \n");
      hypre_printf("\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("Running with these driver parameters:\n");
      hypre_printf("  solver ID    = %d\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   strcpy(paramString, "LS Interface");
   time_index = hypre_InitializeTiming(paramString);
   hypre_BeginTiming(time_index);

   BuildParLaplacian27pt(argc, argv, build_matrix_arg_index, &parcsr_A);

   /*-----------------------------------------------------------
    * Copy the parcsr matrix into the LSI through interface calls
    *-----------------------------------------------------------*/

   ierr = HYPRE_ParCSRMatrixGetComm( parcsr_A, &comm );
   ierr += HYPRE_ParCSRMatrixGetDims( parcsr_A, &M, &N );
   ierr = HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                           &first_local_row, &last_local_row,
                                           &first_local_col, &last_local_col );

   HYPRE_LinSysCore H(hypre_MPI_COMM_WORLD);
   HYPRE_Int numLocalEqns = last_local_row - first_local_row + 1;
   H.createMatricesAndVectors(M, first_local_row + 1, numLocalEqns);

   HYPRE_Int index;
   HYPRE_Int *rowLengths = new HYPRE_Int[numLocalEqns];
   HYPRE_Int **colIndices = new HYPRE_Int*[numLocalEqns];

   local_row = 0;
   for (i = first_local_row; i <= last_local_row; i++)
   {
      ierr += HYPRE_ParCSRMatrixGetRow(parcsr_A, i, &size, &col_ind, &values );
      rowLengths[local_row] = size;
      colIndices[local_row] = new HYPRE_Int[size];
      for (j = 0; j < size; j++) { colIndices[local_row][j] = col_ind[j] + 1; }
      local_row++;
      HYPRE_ParCSRMatrixRestoreRow(parcsr_A, i, &size, &col_ind, &values);
   }
   H.allocateMatrix(colIndices, rowLengths);
   delete [] rowLengths;
   for (i = 0; i < numLocalEqns; i++) { delete [] colIndices[i]; }
   delete [] colIndices;

   HYPRE_Int *newColInd;

   for (i = first_local_row; i <= last_local_row; i++)
   {
      ierr += HYPRE_ParCSRMatrixGetRow(parcsr_A, i, &size, &col_ind, &values );
      newColInd = new HYPRE_Int[size];
      for (j = 0; j < size; j++) { newColInd[j] = col_ind[j] + 1; }
      H.sumIntoSystemMatrix(i + 1, size, (const HYPRE_Real*)values,
                            (const HYPRE_Int*)newColInd);
      delete [] newColInd;
      ierr += HYPRE_ParCSRMatrixRestoreRow(parcsr_A, i, &size, &col_ind, &values);
   }
   H.matrixLoadComplete();
   HYPRE_ParCSRMatrixDestroy(parcsr_A);

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   HYPRE_Real ddata = 1.0;
   HYPRE_Int  status;

   for (i = first_local_row; i <= last_local_row; i++)
   {
      index = i + 1;
      H.sumIntoRHSVector(1, (const HYPRE_Real*) &ddata, (const HYPRE_Int*) &index);
   }

   hypre_EndTiming(time_index);
   strcpy(paramString, "LS Interface");
   hypre_PrintTiming(paramString, hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/

   if ( solver_id == 0 )
   {
      strcpy(paramString, "solver cg");
      H.parameters(1, &paramString);
      if (myid == 0) { hypre_printf("Solver: DS-PCG\n"); }

      strcpy(paramString, "preconditioner diagonal");
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 1 )
   {
      strcpy(paramString, "solver cg");
      H.parameters(1, &paramString);
      if (myid == 0) { hypre_printf("Solver: ParaSails-PCG\n"); }

      strcpy(paramString, "preconditioner parasails");
      H.parameters(1, &paramString);
      strcpy(paramString, "parasailsNlevels 1");
      H.parameters(1, &paramString);
      strcpy(paramString, "parasailsThreshold 0.1");
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 2 )
   {
      strcpy(paramString, "solver cg");
      H.parameters(1, &paramString);
      if (myid == 0) { hypre_printf("Solver: AMG-PCG\n"); }

      strcpy(paramString, "preconditioner boomeramg");
      H.parameters(1, &paramString);
      strcpy(paramString, "amgCoarsenType falgout");
      H.parameters(1, &paramString);
      hypre_sprintf(paramString, "amgStrongThreshold %e", strong_threshold);
      H.parameters(1, &paramString);
      hypre_sprintf(paramString, "amgNumSweeps %d", num_grid_sweeps);
      H.parameters(1, &paramString);
      strcpy(paramString, "amgRelaxType jacobi");
      H.parameters(1, &paramString);
      hypre_sprintf(paramString, "amgRelaxWeight %e", relax_weight);
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 3 )
   {
      strcpy(paramString, "solver cg");
      H.parameters(1, &paramString);
      if (myid == 0) { hypre_printf("Solver: Poly-PCG\n"); }

      strcpy(paramString, "preconditioner poly");
      H.parameters(1, &paramString);
      strcpy(paramString, "polyOrder 9");
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 4 )
   {
      strcpy(paramString, "solver gmres");
      H.parameters(1, &paramString);
      hypre_sprintf(paramString, "gmresDim %d", k_dim);
      H.parameters(1, &paramString);
      if (myid == 0) { hypre_printf("Solver: DS-GMRES\n"); }

      strcpy(paramString, "preconditioner diagonal");
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 5 )
   {
      strcpy(paramString, "solver gmres");
      H.parameters(1, &paramString);
      hypre_sprintf(paramString, "gmresDim %d", k_dim);
      H.parameters(1, &paramString);
      if (myid == 0) { hypre_printf("Solver: PILUT-GMRES\n"); }

      strcpy(paramString, "preconditioner pilut");
      H.parameters(1, &paramString);
      strcpy(paramString, "pilutRowSize 0");
      H.parameters(1, &paramString);
      strcpy(paramString, "pilutDropTol 0.0");
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 6 )
   {
      strcpy(paramString, "solver gmres");
      H.parameters(1, &paramString);
      hypre_sprintf(paramString, "gmresDim %d", k_dim);
      H.parameters(1, &paramString);
      if (myid == 0) { hypre_printf("Solver: AMG-GMRES\n"); }

      strcpy(paramString, "preconditioner boomeramg");
      H.parameters(1, &paramString);
      strcpy(paramString, "amgCoarsenType falgout");
      H.parameters(1, &paramString);
      hypre_sprintf(paramString, "amgStrongThreshold %e", strong_threshold);
      H.parameters(1, &paramString);
      hypre_sprintf(paramString, "amgNumSweeps %d", num_grid_sweeps);
      H.parameters(1, &paramString);
      strcpy(paramString, "amgRelaxType jacobi");
      H.parameters(1, &paramString);
      hypre_sprintf(paramString, "amgRelaxWeight %e", relax_weight);
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 7 )
   {
      strcpy(paramString, "solver gmres");
      H.parameters(1, &paramString);
      hypre_sprintf(paramString, "gmresDim %d", k_dim);
      H.parameters(1, &paramString);
      if (myid == 0) { hypre_printf("Solver: DDILUT-GMRES\n"); }

      strcpy(paramString, "preconditioner ddilut");
      H.parameters(1, &paramString);
      strcpy(paramString, "ddilutFillin 5.0");
      H.parameters(1, &paramString);
      strcpy(paramString, "ddilutDropTol 0.0");
      H.parameters(1, &paramString);
   }
   else if ( solver_id == 8 )
   {
      strcpy(paramString, "solver gmres");
      H.parameters(1, &paramString);
      hypre_sprintf(paramString, "gmresDim %d", k_dim);
      H.parameters(1, &paramString);
      if (myid == 0) { hypre_printf("Solver: POLY-GMRES\n"); }

      strcpy(paramString, "preconditioner poly");
      H.parameters(1, &paramString);
      strcpy(paramString, "polyOrder 5");
      H.parameters(1, &paramString);
   }

   strcpy(paramString, "Krylov Solve");
   time_index = hypre_InitializeTiming(paramString);
   hypre_BeginTiming(time_index);

   H.launchSolver(status, num_iterations);

   hypre_EndTiming(time_index);
   strcpy(paramString, "Solve phase times");
   hypre_PrintTiming(paramString, hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   if (myid == 0)
   {
      hypre_printf("\n Iterations = %d\n", num_iterations);
      hypre_printf("\n");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   delete [] paramString;
   hypre_MPI_Finalize();

   return (0);
}

/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian27pt( HYPRE_Int                  argc,
                       char                *argv[],
                       HYPRE_Int                  arg_index,
                       HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 20;
   ny = 20;
   nz = 20;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Laplacian_27pt:\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(HYPRE_Real,  2, HYPRE_MEMORY_HOST);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
   {
      values[0] = 8.0;
   }
   if (nx * ny == 1 || nx * nz == 1 || ny * nz == 1)
   {
      values[0] = 2.0;
   }
   values[1] = -1.0;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 19

   Interface:    Linear-Algebraic (IJ)

   Compile with: make ex19

   Sample run:   mpirun -np 4 ex19

   Description:  This example solves a 1D steady upwinded advection problem using
                 AIR. With one up sweep and one coarse sweep, AIR converges in one
                 iteration. If the transpose of the interpolation operator is used
                 instead (-restriction 0), the multigrid solve fails to converge
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "ex.h"

#define my_min(a,b)  (((a)<(b)) ? (a) : (b))

int main (int argc, char *argv[])
{
   int i;
   int myid, num_procs;
   int N, n = 33;

   int ilower, iupper;
   int local_size, extra;

   int print_system = 0;

   HYPRE_IJMatrix A;
   HYPRE_ParCSRMatrix parcsr_A;
   HYPRE_IJVector b;
   HYPRE_ParVector par_b;
   HYPRE_IJVector x;
   HYPRE_ParVector par_x;
   HYPRE_IJVector work;
   HYPRE_ParVector par_work;

   HYPRE_Solver solver;
   /* AMG */
   int num_iterations;
   double final_res_norm;
   HYPRE_Int interp_type = 100; /* 1-pt Interp */
   HYPRE_Int ns_coarse = 1, ns_down = 0, ns_up = 1;
   HYPRE_Int **grid_relax_points = NULL;
   HYPRE_Int coarse_threshold = 20;
   HYPRE_Int agg_num_levels = 0; /* AIR does not support aggressive coarsening */
   HYPRE_Int restriction = 1;


   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Initialize HYPRE */
   HYPRE_Initialize();

   /* Print GPU info */
   /* HYPRE_PrintDeviceInfo(); */
#if defined(HYPRE_USING_GPU)
   /* use vendor implementation for SpGEMM */
   HYPRE_SetSpGemmUseVendor(0);
#endif

   /* Parse command line */
   {
      int arg_index = 0;
      int print_usage = 0;

      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-n") == 0 )
         {
            arg_index++;
            n = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-print_system") == 0 )
         {
            arg_index++;
            print_system = 1;
         }
         else if ( strcmp(argv[arg_index], "-help") == 0 )
         {
            print_usage = 1;
            break;
         }
         else if ( strcmp(argv[arg_index], "-ns_coarse") == 0)
         {
            arg_index++;
            ns_coarse = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-ns_down") == 0)
         {
            arg_index++;
            ns_down = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-ns_up") == 0)
         {
            arg_index++;
            ns_up = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-restriction") == 0)
         {
            arg_index++;
            restriction = atoi(argv[arg_index++]);
         }
         else
         {
            arg_index++;
         }
      }

      if (!restriction)
         // Do classical modified interpolation if not doing AIR
         interp_type = 0;

      if ((print_usage) && (myid == 0))
      {
         printf("\n");
         printf("Usage: %s [<options>]\n", argv[0]);
         printf("\n");
         printf("  -n <n>              : problem size (default: 33)\n");
         printf("  -print_system       : print the matrix, rhs, expected solution, and solution\n");
         printf("  -ns_coarse          : the number of sweeps performed for the coarse cycle "
                                        "(default: 1)\n");
         printf("  -ns_down            : the number of sweeps performed for the down cycle "
                                        "(default: 0)\n");
         printf("  -ns_up              : the number of sweeps performed for the up cycle "
                                        "(default: 1)\n");
         printf("  -restriction        : defines the restriction operator, 0 for transpose of "
                                        "interpolation, 1 for AIR-1, 2 for AIR-2 (default: 1)\n");
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   /* Preliminaries: want at least one processor per row */
   N = n; /* global number of rows */

   /* Each processor knows only of its own rows - the range is denoted by ilower
      and upper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   local_size = N / num_procs;
   extra = N - local_size * num_procs;

   ilower = local_size * myid;
   ilower += my_min(myid, extra);

   iupper = local_size * (myid + 1);
   iupper += my_min(myid + 1, extra);
   iupper = iupper - 1;

   /* How many rows do I have? */
   local_size = iupper - ilower + 1;

   /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);

   /* Choose a parallel csr format storage (see the User's Manual) */
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);

   /* Initialize before setting coefficients */
   HYPRE_IJMatrixInitialize(A);

   /* Set matrix entries corresponding to 1D upwind advection */
   {
      int nnz;
      /* OK to use constant-length arrays for CPUs
      double values[2];
      int cols[2];
      */
      double *values = (double *) malloc(2 * sizeof(double));
      int *cols = (int *) malloc(2 * sizeof(int));
      int *tmp = (int *) malloc(2 * sizeof(int));

      for (i = ilower; i <= iupper; i++)
      {
        if (i == 0)
        {
          nnz = 1;
          cols[0] = 0;
          values[0] = 1;
        }
        else
        {
          nnz = 2;
          cols[0] = i - 1;
          cols[1] = i;
          values[0] = -1;
          values[1] = 1;
        }

         /* Set the values for row i */
         tmp[0] = nnz;
         tmp[1] = i;
         HYPRE_IJMatrixSetValues(A, 1, &tmp[0], &tmp[1], cols, values);
      }

      free(values);
      free(cols);
      free(tmp);
   }

   /* Assemble after setting the coefficients */
   HYPRE_IJMatrixAssemble(A);

   /* Note: for the testing of small problems, one may wish to read
      in a matrix in IJ format (for the format, see the output files
      from the -print_system option).
      In this case, one would use the following routine:
      HYPRE_IJMatrixRead( <filename>, MPI_COMM_WORLD,
                          HYPRE_PARCSR, &A );
      <filename>  = IJ.A.out to read in what has been printed out
      by -print_system (processor numbers are omitted).
      A call to HYPRE_IJMatrixRead is an *alternative* to the
      following sequence of HYPRE_IJMatrix calls:
      Create, SetObjectType, Initialize, SetValues, and Assemble
   */


   /* Get the parcsr matrix object to use */
   HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);


   /* Create the solution, work vector, and b */
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &work);
   HYPRE_IJVectorSetObjectType(work, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(work);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);

   HYPRE_IJVectorGetObject(x, (void **) &par_x);
   HYPRE_ParVectorSetRandomValues(par_x, 1);

   HYPRE_IJVectorGetObject(work, (void **) &par_work);
   HYPRE_ParVectorSetRandomValues(par_work, 2);

   HYPRE_IJVectorGetObject(b, (void **) &par_b);
   HYPRE_ParVectorSetConstantValues(par_b, 0);
   HYPRE_ParCSRMatrixMatvec(1, parcsr_A, par_work, 0, par_b);


   /*  Print out the system  - files names will be IJ.out.A.XXXXX
        and IJ.out.b.XXXXX, where XXXXX = processor id */
   if (print_system)
   {
      HYPRE_IJMatrixPrint(A, "IJ.out.A");
      HYPRE_IJVectorPrint(work, "IJ.out.work");
      HYPRE_IJVectorPrint(b, "IJ.out.b");
   }

   /* this is a 2-D 4-by-k array using Double pointers */
   grid_relax_points = hypre_CTAlloc(HYPRE_Int*, 4, HYPRE_MEMORY_HOST);
   grid_relax_points[0] = NULL;
   grid_relax_points[1] = hypre_CTAlloc(HYPRE_Int, ns_down, HYPRE_MEMORY_HOST);
   grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int, ns_up, HYPRE_MEMORY_HOST);
   grid_relax_points[3] = hypre_CTAlloc(HYPRE_Int, ns_coarse, HYPRE_MEMORY_HOST);
   /* down cycle: C */
   for (i = 0; i < ns_down; i++)
   {
     grid_relax_points[1][i] = 1; // C
   }
   /* up cycle: F */
   for (i=0; i<ns_up; i++)
   {
     grid_relax_points[2][0] = -1; // F
   }
   /* coarse: all */
   for (i = 0; i < ns_coarse; i++)
   {
     grid_relax_points[3][i] = 0; // A(ll)
   }

   /* Create solver */
   HYPRE_BoomerAMGCreate(&solver);

   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_BoomerAMGSetPrintLevel(solver, 3);  /* print solve info + parameters */
   HYPRE_BoomerAMGSetRelaxType(solver, 0);   /* Jacobi */
   HYPRE_BoomerAMGSetRelaxOrder(solver, 1);   /* uses C/F relaxation */
   HYPRE_BoomerAMGSetCycleNumSweeps(solver, ns_down, 1);
   HYPRE_BoomerAMGSetCycleNumSweeps(solver, ns_up, 2);
   HYPRE_BoomerAMGSetCycleNumSweeps(solver, ns_coarse, 3);
   HYPRE_BoomerAMGSetGridRelaxPoints(solver, grid_relax_points);
   HYPRE_BoomerAMGSetMaxLevels(solver, 2);  /* maximum number of levels */
   HYPRE_BoomerAMGSetTol(solver, 1e-7);      /* conv. tolerance */
   HYPRE_BoomerAMGSetRestriction(solver, restriction); /* AIR */
   HYPRE_BoomerAMGSetAggNumLevels(solver, agg_num_levels);
   HYPRE_BoomerAMGSetMaxCoarseSize(solver, coarse_threshold);
   HYPRE_BoomerAMGSetInterpType(solver, interp_type);
   HYPRE_BoomerAMGSetMaxIter(solver, 100);


   /* Now setup and solve! */
   HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
   HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);

   /* Run info - needed logging turned on */
   HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
   HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
   if (myid == 0)
   {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %e\n", final_res_norm);
      printf("\n");
   }

   /* Destroy solver */
   HYPRE_BoomerAMGDestroy(solver);

   /* Print solution */
   if (print_system)
      HYPRE_IJVectorPrint(x, "IJ.out.x");

   /* Clean up */
   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJVectorDestroy(work);

   /* Finalize HYPRE */
   HYPRE_Finalize();

   /* Finalize MPI*/
   MPI_Finalize();

   return (0);
}

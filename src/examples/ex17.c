/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 17

   Interface:      Structured interface (Struct)

   Compile with:   make ex17

   Sample run:     mpirun -np 16 ex17 -n 10

   To see options: ex17 -help

   Description:    This code solves an "NDIM-D Laplacian" using CG.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HYPRE_struct_ls.h"
#include "ex.h"

#define NDIM 4
#define NSTENC (2*NDIM+1)

int main (int argc, char *argv[])
{
   int d, i, j;
   int myid, num_procs;
   int n, N, nvol, div, rem;
   int p[NDIM], ilower[NDIM], iupper[NDIM];

   int solver_id;

   HYPRE_StructGrid     grid;
   HYPRE_StructStencil  stencil;
   HYPRE_StructMatrix   A;
   HYPRE_StructVector   b;
   HYPRE_StructVector   x;
   HYPRE_StructSolver   solver;

   int num_iterations;
   double final_res_norm;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Initialize HYPRE */
   HYPRE_Initialize();

   /* Print GPU info */
   /* HYPRE_PrintDeviceInfo(); */

   /* Set defaults */
   n = 10;
   solver_id = 0;

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
         else if ( strcmp(argv[arg_index], "-solver") == 0 )
         {
            arg_index++;
            solver_id = atoi(argv[arg_index++]);
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
         printf("\n");
         printf("Usage: %s [<options>]\n", argv[0]);
         printf("\n");
         printf("  -n <n>              : problem size per processor (default: 33)\n");
         printf("  -solver <ID>        : solver ID\n");
         printf("                        0 - CG (default)\n");
         printf("                        1 - GMRES\n");
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   nvol = pow(n, NDIM);

   /* Figure out the processor grid (N x N x N x N).  The local problem size for
      the interior nodes is indicated by n (n x n x n x n).  p indicates the
      position in the processor grid. */
   N  = pow(num_procs, 1.0 / NDIM) + 1.0e-6;
   div = pow(N, NDIM);
   rem = myid;
   if (num_procs != div)
   {
      printf("Num procs is not a perfect NDIM-th root!\n");
      MPI_Finalize();
      exit(1);
   }
   for (d = NDIM - 1; d >= 0; d--)
   {
      div /= N;
      p[d] = rem / div;
      rem %= div;
   }

   /* Figure out the extents of each processor's piece of the grid. */
   for (d = 0; d < NDIM; d++)
   {
      ilower[d] = p[d] * n;
      iupper[d] = ilower[d] + n - 1;
   }

   /* 1. Set up a grid */
   {
      /* Create an empty 2D grid object */
      HYPRE_StructGridCreate(MPI_COMM_WORLD, NDIM, &grid);

      /* Add a new box to the grid */
      HYPRE_StructGridSetExtents(grid, ilower, iupper);

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      HYPRE_StructGridAssemble(grid);
   }

   /* 2. Define the discretization stencil */
   {
      /* Create an empty NDIM-D, NSTENC-pt stencil object */
      HYPRE_StructStencilCreate(NDIM, NSTENC, &stencil);

      /* Define the geometry of the stencil */
      {
         int entry;
         int offset[NDIM];

         entry = 0;
         for (d = 0; d < NDIM; d++)
         {
            offset[d] = 0;
         }
         HYPRE_StructStencilSetElement(stencil, entry++, offset);
         for (d = 0; d < NDIM; d++)
         {
            offset[d] = -1;
            HYPRE_StructStencilSetElement(stencil, entry++, offset);
            offset[d] =  1;
            HYPRE_StructStencilSetElement(stencil, entry++, offset);
            offset[d] =  0;
         }
      }
   }

   /* 3. Set up a Struct Matrix */
   {
      int nentries = NSTENC;
      int nvalues  = nentries * nvol;
      double *values;
      int stencil_indices[NSTENC];

      /* Create an empty matrix object */
      HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);

      /* Indicate that the matrix coefficients are ready to be set */
      HYPRE_StructMatrixInitialize(A);

      values = (double*) calloc(nvalues, sizeof(double));

      for (j = 0; j < nentries; j++)
      {
         stencil_indices[j] = j;
      }

      /* Set the standard stencil at each grid point; fix boundaries later */
      for (i = 0; i < nvalues; i += nentries)
      {
         values[i] = NSTENC; /* Use absolute row sum */
         for (j = 1; j < nentries; j++)
         {
            values[i + j] = -1.0;
         }
      }

      HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
                                     stencil_indices, values);

      free(values);
   }

   /* 4. Incorporate zero boundary conditions: go along each edge of the domain
         and set the stencil entry that reaches to the boundary to zero.*/
   {
      int bc_ilower[NDIM];
      int bc_iupper[NDIM];
      int nentries = 1;
      int nvalues  = nentries * nvol / n; /* number of stencil entries times the
                                         length of one side of my grid box */
      double *values;
      int stencil_indices[1];

      values = (double*) calloc(nvalues, sizeof(double));
      for (j = 0; j < nvalues; j++)
      {
         values[j] = 0.0;
      }

      for (d = 0; d < NDIM; d++)
      {
         bc_ilower[d] = ilower[d];
         bc_iupper[d] = iupper[d];
      }
      stencil_indices[0] = 1;
      for (d = 0; d < NDIM; d++)
      {
         /* lower boundary in dimension d */
         if (p[d] == 0)
         {
            bc_iupper[d] = ilower[d];
            HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                           stencil_indices, values);
            bc_iupper[d] = iupper[d];
         }
         stencil_indices[0]++;

         /* upper boundary in dimension d */
         if (p[d] == N - 1)
         {
            bc_ilower[d] = iupper[d];
            HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                           stencil_indices, values);
            bc_ilower[d] = ilower[d];
         }
         stencil_indices[0]++;
      }

      free(values);
   }

   /* This is a collective call finalizing the matrix assembly.
      The matrix is now ``ready to be used'' */
   HYPRE_StructMatrixAssemble(A);

   /* 5. Set up Struct Vectors for b and x */
   {
      int     nvalues = nvol;
      double *values;

      values = (double*) calloc(nvalues, sizeof(double));

      /* Create an empty vector object */
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);

      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_StructVectorInitialize(b);
      HYPRE_StructVectorInitialize(x);

      /* Set the values */
      for (i = 0; i < nvalues; i ++)
      {
         values[i] = 1.0;
      }
      HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

      for (i = 0; i < nvalues; i ++)
      {
         values[i] = 0.0;
      }
      HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);

      free(values);

      /* This is a collective call finalizing the vector assembly.
         The vector is now ``ready to be used'' */
      HYPRE_StructVectorAssemble(b);
      HYPRE_StructVectorAssemble(x);
   }

#if 0
   HYPRE_StructMatrixPrint("ex17.out.A", A, 0);
   HYPRE_StructVectorPrint("ex17.out.b", b, 0);
   HYPRE_StructVectorPrint("ex17.out.x0", x, 0);
#endif

   /* 6. Set up and use a struct solver
      (Solver options can be found in the Reference Manual.) */
   if (solver_id == 0)
   {
      HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructPCGSetMaxIter(solver, 100);
      HYPRE_StructPCGSetTol(solver, 1.0e-06);
      HYPRE_StructPCGSetTwoNorm(solver, 1);
      HYPRE_StructPCGSetRelChange(solver, 0);
      HYPRE_StructPCGSetPrintLevel(solver, 2); /* print each CG iteration */
      HYPRE_StructPCGSetLogging(solver, 1);

      /* No preconditioner */

      HYPRE_StructPCGSetup(solver, A, b, x);
      HYPRE_StructPCGSolve(solver, A, b, x);

      /* Get some info on the run */
      HYPRE_StructPCGGetNumIterations(solver, &num_iterations);
      HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

      /* Clean up */
      HYPRE_StructPCGDestroy(solver);
   }

   if (myid == 0)
   {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %g\n", final_res_norm);
      printf("\n");
   }

   /* Free memory */
   HYPRE_StructGridDestroy(grid);
   HYPRE_StructStencilDestroy(stencil);
   HYPRE_StructMatrixDestroy(A);
   HYPRE_StructVectorDestroy(b);
   HYPRE_StructVectorDestroy(x);

   /* Finalize HYPRE */
   HYPRE_Finalize();

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}

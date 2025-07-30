/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 18

   Interface:      SStructured interface (SStruct)

   Compile with:   make ex18

   Sample run:     mpirun -np 16 ex18 -n 4

   To see options: ex18 -help

   Description:    This code solves an "NDIM-D Laplacian" using CG.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HYPRE_sstruct_ls.h"
#include "ex.h"

#define NDIM   4
#define NPARTS 1
#define NVARS  2
#define NSTENC NVARS*(2*NDIM+1)

int main (int argc, char *argv[])
{
   int d, i, j;
   int myid, num_procs;
   int n, N, nvol, div, rem;
   int p[NDIM], ilower[NDIM], iupper[NDIM];

   int solver_id, object_type = HYPRE_SSTRUCT;

   HYPRE_SStructGrid     grid;
   HYPRE_SStructStencil  stencil0, stencil1;
   HYPRE_SStructGraph    graph;
   HYPRE_SStructMatrix   A;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   x;

   HYPRE_SStructSolver   solver;

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
   n = 4;
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
         printf("  -n <n>         : problem size per processor (default: 4)\n");
         printf("  -solver <ID>   : solver ID\n");
         printf("                   0 - CG (default)\n");
         printf("                   1 - GMRES\n");
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
      int part = 0;
      HYPRE_SStructVariable vartypes[NVARS] = {HYPRE_SSTRUCT_VARIABLE_CELL,
                                               HYPRE_SSTRUCT_VARIABLE_CELL
                                              };

      /* Create an empty 2D grid object */
      HYPRE_SStructGridCreate(MPI_COMM_WORLD, NDIM, NPARTS, &grid);

      /* Add a new box to the grid */
      HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);

      /* Set the variable type and number of variables on each part. */
      HYPRE_SStructGridSetVariables(grid, part, NVARS, vartypes);

      /* The grid is now ready to use */
      HYPRE_SStructGridAssemble(grid);
   }

   /* 2. Define the discretization stencil */
   {
      /* Create two empty NDIM-D, NSTENC-pt stencil objects */
      HYPRE_SStructStencilCreate(NDIM, NSTENC, &stencil0);
      HYPRE_SStructStencilCreate(NDIM, NSTENC, &stencil1);

      /* Define the geometry of the stencil */
      {
         int entry, var0 = 0, var1 = 1;
         int offset[NDIM];

         entry = 0;
         for (d = 0; d < NDIM; d++)
         {
            offset[d] = 0;
         }
         HYPRE_SStructStencilSetEntry(stencil0, entry, offset, var0);
         HYPRE_SStructStencilSetEntry(stencil1, entry, offset, var1);
         entry++;
         HYPRE_SStructStencilSetEntry(stencil0, entry, offset, var1);
         HYPRE_SStructStencilSetEntry(stencil1, entry, offset, var0);
         entry++;
         for (d = 0; d < NDIM; d++)
         {
            offset[d] = -1;
            HYPRE_SStructStencilSetEntry(stencil0, entry, offset, var0);
            HYPRE_SStructStencilSetEntry(stencil1, entry, offset, var1);
            entry++;
            HYPRE_SStructStencilSetEntry(stencil0, entry, offset, var1);
            HYPRE_SStructStencilSetEntry(stencil1, entry, offset, var0);
            entry++;
            offset[d] =  1;
            HYPRE_SStructStencilSetEntry(stencil0, entry, offset, var0);
            HYPRE_SStructStencilSetEntry(stencil1, entry, offset, var1);
            entry++;
            HYPRE_SStructStencilSetEntry(stencil0, entry, offset, var1);
            HYPRE_SStructStencilSetEntry(stencil1, entry, offset, var0);
            entry++;
            offset[d] =  0;
         }
      }
   }

   /* 3. Set up the Graph */
   {
      int part = 0;
      int var0 = 0, var1 = 1;

      /* Create the graph object */
      HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);

      /* Set up the object type (see Matrix and VectorSetObjectType below) */
      HYPRE_SStructGraphSetObjectType(graph, object_type);

      /* Set the stencil */
      HYPRE_SStructGraphSetStencil(graph, part, var0, stencil0);
      HYPRE_SStructGraphSetStencil(graph, part, var1, stencil1);

      /* Assemble the graph */
      HYPRE_SStructGraphAssemble(graph);
   }

   /* 4. Set up the Matrix */
   {
      int part = 0;
      int var0 = 0, var1 = 1;
      int nentries  = NSTENC / NVARS;
      int nvalues   = nentries * nvol;
      double *values;
      int stencil_indices[NSTENC];

      /* Create an empty matrix object */
      HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);

      /* Set up the object type */
      HYPRE_SStructMatrixSetObjectType(A, object_type);

      /* Get ready to set values */
      HYPRE_SStructMatrixInitialize(A);

      values = (double*) calloc(nvalues, sizeof(double));

      /* Set intra-variable values; fix boundaries later */
      for (j = 0; j < nentries; j++)
      {
         stencil_indices[j] = 2 * j;
      }
      for (i = 0; i < nvalues; i += nentries)
      {
         values[i]   = 1.1 * (NSTENC / NVARS); /* Diagonal: Use absolute row sum */
         for (j = 1; j < nentries; j++)
         {
            values[i + j] = -1.0;
         }
      }
      HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper, var0,
                                      nentries, stencil_indices, values);
      HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper, var1,
                                      nentries, stencil_indices, values);

      /* Set inter-variable values; fix boundaries later */
      for (j = 0; j < nentries; j++)
      {
         stencil_indices[j] = 2 * j + 1;
      }
      for (i = 0; i < nvalues; i += nentries)
      {
         values[i] = -0.1;
         for (j = 1; j < nentries; j++)
         {
            values[i + j] = -0.1;
         }
      }
      HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper, var0,
                                      nentries, stencil_indices, values);
      HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper, var1,
                                      nentries, stencil_indices, values);

      free(values);
   }

   /* 5. Incorporate zero boundary conditions: go along each edge of the domain
         and set the stencil entry that reaches to the boundary to zero.*/
   {
      int part = 0;
      int var0 = 0, var1 = 1;
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
      stencil_indices[0] = NVARS;
      for (d = 0; d < NDIM; d++)
      {
         /* lower boundary in dimension d */
         if (p[d] == 0)
         {
            bc_iupper[d] = ilower[d];
            for (i = 0; i < NVARS; i++)
            {
               HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, var0,
                                               nentries, stencil_indices, values);
               HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, var1,
                                               nentries, stencil_indices, values);
               stencil_indices[0]++;
            }
            bc_iupper[d] = iupper[d];
         }
         else
         {
            stencil_indices[0] += NVARS;
         }

         /* upper boundary in dimension d */
         if (p[d] == N - 1)
         {
            bc_ilower[d] = iupper[d];
            for (i = 0; i < NVARS; i++)
            {
               HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, var0,
                                               nentries, stencil_indices, values);
               HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, var1,
                                               nentries, stencil_indices, values);
               stencil_indices[0]++;
            }
            bc_ilower[d] = ilower[d];
         }
         else
         {
            stencil_indices[0] += NVARS;
         }
      }

      free(values);
   }

   /* The matrix is now ready to use */
   HYPRE_SStructMatrixAssemble(A);

   /* 6. Set up Vectors for b and x */
   {
      int part = 0;
      int var0 = 0, var1 = 1;
      int nvalues = NVARS * nvol;
      double *values;

      values = (double*) calloc(nvalues, sizeof(double));

      /* Create an empty vector object */
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);

      /* Set up the object type */
      HYPRE_SStructVectorSetObjectType(b, object_type);
      HYPRE_SStructVectorSetObjectType(x, object_type);

      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_SStructVectorInitialize(b);
      HYPRE_SStructVectorInitialize(x);

      /* Set the values */
      for (i = 0; i < nvalues; i ++)
      {
         values[i] = 1.0;
      }
      HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var0, values);
      HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var1, values);

      for (i = 0; i < nvalues; i ++)
      {
         values[i] = 0.0;
      }
      HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var0, values);
      HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var1, values);

      free(values);

      /* The vector is now ready to use */
      HYPRE_SStructVectorAssemble(b);
      HYPRE_SStructVectorAssemble(x);
   }

#if 0
   HYPRE_SStructMatrixPrint("ex18.out.A", A, 0);
   HYPRE_SStructVectorPrint("ex18.out.b", b, 0);
   HYPRE_SStructVectorPrint("ex18.out.x0", x, 0);
#endif

   /* 7. Set up and use a struct solver */
   if (solver_id == 0)
   {
      HYPRE_SStructPCGCreate(MPI_COMM_WORLD, &solver);
      HYPRE_SStructPCGSetMaxIter(solver, 100);
      HYPRE_SStructPCGSetTol(solver, 1.0e-06);
      HYPRE_SStructPCGSetTwoNorm(solver, 1);
      HYPRE_SStructPCGSetRelChange(solver, 0);
      HYPRE_SStructPCGSetPrintLevel(solver, 2); /* print each CG iteration */
      HYPRE_SStructPCGSetLogging(solver, 1);

      /* No preconditioner */

      HYPRE_SStructPCGSetup(solver, A, b, x);
      HYPRE_SStructPCGSolve(solver, A, b, x);

      /* Get some info on the run */
      HYPRE_SStructPCGGetNumIterations(solver, &num_iterations);
      HYPRE_SStructPCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

      /* Clean up */
      HYPRE_SStructPCGDestroy(solver);
   }

   if (myid == 0)
   {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %g\n", final_res_norm);
      printf("\n");
   }

   /* Free memory */
   HYPRE_SStructGridDestroy(grid);
   HYPRE_SStructGraphDestroy(graph);
   HYPRE_SStructStencilDestroy(stencil0);
   HYPRE_SStructStencilDestroy(stencil1);
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);

   /* Finalize HYPRE */
   HYPRE_Finalize();

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}

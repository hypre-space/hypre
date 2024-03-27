/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 16

   Interface:      Semi-Structured interface (SStruct)

   Compile with:   make ex16

   Sample run:     mpirun -np 4 ex16 -n 10

   To see options: ex16 -help

   Description:    This code solves the 2D Laplace equation using a high order
                   Q3 finite element discretization.  Specifically, we solve
                   -Delta u = 1 with zero boundary conditions on a unit square
                   domain meshed with a uniform grid.  The mesh is distributed
                   across an N x N process grid, with each processor containing
                   an n x n sub-mesh of data, so the global mesh is nN x nN.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HYPRE_sstruct_mv.h"
#include "HYPRE_sstruct_ls.h"
#include "HYPRE.h"
#include "ex.h"

#ifdef HYPRE_EXVIS
#include "vis.c"
#endif

/*
   This routine computes the stiffness matrix for the Laplacian on a square of
   size h, using bi-cubic elements with degrees of freedom in lexicographical
   ordering.  So, the element looks as follows:

                          [12]-[13]-[14]-[15]
                            |              |
                           [8]  [9] [10] [11]
                            |              |
                           [4]  [5]  [6]  [7]
                            |              |
                           [0]--[1]--[2]--[3]
*/
void ComputeFEMQ3 (double S[16][16], double F[16], double h)
{
   int i, j;
   double s = 1.0 / 33600;
   double h2_64 = h * h / 64;

   S[ 0][ 0] = 18944 * s;
   S[ 0][ 1] = -4770 * s;
   S[ 0][ 2] = 792 * s;
   S[ 0][ 3] = 574 * s;
   S[ 0][ 4] = -4770 * s;
   S[ 0][ 5] = -18711 * s;
   S[ 0][ 6] = 6075 * s;
   S[ 0][ 7] = -2439 * s;
   S[ 0][ 8] = 792 * s;
   S[ 0][ 9] = 6075 * s;
   S[ 0][10] = -1944 * s;
   S[ 0][11] = 747 * s;
   S[ 0][12] = 574 * s;
   S[ 0][13] = -2439 * s;
   S[ 0][14] = 747 * s;
   S[ 0][15] = -247 * s;

   S[ 1][ 1] = 75600 * s;
   S[ 1][ 2] = -25002 * s;
   S[ 1][ 3] = 792 * s;
   S[ 1][ 4] = -18711 * s;
   S[ 1][ 5] = -39852 * s;
   S[ 1][ 6] = -7047 * s;
   S[ 1][ 7] = 6075 * s;
   S[ 1][ 8] = 6075 * s;
   S[ 1][ 9] = 9720 * s;
   S[ 1][10] = 3159 * s;
   S[ 1][11] = -1944 * s;
   S[ 1][12] = -2439 * s;
   S[ 1][13] = -108 * s;
   S[ 1][14] = -2295 * s;
   S[ 1][15] = 747 * s;

   S[ 2][ 2] = 75600 * s;
   S[ 2][ 3] = -4770 * s;
   S[ 2][ 4] = 6075 * s;
   S[ 2][ 5] = -7047 * s;
   S[ 2][ 6] = -39852 * s;
   S[ 2][ 7] = -18711 * s;
   S[ 2][ 8] = -1944 * s;
   S[ 2][ 9] = 3159 * s;
   S[ 2][10] = 9720 * s;
   S[ 2][11] = 6075 * s;
   S[ 2][12] = 747 * s;
   S[ 2][13] = -2295 * s;
   S[ 2][14] = -108 * s;
   S[ 2][15] = -2439 * s;

   S[ 3][ 3] = 18944 * s;
   S[ 3][ 4] = -2439 * s;
   S[ 3][ 5] = 6075 * s;
   S[ 3][ 6] = -18711 * s;
   S[ 3][ 7] = -4770 * s;
   S[ 3][ 8] = 747 * s;
   S[ 3][ 9] = -1944 * s;
   S[ 3][10] = 6075 * s;
   S[ 3][11] = 792 * s;
   S[ 3][12] = -247 * s;
   S[ 3][13] = 747 * s;
   S[ 3][14] = -2439 * s;
   S[ 3][15] = 574 * s;

   S[ 4][ 4] = 75600 * s;
   S[ 4][ 5] = -39852 * s;
   S[ 4][ 6] = 9720 * s;
   S[ 4][ 7] = -108 * s;
   S[ 4][ 8] = -25002 * s;
   S[ 4][ 9] = -7047 * s;
   S[ 4][10] = 3159 * s;
   S[ 4][11] = -2295 * s;
   S[ 4][12] = 792 * s;
   S[ 4][13] = 6075 * s;
   S[ 4][14] = -1944 * s;
   S[ 4][15] = 747 * s;

   S[ 5][ 5] = 279936 * s;
   S[ 5][ 6] = -113724 * s;
   S[ 5][ 7] = 9720 * s;
   S[ 5][ 8] = -7047 * s;
   S[ 5][ 9] = -113724 * s;
   S[ 5][10] = 24057 * s;
   S[ 5][11] = 3159 * s;
   S[ 5][12] = 6075 * s;
   S[ 5][13] = 9720 * s;
   S[ 5][14] = 3159 * s;
   S[ 5][15] = -1944 * s;

   S[ 6][ 6] = 279936 * s;
   S[ 6][ 7] = -39852 * s;
   S[ 6][ 8] = 3159 * s;
   S[ 6][ 9] = 24057 * s;
   S[ 6][10] = -113724 * s;
   S[ 6][11] = -7047 * s;
   S[ 6][12] = -1944 * s;
   S[ 6][13] = 3159 * s;
   S[ 6][14] = 9720 * s;
   S[ 6][15] = 6075 * s;

   S[ 7][ 7] = 75600 * s;
   S[ 7][ 8] = -2295 * s;
   S[ 7][ 9] = 3159 * s;
   S[ 7][10] = -7047 * s;
   S[ 7][11] = -25002 * s;
   S[ 7][12] = 747 * s;
   S[ 7][13] = -1944 * s;
   S[ 7][14] = 6075 * s;
   S[ 7][15] = 792 * s;

   S[ 8][ 8] = 75600 * s;
   S[ 8][ 9] = -39852 * s;
   S[ 8][10] = 9720 * s;
   S[ 8][11] = -108 * s;
   S[ 8][12] = -4770 * s;
   S[ 8][13] = -18711 * s;
   S[ 8][14] = 6075 * s;
   S[ 8][15] = -2439 * s;

   S[ 9][ 9] = 279936 * s;
   S[ 9][10] = -113724 * s;
   S[ 9][11] = 9720 * s;
   S[ 9][12] = -18711 * s;
   S[ 9][13] = -39852 * s;
   S[ 9][14] = -7047 * s;
   S[ 9][15] = 6075 * s;

   S[10][10] = 279936 * s;
   S[10][11] = -39852 * s;
   S[10][12] = 6075 * s;
   S[10][13] = -7047 * s;
   S[10][14] = -39852 * s;
   S[10][15] = -18711 * s;

   S[11][11] = 75600 * s;
   S[11][12] = -2439 * s;
   S[11][13] = 6075 * s;
   S[11][14] = -18711 * s;
   S[11][15] = -4770 * s;

   S[12][12] = 18944 * s;
   S[12][13] = -4770 * s;
   S[12][14] = 792 * s;
   S[12][15] = 574 * s;

   S[13][13] = 75600 * s;
   S[13][14] = -25002 * s;
   S[13][15] = 792 * s;

   S[14][14] = 75600 * s;
   S[14][15] = -4770 * s;

   S[15][15] = 18944 * s;

   /* The stiffness matrix is symmetric */
   for (i = 1; i < 16; i++)
      for (j = 0; j < i; j++)
      {
         S[i][j] = S[j][i];
      }

   F[ 0] = h2_64;
   F[ 1] = 3 * h2_64;
   F[ 2] = 3 * h2_64;
   F[ 3] = h2_64;
   F[ 4] = 3 * h2_64;
   F[ 5] = 9 * h2_64;
   F[ 6] = 9 * h2_64;
   F[ 7] = 3 * h2_64;
   F[ 8] = 3 * h2_64;
   F[ 9] = 9 * h2_64;
   F[10] = 9 * h2_64;
   F[11] = 3 * h2_64;
   F[12] = h2_64;
   F[13] = 3 * h2_64;
   F[14] = 3 * h2_64;
   F[15] = h2_64;
}


int main (int argc, char *argv[])
{
   int myid, num_procs;
   int n, N, pi, pj;
   double h;
   int vis;

   HYPRE_SStructGrid     grid;
   HYPRE_SStructGraph    graph;
   HYPRE_SStructMatrix   A;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   x;

   HYPRE_Solver          solver;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Initialize HYPRE */
   HYPRE_Initialize();

   /* Print GPU info */
   /* HYPRE_PrintDeviceInfo(); */

   /* Set default parameters */
   n = 10;
   vis = 0;

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
         else if ( strcmp(argv[arg_index], "-vis") == 0 )
         {
            arg_index++;
            vis = 1;
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
         printf("  -n <n>           : problem size per processor (default: 10)\n");
         printf("  -vis             : save the solution for GLVis visualization\n");
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   /* Figure out the processor grid (N x N).  The local problem size is n^2,
      while pi and pj indicate the position in the processor grid. */
   N  = pow(num_procs, 1.0 / 2.0) + 0.5;
   if (num_procs != N * N)
   {
      if (myid == 0)
      {
         printf("Can't run on %d processors, try %d.\n", num_procs, N * N);
      }
      MPI_Finalize();
      exit(1);
   }
   h  = 1.0 / (N * n);
   pj = myid / N;
   pi = myid - pj * N;

   /* 1. Set up the grid.  For simplicity we use only one part to represent the
         unit square. */
   {
      int ndim = 2;
      int nparts = 1;

      /* Create an empty 2D grid object */
      HYPRE_SStructGridCreate(MPI_COMM_WORLD, ndim, nparts, &grid);

      /* Set the extents of the grid - each processor sets its grid boxes. */
      {
         int part = 0;
         int ilower[2] = {1 + pi * n, 1 + pj * n};
         int iupper[2] = {n + pi * n, n + pj * n};

         HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
      }

      /* Set the variable type and number of variables on each part.  There is
         one variable of type NODE, two of type XFACE, two of type YFACE, and
         four of type CELL. */
      {
         int i;
         int nvars = 9;

         HYPRE_SStructVariable vars[9] = {HYPRE_SSTRUCT_VARIABLE_NODE,
                                          HYPRE_SSTRUCT_VARIABLE_XFACE,
                                          HYPRE_SSTRUCT_VARIABLE_XFACE,
                                          HYPRE_SSTRUCT_VARIABLE_YFACE,
                                          HYPRE_SSTRUCT_VARIABLE_YFACE,
                                          HYPRE_SSTRUCT_VARIABLE_CELL,
                                          HYPRE_SSTRUCT_VARIABLE_CELL,
                                          HYPRE_SSTRUCT_VARIABLE_CELL,
                                          HYPRE_SSTRUCT_VARIABLE_CELL
                                         };
         for (i = 0; i < nparts; i++)
         {
            HYPRE_SStructGridSetVariables(grid, i, nvars, vars);
         }
      }

      /* Set the ordering of the variables in the finite element problem.  This
         is done by listing the variable numbers and offset directions relative
         to the element's center.  See the Reference Manual for more details.
         The ordering and location of the nine variables in each element is as
         follows (notation is [order# : variable#]):

                          [12:0]-[13:3]-[14:4]-[15:0]
                             |                    |
                             |                    |
                           [8:2]  [9:7] [10:8] [11:2]
                             |                    |
                             |                    |
                           [4:1]  [5:5]  [6:6]  [7:1]
                             |                    |
                             |                    |
                           [0:0]--[1:3]--[2:4]--[3:0]
      */
      {
         int part = 0;
         int ordering[48] = { 0, -1, -1,    3,  0, -1,    4,  0, -1,    0, +1, -1,
                              1, -1,  0,    5,  0,  0,    6,  0,  0,    1, +1,  0,
                              2, -1,  0,    7,  0,  0,    8,  0,  0,    2, +1,  0,
                              0, -1, +1,    3,  0, +1,    4,  0, +1,    0, +1, +1
                            };

         HYPRE_SStructGridSetFEMOrdering(grid, part, ordering);
      }

      /* Now the grid is ready to be used */
      HYPRE_SStructGridAssemble(grid);
   }

   /* 2. Set up the Graph - this determines the non-zero structure of the
         matrix. */
   {
      int part = 0;

      /* Create the graph object */
      HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);

      /* See MatrixSetObjectType below */
      HYPRE_SStructGraphSetObjectType(graph, HYPRE_PARCSR);

      /* Indicate that this problem uses finite element stiffness matrices and
         load vectors, instead of stencils. */
      HYPRE_SStructGraphSetFEM(graph, part);

      /* The local stiffness matrix is full, so there is no need to call
         HYPRE_SStructGraphSetFEMSparsity() to set its sparsity pattern. */

      /* Assemble the graph */
      HYPRE_SStructGraphAssemble(graph);
   }

   /* 3. Set up the SStruct Matrix and right-hand side vector */
   {
      int part = 0;

      /* Create the matrix object */
      HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);
      /* Use a ParCSR storage */
      HYPRE_SStructMatrixSetObjectType(A, HYPRE_PARCSR);
      /* Indicate that the matrix coefficients are ready to be set */
      HYPRE_SStructMatrixInitialize(A);

      /* Create an empty vector object */
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
      /* Use a ParCSR storage */
      HYPRE_SStructVectorSetObjectType(b, HYPRE_PARCSR);
      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_SStructVectorInitialize(b);

      /* Set the matrix and vector entries by finite element assembly */
      {
         /* Local stifness matrix and load vector */
         double S[16][16], F[16];

         int i, j;
         int index[2];

         for (j = 1; j <= n; j++)
         {
            for (i = 1; i <= n; i++)
            {
               index[0] = i + pi * n;
               index[1] = j + pj * n;

               /* Compute the FEM matrix and rhs */
               ComputeFEMQ3(S, F, h);

               /* Set boundary conditions */
               {
                  int ii, jj, bdy, dd;
                  int set_bc[4] = {0, 0, 0, 0};
                  int bc_dofs[4][4] =
                  {
                     { 0,  4,  8, 12},  /* x = 0 boundary */
                     { 0,  1,  2,  3},  /* y = 0 boundary */
                     { 3,  7, 11, 15},  /* x = 1 boundary */
                     {12, 13, 14, 15}   /* y = 1 boundary */
                  };

                  /* Determine the boundary conditions to be set */
                  if (index[0] == 1)   { set_bc[0] = 1; } /* x = 0 boundary */
                  if (index[1] == 1)   { set_bc[1] = 1; } /* y = 0 boundary */
                  if (index[0] == N * n) { set_bc[2] = 1; } /* x = 1 boundary */
                  if (index[1] == N * n) { set_bc[3] = 1; } /* y = 1 boundary */

                  /* Modify the FEM matrix and rhs on each boundary by setting
                     rows and columns of S to the identity and F to zero */
                  for (bdy = 0; bdy < 4; bdy++)
                  {
                     /* Only modify if boundary condition needs to be set */
                     if (set_bc[bdy])
                     {
                        for (dd = 0; dd < 4; dd++)
                        {
                           for (jj = 0; jj < 16; jj++)
                           {
                              ii = bc_dofs[bdy][dd];
                              S[ii][jj] = 0.0; /* row */
                              S[jj][ii] = 0.0; /* col */
                           }
                           S[ii][ii] = 1.0; /* diagonal */
                           F[ii]     = 0.0; /* rhs */
                        }
                     }
                  }
               }

               /* Add this elements contribution to the matrix */
               HYPRE_SStructMatrixAddFEMValues(A, part, index, &S[0][0]);

               /* Add this elements contribution to the rhs */
               HYPRE_SStructVectorAddFEMValues(b, part, index, F);
            }
         }
      }
   }

   /* Collective calls finalizing the matrix and vector assembly */
   HYPRE_SStructMatrixAssemble(A);
   HYPRE_SStructVectorAssemble(b);

   /* 4. Set up SStruct Vector for the solution vector x */
   {
      int part = 0;
      int var, nvars = 9;
      int nvalues = (n + 1) * (n + 1);
      double *values;

      values = (double*) calloc(nvalues, sizeof(double));

      /* Create an empty vector object */
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);
      /* Set the object type to ParCSR */
      HYPRE_SStructVectorSetObjectType(x, HYPRE_PARCSR);
      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_SStructVectorInitialize(x);

      /* Set the values for the initial guess one variable at a time.  Since the
         SetBoxValues() calls below set the values to the right and up from the
         cell center, ilower needs to be adjusted. */
      for (var = 0; var < nvars; var++)
      {
         int ilower[2] = {1 + pi * n, 1 + pj * n};
         int iupper[2] = {n + pi * n, n + pj * n};

         switch (var)
         {
            case 0: /* NODE */
               ilower[0]--;
               ilower[1]--;
               break;
            case 1: case 2: /* XFACE */
               ilower[0]--;
               break;
            case 3: case 4: /* YFACE */
               ilower[1]--;
               break;
         }

         HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
      }

      free(values);

      /* Finalize the vector assembly */
      HYPRE_SStructVectorAssemble(x);
   }

   /* 5. Set up and call the solver (Solver options can be found in the
         Reference Manual.) */
   {
      double final_res_norm;
      int its;

      HYPRE_ParCSRMatrix    par_A;
      HYPRE_ParVector       par_b;
      HYPRE_ParVector       par_x;

      /* Extract the ParCSR objects needed in the solver */
      HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
      HYPRE_SStructVectorGetObject(b, (void **) &par_b);
      HYPRE_SStructVectorGetObject(x, (void **) &par_x);

      /* Here we construct a BoomerAMG solver.  See the other SStruct examples
         as well as the Reference manual for additional solver choices. */
      HYPRE_BoomerAMGCreate(&solver);
      HYPRE_BoomerAMGSetCoarsenType(solver, 6);
      HYPRE_BoomerAMGSetStrongThreshold(solver, 0.25);
      HYPRE_BoomerAMGSetTol(solver, 1e-6);
      HYPRE_BoomerAMGSetPrintLevel(solver, 2);
      HYPRE_BoomerAMGSetMaxIter(solver, 50);

      /* call the setup */
      HYPRE_BoomerAMGSetup(solver, par_A, par_b, par_x);

      /* call the solve */
      HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);

      /* get some info */
      HYPRE_BoomerAMGGetNumIterations(solver, &its);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver,
                                                  &final_res_norm);
      /* clean up */
      HYPRE_BoomerAMGDestroy(solver);

      /* Gather the solution vector */
      HYPRE_SStructVectorGather(x);

      /* Save the solution for GLVis visualization, see vis/glvis-ex16.sh */
      if (vis)
      {
#ifdef HYPRE_EXVIS
         FILE *file;
         char  filename[255];

         int part = 0;
         int i, j, k, index[2];
         int nvalues = n * n * 16;
         double X[16], *values;

         /* GLVis-to-hypre local renumbering */
         int g2h[16] = {0, 3, 15, 12, 1, 2, 7, 11, 14, 13, 8, 4, 5, 6, 9, 10};

         values = (double*) calloc(nvalues, sizeof(double));

         nvalues = 0;
         for (j = 1; j <= n; j++)
         {
            for (i = 1; i <= n; i++)
            {
               index[0] = i + pi * n;
               index[1] = j + pj * n;

               /* Get local element solution values X */
               HYPRE_SStructVectorGetFEMValues(x, part, index, X);

               /* Copy local solution X into values array */
               for (k = 0; k < 16; k++)
               {
                  values[nvalues] = X[g2h[k]];
                  nvalues++;
               }
            }
         }

         sprintf(filename, "%s.%06d", "vis/ex16.sol", myid);
         if ((file = fopen(filename, "w")) == NULL)
         {
            printf("Error: can't open output file %s\n", filename);
            MPI_Finalize();
            exit(1);
         }

         /* Finite element space header */
         fprintf(file, "FiniteElementSpace\n");
         fprintf(file, "FiniteElementCollection: Local_Quad_Q3\n");
         fprintf(file, "VDim: 1\n");
         fprintf(file, "Ordering: 0\n\n");

         /* Save solution with replicated shared data */
         for (i = 0; i < nvalues; i++)
         {
            fprintf(file, "%.14e\n", values[i]);
         }

         fflush(file);
         fclose(file);
         free(values);

         /* Save local finite element mesh */
         GLVis_PrintLocalSquareMesh("vis/ex16.mesh", n, n, h,
                                    pi * h * n, pj * h * n, myid);

         /* Additional visualization data */
         GLVis_PrintData("vis/ex16.data", myid, num_procs);
#endif
      }

      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", its);
         printf("Final Relative Residual Norm = %g\n", final_res_norm);
         printf("\n");
      }
   }

   /* Free memory */
   HYPRE_SStructGridDestroy(grid);
   HYPRE_SStructGraphDestroy(graph);
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);

   /* Finalize HYPRE */
   HYPRE_Finalize();

   /* Finalize MPI */
   MPI_Finalize();

   return 0;
}

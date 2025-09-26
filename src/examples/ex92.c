/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 15

   Interface:      Semi-Structured interface (SStruct)

   Compile with:   make ex15

   Sample run:     mpirun -np 8 ex15 -n 10

   To see options: ex15 -help

   Description:

                   This code solves a 3D electromagnetic diffusion (definite
                   curl-curl) problem using the lowest order Nedelec, or "edge"
                   finite element discretization on a uniform hexahedral meshing
                   of the unit cube with a refinement region of factor r in the
                   first octant of the mesh.  The right-hand-side corresponds to
                   a unit vector force and we use uniform zero Dirichlet
                   boundary conditions.  The overall problem reads: curl alpha
                   curl E + beta E = 1, with E x n = 0 on the boundary, where
                   alpha and beta are piecewise-constant material coefficients.

                   The grid is constructed from two parts with a global coarse
                   grid of size N x N x N on part 0 and a global fine grid of
                   size R x R x R on part 1.  Each part is distributed across an
                   M x M x M processor grid, hence the number of processors
                   specified should be a perfect cube.  The value of N is given
                   by N=(2*s)*M, where s is an input parameter, and R=r*(N/2) is
                   the size of the refined octant.  The index spaces on the two
                   parts are aligned such that cell (1,1) on the fine level is
                   in the lower-left corner of cell (1,1) on the coarse level.

                   This example code is mainly meant as an illustration of using
                   the Auxiliary-space Maxwell Solver (AMS) through the SStruct
                   interface on an AMR mesh.  We recommend viewing Example 15
                   before viewing this example.  */

#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_sstruct_mv.h"
#include "HYPRE_sstruct_ls.h"
#include "_hypre_parcsr_ls.h"
#include "HYPRE.h"

#include "vis.c"

int optionAlpha, optionBeta;

/* Curl-curl coefficient alpha = mu^{-1} */
double alpha(double x, double y, double z)
{
   switch (optionAlpha)
   {
      case 0: /* uniform coefficient */
         return 1.0;
      case 1: /* smooth coefficient */
         return x * x + exp(y) + sin(z);
      case 2: /* small outside of an interior cube */
         if ((fabs(x - 0.5) < 0.25) && (fabs(y - 0.5) < 0.25) && (fabs(z - 0.5) < 0.25))
         {
            return 1.0;
         }
         else
         {
            return 1.0e-6;
         }
      case 3: /* small outside of an interior ball */
         if (((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5)) < 0.0625)
         {
            return 1.0;
         }
         else
         {
            return 1.0e-6;
         }
      case 4: /* random coefficient */
         return hypre_Rand();
      default:
         return 1.0;
   }
}

/* Mass coefficient beta = sigma */
double beta(double x, double y, double z)
{
   switch (optionBeta)
   {
      case 0: /* uniform coefficient */
         return 1.0;
      case 1: /* smooth coefficient */
         return x * x + exp(y) + sin(z);
      case 2:/* small outside of interior cube */
         if ((fabs(x - 0.5) < 0.25) && (fabs(y - 0.5) < 0.25) && (fabs(z - 0.5) < 0.25))
         {
            return 1.0;
         }
         else
         {
            return 1.0e-6;
         }
      case 3: /* small outside of an interior ball */
         if (((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5)) < 0.0625)
         {
            return 1.0;
         }
         else
         {
            return 1.0e-6;
         }
      case 4: /* random coefficient */
         return hypre_Rand();
      default:
         return 1.0;
   }
}

/*
   This routine computes the lowest order Nedelec, or "edge" finite element
   stiffness matrix and load vector on a cube of size h.  The 12 edges {e_i}
   are numbered in terms of the vertices as follows:

           [7]------[6]
           /|       /|     e_0 = 01, e_1 = 12, e_2  = 32, e_3  = 03,
          / |      / |     e_4 = 45, e_5 = 56, e_6  = 76, e_7  = 47,
        [4]------[5] |     e_8 = 04, e_9 = 15, e_10 = 26, e_11 = 37.
         | [3]----|-[2]
         | /      | /      The edges are oriented from first to the
         |/       |/       second vertex, e.g. e_0 is from [0] to [1].
        [0]------[1]

   We allow for different scaling of the curl-curl and the mass parts of the
   matrix with coefficients alpha and beta respectively:

         S_ij = alpha (curl phi_i,curl phi_j) + beta (phi_i, phi_j).

   The load vector corresponding to a right-hand side of {1,1,1} is

                        F_j = (1,phi_j) = h^2/4.
*/
void ComputeFEMND1(double S[12][12], double F[12], index[3], double h)
{
   int i, j;

   double h2_4 = h * h / 4;

   double x = index[0] * h - h / 2;
   double y = index[1] * h - h / 2;
   double z = index[2] * h - h / 2;

   double cS1 = alpha(x, y, z) / (6.0 * h), cS2 = 2 * cS1, cS4 = 2 * cS2;
   double cM1 = beta(x, y, z) * h / 36.0,   cM2 = 2 * cM1, cM4 = 2 * cM2;

   S[ 0][ 0] =  cS4 + cM4;   S[ 0][ 1] =  cS2;         S[ 0][ 2] = -cS1 + cM2;
   S[ 0][ 3] = -cS2;         S[ 0][ 4] = -cS1 + cM2;   S[ 0][ 5] =  cS1;
   S[ 0][ 6] = -cS2 + cM1;   S[ 0][ 7] = -cS1;         S[ 0][ 8] = -cS2;
   S[ 0][ 9] =  cS2;         S[ 0][10] =  cS1;         S[ 0][11] = -cS1;

   S[ 1][ 1] =  cS4 + cM4;   S[ 1][ 2] = -cS2;         S[ 1][ 3] = -cS1 + cM2;
   S[ 1][ 4] =  cS1;         S[ 1][ 5] = -cS1 + cM2;   S[ 1][ 6] = -cS1;
   S[ 1][ 7] = -cS2 + cM1;   S[ 1][ 8] = -cS1;         S[ 1][ 9] = -cS2;
   S[ 1][10] =  cS2;         S[ 1][11] =  cS1;

   S[ 2][ 2] =  cS4 + cM4;   S[ 2][ 3] =  cS2;         S[ 2][ 4] = -cS2 + cM1;
   S[ 2][ 5] = -cS1;         S[ 2][ 6] = -cS1 + cM2;   S[ 2][ 7] =  cS1;
   S[ 2][ 8] = -cS1;         S[ 2][ 9] =  cS1;         S[ 2][10] =  cS2;
   S[ 2][11] = -cS2;

   S[ 3][ 3] =  cS4 + cM4;   S[ 3][ 4] = -cS1;         S[ 3][ 5] = -cS2 + cM1;
   S[ 3][ 6] =  cS1;         S[ 3][ 7] = -cS1 + cM2;   S[ 3][ 8] = -cS2;
   S[ 3][ 9] = -cS1;         S[ 3][10] =  cS1;         S[ 3][11] =  cS2;

   S[ 4][ 4] =  cS4 + cM4;   S[ 4][ 5] =  cS2;         S[ 4][ 6] = -cS1 + cM2;
   S[ 4][ 7] = -cS2;         S[ 4][ 8] =  cS2;         S[ 4][ 9] = -cS2;
   S[ 4][10] = -cS1;         S[ 4][11] =  cS1;

   S[ 5][ 5] =  cS4 + cM4;   S[ 5][ 6] = -cS2;         S[ 5][ 7] = -cS1 + cM2;
   S[ 5][ 8] =  cS1;         S[ 5][ 9] =  cS2;         S[ 5][10] = -cS2;
   S[ 5][11] = -cS1;

   S[ 6][ 6] =  cS4 + cM4;   S[ 6][ 7] =  cS2;         S[ 6][ 8] =  cS1;
   S[ 6][ 9] = -cS1;         S[ 6][10] = -cS2;         S[ 6][11] =  cS2;

   S[ 7][ 7] =  cS4 + cM4;   S[ 7][ 8] =  cS2;         S[ 7][ 9] =  cS1;
   S[ 7][10] = -cS1;         S[ 7][11] = -cS2;

   S[ 8][ 8] =  cS4 + cM4;   S[ 8][ 9] = -cS1 + cM2;   S[ 8][10] = -cS2 + cM1;
   S[ 8][11] = -cS1 + cM2;

   S[ 9][ 9] =  cS4 + cM4;   S[ 9][10] = -cS1 + cM2;   S[ 9][11] = -cS2 + cM1;

   S[10][10] =  cS4 + cM4;   S[10][11] = -cS1 + cM2;

   S[11][11] =  cS4 + cM4;

   /* The stiffness matrix is symmetric */
   for (i = 1; i < 12; i++)
      for (j = 0; j < i; j++)
      {
         S[i][j] = S[j][i];
      }

   for (i = 0; i < 12; i++)
   {
      F[i] = h2_4;
   }
}


/* Set Dirichlet boundary conditions for a set of edges */
void SetDirBC(double S[12][12], double F[12], int n_edges, int *bc_edges)
{
   int ii, jj;
   for (ii = 0; ii < n_edges; ii++)
   {
      for (jj = 0; jj < 12; jj++)
      {
         S[bc_edges[ii]][jj] = S[jj][bc_edges[ii]] = 0.0;
      }
      S[bc_edges[ii]][bc_edges[ii]] = 1.0;
      F[bc_edges[ii]] = 0.0;
   }
}


int main (int argc, char *argv[])
{
   int myid, num_procs;
   int n, N, pi, pj, pk;
   double h;
   int vis;

   double tol, theta;
   int maxit, cycle_type;
   int rlx_type, rlx_sweeps, rlx_weight, rlx_omega;
   int amg_coarsen_type, amg_agg_levels, amg_rlx_type;
   int amg_interp_type, amg_Pmax;
   int singular_problem ;

   int time_index;

   HYPRE_SStructGrid     edge_grid;
   HYPRE_SStructGraph    A_graph;
   HYPRE_SStructMatrix   A;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   x;
   HYPRE_SStructGrid     node_grid;
   HYPRE_SStructGraph    G_graph;
   HYPRE_SStructStencil  G_stencil[3];
   HYPRE_SStructMatrix   G;
   HYPRE_SStructVector   xcoord, ycoord, zcoord;

   HYPRE_Solver          solver, precond;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Set default parameters */
   s                = 5;
   vis              = 0;
   optionAlpha      = 0;
   optionBeta       = 0;
   maxit            = 100;
   tol              = 1e-6;
   cycle_type       = 13;
   rlx_type         = 2;
   rlx_sweeps       = 1;
   rlx_weight       = 1.0;
   rlx_omega        = 1.0;
   amg_coarsen_type = 10;
   amg_agg_levels   = 1;
   amg_rlx_type     = 6;
   theta            = 0.25;
   amg_interp_type  = 6;
   amg_Pmax         = 4;
   singular_problem = 0;

   /* Parse command line */
   {
      int arg_index = 0;
      int print_usage = 0;

      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-s") == 0 )
         {
            arg_index++;
            s = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-a") == 0 )
         {
            arg_index++;
            optionAlpha = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-b") == 0 )
         {
            arg_index++;
            optionBeta = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-vis") == 0 )
         {
            arg_index++;
            vis = 1;
         }
         else if ( strcmp(argv[arg_index], "-maxit") == 0 )
         {
            arg_index++;
            maxit = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-tol") == 0 )
         {
            arg_index++;
            tol = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-type") == 0 )
         {
            arg_index++;
            cycle_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rlx") == 0 )
         {
            arg_index++;
            rlx_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rlxn") == 0 )
         {
            arg_index++;
            rlx_sweeps = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rlxw") == 0 )
         {
            arg_index++;
            rlx_weight = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rlxo") == 0 )
         {
            arg_index++;
            rlx_omega = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-ctype") == 0 )
         {
            arg_index++;
            amg_coarsen_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-amgrlx") == 0 )
         {
            arg_index++;
            amg_rlx_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-agg") == 0 )
         {
            arg_index++;
            amg_agg_levels = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-itype") == 0 )
         {
            arg_index++;
            amg_interp_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-pmax") == 0 )
         {
            arg_index++;
            amg_Pmax = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-sing") == 0 )
         {
            arg_index++;
            singular_problem = 1;
         }
         else if ( strcmp(argv[arg_index], "-theta") == 0 )
         {
            arg_index++;
            theta = atof(argv[arg_index++]);
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
         printf("  -s <n>          : scale factor for base grid (default: 5)\n");
         printf("  -a <alpha_opt>  : curl-curl coefficient choice (default: 0)\n");
         printf("  -b <beta_opt>   : mass coefficient choice (default: 0)\n");
         printf("  -vis            : save the solution for GLVis visualization\n");
         printf("\n");
         printf("PCG-AMS solver options:                                 \n");
         printf("  -maxit <num>    : maximum number of iterations (100)  \n");
         printf("  -tol <num>      : convergence tolerance (1e-6)        \n");
         printf("  -type <num>     : 3-level cycle type (0-8, 11-14)     \n");
         printf("  -theta <num>    : BoomerAMG threshold (0.25)          \n");
         printf("  -ctype <num>    : BoomerAMG coarsening type           \n");
         printf("  -agg <num>      : Levels of BoomerAMG agg. coarsening \n");
         printf("  -amgrlx <num>   : BoomerAMG relaxation type           \n");
         printf("  -itype <num>    : BoomerAMG interpolation type        \n");
         printf("  -pmax <num>     : BoomerAMG interpolation truncation  \n");
         printf("  -rlx <num>      : relaxation type                     \n");
         printf("  -rlxn <num>     : number of relaxation sweeps         \n");
         printf("  -rlxw <num>     : damping parameter (usually <=1)     \n");
         printf("  -rlxo <num>     : SOR parameter (usually in (0,2))    \n");
         printf("  -sing           : curl-curl only (singular) problem   \n");
         printf("\n");
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   /* Figure out the processor grid (M x M x M).  The local problem size is
      based on s and r, while pi, pj and pk indicate the position in the
      processor grid. */
   M  = pow(num_procs, 1.0 / 3.0) + 0.5;
   if (num_procs != M * M * M)
   {
      if (myid == 0) printf("Can't run on %d processors, try %d.\n",
                               num_procs, M * M * M);
      MPI_Finalize();
      exit(1);
   }
   n0 = 2 * s;            /* size of box0 is n0 x n0 x n0 */
   n1 = r * s;            /* size of box1 is n1 x n1 x n1 */
   h0 = 1.0 / (M * n0);   /* grid spacing on part 0 */
   h1 = h0 / r;           /* grid spacing on part 1 */
   pk = myid / (M * M);
   pj = myid / M - pk * M;
   pi = myid - pj * M - pk * M * M;

   /* Start timing */
   time_index = hypre_InitializeTiming("SStruct Setup");
   hypre_BeginTiming(time_index);

   {
      int ndim         = 3;
      int nparts       = 2;
      int nboxes       = 2;
      int bpart[2]     = {0, 1};
      int bilo[2][3]   = {{pi*n0 + 1, pj*n0 + 1, pk*n0 + 1},
         {pi*n1 + 1, pj*n1 + 1, pk*n1 + 1}
      };
      int biup[2][3]   = {{pi*n0 + n0, pj*n0 + n0, pk*n0 + n0},
         {pi*n1 + n1, pj*n1 + n1, pk*n1 + n1}
      };
      int bbndlo[2][3] = {{1, 1, 1},          {1, 1, 1}};
      int bbndup[2][3] = {{M * n0, M * n0, M * n0}, {M*n0 * r, M*n0 * r, M*n0 * r}};

      double bh[2]     = {h0, h1};

      /* 1. Set up the edge and nodal grids.  Note that we do this
         simultaneously to make sure that they have the same extents. */

      /* Create empty 3D grid objects */
      HYPRE_SStructGridCreate(MPI_COMM_WORLD, ndim, nparts, &node_grid);
      HYPRE_SStructGridCreate(MPI_COMM_WORLD, ndim, nparts, &edge_grid);

      object_type = HYPRE_PARCSR;
      HYPRE_SStructGridSetObjectType(node_grid, object_type);
      HYPRE_SStructGridSetObjectType(edge_grid, object_type);

      /* Set the extents of the grid - each processor sets its grid boxes. */
      for (i = 0; i < nboxes; i++)
      {
         HYPRE_SStructGridSetExtents(node_grid, bpart[i], bilo[i], biup[i]);
         HYPRE_SStructGridSetExtents(edge_grid, bpart[i], bilo[i], biup[i]);
      }

      /* Set the variable type and number of variables on each grid. */
      {
         int nnodevars = 1;
         int nedgevars = 3;
         HYPRE_SStructVariable nodevars[1] = {HYPRE_SSTRUCT_VARIABLE_NODE};
         HYPRE_SStructVariable edgevars[3] = {HYPRE_SSTRUCT_VARIABLE_XEDGE,
                                              HYPRE_SSTRUCT_VARIABLE_YEDGE,
                                              HYPRE_SSTRUCT_VARIABLE_ZEDGE
                                             };

         for (part = 0; part < nparts; part++)
         {
            HYPRE_SStructGridSetVariables(node_grid, part, nnodevars, nodevars);
            HYPRE_SStructGridSetVariables(edge_grid, part, nedgevars, edgevars);
         }
      }

      /* Declare fine part 1 to be a refinement of coarse part 0 with a given
       * refinement factor (rfactors) and index alignment */
      {
         int coarse_part = bpart[0];
         int fine_part   = bpart[1];
         int coarse_index[3] = {1, 1, 1};
         int fine_index[3]   = {1, 1, 1};
         int rfactors[3]     = {r, r, r};

         HYPRE_SStructGridSetAMRPart(grid, coarse_part, fine_part,
                                     coarse_index, fine_index, rfactors);
      }

      /* Now the grids are ready to be used */
      HYPRE_SStructGridAssemble(node_grid);
      HYPRE_SStructGridAssemble(edge_grid);

      /* 2. Create the finite element stiffness matrix A and load vector b. */

      /* Set the ordering of the variables in the finite element problem.  This
         is done by listing the variable offset directions relative to the
         element's center.  See the Reference Manual for more details. */
      {
         int ordering[48] =       { 0,  0, -1, -1,    /* x-edge [0]-[1] */
                                    1, +1,  0, -1,    /* y-edge [1]-[2] */
                                    /*     [7]------[6]  */    0,  0, +1, -1,    /* x-edge [3]-[2] */
                                    /*     /|       /|   */    1, -1,  0, -1,    /* y-edge [0]-[3] */
                                    /*    / |      / |   */    0,  0, -1, +1,    /* x-edge [4]-[5] */
                                    /*  [4]------[5] |   */    1, +1,  0, +1,    /* y-edge [5]-[6] */
                                    /*   | [3]----|-[2]  */    0,  0, +1, +1,    /* x-edge [7]-[6] */
                                    /*   | /      | /    */    1, -1,  0, +1,    /* y-edge [4]-[7] */
                                    /*   |/       |/     */    2, -1, -1,  0,    /* z-edge [0]-[4] */
                                    /*  [0]------[1]     */    2, +1, -1,  0,    /* z-edge [1]-[5] */
                                    2, +1, +1,  0,    /* z-edge [2]-[6] */
                                    2, -1, +1,  0
                                  };  /* z-edge [3]-[7] */

         for (part = 0; part < nparts; part++)
         {
            HYPRE_SStructGridSetFEMOrdering(edge_grid, part, ordering);
         }
      }

      /* Set up the Graph - determines the nonzero structure of the matrix. */
      {
         /* Create the graph object */
         HYPRE_SStructGraphCreate(MPI_COMM_WORLD, edge_grid, &A_graph);

         /* Indicate that this problem uses finite element stiffness matrices and
            load vectors, instead of stencils. */
         for (part = 0; part < nparts; part++)
         {
            HYPRE_SStructGraphSetFEM(A_graph, part);
         }

         /* The edge finite element matrix is full, so there is no need to call the
            HYPRE_SStructGraphSetFEMSparsity() function. */

         /* Assemble the graph */
         HYPRE_SStructGraphAssemble(A_graph);
      }

      /* Set up the SStruct Matrix and right-hand side vector */
      {
         /* Create the matrix object */
         HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, A_graph, &A);
         /* Indicate that the matrix coefficients are ready to be set */
         HYPRE_SStructMatrixInitialize(A);

         /* Create an empty vector object */
         HYPRE_SStructVectorCreate(MPI_COMM_WORLD, edge_grid, &b);
         /* Indicate that the vector coefficients are ready to be set */
         HYPRE_SStructVectorInitialize(b);
      }

      /* Set the matrix and vector entries by finite element assembly */
      {
         /* local stiffness matrix and load vector */
         double S[12][12], F[12];
         int index[3];

         for (i = 0; i < nboxes; i++)
         {
            for (index[0] = bilo[i][0]; index[0] <= biup[i][0]; index[0]++)
               for (index[1] = bilo[i][1]; index[1] <= biup[i][1]; index[1]++)
                  for (index[2] = bilo[i][2]; index[2] <= biup[i][2]; index[2]++)
                  {
                     /* Compute the FEM matrix and r.h.s. for cell index with
                        coefficients evaluated at the cell center. */
                     ComputeFEMND1(S, F, index, bh[i]);

                     /* Eliminate boundary conditions on x = 0 */
                     if (index[0] == bbndlo[i][0])
                     {
                        int bc_edges[4] = { 3, 11, 7, 8 };
                        SetDirBC(S, F, 4, bc_edges);
                     }
                     /* Eliminate boundary conditions on y = 0 */
                     if (index[1] == bbndlo[i][1])
                     {
                        int bc_edges[4] = { 0, 9, 4, 8 };
                        SetDirBC(S, F, 4, bc_edges);
                     }
                     /* Eliminate boundary conditions on z = 0 */
                     if (index[2] == bbndlo[i][2])
                     {
                        int bc_edges[4] = { 0, 1, 2, 3 };
                        SetDirBC(S, F, 4, bc_edges);
                     }
                     /* Eliminate boundary conditions on x = 1 */
                     if (index[0] == bbndup[i][0])
                     {
                        int bc_edges[4] = { 1, 10, 5, 9 };
                        SetDirBC(S, F, 4, bc_edges);
                     }
                     /* Eliminate boundary conditions on y = 1 */
                     if (index[1] == bbndup[i][1])
                     {
                        int bc_edges[4] = { 2, 10, 6, 11 };
                        SetDirBC(S, F, 4, bc_edges);
                     }
                     /* Eliminate boundary conditions on z = 1 */
                     if (index[2] == bbndup[i][2])
                     {
                        int bc_edges[4] = { 4, 5, 6, 7 };
                        SetDirBC(S, F, 4, bc_edges);
                     }

                     /* Assemble the matrix */
                     HYPRE_SStructMatrixAddFEMValues(A, part, index, &S[0][0]);

                     /* Assemble the vector */
                     HYPRE_SStructVectorAddFEMValues(b, part, index, F);
                  }
         }
      }

      /* Collective calls finalizing the matrix and vector assembly */
      HYPRE_SStructMatrixAssemble(A);
      HYPRE_SStructVectorAssemble(b);

      /* 3. Create the discrete gradient matrix G, which is needed in AMS. */
      {
         int stencil_size = 2;

         /* Define the discretization stencil relating the edges and nodes of the
            grid. */
         {
            int ndim = 3;
            int entry;
            int var = 0; /* the node variable */

            /* The discrete gradient stencils connect edge to node variables. */
            int Gx_offsets[2][3] = {{-1, 0, 0}, {0, 0, 0}}; /* x-edge [7]-[6] */
            int Gy_offsets[2][3] = {{0, -1, 0}, {0, 0, 0}}; /* y-edge [5]-[6] */
            int Gz_offsets[2][3] = {{0, 0, -1}, {0, 0, 0}}; /* z-edge [2]-[6] */

            HYPRE_SStructStencilCreate(ndim, stencil_size, &G_stencil[0]);
            HYPRE_SStructStencilCreate(ndim, stencil_size, &G_stencil[1]);
            HYPRE_SStructStencilCreate(ndim, stencil_size, &G_stencil[2]);

            for (entry = 0; entry < stencil_size; entry++)
            {
               HYPRE_SStructStencilSetEntry(G_stencil[0], entry, Gx_offsets[entry], var);
               HYPRE_SStructStencilSetEntry(G_stencil[1], entry, Gy_offsets[entry], var);
               HYPRE_SStructStencilSetEntry(G_stencil[2], entry, Gz_offsets[entry], var);
            }
         }
      }

      /* Set up the Graph - determines the nonzero structure of the matrix. */
      {
         int nvars = 3;
         int var; /* the edge variables */

         /* Create the discrete gradient graph object */
         HYPRE_SStructGraphCreate(MPI_COMM_WORLD, edge_grid, &G_graph);

         /* Since the discrete gradient relates edge and nodal variables (it is a
            rectangular matrix), we have to specify the domain (column) grid. */
         HYPRE_SStructGraphSetDomainGrid(G_graph, node_grid);

         /* Tell the graph which stencil to use for each edge variable on each
            part (we only have one part). */
         for (var = 0; var < nvars; var++)
         {
            HYPRE_SStructGraphSetStencil(G_graph, part, var, G_stencil[var]);
         }

         /* Assemble the graph */
         HYPRE_SStructGraphAssemble(G_graph);
      }

      /* Set up the SStruct Matrix */
      {
         /* Create the matrix object */
         HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, G_graph, &G);
         /* Indicate that the matrix coefficients are ready to be set */
         HYPRE_SStructMatrixInitialize(G);
      }

      /* Set the discrete gradient values, assuming a "natural" orientation of
         the edges (i.e. one in agreement with the coordinate directions). */
      {
         int i;
         int nedges = n * (n + 1) * (n + 1);
         double *values;
         int stencil_indices[2] = {0, 1}; /* the nodes of each edge */

         values = calloc(2 * nedges, sizeof(double));

         /* The edge orientation is fixed: from first to second node */
         for (i = 0; i < nedges; i++)
         {
            values[2 * i]   = -1.0;
            values[2 * i + 1] =  1.0;
         }

         /* Set the values in the discrete gradient x-edges */
         {
            int var = 0;
            int ilower[3] = {1 + pi * n, 0 + pj * n, 0 + pk * n};
            int iupper[3] = {n + pi * n, n + pj * n, n + pk * n};
            HYPRE_SStructMatrixSetBoxValues(G, part, ilower, iupper, var,
                                            stencil_size, stencil_indices,
                                            values);
         }
         /* Set the values in the discrete gradient y-edges */
         {
            int var = 1;
            int ilower[3] = {0 + pi * n, 1 + pj * n, 0 + pk * n};
            int iupper[3] = {n + pi * n, n + pj * n, n + pk * n};
            HYPRE_SStructMatrixSetBoxValues(G, part, ilower, iupper, var,
                                            stencil_size, stencil_indices,
                                            values);
         }
         /* Set the values in the discrete gradient z-edges */
         {
            int var = 2;
            int ilower[3] = {0 + pi * n, 0 + pj * n, 1 + pk * n};
            int iupper[3] = {n + pi * n, n + pj * n, n + pk * n};
            HYPRE_SStructMatrixSetBoxValues(G, part, ilower, iupper, var,
                                            stencil_size, stencil_indices,
                                            values);
         }

         free(values);
      }

      /* Finalize the matrix assembly */
      HYPRE_SStructMatrixAssemble(G);
   }

   /* 4. Create the vectors of nodal coordinates xcoord, ycoord and zcoord,
         which are needed in AMS. */
   {
      int i, j, k;
      int part = 0;
      int var = 0; /* the node variable */
      int index[3];
      double xval, yval, zval;

      /* Create empty vector objects */
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, node_grid, &xcoord);
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, node_grid, &ycoord);
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, node_grid, &zcoord);
      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_SStructVectorInitialize(xcoord);
      HYPRE_SStructVectorInitialize(ycoord);
      HYPRE_SStructVectorInitialize(zcoord);

      /* Compute and set the coordinates of the nodes */
      for (i = 0; i <= n; i++)
         for (j = 0; j <= n; j++)
            for (k = 0; k <= n; k++)
            {
               index[0] = i + pi * n; index[1] = j + pj * n; index[2] = k + pk * n;

               xval = index[0] * h;
               yval = index[1] * h;
               zval = index[2] * h;

               HYPRE_SStructVectorSetValues(xcoord, part, index, var, &xval);
               HYPRE_SStructVectorSetValues(ycoord, part, index, var, &yval);
               HYPRE_SStructVectorSetValues(zcoord, part, index, var, &zval);
            }

      /* Finalize the vector assembly */
      HYPRE_SStructVectorAssemble(xcoord);
      HYPRE_SStructVectorAssemble(ycoord);
      HYPRE_SStructVectorAssemble(zcoord);
   }

   /* 5. Set up a SStruct Vector for the solution vector x */
   {
      int part = 0;
      int nvalues = n * (n + 1) * (n + 1);
      double *values;

      values = calloc(nvalues, sizeof(double));

      /* Create an empty vector object */
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, edge_grid, &x);
      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_SStructVectorInitialize(x);

      /* Set the values for the initial guess x-edge */
      {
         int var = 0;
         int ilower[3] = {1 + pi * n, 0 + pj * n, 0 + pk * n};
         int iupper[3] = {n + pi * n, n + pj * n, n + pk * n};
         HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
      }
      /* Set the values for the initial guess y-edge */
      {
         int var = 1;
         int ilower[3] = {0 + pi * n, 1 + pj * n, 0 + pk * n};
         int iupper[3] = {n + pi * n, n + pj * n, n + pk * n};
         HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
      }
      /* Set the values for the initial guess z-edge */
      {
         int var = 2;
         int ilower[3] = {0 + pi * n, 0 + pj * n, 1 + pk * n};
         int iupper[3] = {n + pi * n, n + pj * n, n + pk * n};
         HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
      }

      free(values);

      /* Finalize the vector assembly */
      HYPRE_SStructVectorAssemble(x);
   }

   /* Finalize current timing */
   hypre_EndTiming(time_index);
   hypre_PrintTiming("SStruct phase times", MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /* 6. Set up and call the PCG-AMS solver (Solver options can be found in the
         Reference Manual.) */
   {
      double final_res_norm;
      int its;

      HYPRE_ParCSRMatrix    par_A;
      HYPRE_ParVector       par_b;
      HYPRE_ParVector       par_x;

      HYPRE_ParCSRMatrix    par_G;
      HYPRE_ParVector       par_xcoord;
      HYPRE_ParVector       par_ycoord;
      HYPRE_ParVector       par_zcoord;

      /* Extract the ParCSR objects needed in the solver */
      HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
      HYPRE_SStructVectorGetObject(b, (void **) &par_b);
      HYPRE_SStructVectorGetObject(x, (void **) &par_x);
      HYPRE_SStructMatrixGetObject(G, (void **) &par_G);
      HYPRE_SStructVectorGetObject(xcoord, (void **) &par_xcoord);
      HYPRE_SStructVectorGetObject(ycoord, (void **) &par_ycoord);
      HYPRE_SStructVectorGetObject(zcoord, (void **) &par_zcoord);

      if (myid == 0)
         printf("Problem size: %d\n\n",
                hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix*)par_A));

      /* Start timing */
      time_index = hypre_InitializeTiming("AMS Setup");
      hypre_BeginTiming(time_index);

      /* Create solver */
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_PCGSetMaxIter(solver, maxit); /* max iterations */
      HYPRE_PCGSetTol(solver, tol); /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 0); /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
      HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

      /* Create AMS preconditioner */
      HYPRE_AMSCreate(&precond);

      /* Set AMS parameters */
      HYPRE_AMSSetMaxIter(precond, 1);
      HYPRE_AMSSetTol(precond, 0.0);
      HYPRE_AMSSetCycleType(precond, cycle_type);
      HYPRE_AMSSetPrintLevel(precond, 1);

      /* Set discrete gradient */
      HYPRE_AMSSetDiscreteGradient(precond, par_G);

      /* Set vertex coordinates */
      HYPRE_AMSSetCoordinateVectors(precond,
                                    par_xcoord, par_ycoord, par_zcoord);

      if (singular_problem)
      {
         HYPRE_AMSSetBetaPoissonMatrix(precond, NULL);
      }

      /* Smoothing and AMG options */
      HYPRE_AMSSetSmoothingOptions(precond,
                                   rlx_type, rlx_sweeps,
                                   rlx_weight, rlx_omega);
      HYPRE_AMSSetAlphaAMGOptions(precond,
                                  amg_coarsen_type, amg_agg_levels,
                                  amg_rlx_type, theta, amg_interp_type,
                                  amg_Pmax);
      HYPRE_AMSSetBetaAMGOptions(precond,
                                 amg_coarsen_type, amg_agg_levels,
                                 amg_rlx_type, theta, amg_interp_type,
                                 amg_Pmax);

      /* Set the PCG preconditioner */
      HYPRE_PCGSetPrecond(solver,
                          (HYPRE_PtrToSolverFcn) HYPRE_AMSSolve,
                          (HYPRE_PtrToSolverFcn) HYPRE_AMSSetup,
                          precond);

      /* Call the setup */
      HYPRE_ParCSRPCGSetup(solver, par_A, par_b, par_x);

      /* Finalize current timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Start timing again */
      time_index = hypre_InitializeTiming("AMS Solve");
      hypre_BeginTiming(time_index);

      /* Call the solve */
      HYPRE_ParCSRPCGSolve(solver, par_A, par_b, par_x);

      /* Finalize current timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Get some info */
      HYPRE_PCGGetNumIterations(solver, &its);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

      /* Clean up */
      HYPRE_AMSDestroy(precond);
      HYPRE_ParCSRPCGDestroy(solver);

      /* Gather the solution vector */
      HYPRE_SStructVectorGather(x);

      /* Save the solution for GLVis visualization, see vis/glvis-ex15.sh */
      if (vis)
      {
         FILE *file;
         char  filename[255];

         int part = 0;
         int nvalues = n * (n + 1) * (n + 1);
         double *xvalues, *yvalues, *zvalues;

         xvalues = calloc(nvalues, sizeof(double));
         yvalues = calloc(nvalues, sizeof(double));
         zvalues = calloc(nvalues, sizeof(double));

         /* Get local solution in the x-edges */
         {
            int var = 0;
            int ilower[3] = {1 + pi * n, 0 + pj * n, 0 + pk * n};
            int iupper[3] = {n + pi * n, n + pj * n, n + pk * n};
            HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                            var, xvalues);
         }
         /* Get local solution in the y-edges */
         {
            int var = 1;
            int ilower[3] = {0 + pi * n, 1 + pj * n, 0 + pk * n};
            int iupper[3] = {n + pi * n, n + pj * n, n + pk * n};
            HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                            var, yvalues);
         }
         /* Get local solution in the z-edges */
         {
            int var = 2;
            int ilower[3] = {0 + pi * n, 0 + pj * n, 1 + pk * n};
            int iupper[3] = {n + pi * n, n + pj * n, n + pk * n};
            HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                            var, zvalues);
         }

         sprintf(filename, "%s.%06d", "vis/ex15.sol", myid);
         if ((file = fopen(filename, "w")) == NULL)
         {
            printf("Error: can't open output file %s\n", filename);
            MPI_Finalize();
            exit(1);
         }

         /* Finite element space header */
         fprintf(file, "FiniteElementSpace\n");
         fprintf(file, "FiniteElementCollection: Local_Hex_ND1\n");
         fprintf(file, "VDim: 1\n");
         fprintf(file, "Ordering: 0\n\n");

         /* Save solution with replicated shared data, i.e., element by element,
            using the same numbering as the local finite element unknowns. */
         {
            int i, j, k, s;

            /* Initial x-, y- and z-edge indices in the values arrays */
            int oi[4] = { 0, n, n*(n + 1), n*(n + 1) + n }; /* e_0, e_2,  e_4,  e_6 */
            int oj[4] = { 0, 1, n*(n + 1), n*(n + 1) + 1 }; /* e_3, e_1,  e_7,  e_5 */
            int ok[4] = { 0, 1,     n + 1,       n + 2 }; /* e_8, e_9, e_11, e_10 */
            /* Loop over the cells while updating the above offsets */
            for (k = 0; k < n; k++)
            {
               for (j = 0; j < n; j++)
               {
                  for (i = 0; i < n; i++)
                  {
                     fprintf(file,
                             "%.14e\n%.14e\n%.14e\n%.14e\n"
                             "%.14e\n%.14e\n%.14e\n%.14e\n"
                             "%.14e\n%.14e\n%.14e\n%.14e\n",
                             xvalues[oi[0]], yvalues[oj[1]], xvalues[oi[1]], yvalues[oj[0]],
                             xvalues[oi[2]], yvalues[oj[3]], xvalues[oi[3]], yvalues[oj[2]],
                             zvalues[ok[0]], zvalues[ok[1]], zvalues[ok[3]], zvalues[ok[2]]);

                     for (s = 0; s < 4; s++) { oi[s]++, oj[s]++, ok[s]++; }
                  }
                  for (s = 0; s < 4; s++) { oj[s]++, ok[s]++; }
               }
               for (s = 0; s < 4; s++) { oi[s] += n, ok[s] += n + 1; }
            }
         }

         fflush(file);
         fclose(file);
         free(xvalues);
         free(yvalues);
         free(zvalues);

         /* Save local finite element mesh */
         GLVis_PrintLocalCubicMesh("vis/ex15.mesh", n, n, n, h,
                                   pi * h * n, pj * h * n, pk * h * n, myid);

         /* Additional visualization data */
         if (myid == 0)
         {
            sprintf(filename, "%s", "vis/ex15.data");
            file = fopen(filename, "w");
            fprintf(file, "np %d\n", num_procs);
            fflush(file);
            fclose(file);
         }
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
   HYPRE_SStructGridDestroy(edge_grid);
   HYPRE_SStructGraphDestroy(A_graph);
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);
   HYPRE_SStructGridDestroy(node_grid);
   HYPRE_SStructGraphDestroy(G_graph);
   HYPRE_SStructStencilDestroy(G_stencil[0]);
   HYPRE_SStructStencilDestroy(G_stencil[1]);
   HYPRE_SStructStencilDestroy(G_stencil[2]);
   HYPRE_SStructMatrixDestroy(G);
   HYPRE_SStructVectorDestroy(xcoord);
   HYPRE_SStructVectorDestroy(ycoord);
   HYPRE_SStructVectorDestroy(zcoord);

   /* Finalize MPI */
   MPI_Finalize();

   return 0;
}

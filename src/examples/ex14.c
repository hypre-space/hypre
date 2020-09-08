/*
   Example 14

   Interface:      Semi-Structured interface (SStruct)

   Compile with:   make ex14

   Sample run:     mpirun -np 6 ex14 -n 10

   To see options: ex14 -help

   Description:    This code solves the 2D Laplace equation using bilinear
                   finite element discretization on a mesh with an "enhanced
                   connectivity" point.  Specifically, we solve -Delta u = 1
                   with zero boundary conditions on a star-shaped domain
                   consisting of identical rhombic parts each meshed with a
                   uniform n x n grid.  Every part is assigned to a different
                   processor and all parts meet at the origin, equally
                   subdividing the 2*pi angle there. The case of six processors
                   (parts) looks as follows:

                                                +
                                               / \
                                              /   \
                                             /     \
                                   +--------+   1   +---------+
                                    \        \     /         /
                                     \    2   \   /    0    /
                                      \        \ /         /
                                       +--------+---------+
                                      /        / \         \
                                     /    3   /   \    5    \
                                    /        /     \         \
                                   +--------+   4   +---------+
                                             \     /
                                              \   /
                                               \ /
                                                +

                   Note that in this problem we use nodal variables, which are
                   shared between the different parts.  The node at the origin,
                   for example, belongs to all parts as illustrated below:

                                                .
                                               / \
                                              .   .
                                             / \ / \
                                            o   .   *
                                  .---.---o  \ / \ /  *---.---.
                                   \   \   \  o   *  /   /   /
                                    .---.---o  \ /  *---.---.
                                     \   \   \  x  /   /   /
                                      @---@---x   x---z---z
                                      @---@---x   x---z---z
                                     /   /   /  x  \   \   \
                                    .---.---a  / \  #---.---.
                                   /   /   /  a   #  \   \   \
                                  .---.---a  / \ / \  #---.---.
                                            a   .   #
                                             \ / \ /
                                              .   .
                                               \ /
                                                .

                   This example is a identical to Example 13, except that it
                   uses the SStruct FEM input functions instead of stencils to
                   describe the problem.  This is the recommended way to set up a
                   finite element problem in the SStruct interface.
*/

#include "_hypre_utilities.h"
#include "HYPRE_sstruct_mv.h"
#include "HYPRE_sstruct_ls.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#include "vis.c"

/*
   This routine computes the bilinear finite element stiffness matrix and
   load vector on a rhombus with angle gamma. Specifically, let R be the
   rhombus
                              [3]------[2]
                              /        /
                             /        /
                           [0]------[1]

   with sides of length h. The finite element stiffness matrix

                  S_ij = (grad phi_i,grad phi_j)_R

   with bilinear finite element functions {phi_i} has the form

                       /  4-k    -1  -2+k    -1 \
               alpha . |   -1   4+k    -1  -2-k |
                       | -2+k    -1   4-k    -1 |
                       \   -1  -2-k    -1   4+k /

   where alpha = 1/(6*sin(gamma)) and k = 3*cos(gamma). The load vector
   corresponding to a right-hand side of 1 is

                  F_j = (1,phi_j)_R = h^2/4 * sin(gamma)
*/
void ComputeFEMRhombus (double S[4][4], double F[4], double gamma, double h)
{
   int i, j;

   double h2_4 = h*h/4;
   double sing = sin(gamma);
   double alpha = 1/(6*sing);
   double k = 3*cos(gamma);

   S[0][0] = alpha * (4-k);
   S[0][1] = alpha * (-1);
   S[0][2] = alpha * (-2+k);
   S[0][3] = alpha * (-1);
   S[1][1] = alpha * (4+k);
   S[1][2] = alpha * (-1);
   S[1][3] = alpha * (-2-k);
   S[2][2] = alpha * (4-k);
   S[2][3] = alpha * (-1);
   S[3][3] = alpha * (4+k);

   /* The stiffness matrix is symmetric */
   for (i = 1; i < 4; i++)
      for (j = 0; j < i; j++)
         S[i][j] = S[j][i];

   for (i = 0; i < 4; i++)
      F[i] = h2_4*sing;
}


int main (int argc, char *argv[])
{
   int myid, num_procs;
   int n;
   double gamma, h;
   int vis;

   HYPRE_SStructGrid     grid;
   HYPRE_SStructGraph    graph;
   HYPRE_SStructMatrix   A;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   x;

   HYPRE_Int             solver_id = 5;
   HYPRE_Int             print_system = 0;
   HYPRE_Int             time_index;
   HYPRE_Int             object_type;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

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
         else if ( strcmp(argv[arg_index], "-solver") == 0 )
         {
            arg_index++;
            solver_id = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-vis") == 0 )
         {
            arg_index++;
            vis = 1;
         }
         else if ( strcmp(argv[arg_index], "-print") == 0 )
         {
            arg_index++;
            print_system = 1;
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
         hypre_printf("  -n <n>              : problem size per processor (default: 10)\n");
         hypre_printf("  -vis                : save the solution for GLVis visualization\n");
         hypre_printf("  -solver <ID>        : solver ID (default = 5)\n");
         hypre_printf("                         0 - SMG split solver\n");
         hypre_printf("                         1 - PFMG split solver\n");
         hypre_printf("                         4 - SSAMG\n");
         hypre_printf("                         5 - BoomerAMG\n");
         hypre_printf("                         8 - 1-step Jacobi split solver\n");
         hypre_printf("                        10 - PCG with SMG split precond\n");
         hypre_printf("                        11 - PCG with PFMG split precond\n");
         hypre_printf("                        14 - PCG with SSAMG precond\n");
         hypre_printf("                        18 - PCG with diagonal scaling\n");
         hypre_printf("                        19 - PCG\n");
         hypre_printf("                        20 - PCG with BoomerAMG precond\n");
         hypre_printf("                        21 - PCG with EUCLID precond\n");
         hypre_printf("                        22 - PCG with ParaSails precond\n");
         hypre_printf("                        28 - PCG with diagonal scaling\n");
         hypre_printf("  -print             : print out the system\n");
         hypre_printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   /* Set object type */
   if (solver_id == 5)
   {
      object_type = HYPRE_PARCSR;
   }
   else
   {
      object_type = HYPRE_SSTRUCT;
   }

   /* Set the rhombus angle, gamma, and the mesh size, h, depending on the
      number of processors np and the given n */
   if (num_procs < 3)
   {
      if (myid ==0) hypre_printf("Must run with at least 3 processors!\n");
      MPI_Finalize();
      exit(1);
   }
   gamma = 2*M_PI/num_procs;
   h = 1.0/n;

   /* 1. Set up the grid.  We will set up the grid so that processor X owns
         part X.  Note that each part has its own index space numbering. Later
         we relate the parts to each other. */
   {
      int ndim = 2;
      int nparts = num_procs;

      /* Create an empty 2D grid object */
      HYPRE_SStructGridCreate(MPI_COMM_WORLD, ndim, nparts, &grid);

      /* Set the extents of the grid - each processor sets its grid boxes.  Each
         part has its own relative index space numbering */
      {
         int part = myid;
         int ilower[2] = {1,1}; /* lower-left cell touching the origin */
         int iupper[2] = {n,n}; /* upper-right cell */

         HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
      }

      /* Set the variable type and number of variables on each part.  These need
         to be set in each part which is neighboring or contains boxes owned by
         the processor. */
      {
         int i;
         int nvars = 1;

         HYPRE_SStructVariable vartypes[1] = {HYPRE_SSTRUCT_VARIABLE_NODE};
         for (i = 0; i < nparts; i++)
            HYPRE_SStructGridSetVariables(grid, i, nvars, vartypes);
      }

      /* Set the ordering of the variables in the finite element problem.  This
         is done by listing the variable offset directions relative to the
         element's center.  See the Reference Manual for more details. */
      {
         int part = myid;
         int ordering[12] = { 0, -1, -1,    /*    [3]------[2] */
                              0, +1, -1,    /*    /        /   */
                              0, +1, +1,    /*   /        /    */
                              0, -1, +1 };  /* [0]------[1]    */

         HYPRE_SStructGridSetFEMOrdering(grid, part, ordering);
      }

      /* Now we need to set the spatial relation between each of the parts.
         Since we are using nodal variables, we have to use SetSharedPart to
         establish the connection at the origin. */
      {
         /* Relation to the clockwise-previous neighbor part, e.g. 0 and 1 for
            the case of 6 parts.  Note that we could have used SetNeighborPart
            here instead of SetSharedPart. */
         {
            int part = myid;
            /* the box of cells intersecting the boundary in the current part */
            int ilower[2] = {1,1}, iupper[2] = {1,n};
            /* share all data on the left side of the box */
            int offset[2] = {-1,0};

            int shared_part = (myid+1) % num_procs;
            /* the box of cells intersecting the boundary in the neighbor */
            int shared_ilower[2] = {1,1}, shared_iupper[2] = {n,1};
            /* share all data on the bottom of the box */
            int shared_offset[2] = {0,-1};

            /* x/y-direction on the current part is -y/x on the neighbor */
            int index_map[2] = {1,0};
            int index_dir[2] = {-1,1};

            HYPRE_SStructGridSetSharedPart(grid, part, ilower, iupper, offset,
                                           shared_part, shared_ilower,
                                           shared_iupper, shared_offset,
                                           index_map, index_dir);
         }

         /* Relation to the clockwise-following neighbor part, e.g. 0 and 5 for
            the case of 6 parts.  Note that we could have used SetNeighborPart
            here instead of SetSharedPart. */
         {
            int part = myid;
            /* the box of cells intersecting the boundary in the current part */
            int ilower[2] = {1,1}, iupper[2] = {n,1};
            /* share all data on the bottom of the box */
            int offset[2] = {0,-1};

            int shared_part = (myid+num_procs-1) % num_procs;
            /* the box of cells intersecting the boundary in the neighbor */
            int shared_ilower[2] = {1,1}, shared_iupper[2] = {1,n};
            /* share all data on the left side of the box */
            int shared_offset[2] = {-1,0};

            /* x/y-direction on the current part is y/-x on the neighbor */
            int index_map[2] = {1,0};
            int index_dir[2] = {1,-1};

            HYPRE_SStructGridSetSharedPart(grid, part, ilower, iupper, offset,
                                           shared_part, shared_ilower,
                                           shared_iupper, shared_offset,
                                           index_map, index_dir);
         }

         /* Relation to all other parts, e.g. 0 and 2,3,4.  This can be
            described only by SetSharedPart. */
         {
            int part = myid;
            /* the (one cell) box that touches the origin */
            int ilower[2] = {1,1}, iupper[2] = {1,1};
            /* share all data in the bottom left corner (i.e. the origin) */
            int offset[2] = {-1,-1};

            int shared_part;
            /* the box of one cell that touches the origin */
            int shared_ilower[2] = {1,1}, shared_iupper[2] = {1,1};
            /* share all data in the bottom left corner (i.e. the origin) */
            int shared_offset[2] = {-1,-1};

            /* x/y-direction on the current part is -x/-y on the neighbor, but
               in this case the arguments are not really important since we are
               only sharing a point */
            int index_map[2] = {0,1};
            int index_dir[2] = {-1,-1};

            for (shared_part = 0; shared_part < myid-1; shared_part++)
               HYPRE_SStructGridSetSharedPart(grid, part, ilower, iupper, offset,
                                              shared_part, shared_ilower,
                                              shared_iupper, shared_offset,
                                              index_map, index_dir);

            for (shared_part = myid+2; shared_part < num_procs; shared_part++)
               HYPRE_SStructGridSetSharedPart(grid, part, ilower, iupper, offset,
                                              shared_part, shared_ilower,
                                              shared_iupper, shared_offset,
                                              index_map, index_dir);
         }
      }

      /* Now the grid is ready to be used */
      HYPRE_SStructGridAssemble(grid);
   }

   /* 2. Set up the Graph - this determines the non-zero structure of the
         matrix. */
   {
      int part;

      /* Create the graph object */
      HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);

      /* See MatrixSetObjectType below */
      HYPRE_SStructGraphSetObjectType(graph, object_type);

      /* Indicate that this problem uses finite element stiffness matrices and
         load vectors, instead of stencils. */
      for (part = 0; part < num_procs; part++)
         HYPRE_SStructGraphSetFEM(graph, part);

      /* The local stiffness matrix is full, so there is no need to call
         HYPRE_SStructGraphSetFEMSparsity to set its sparsity pattern. */

      /* Assemble the graph */
      HYPRE_SStructGraphAssemble(graph);
   }

   /* 3. Set up the SStruct Matrix and right-hand side vector */
   {
      int part = myid;

      /* Create the matrix object */
      HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);
      /* Use a ParCSR storage */
      HYPRE_SStructMatrixSetObjectType(A, object_type);
      /* Indicate that the matrix coefficients are ready to be set */
      HYPRE_SStructMatrixInitialize(A);

      /* Create an empty vector object */
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
      /* Use a ParCSR storage */
      HYPRE_SStructVectorSetObjectType(b, object_type);
      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_SStructVectorInitialize(b);

      /* Set the matrix and vector entries by finite element assembly */
      {
         /* local stifness matrix and load vector */
         double S[4][4], F[4];

         int i, j, k;
         int index[2];

         /* set the values in the interior cells */
         {
            ComputeFEMRhombus(S, F, gamma, h);

            for (i = 1; i <= n; i++)
               for (j = 1; j <= n; j++)
               {
                  index[0] = i;
                  index[1] = j;
                  HYPRE_SStructMatrixAddFEMValues(A, part, index, &S[0][0]);
                  HYPRE_SStructVectorAddFEMValues(b, part, index, F);
               }
         }

         /* cells having nodes 1,2 on the domain boundary */
         {
            ComputeFEMRhombus(S, F, gamma, h);

            /* eliminate nodes 1,2 from S and F */
            for (k = 0; k < 4; k++)
            {
               S[1][k] = S[k][1] = 0.0;
               S[2][k] = S[k][2] = 0.0;
            }
            S[1][1] = 1.0;
            S[2][2] = 1.0;
            F[1] = 0.0;
            F[2] = 0.0;

            for (i = n; i <= n; i++)
               for (j = 1; j <= n; j++)
               {
                  index[0] = i;
                  index[1] = j;
                  HYPRE_SStructMatrixAddFEMValues(A, part, index, &S[0][0]);
                  HYPRE_SStructVectorAddFEMValues(b, part, index, F);
               }
         }

         /* cells having nodes 2,3 on the domain boundary */
         {
            ComputeFEMRhombus(S, F, gamma, h);

            /* eliminate nodes 2,3 from S and F */
            for (k = 0; k < 4; k++)
            {
               S[2][k] = S[k][2] = 0.0;
               S[3][k] = S[k][3] = 0.0;
            }
            S[2][2] = 1.0;
            S[3][3] = 1.0;
            F[2] = 0.0;
            F[3] = 0.0;

            for (i = 1; i <= n; i++)
               for (j = n; j <= n; j++)
               {
                  index[0] = i;
                  index[1] = j;
                  HYPRE_SStructMatrixAddFEMValues(A, part, index, &S[0][0]);
                  HYPRE_SStructVectorAddFEMValues(b, part, index, F);
               }

         }

         /* cells having nodes 1,2,3 on the domain boundary */
         {
            ComputeFEMRhombus(S, F, gamma, h);

            /* eliminate nodes 2,3 from S and F */
            for (k = 0; k < 4; k++)
            {
               S[1][k] = S[k][1] = 0.0;
               S[2][k] = S[k][2] = 0.0;
               S[3][k] = S[k][3] = 0.0;
            }
            S[1][1] = 1.0;
            S[2][2] = 1.0;
            S[3][3] = 1.0;
            F[1] = 0.0;
            F[2] = 0.0;
            F[3] = 0.0;

            for (i = n; i <= n; i++)
               for (j = n; j <= n; j++)
               {
                  index[0] = i;
                  index[1] = j;
                  HYPRE_SStructMatrixAddFEMValues(A, part, index, &S[0][0]);
                  HYPRE_SStructVectorAddFEMValues(b, part, index, F);
               }
         }
      }
   }

   /* Collective calls finalizing the matrix and vector assembly */
   HYPRE_SStructMatrixAssemble(A);
   HYPRE_SStructVectorAssemble(b);

   if (print_system)
   {
      HYPRE_SStructVectorGather(b);
      HYPRE_SStructMatrixPrint("sstruct.out.A", A, 0);
      HYPRE_SStructVectorPrint("sstruct.out.b", b, 0);
   }

   /* 4. Set up SStruct Vector for the solution vector x */
   {
      int part = myid;
      int var = 0;
      int nvalues = (n+1)*(n+1);
      double *values;

      /* Since the SetBoxValues() calls below set the values of the nodes in
         the upper-right corners of the cells, the nodal box should start
         from (0,0) instead of (1,1). */
      int ilower[2] = {0,0};
      int iupper[2] = {n,n};

      values = (double*) calloc(nvalues, sizeof(double));

      /* Create an empty vector object */
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);
      /* Set the object type to ParCSR */
      HYPRE_SStructVectorSetObjectType(x, object_type);
      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_SStructVectorInitialize(x);
      /* Set the values for the initial guess */
      HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);

      free(values);

      /* Finalize the vector assembly */
      HYPRE_SStructVectorAssemble(x);
   }

   /* 5. Set up and call the solver (Solver options can be found in the
         Reference Manual.) */
   HYPRE_Real final_res_norm;
   HYPRE_Int  its;

   if (solver_id == 5)
   {
      HYPRE_Solver          solver;
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
      HYPRE_BoomerAMGSetOldDefault(solver);
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
   }
   else if ((solver_id >= 10) && (solver_id < 20))
   {
      HYPRE_SStructSolver  sstruct_solver;
      HYPRE_Int  krylov_max_iter = 100;
      HYPRE_Int  krylov_print_level = 3;
      HYPRE_Real tol = 1.0e-6;
      HYPRE_Int  rel_change = 0;

      HYPRE_SStructSolver  sstruct_precond;
      HYPRE_Int    max_levels = 20;
      HYPRE_Int    relax_type = 0;
      HYPRE_Int    usr_jacobi_weight = 0;
      HYPRE_Real   jacobi_weight = 1.0;
      HYPRE_Int    n_pre = 0;
      HYPRE_Int    n_pos = 0;

      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_SStructPCGCreate(hypre_MPI_COMM_WORLD, &sstruct_solver);
      HYPRE_PCGSetMaxIter( (HYPRE_Solver) sstruct_solver, krylov_max_iter );
      HYPRE_PCGSetTol( (HYPRE_Solver) sstruct_solver, tol );
      HYPRE_PCGSetTwoNorm( (HYPRE_Solver) sstruct_solver, 1 );
      HYPRE_PCGSetRelChange( (HYPRE_Solver) sstruct_solver, rel_change );
      HYPRE_PCGSetPrintLevel( (HYPRE_Solver) sstruct_solver, krylov_print_level );

      if (solver_id == 14)
      {
         /* use SSAMG solver as preconditioner */
         HYPRE_SStructSSAMGCreate(hypre_MPI_COMM_WORLD, &sstruct_precond);
         HYPRE_SStructSSAMGSetMaxIter(sstruct_precond, 1);
         HYPRE_SStructSSAMGSetMaxLevels(sstruct_precond, max_levels);
         HYPRE_SStructSSAMGSetTol(sstruct_precond, 0.0);
         HYPRE_SStructSSAMGSetZeroGuess(sstruct_precond);
         HYPRE_SStructSSAMGSetRelaxType(sstruct_precond, relax_type);
         if (usr_jacobi_weight)
         {
            HYPRE_SStructSSAMGSetRelaxWeight(sstruct_precond, jacobi_weight);
         }
         HYPRE_SStructSSAMGSetNumPreRelax(sstruct_precond, n_pre);
         HYPRE_SStructSSAMGSetNumPostRelax(sstruct_precond, n_pos);
         HYPRE_SStructSSAMGSetPrintLevel(sstruct_precond, 1);
         HYPRE_SStructSSAMGSetLogging(sstruct_precond, 1);

         HYPRE_PCGSetPrecond( (HYPRE_Solver) sstruct_solver,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_SStructSSAMGSetup,
                              (HYPRE_Solver) sstruct_precond);
      }
      HYPRE_PCGSetup( (HYPRE_Solver) sstruct_solver, (HYPRE_Matrix) A,
                      (HYPRE_Vector) b, (HYPRE_Vector) x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_PCGSolve( (HYPRE_Solver) sstruct_solver, (HYPRE_Matrix) A,
                      (HYPRE_Vector) b, (HYPRE_Vector) x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_PCGGetNumIterations( (HYPRE_Solver) sstruct_solver, &its );
      HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver) sstruct_solver, &final_res_norm );
      HYPRE_SStructPCGDestroy(sstruct_solver);

      if (solver_id == 14)
      {
         HYPRE_SStructSSAMGDestroy(sstruct_precond);
      }
   }
   else
   {
      if (myid ==0)
      {
         hypre_printf("Solver ID not supported: %d\n", solver_id);
      }
      MPI_Finalize();
      exit(1);
   }

   if (myid == 0)
   {
      hypre_printf("\n");
      hypre_printf("Iterations = %d\n", its);
      hypre_printf("Final Relative Residual Norm = %g\n", final_res_norm);
      hypre_printf("\n");
   }

   /* Save the solution for GLVis visualization, see vis/glvis-ex13.sh */
   if (vis)
   {
      FILE *file;
      char filename[255];

      int i, part = myid, var = 0;
      int nvalues = (n+1)*(n+1);
      double *values = (double*) calloc(nvalues, sizeof(double));
      int ilower[2] = {0,0};
      int iupper[2] = {n,n};

      /* get all local data (including a local copy of the shared values) */
      HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                      var, values);

      hypre_sprintf(filename, "%s.%06d", "vis/ex14.sol", myid);
      if ((file = fopen(filename, "w")) == NULL)
      {
         hypre_printf("Error: can't open output file %s\n", filename);
         MPI_Finalize();
         exit(1);
      }

      /* finite element space header */
      hypre_fprintf(file, "FiniteElementSpace\n");
      hypre_fprintf(file, "FiniteElementCollection: H1_2D_P1\n");
      hypre_fprintf(file, "VDim: 1\n");
      hypre_fprintf(file, "Ordering: 0\n\n");

      /* save solution */
      for (i = 0; i < nvalues; i++)
      {
         hypre_fprintf(file, "%.14e\n", values[i]);
      }

      fflush(file);
      fclose(file);
      free(values);

      /* save local finite element mesh */
      GLVis_PrintLocalRhombusMesh("vis/ex14.mesh", n, myid, gamma);

      /* additional visualization data */
      GLVis_PrintData("vis/ex14.data", myid, num_procs);
   }

   /* Free memory */
   HYPRE_SStructGridDestroy(grid);
   HYPRE_SStructGraphDestroy(graph);
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);

   /* Finalize MPI */
   MPI_Finalize();

   return 0;
}

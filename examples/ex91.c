/*
   Example 17

   Interface:      SStructured interface (SStruct)

   Compile with:   make ex17

   Sample run:     mpirun -np 16 ex17 -n 40 -r 2 -solver 1

   To see options: ex17 -help

   Description:

                   This example uses the sstruct interface to solve a 2D
                   Laplacian problem on a cell-centered grid with two refined
                   regions.  The discretization is based on a standard 5-point
                   finite difference scheme and illustrates the use of the
                   SStructGridSetAMR routines to define interpolation at ghost
                   cells to build the linear system.  The grid is as follows:


                   *************************************************
                   *           |           |           |      (N,N)*
                   *           |           |           |           *
                   *           |           |           |           *
                   *           |           |           |           *
                   *           |           |           |           *
                   *-----------|-----------*************-----------*
                   *           |           *     |     *           *
                   *           |           *     |     *           *
                   *           |           *-----|-----*           *
                   *           |           *     |     *           *
                   *           |           *     |     *           *
                   *************************************-----------*
                   *     |     |     |     *   ^       |           *
                   *     |     |     |     *   |       |           *
                   *-----|-----|-----|-----*<--part 1  |           *
                   *     |     |     |     *           |           *
                   *     |     |     |     *           |           *
                   *-----|-----|-----|-----*-----------|-----------*
                   *     |     |     |     *           |           *
                   *     |     |     |     *           |           *
                   *-----|-----|-----|-----*           |  part 0   *
                   *     |     |     |     *           |           *
                   *(1,1)|     |     |     *           |           *
                   *************************************************

                   The grid is constructed from two parts.  Part 0 is the coarse
                   level and consists of one global box of size N x N.  Part 1
                   is a refinement of part 0 by factor r (pictured above is r=2)
                   and consists of a global box of size r*(N/2) x r*(N/2) and a
                   second box of size r*(N/4) x r*(N/4) .  The value of N is
                   given by N=(4*n)*M, where n is an input parameter and M is
                   based on the number of processors (see below).

                   The discretization is given by the standard 4,-1 stencil on
                   part 0 and the same stencil scaled by r^2 on part 1.  The
                   coupling between parts is determined by interpolation and
                   restriction operators, P and R, through the Petrov-Galerkin
                   product, RAP.  These operators are set up through a reference
                   coarse-fine interpolation template, followed by fix-ups at
                   certain grid locations.  The index spaces on the two parts
                   are aligned such that cell (1,1) on the fine level is in the
                   lower-left corner of cell (1,1) on the coarse level.  The
                   boundary conditions are zero Dirichlet conditions.

                   The grid is parallelized by distributing each of the three
                   main grid boxes on an M x M process grid, so the number of
                   processes specified must be a perfect square.  The solvers
                   available are various ParCSR solvers, including BoomerAMG.
*/

#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE_sstruct_ls.h"

#ifdef M_PI
  #define PI M_PI
#else
  #define PI 3.14159265358979
#endif

#include "vis.c"

#define RMAX 10

/* Macro to evaluate a function F in the grid point (i,j) */
#define Eval(F,i,j) (F( (ilower[0]+(i))*h, (ilower[1]+(j))*h ))
#define bcEval(F,i,j) (F( (bc_ilower[0]+(i))*h, (bc_ilower[1]+(j))*h ))

int optionK, optionB, optionC, optionU0, optionF;

/* Diffusion coefficient */
double K(double x, double y)
{
   switch (optionK)
   {
      case 0:
         return 1.0;
      case 1:
         return x*x+exp(y);
      case 2:
         if ((fabs(x-0.5) < 0.25) && (fabs(y-0.5) < 0.25))
            return 100.0;
         else
            return 1.0;
      case 3:
         if (((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)) < 0.0625)
            return 10.0;
         else
            return 1.0;
      default:
         return 1.0;
   }
}

/* Convection vector, first component */
double B1(double x, double y)
{
   switch (optionB)
   {
      case 0:
         return 0.0;
      case 1:
         return -0.1;
      case 2:
         return 0.25;
      case 3:
         return 1.0;
      default:
         return 0.0;
   }
}

/* Convection vector, second component */
double B2(double x, double y)
{
   switch (optionB)
   {
      case 0:
         return 0.0;
      case 1:
         return 0.1;
      case 2:
         return -0.25;
      case 3:
         return 1.0;
      default:
         return 0.0;
   }
}

/* Reaction coefficient */
double C(double x, double y)
{
   switch (optionC)
   {
      case 0:
         return 0.0;
      case 1:
         return 10.0;
      case 2:
         return 100.0;
      default:
         return 0.0;
   }
}

/* Boundary condition */
double U0(double x, double y)
{
   switch (optionU0)
   {
      case 0:
         return 0.0;
      case 1:
         return (x+y)/100;
      case 2:
         return (sin(5*PI*x)+sin(5*PI*y))/1000;
      default:
         return 0.0;
   }
}

/* Right-hand side */
double F(double x, double y)
{
   switch (optionF)
   {
      case 0:
         return 1.0;
      case 1:
         return 0.0;
      case 2:
         return 2*PI*PI*sin(PI*x)*sin(PI*y);
      case 3:
         if ((fabs(x-0.5) < 0.25) && (fabs(y-0.5) < 0.25))
            return -1.0;
         else
            return 1.0;
      case 4:
         if (((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)) < 0.0625)
            return -1.0;
         else
            return 1.0;
      default:
         return 1.0;
   }
}

int main (int argc, char *argv[])
{
   int i, j, k;

   int myid, num_procs;

   int n, M, pi, pj;
   double h, h2;
   int ilower[2], iupper[2];
   int coarse_index[2], fine_index[2];

   int solver_id;
   int n_pre, n_post;
   int rap, relax, skip, sym;
   int time_index;

   int object_type;

   int num_iterations;
   double final_res_norm;

   int vis;

   HYPRE_SStructGrid     grid;
   HYPRE_SStructStencil  stencil;
   HYPRE_SStructGraph    graph;
   HYPRE_SStructMatrix   A;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   x;

   /* We are using struct solvers for this example */
   HYPRE_StructSolver   solver;
   HYPRE_StructSolver   precond;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Set default parameters */
   n         = 40;
   r         = 2;
   solver_id = 1;
   n_pre     = 1;
   n_post    = 1;

   vis       = 0;

   optionK   = 0;
   optionB   = 0;
   optionC   = 0;
   optionU0  = 0;
   optionF   = 0;

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
         else if ( strcmp(argv[arg_index], "-K") == 0 )
         {
            arg_index++;
            optionK = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-B") == 0 )
         {
            arg_index++;
            optionB = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-C") == 0 )
         {
            arg_index++;
            optionC = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-U0") == 0 )
         {
            arg_index++;
            optionU0 = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-F") == 0 )
         {
            arg_index++;
            optionF = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-solver") == 0 )
         {
            arg_index++;
            solver_id = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-v") == 0 )
         {
            arg_index++;
            n_pre = atoi(argv[arg_index++]);
            n_post = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rap") == 0 )
         {
            arg_index++;
            rap = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-relax") == 0 )
         {
            arg_index++;
            relax = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-skip") == 0 )
         {
            arg_index++;
            skip = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-sym") == 0 )
         {
            arg_index++;
            sym = atoi(argv[arg_index++]);
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
         printf("  -n  <n>             : problem size per processor (default: 8)\n");
         printf("  -K  <K>             : choice for the diffusion coefficient (default: 1)\n");
         printf("  -B  <B>             : choice for the convection vector (default: 0)\n");
         printf("  -C  <C>             : choice for the reaction coefficient (default: 0)\n");
         printf("  -U0 <U0>            : choice for the boundary condition (default: 0)\n");
         printf("  -F  <F>             : choice for the right-hand side (default: 1) \n");
         printf("  -solver <ID>        : solver ID\n");
         printf("                        0  - SMG \n");
         printf("                        1  - PFMG\n");
         printf("                        10 - CG with SMG precond (default)\n");
         printf("                        11 - CG with PFMG precond\n");
         printf("                        17 - CG with 2-step Jacobi\n");
         printf("                        18 - CG with diagonal scaling\n");
         printf("                        19 - CG\n");
         printf("                        30 - GMRES with SMG precond\n");
         printf("                        31 - GMRES with PFMG precond\n");
         printf("                        37 - GMRES with 2-step Jacobi\n");
         printf("                        38 - GMRES with diagonal scaling\n");
         printf("                        39 - GMRES\n");
         printf("  -v <n_pre> <n_post> : number of pre and post relaxations\n");
         printf("  -rap <r>            : coarse grid operator type\n");
         printf("                        0 - Galerkin (default)\n");
         printf("                        1 - non-Galerkin ParFlow operators\n");
         printf("                        2 - Galerkin, general operators\n");
         printf("  -relax <r>          : relaxation type\n");
         printf("                        0 - Jacobi\n");
         printf("                        1 - Weighted Jacobi (default)\n");
         printf("                        2 - R/B Gauss-Seidel\n");
         printf("                        3 - R/B Gauss-Seidel (nonsymmetric)\n");
         printf("  -skip <s>           : skip levels in PFMG (0 or 1)\n");
         printf("  -sym <s>            : symmetric storage (1) or not (0)\n");
         printf("  -vis                : save the solution for GLVis visualization\n");
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   /* Figure out the processor grid (M x M).  The local
      problem size is indicated by n (n x n). pi and pj
      indicate position in the processor grid. */
   M  = sqrt(num_procs);
   n0 = 4*n;            /* size of box0 is n0 x n0 */
   n1 = 2*n*r;          /* size of box1 is n1 x n1 */
   n2 = 1*n*r;          /* size of box2 is n2 x n2 */
   h0 = 1.0 / (n0*M);   /* grid spacing on part 0  */
   h1 = h0 / r;         /* grid spacing on part 1  */
   pj = myid / M;
   pi = myid - pj*M;

   /* Define the cells owned by the current processor (each processor's
      piece of the global grid) */
   ilower0[0] = pi*n0+1;
   ilower0[1] = pj*n0+1;
   iupper0[0] = ilower0[0] + n0-1;
   iupper0[1] = ilower0[1] + n0-1;
   ilower1[0] = pi*n1+1;
   ilower1[1] = pj*n1+1;
   iupper1[0] = ilower1[0] + n1-1;
   iupper1[1] = ilower1[1] + n1-1;
   ilower2[0] = n1+pi*n2+1;
   ilower2[1] = n1+pj*n2+1;
   iupper2[0] = ilower2[0] + n2-1;
   iupper2[1] = ilower2[1] + n2-1;

   /* Define the indexes where the two parts align */
   coarse_index[0] = coarse_index[1] = 1;
   fine_index[0]   = fine_index[1]   = 1;

   /* Define the refinement factors */
   rfactors[0] = rfactors[1] = r;

   /* 1. Set up a 2D grid */
   {
      int ndim = 2;
      int nparts = 2;
      int nvars = 1;
      int coarse_part = 0;
      int fine_part = 1;
      int var = 0;
      int i;

      /* Create an empty 2D grid object */
      HYPRE_SStructGridCreate(MPI_COMM_WORLD, ndim, nparts, &grid);

      /* Set the object type for the system.  See GetAMRObjects() below. */
      object_type = HYPRE_PARCSR;
      HYPRE_SStructGridSetObjectType(grid, object_type);

      /* Add the box on the coarse grid part 0 */
      HYPRE_SStructGridSetExtents(grid, coarse_part, ilower0, iupper0);

      /* Add the boxes on the fine grid part 1 */
      part = 1;
      HYPRE_SStructGridSetExtents(grid, fine_part, ilower1, iupper1);
      HYPRE_SStructGridSetExtents(grid, fine_part, ilower2, iupper2);

      /* Set the variable type for each part */
      {
         HYPRE_SStructVariable vartypes[1] = {HYPRE_SSTRUCT_VARIABLE_CELL};

         for (i = 0; i< nparts; i++)
            HYPRE_SStructGridSetVariables(grid, i, nvars, vartypes);
      }

      /* Declare fine part 1 to be a refinement of coarse part 0 with a given
       * refinement factor (rfactors) and alignment */
      HYPRE_SStructGridSetAMRPart(grid, coarse_part, fine_part,
                                  coarse_index, fine_index, rfactors);


      /* Define interpolation and restriction on the reference coarse-fine
       * template and fix up interpolation at certain specific grid locations.
       * For interpolation, use bilinear between both coarse and fine real
       * variables.  For restriction, decouple fine slave variables from all
       * real variables (injection). */
      {
         int    row, nvalues, vars[4], cf[4], indexes[8];
         double values[4];
         int    edge;
         double v1[RMAX][2], s;

         /* Set linear interpolation values and indexes in 1D.
          *
          * Examples:
          *   r=2: v1 =                 [3/4,1/4,0] [3/4,0,1/4]
          *   r=3: v1 =             [2/3,1/3,0] [1,0,0] [2/3,0,1/3]
          *   r=4: v1 =     [5/8,3/8,0] [7/8,1/8,0] [7/8,0,1/8] [5/8,0,3/8]
          *   r=5: v1 = [3/5,2/5,0] [4/5,1/5,0] [1,0,0] [4/5,0,1/5] [3/5,0,2/5]
          */

         for (i = 0; i < r/2; i++)
         {
            j = r-1-i;
            s = (r+1 + 2*i)/(2*r);
            v1[i][0] = s;    v1[i][1] = 1-s;  v1[i][2] = 0;
            v1[j][0] = s;    v1[j][1] = 0;    v1[j][2] = 1-s;
         }
         if (r%2) /* odd case - set middle values */
         {
            i = r/2;
            v1[i][0] = 1;    v1[i][1] = 0;    v1[i][2] = 0;
         }

         /* Set up interpolation in the reference coarse-fine template. */
         vars[0] = 0;  cf[0] = 0;  /* coarse */
         vars[1] = 0;  cf[1] = 0;  /* coarse */
         vars[2] = 0;  cf[2] = 0;  /* coarse */
         for (i = 0; i < r; i++)
         {
            vars[3+i] = 0;  cf[3+i] = 1;  /* fine */
         }
         t = 2/(r+1);
         for (edge = 0; edge < 4; edge++)
         {
            cvalues[0] = 0;
            cvalues[1] = 0;
            cvalues[2] = 0;
            for (i = 0; i < r; i++)
            {
               values[0] = t*v1[i][0];
               values[1] = t*v1[i][1];
               values[2] = t*v1[i][2];
               values[3] = (1-t);
               switch (edge)
               {
                  case 0: /* west edge */
                     index[0]   = -1;    index[1]   =  i;
                     indexes[0] = -1;    indexes[1] =  0;  /* coarse */
                     indexes[2] = -1;    indexes[3] = -1;  /* coarse */
                     indexes[4] = -1;    indexes[5] =  1;  /* coarse */
                     indexes[6] =  0;    indexes[7] =  i;  /* fine */
                     cindex[0]  = -1;    cindex[1]  =  0;
                     break;
                  case 1: /* east edge */
                     index[0]   =  r;    index[1]   =  i;
                     indexes[0] =  1;    indexes[1] =  0;  /* coarse */
                     indexes[2] =  1;    indexes[3] = -1;  /* coarse */
                     indexes[4] =  1;    indexes[5] =  1;  /* coarse */
                     indexes[6] =  r;    indexes[7] =  i;  /* fine */
                     cindex[0]  =  1;    cindex[1]  =  0;
                     break;
                  case 2: /* south edge */
                     index[0]   =  i;    index[1]   = -1;
                     indexes[0] =  0;    indexes[1] = -1;  /* coarse */
                     indexes[2] = -1;    indexes[3] = -1;  /* coarse */
                     indexes[4] =  1;    indexes[5] = -1;  /* coarse */
                     indexes[6] =  i;    indexes[7] =  0;  /* fine */
                     cindex[0]  =  0;    cindex[1]  = -1;
                     break;
                  case 3: /* north edge */
                     index[0]   =  i;    index[1]   =  r;
                     indexes[0] =  0;    indexes[1] =  1;  /* coarse */
                     indexes[2] = -1;    indexes[3] =  1;  /* coarse */
                     indexes[4] =  1;    indexes[5] =  1;  /* coarse */
                     indexes[6] =  i;    indexes[7] =  r;  /* fine */
                     cindex[0]  =  0;    cindex[1]  =  1;
                     break;
               }
               cvalues[0]  += values[0];
               cvalues[1]  += values[1];
               cvalues[2]  += values[2];
               cvalues[3+i] = values[3];
               cindexes[0]     = indexes[0];    cindexes[1]     = indexes[1];
               cindexes[2]     = indexes[2];    cindexes[3]     = indexes[3];
               cindexes[4]     = indexes[4];    cindexes[5]     = indexes[5];
               cindexes[6+2*i] = indexes[6];    cindexes[7+2*i] = indexes[7];

               nvalues = 4;
               HYPRE_SStructGridSetAMRRefInterp(
                  grid, coarse_part, 1, var, index,
                  nvalues, vars, cf, indexes, values);
               nvalues = 0;
               HYPRE_SStructGridSetAMRRefRestrictT(
                  grid, coarse_part, 1, var, index,
                  nvalues, vars, cf, indexes, values);
            }
            nvalues = 3+r;
            HYPRE_SStructGridSetAMRRefInterp(
               grid, coarse_part, 0, var, cindex,
               nvalues, vars, cf, cindexes, cvalues);
         }

#if 0
         /* Fix up interpolation at certain specific grid locations (later) */
         HYPRE_SStructGridSetAMRInterp(
            grid, coarse_part, coarse_index, 1, var, index,
            nvalues, vars, cf, indexes, values);
#endif
      }

#if 0
      HYPRE_SStructGetAMRObjects(HYPRE_SStructMatrix   matrix,
                                 HYPRE_SStructVector   rhs,
                                 void                **matrix_object,
                                 void                **rhs_object);
#endif

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      HYPRE_SStructGridAssemble(grid);
   }

   /* 2. Define the discretization stencil */
   {
      int ndim = 2;
      int var = 0;

      if (sym == 0)
      {
         /* Define the geometry of the stencil */
         int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};

         /* Create an empty 2D, 5-pt stencil object */
         HYPRE_SStructStencilCreate(ndim, 5, &stencil);

         /* Assign stencil entries */
         for (i = 0; i < 5; i++)
            HYPRE_SStructStencilSetEntry(stencil, i, offsets[i], var);
      }
      else /* Symmetric storage */
      {
         /* Define the geometry of the stencil */
         int offsets[3][2] = {{0,0}, {1,0}, {0,1}};

         /* Create an empty 2D, 3-pt stencil object */
         HYPRE_SStructStencilCreate(ndim, 3, &stencil);

         /* Assign stencil entries */
         for (i = 0; i < 3; i++)
            HYPRE_SStructStencilSetEntry(stencil, i, offsets[i], var);
      }
   }

   /* 3. Set up the Graph  - this determines the non-zero structure
      of the matrix */
   {
      int var = 0;
      int part = 0;

      /* Create the graph object */
      HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);

      /* Now we need to tell the graph which stencil to use for each
         variable on each part (we only have one variable and one part)*/
      HYPRE_SStructGraphSetStencil(graph, part, var, stencil);

      /* Assemble the graph */
      HYPRE_SStructGraphAssemble(graph);
   }

   /* 4. Set up SStruct Vectors for b and x */
   {
      double *values;

      /* We have one part and one variable. */
      int part = 0;
      int var = 0;

      /* Create an empty vector object */
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);

      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_SStructVectorInitialize(b);
      HYPRE_SStructVectorInitialize(x);

      values = calloc((n*n), sizeof(double));

      /* Set the values of b in left-to-right, bottom-to-top order */
      for (k = 0, j = 0; j < n; j++)
         for (i = 0; i < n; i++, k++)
            values[k] = h2 * Eval(F,i,j);
      HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);

      /* Set x = 0 */
      for (i = 0; i < (n*n); i ++)
         values[i] = 0.0;
      HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);

      free(values);

      /* Assembling is postponed since the vectors will be further modified */
   }

   /* 4. Set up a SStruct Matrix */
   {
      /* We have one part and one variable. */
      int part = 0;
      int var = 0;

      /* Create an empty matrix object */
      HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);

      /* Use symmetric storage? The function below is for symmetric stencil entries
         (use HYPRE_SStructMatrixSetNSSymmetric for non-stencil entries) */
      HYPRE_SStructMatrixSetSymmetric(A, part, var, var, sym);

      /* Indicate that the matrix coefficients are ready to be set */
      HYPRE_SStructMatrixInitialize(A);

      /* Set the stencil values in the interior. Here we set the values
         at every node. We will modify the boundary nodes later. */
      if (sym == 0)
      {
         int stencil_indices[5] = {0, 1, 2, 3, 4}; /* labels correspond
                                                      to the offsets */
         double *values;

         values = calloc(5*(n*n), sizeof(double));

         /* The order is left-to-right, bottom-to-top */
         for (k = 0, j = 0; j < n; j++)
            for (i = 0; i < n; i++, k+=5)
            {
               values[k+1] = - Eval(K,i-0.5,j) - Eval(B1,i-0.5,j);

               values[k+2] = - Eval(K,i+0.5,j) + Eval(B1,i+0.5,j);

               values[k+3] = - Eval(K,i,j-0.5) - Eval(B2,i,j-0.5);

               values[k+4] = - Eval(K,i,j+0.5) + Eval(B2,i,j+0.5);

               values[k] = h2 * Eval(C,i,j)
                  + Eval(K ,i-0.5,j) + Eval(K ,i+0.5,j)
                  + Eval(K ,i,j-0.5) + Eval(K ,i,j+0.5)
                  - Eval(B1,i-0.5,j) + Eval(B1,i+0.5,j)
                  - Eval(B2,i,j-0.5) + Eval(B2,i,j+0.5);
            }

         HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                         var, 5,
                                         stencil_indices, values);

         free(values);
      }
      else /* Symmetric storage */
      {
         int stencil_indices[3] = {0, 1, 2};
         double *values;

         values = calloc(3*(n*n), sizeof(double));

         /* The order is left-to-right, bottom-to-top */
         for (k = 0, j = 0; j < n; j++)
            for (i = 0; i < n; i++, k+=3)
            {
               values[k+1] = - Eval(K,i+0.5,j);
               values[k+2] = - Eval(K,i,j+0.5);
               values[k] = h2 * Eval(C,i,j)
                  + Eval(K,i+0.5,j) + Eval(K,i,j+0.5)
                  + Eval(K,i-0.5,j) + Eval(K,i,j-0.5);
            }

         HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                         var, 3,
                                         stencil_indices, values);

         free(values);
      }
   }

   /* 5. Set the boundary conditions, while eliminating the coefficients
         reaching ouside of the domain boundary. We must modify the matrix
         stencil and the corresponding rhs entries. */
   {
      int bc_ilower[2];
      int bc_iupper[2];

      int stencil_indices[5] = {0, 1, 2, 3, 4};
      double *values, *bvalues;

      int nentries;

      /* We have one part and one variable. */
      int part = 0;
      int var = 0;

      if (sym == 0)
         nentries = 5;
      else
         nentries = 3;

      values  = calloc(nentries*n, sizeof(double));
      bvalues = calloc(n, sizeof(double));

      /* The stencil at the boundary nodes is 1-0-0-0-0. Because
         we have I x_b = u_0; */
      for (i = 0; i < nentries*n; i += nentries)
      {
         values[i] = 1.0;
         for (j = 1; j < nentries; j++)
            values[i+j] = 0.0;
      }

      /* Processors at y = 0 */
      if (pj == 0)
      {
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n;

         bc_iupper[0] = bc_ilower[0] + n-1;
         bc_iupper[1] = bc_ilower[1];

         /* Modify the matrix */
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);

         /* Put the boundary conditions in b */
         for (i = 0; i < n; i++)
            bvalues[i] = bcEval(U0,i,0);

         HYPRE_SStructVectorSetBoxValues(b, part, bc_ilower,
                                         bc_iupper, var, bvalues);
      }

      /* Processors at y = 1 */
      if (pj == N-1)
      {
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n + n-1;

         bc_iupper[0] = bc_ilower[0] + n-1;
         bc_iupper[1] = bc_ilower[1];

         /* Modify the matrix */
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);

         /* Put the boundary conditions in b */
         for (i = 0; i < n; i++)
            bvalues[i] = bcEval(U0,i,0);

         HYPRE_SStructVectorSetBoxValues(b, part, bc_ilower, bc_iupper, var, bvalues);
      }

      /* Processors at x = 0 */
      if (pi == 0)
      {
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n-1;

         /* Modify the matrix */
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);

         /* Put the boundary conditions in b */
         for (j = 0; j < n; j++)
            bvalues[j] = bcEval(U0,0,j);

         HYPRE_SStructVectorSetBoxValues(b, part, bc_ilower, bc_iupper,
                                         var, bvalues);
      }

      /* Processors at x = 1 */
      if (pi == N-1)
      {
         bc_ilower[0] = pi*n + n-1;
         bc_ilower[1] = pj*n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n-1;

         /* Modify the matrix */
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);

         /* Put the boundary conditions in b */
         for (j = 0; j < n; j++)
            bvalues[j] = bcEval(U0,0,j);

         HYPRE_SStructVectorSetBoxValues(b, part, bc_ilower, bc_iupper,
                                         var, bvalues);
      }

      /* Recall that the system we are solving is:
         [A_ii 0; 0 I] [x_i ; x_b] = [b_i - A_ib u_0; u_0].
         This requires removing the connections between the interior
         and boundary nodes that we have set up when we set the
         5pt stencil at each node. We adjust for removing
         these connections by appropriately modifying the rhs.
         For the symm ordering scheme, just do the top and right
         boundary */

      /* Processors at y = 0, neighbors of boundary nodes */
      if (pj == 0)
      {
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n + 1;

         bc_iupper[0] = bc_ilower[0] + n-1;
         bc_iupper[1] = bc_ilower[1];

         stencil_indices[0] = 3;

         /* Modify the matrix */
         for (i = 0; i < n; i++)
            bvalues[i] = 0.0;

         if (sym == 0)
            HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                            var, 1,
                                            stencil_indices, bvalues);

         /* Eliminate the boundary conditions in b */
         for (i = 0; i < n; i++)
            bvalues[i] = bcEval(U0,i,-1) * (bcEval(K,i,-0.5)+bcEval(B2,i,-0.5));

         if (pi == 0)
            bvalues[0] = 0.0;

         if (pi == N-1)
            bvalues[n-1] = 0.0;

         /* Note the use of AddToBoxValues (because we have already set values
            at these nodes) */
         HYPRE_SStructVectorAddToBoxValues(b, part, bc_ilower, bc_iupper,
                                           var, bvalues);
      }

      /* Processors at x = 0, neighbors of boundary nodes */
      if (pi == 0)
      {
         bc_ilower[0] = pi*n + 1;
         bc_ilower[1] = pj*n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n-1;

         stencil_indices[0] = 1;

         /* Modify the matrix */
         for (j = 0; j < n; j++)
            bvalues[j] = 0.0;

         if (sym == 0)
            HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                            var, 1,
                                            stencil_indices, bvalues);

         /* Eliminate the boundary conditions in b */
         for (j = 0; j < n; j++)
            bvalues[j] = bcEval(U0,-1,j) * (bcEval(K,-0.5,j)+bcEval(B1,-0.5,j));

         if (pj == 0)
            bvalues[0] = 0.0;

         if (pj == N-1)
            bvalues[n-1] = 0.0;

         HYPRE_SStructVectorAddToBoxValues(b, part, bc_ilower, bc_iupper, var, bvalues);
      }

      /* Processors at y = 1, neighbors of boundary nodes */
      if (pj == N-1)
      {
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n + (n-1) -1;

         bc_iupper[0] = bc_ilower[0] + n-1;
         bc_iupper[1] = bc_ilower[1];

         if (sym == 0)
            stencil_indices[0] = 4;
         else
            stencil_indices[0] = 2;

         /* Modify the matrix */
         for (i = 0; i < n; i++)
            bvalues[i] = 0.0;

         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, var, 1,
                                         stencil_indices, bvalues);

         /* Eliminate the boundary conditions in b */
         for (i = 0; i < n; i++)
            bvalues[i] = bcEval(U0,i,1) * (bcEval(K,i,0.5)+bcEval(B2,i,0.5));

         if (pi == 0)
            bvalues[0] = 0.0;

         if (pi == N-1)
            bvalues[n-1] = 0.0;

         HYPRE_SStructVectorAddToBoxValues(b, part, bc_ilower, bc_iupper,
                                           var, bvalues);
      }

      /* Processors at x = 1, neighbors of boundary nodes */
      if (pi == N-1)
      {
         bc_ilower[0] = pi*n + (n-1) - 1;
         bc_ilower[1] = pj*n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n-1;

         if (sym == 0)
            stencil_indices[0] = 2;
         else
            stencil_indices[0] = 1;

         /* Modify the matrix */
         for (j = 0; j < n; j++)
            bvalues[j] = 0.0;

         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, 1,
                                         stencil_indices, bvalues);

         /* Eliminate the boundary conditions in b */
         for (j = 0; j < n; j++)
            bvalues[j] = bcEval(U0,1,j) * (bcEval(K,0.5,j)+bcEval(B1,0.5,j));

         if (pj == 0)
            bvalues[0] = 0.0;

         if (pj == N-1)
            bvalues[n-1] = 0.0;

         HYPRE_SStructVectorAddToBoxValues(b, part, bc_ilower, bc_iupper, var, bvalues);
      }

      free(values);
      free(bvalues);
   }

   /* Finalize the vector and matrix assembly */
   HYPRE_SStructMatrixAssemble(A);
   HYPRE_SStructVectorAssemble(b);
   HYPRE_SStructVectorAssemble(x);

   /* 6. Set up and use a solver */
   {
      HYPRE_StructMatrix    sA;
      HYPRE_StructVector    sb;
      HYPRE_StructVector    sx;

      /* Because we are using a struct solver, we need to get the
         object of the matrix and vectors to pass in to the struct solvers */

      HYPRE_SStructMatrixGetObject(A, (void **) &sA);
      HYPRE_SStructVectorGetObject(b, (void **) &sb);
      HYPRE_SStructVectorGetObject(x, (void **) &sx);

      if (solver_id == 0) /* SMG */
      {
         /* Start timing */
         time_index = hypre_InitializeTiming("SMG Setup");
         hypre_BeginTiming(time_index);

         /* Options and setup */
         HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver);
         HYPRE_StructSMGSetMemoryUse(solver, 0);
         HYPRE_StructSMGSetMaxIter(solver, 50);
         HYPRE_StructSMGSetTol(solver, 1.0e-06);
         HYPRE_StructSMGSetRelChange(solver, 0);
         HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
         HYPRE_StructSMGSetNumPostRelax(solver, n_post);
         HYPRE_StructSMGSetPrintLevel(solver, 1);
         HYPRE_StructSMGSetLogging(solver, 1);
         HYPRE_StructSMGSetup(solver, sA, sb, sx);

         /* Finalize current timing */
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         /* Start timing again */
         time_index = hypre_InitializeTiming("SMG Solve");
         hypre_BeginTiming(time_index);

         /* Solve */
         HYPRE_StructSMGSolve(solver, sA, sb, sx);
         hypre_EndTiming(time_index);
         /* Finalize current timing */

         hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         /* Get info and release memory */
         HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
         HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         HYPRE_StructSMGDestroy(solver);
      }

      if (solver_id == 1) /* PFMG */
      {
         /* Start timing */
         time_index = hypre_InitializeTiming("PFMG Setup");
         hypre_BeginTiming(time_index);

         /* Options and setup */
         HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &solver);
         HYPRE_StructPFMGSetMaxIter(solver, 50);
         HYPRE_StructPFMGSetTol(solver, 1.0e-06);
         HYPRE_StructPFMGSetRelChange(solver, 0);
         HYPRE_StructPFMGSetRAPType(solver, rap);
         HYPRE_StructPFMGSetRelaxType(solver, relax);
         HYPRE_StructPFMGSetNumPreRelax(solver, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(solver, n_post);
         HYPRE_StructPFMGSetSkipRelax(solver, skip);
         HYPRE_StructPFMGSetPrintLevel(solver, 1);
         HYPRE_StructPFMGSetLogging(solver, 1);
         HYPRE_StructPFMGSetup(solver, sA, sb, sx);

         /* Finalize current timing */
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         /* Start timing again */
         time_index = hypre_InitializeTiming("PFMG Solve");
         hypre_BeginTiming(time_index);

         /* Solve */
         HYPRE_StructPFMGSolve(solver, sA, sb, sx);

         /* Finalize current timing */
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         /* Get info and release memory */
         HYPRE_StructPFMGGetNumIterations(solver, &num_iterations);
         HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         HYPRE_StructPFMGDestroy(solver);
      }

      /* Preconditioned CG */
      if ((solver_id > 9) && (solver_id < 20))
      {
         time_index = hypre_InitializeTiming("PCG Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
         HYPRE_StructPCGSetMaxIter(solver, 200 );
         HYPRE_StructPCGSetTol(solver, 1.0e-06 );
         HYPRE_StructPCGSetTwoNorm(solver, 1 );
         HYPRE_StructPCGSetRelChange(solver, 0 );
         HYPRE_StructPCGSetPrintLevel(solver, 2 );

         if (solver_id == 10)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, 0);
            HYPRE_StructSMGSetLogging(precond, 0);
            HYPRE_StructPCGSetPrecond(solver,
                                      HYPRE_StructSMGSolve,
                                      HYPRE_StructSMGSetup,
                                      precond);
         }

         else if (solver_id == 11)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            HYPRE_StructPFMGSetPrintLevel(precond, 0);
            HYPRE_StructPFMGSetLogging(precond, 0);
            HYPRE_StructPCGSetPrecond(solver,
                                      HYPRE_StructPFMGSolve,
                                      HYPRE_StructPFMGSetup,
                                      precond);
         }

         else if (solver_id == 17)
         {
            /* use two-step Jacobi as preconditioner */
            HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &precond);
            HYPRE_StructJacobiSetMaxIter(precond, 2);
            HYPRE_StructJacobiSetTol(precond, 0.0);
            HYPRE_StructJacobiSetZeroGuess(precond);
            HYPRE_StructPCGSetPrecond( solver,
                                       HYPRE_StructJacobiSolve,
                                       HYPRE_StructJacobiSetup,
                                       precond);
         }

         else if (solver_id == 18)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_StructPCGSetPrecond(solver,
                                      HYPRE_StructDiagScale,
                                      HYPRE_StructDiagScaleSetup,
                                      precond);
         }

         /* PCG Setup */
         HYPRE_StructPCGSetup(solver, sA, sb, sx );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("PCG Solve");
         hypre_BeginTiming(time_index);

         /* PCG Solve */
         HYPRE_StructPCGSolve(solver, sA, sb, sx);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         /* Get info and release memory */
         HYPRE_StructPCGGetNumIterations( solver, &num_iterations );
         HYPRE_StructPCGGetFinalRelativeResidualNorm( solver, &final_res_norm );
         HYPRE_StructPCGDestroy(solver);

         if (solver_id == 10)
         {
            HYPRE_StructSMGDestroy(precond);
         }
         else if (solver_id == 11 )
         {
            HYPRE_StructPFMGDestroy(precond);
         }
         else if (solver_id == 17)
         {
            HYPRE_StructJacobiDestroy(precond);
         }
      }

      /* Preconditioned GMRES */
      if ((solver_id > 29) && (solver_id < 40))
      {
         time_index = hypre_InitializeTiming("GMRES Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver);

         /* Note that GMRES can be used with all the interfaces - not
            just the struct.  So here we demonstrate the
            more generic GMRES interface functions. Since we have chosen
            a struct solver then we must type cast to the more generic
            HYPRE_Solver when setting options with these generic functions.
            Note that one could declare the solver to be
            type HYPRE_Solver, and then the casting would not be necessary.*/

         HYPRE_GMRESSetMaxIter((HYPRE_Solver) solver, 500 );
         HYPRE_GMRESSetKDim((HYPRE_Solver) solver,30);
         HYPRE_GMRESSetTol((HYPRE_Solver) solver, 1.0e-06 );
         HYPRE_GMRESSetPrintLevel((HYPRE_Solver) solver, 2 );
         HYPRE_GMRESSetLogging((HYPRE_Solver) solver, 1 );

         if (solver_id == 30)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, 0);
            HYPRE_StructSMGSetLogging(precond, 0);
            HYPRE_StructGMRESSetPrecond(solver,
                                        HYPRE_StructSMGSolve,
                                        HYPRE_StructSMGSetup,
                                        precond);
         }

         else if (solver_id == 31)
         {
            /* use symmetric PFMG as preconditioner */
            HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            HYPRE_StructPFMGSetPrintLevel(precond, 0);
            HYPRE_StructPFMGSetLogging(precond, 0);
            HYPRE_StructGMRESSetPrecond( solver,
                                         HYPRE_StructPFMGSolve,
                                         HYPRE_StructPFMGSetup,
                                         precond);
         }

         else if (solver_id == 37)
         {
            /* use two-step Jacobi as preconditioner */
            HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &precond);
            HYPRE_StructJacobiSetMaxIter(precond, 2);
            HYPRE_StructJacobiSetTol(precond, 0.0);
            HYPRE_StructJacobiSetZeroGuess(precond);
            HYPRE_StructGMRESSetPrecond( solver,
                                         HYPRE_StructJacobiSolve,
                                         HYPRE_StructJacobiSetup,
                                         precond);
         }

         else if (solver_id == 38)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            HYPRE_StructGMRESSetPrecond( solver,
                                         HYPRE_StructDiagScale,
                                         HYPRE_StructDiagScaleSetup,
                                         precond);
         }

         /* GMRES Setup */
         HYPRE_StructGMRESSetup(solver, sA, sb, sx );

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("GMRES Solve");
         hypre_BeginTiming(time_index);

         /* GMRES Solve */
         HYPRE_StructGMRESSolve(solver, sA, sb, sx);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         /* Get info and release memory */
         HYPRE_StructGMRESGetNumIterations(solver, &num_iterations);
         HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         HYPRE_StructGMRESDestroy(solver);

         if (solver_id == 30)
         {
            HYPRE_StructSMGDestroy(precond);
         }
         else if (solver_id == 31)
         {
            HYPRE_StructPFMGDestroy(precond);
         }
         else if (solver_id == 37)
         {
            HYPRE_StructJacobiDestroy(precond);
         }
      }

   }

   /* Save the solution for GLVis visualization, see vis/glvis-ex7.sh */
   if (vis)
   {
      FILE *file;
      char filename[255];

      int part = 0, var = 0;
      int nvalues = n*n;
      double *values = calloc(nvalues, sizeof(double));

      /* get all local data (including a local copy of the shared values) */
      HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                      var, values);

      sprintf(filename, "%s.%06d", "vis/ex7.sol", myid);
      if ((file = fopen(filename, "w")) == NULL)
      {
         printf("Error: can't open output file %s\n", filename);
         MPI_Finalize();
         exit(1);
      }

      /* save solution with global unknown numbers */
      k = 0;
      for (j = 0; j < n; j++)
         for (i = 0; i < n; i++)
            fprintf(file, "%06d %.14e\n", pj*N*n*n+pi*n+j*N*n+i, values[k++]);

      fflush(file);
      fclose(file);
      free(values);

      /* save global finite element mesh */
      if (myid == 0)
         GLVis_PrintGlobalSquareMesh("vis/ex7.mesh", N*n-1);
   }

   if (myid == 0)
   {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %e\n", final_res_norm);
      printf("\n");
   }

   /* Free memory */
   HYPRE_SStructGridDestroy(grid);
   HYPRE_SStructStencilDestroy(stencil);
   HYPRE_SStructGraphDestroy(graph);
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}

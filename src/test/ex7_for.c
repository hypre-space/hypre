/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/

/*
   Example 7 -- FORTRAN Test Version

   Interface:      SStructured interface (SStruct)

   Compile with:   make ex7

   Sample run:     mpirun -np 16 ex7_for -n 33 -solver 10 -K 3 -B 0 -C 1 -U0 2 -F 4

   To see options: ex7 -help

   Description:    This example uses the sstruct interface to solve the same
                   problem as was solved in Example 4 with the struct interface.
                   Therefore, there is only one part and one variable.

                   This code solves the convection-reaction-diffusion problem
                   div (-K grad u + B u) + C u = F in the unit square with
                   boundary condition u = U0.  The domain is split into N x N
                   processor grid.  Thus, the given number of processors should
                   be a perfect square. Each processor has a n x n grid, with
                   nodes connected by a 5-point stencil.  We use cell-centered
                   variables, and, therefore, the nodes are not shared.

                   To incorporate the boundary conditions, we do the following:
                   Let x_i and x_b be the interior and boundary parts of the
                   solution vector x. If we split the matrix A as
                             A = [A_ii A_ib; A_bi A_bb],
                   then we solve
                             [A_ii 0; 0 I] [x_i ; x_b] = [b_i - A_ib u_0; u_0].
                   Note that this differs from Example 3 in that we
                   are actually solving for the boundary conditions (so they
                   may not be exact as in ex3, where we only solved for the
                   interior).  This approach is useful for more general types
                   of b.c.

                   As in the previous example (Example 6), we use a structured
                   solver.  A number of structured solvers are available.
                   More information can be found in the Solvers and Preconditioners
                   chapter of the User's Manual.
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

/* Macro to evaluate a function F in the grid point (i,j) */
#define Eval(F,i,j) (F( (ilower[0]+(i))*h, (ilower[1]+(j))*h ))
#define bcEval(F,i,j) (F( (bc_ilower[0]+(i))*h, (bc_ilower[1]+(j))*h ))

#ifdef HYPRE_FORTRAN
#include "fortran.h"
#include "hypre_struct_fortran_test.h"
#include "hypre_sstruct_fortran_test.h"
#endif

HYPRE_Int optionK, optionB, optionC, optionU0, optionF;

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

HYPRE_Int main (HYPRE_Int argc, char *argv[])
{
   HYPRE_Int i, j, k;

   HYPRE_Int myid, num_procs;

   HYPRE_Int n, N, pi, pj;
   double h, h2;
   HYPRE_Int ilower[2], iupper[2];

   HYPRE_Int solver_id;
   HYPRE_Int n_pre, n_post;
   HYPRE_Int rap, relax, skip, sym;
   HYPRE_Int time_index;

   HYPRE_Int object_type;

   HYPRE_Int num_iterations;
   double final_res_norm;

   HYPRE_Int print_solution;

   /* We are using struct solvers for this example */
#ifdef HYPRE_FORTRAN
   hypre_F90_Obj grid;
   hypre_F90_Obj stencil;
   hypre_F90_Obj graph;
   hypre_F90_Obj A;
   hypre_F90_Obj b;
   hypre_F90_Obj x;

   hypre_F90_Obj solver;
   hypre_F90_Obj precond;
   HYPRE_Int      precond_id;

   hypre_F90_Obj long_temp_COMM;
   HYPRE_Int      temp_COMM;

   HYPRE_Int      zero = 0;
   HYPRE_Int      one = 1;
   HYPRE_Int      two = 2;
   HYPRE_Int      three = 3;
   HYPRE_Int      five = 5;
   HYPRE_Int      fifty = 50;
   HYPRE_Int      twohundred = 200;
   HYPRE_Int      fivehundred = 500;

   double   zerodot = 0.;
   double   tol = 1.e-6;
#else
   HYPRE_SStructGrid     grid;
   HYPRE_SStructStencil  stencil;
   HYPRE_SStructGraph    graph;
   HYPRE_SStructMatrix   A;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   x;

   HYPRE_StructSolver   solver;
   HYPRE_StructSolver   precond;
#endif

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   /* Set default parameters */
   n         = 33;
   optionK   = 0;
   optionB   = 0;
   optionC   = 0;
   optionU0  = 0;
   optionF   = 0;
   solver_id = 10;
   n_pre     = 1;
   n_post    = 1;
   rap       = 0;
   relax     = 1;
   skip      = 0;
   sym       = 0;

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
         hypre_printf("  -n  <n>             : problem size per processor (default: 8)\n");
         hypre_printf("  -K  <K>             : choice for the diffusion coefficient (default: 1.0)\n");
         hypre_printf("  -B  <B>             : choice for the convection vector (default: 0.0)\n");
         hypre_printf("  -C  <C>             : choice for the reaction coefficient (default: 0.0)\n");
         hypre_printf("  -U0 <U0>            : choice for the boundary condition (default: 0.0)\n");
         hypre_printf("  -F  <F>             : choice for the right-hand side (default: 1.0) \n");
         hypre_printf("  -solver <ID>        : solver ID\n");
         hypre_printf("                        0  - SMG \n");
         hypre_printf("                        1  - PFMG\n");
         hypre_printf("                        10 - CG with SMG precond (default)\n");
         hypre_printf("                        11 - CG with PFMG precond\n");
         hypre_printf("                        17 - CG with 2-step Jacobi\n");
         hypre_printf("                        18 - CG with diagonal scaling\n");
         hypre_printf("                        19 - CG\n");
         hypre_printf("                        30 - GMRES with SMG precond\n");
         hypre_printf("                        31 - GMRES with PFMG precond\n");
         hypre_printf("                        37 - GMRES with 2-step Jacobi\n");
         hypre_printf("                        38 - GMRES with diagonal scaling\n");
         hypre_printf("                        39 - GMRES\n");
         hypre_printf("  -v <n_pre> <n_post> : number of pre and post relaxations\n");
         hypre_printf("  -rap <r>            : coarse grid operator type\n");
         hypre_printf("                        0 - Galerkin (default)\n");
         hypre_printf("                        1 - non-Galerkin ParFlow operators\n");
         hypre_printf("                        2 - Galerkin, general operators\n");
         hypre_printf("  -relax <r>          : relaxation type\n");
         hypre_printf("                        0 - Jacobi\n");
         hypre_printf("                        1 - Weighted Jacobi (default)\n");
         hypre_printf("                        2 - R/B Gauss-Seidel\n");
         hypre_printf("                        3 - R/B Gauss-Seidel (nonsymmetric)\n");
         hypre_printf("  -skip <s>           : skip levels in PFMG (0 or 1)\n");
         hypre_printf("  -sym <s>            : symmetric storage (1) or not (0)\n");
         hypre_printf("  -print_solution     : print the solution vector\n");
         hypre_printf("\n");
      }

      if (print_usage)
      {
         hypre_MPI_Finalize();
         return (0);
      }
   }

   /* Convection produces non-symmetric matrices */
   if (optionB && sym)
      optionB = 0;

   /* Figure out the processor grid (N x N).  The local
      problem size is indicated by n (n x n). pi and pj
      indicate position in the processor grid. */
   N  = sqrt(num_procs);
   h  = 1.0 / (N*n-1);
   h2 = h*h;
   pj = myid / N;
   pi = myid - pj*N;

   /* Define the nodes owned by the current processor (each processor's
      piece of the global grid) */
   ilower[0] = pi*n;
   ilower[1] = pj*n;
   iupper[0] = ilower[0] + n-1;
   iupper[1] = ilower[1] + n-1;

   /* 1. Set up a 2D grid */
   {
      HYPRE_Int ndim = 2;
      HYPRE_Int nparts = 1;
      HYPRE_Int nvars = 1;
      HYPRE_Int part = 0;
      HYPRE_Int i;

      /* Create an empty 2D grid object */
#ifdef HYPRE_FORTRAN
      temp_COMM = (HYPRE_Int) hypre_MPI_COMM_WORLD;
      long_temp_COMM = (hypre_F90_Obj) hypre_MPI_COMM_WORLD;
      HYPRE_SStructGridCreate(&temp_COMM, &ndim, &nparts, &grid);
#else
      HYPRE_SStructGridCreate(hypre_MPI_COMM_WORLD, ndim, nparts, &grid);
#endif

      /* Add a new box to the grid */
#ifdef HYPRE_FORTRAN
      HYPRE_SStructGridSetExtents(&grid, &part, &ilower[0], &iupper[0]);
#else
      HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
#endif

      /* Set the variable type for each part */
      {
#ifdef HYPRE_FORTRAN
         hypre_F90_Obj vartypes[1] = {HYPRE_SSTRUCT_VARIABLE_CELL};
#else
         HYPRE_SStructVariable vartypes[1] = {HYPRE_SSTRUCT_VARIABLE_CELL};
#endif

         for (i = 0; i< nparts; i++)
#ifdef HYPRE_FORTRAN
            HYPRE_SStructGridSetVariables(&grid, &i, &nvars, &vartypes[0]);
#else
            HYPRE_SStructGridSetVariables(grid, i, nvars, vartypes);
#endif
      }

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
#ifdef HYPRE_FORTRAN
      HYPRE_SStructGridAssemble(&grid);
#else
      HYPRE_SStructGridAssemble(grid);
#endif
   }

   /* 2. Define the discretization stencil */
   {
      HYPRE_Int ndim = 2;
      HYPRE_Int var = 0;

      if (sym == 0)
      {
         /* Define the geometry of the stencil */
         HYPRE_Int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};

         /* Create an empty 2D, 5-pt stencil object */
#ifdef HYPRE_FORTRAN
         HYPRE_SStructStencilCreate(&ndim, &five, &stencil);
#else
         HYPRE_SStructStencilCreate(ndim, 5, &stencil);
#endif

         /* Assign stencil entries */
         for (i = 0; i < 5; i++)
#ifdef HYPRE_FORTRAN
            HYPRE_SStructStencilSetEntry(&stencil, &i, offsets[i], &var);
#else
            HYPRE_SStructStencilSetEntry(stencil, i, offsets[i], var);
#endif
      }
      else /* Symmetric storage */
      {
         /* Define the geometry of the stencil */
         HYPRE_Int offsets[3][2] = {{0,0}, {1,0}, {0,1}};

         /* Create an empty 2D, 3-pt stencil object */
#ifdef HYPRE_FORTRAN
         HYPRE_SStructStencilCreate(&ndim, &three, &stencil);
#else
         HYPRE_SStructStencilCreate(ndim, 3, &stencil);
#endif

         /* Assign stencil entries */
         for (i = 0; i < 3; i++)
#ifdef HYPRE_FORTRAN
            HYPRE_SStructStencilSetEntry(&stencil, &i, offsets[i], &var);
#else
            HYPRE_SStructStencilSetEntry(stencil, i, offsets[i], var);
#endif
      }
   }

   /* 3. Set up the Graph  - this determines the non-zero structure
      of the matrix */
   {
      HYPRE_Int var = 0;
      HYPRE_Int part = 0;

      /* Create the graph object */
#ifdef HYPRE_FORTRAN
      HYPRE_SStructGraphCreate(&temp_COMM, &grid, &graph);
#else
      HYPRE_SStructGraphCreate(hypre_MPI_COMM_WORLD, grid, &graph);
#endif

      /* Now we need to tell the graph which stencil to use for each
         variable on each part (we only have one variable and one part)*/
#ifdef HYPRE_FORTRAN
      HYPRE_SStructGraphSetStencil(&graph, &part, &var, &stencil);
#else
      HYPRE_SStructGraphSetStencil(graph, part, var, stencil);
#endif

      /* Here we could establish connections between parts if we
         had more than one part. */

      /* Assemble the graph */
#ifdef HYPRE_FORTRAN
      HYPRE_SStructGraphAssemble(&graph);
#else
      HYPRE_SStructGraphAssemble(graph);
#endif
   }

   /* 4. Set up SStruct Vectors for b and x */
   {
      double *values;

      /* We have one part and one variable. */
      HYPRE_Int part = 0;
      HYPRE_Int var = 0;

      /* Create an empty vector object */
#ifdef HYPRE_FORTRAN
      HYPRE_SStructVectorCreate(&temp_COMM, &grid, &b);
      HYPRE_SStructVectorCreate(&temp_COMM, &grid, &x);
#else
      HYPRE_SStructVectorCreate(hypre_MPI_COMM_WORLD, grid, &b);
      HYPRE_SStructVectorCreate(hypre_MPI_COMM_WORLD, grid, &x);
#endif

      /* Set the object type (by default HYPRE_SSTRUCT). This determines the
         data structure used to store the matrix.  If you want to use unstructured
         solvers, e.g. BoomerAMG, the object type should be HYPRE_PARCSR.
         If the problem is purely structured (with one part), you may want to use
         HYPRE_STRUCT to access the structured solvers. Here we have a purely
         structured example. */
      object_type = HYPRE_STRUCT;
#ifdef HYPRE_FORTRAN
      HYPRE_SStructVectorSetObjectType(&b, &object_type);
      HYPRE_SStructVectorSetObjectType(&x, &object_type);
#else
      HYPRE_SStructVectorSetObjectType(b, object_type);
      HYPRE_SStructVectorSetObjectType(x, object_type);
#endif

      /* Indicate that the vector coefficients are ready to be set */
#ifdef HYPRE_FORTRAN
      HYPRE_SStructVectorInitialize(&b);
      HYPRE_SStructVectorInitialize(&x);
#else
      HYPRE_SStructVectorInitialize(b);
      HYPRE_SStructVectorInitialize(x);
#endif

      values = calloc((n*n), sizeof(double));

      /* Set the values of b in left-to-right, bottom-to-top order */
      for (k = 0, j = 0; j < n; j++)
         for (i = 0; i < n; i++, k++)
            values[k] = h2 * Eval(F,i,j);
#ifdef HYPRE_FORTRAN
      HYPRE_SStructVectorSetBoxValues(&b, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
      HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);
#endif

      /* Set x = 0 */
      for (i = 0; i < (n*n); i ++)
         values[i] = 0.0;
#ifdef HYPRE_FORTRAN
      HYPRE_SStructVectorSetBoxValues(&x, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
      HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
#endif

      free(values);

      /* Assembling is postponed since the vectors will be further modified */
   }

   /* 4. Set up a SStruct Matrix */
   {
      /* We have one part and one variable. */
      HYPRE_Int part = 0;
      HYPRE_Int var = 0;

      /* Create an empty matrix object */
#ifdef HYPRE_FORTRAN
      HYPRE_SStructMatrixCreate(&temp_COMM, &graph, &A);
#else
      HYPRE_SStructMatrixCreate(hypre_MPI_COMM_WORLD, graph, &A);
#endif

      /* Use symmetric storage? The function below is for symmetric stencil entries
         (use HYPRE_SStructMatrixSetNSSymmetric for non-stencil entries) */
#ifdef HYPRE_FORTRAN
      HYPRE_SStructMatrixSetSymmetric(&A, &part, &var, &var, &sym);
#else
      HYPRE_SStructMatrixSetSymmetric(A, part, var, var, sym);
#endif

      /* As with the vectors,  set the object type for the vectors
         to be the struct type */
      object_type = HYPRE_STRUCT;
#ifdef HYPRE_FORTRAN
      HYPRE_SStructMatrixSetObjectType(&A, &object_type);
#else
      HYPRE_SStructMatrixSetObjectType(A, object_type);
#endif

      /* Indicate that the matrix coefficients are ready to be set */
#ifdef HYPRE_FORTRAN
      HYPRE_SStructMatrixInitialize(&A);
#else
      HYPRE_SStructMatrixInitialize(A);
#endif

      /* Set the stencil values in the interior. Here we set the values
         at every node. We will modify the boundary nodes later. */
      if (sym == 0)
      {
         HYPRE_Int stencil_indices[5] = {0, 1, 2, 3, 4}; /* labels correspond
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

#ifdef HYPRE_FORTRAN
         HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                         &var, &five,
                                         &stencil_indices[0], &values[0]);
#else
         HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                         var, 5,
                                         stencil_indices, values);
#endif

         free(values);
      }
      else /* Symmetric storage */
      {
         HYPRE_Int stencil_indices[3] = {0, 1, 2};
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

#ifdef HYPRE_FORTRAN
         HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                         &var, &three,
                                         &stencil_indices[0], &values[0]);
#else
         HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                         var, 3,
                                         stencil_indices, values);
#endif

         free(values);
      }
   }

   /* 5. Set the boundary conditions, while eliminating the coefficients
         reaching ouside of the domain boundary. We must modify the matrix
         stencil and the corresponding rhs entries. */
   {
      HYPRE_Int bc_ilower[2];
      HYPRE_Int bc_iupper[2];

      HYPRE_Int stencil_indices[5] = {0, 1, 2, 3, 4};
      double *values, *bvalues;

      HYPRE_Int nentries;

      /* We have one part and one variable. */
      HYPRE_Int part = 0;
      HYPRE_Int var = 0;

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
#ifdef HYPRE_FORTRAN
         HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                         &var, &nentries,
                                         &stencil_indices[0], &values[0]);
#else
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);
#endif

         /* Put the boundary conditions in b */
         for (i = 0; i < n; i++)
            bvalues[i] = bcEval(U0,i,0);

#ifdef HYPRE_FORTRAN
         HYPRE_SStructVectorSetBoxValues(&b, &part, &bc_ilower[0],
                                         &bc_iupper[0], &var, &bvalues[0]);
#else
         HYPRE_SStructVectorSetBoxValues(b, part, bc_ilower,
                                         bc_iupper, var, bvalues);
#endif
      }

      /* Processors at y = 1 */
      if (pj == N-1)
      {
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n + n-1;

         bc_iupper[0] = bc_ilower[0] + n-1;
         bc_iupper[1] = bc_ilower[1];

         /* Modify the matrix */
#ifdef HYPRE_FORTRAN
         HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                         &var, &nentries,
                                         &stencil_indices[0], &values[0]);
#else
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);
#endif

         /* Put the boundary conditions in b */
         for (i = 0; i < n; i++)
            bvalues[i] = bcEval(U0,i,0);

#ifdef HYPRE_FORTRAN
         HYPRE_SStructVectorSetBoxValues(&b, &part, &bc_ilower[0], 
                                         &bc_iupper[0], &var, &bvalues[0]);
#else
         HYPRE_SStructVectorSetBoxValues(b, part, bc_ilower, bc_iupper, var, bvalues);
#endif
      }

      /* Processors at x = 0 */
      if (pi == 0)
      {
         bc_ilower[0] = pi*n;
         bc_ilower[1] = pj*n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n-1;

         /* Modify the matrix */
#ifdef HYPRE_FORTRAN
         HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                         &var, &nentries,
                                         &stencil_indices[0], &values[0]);
#else
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);
#endif

         /* Put the boundary conditions in b */
         for (j = 0; j < n; j++)
            bvalues[j] = bcEval(U0,0,j);

#ifdef HYPRE_FORTRAN
         HYPRE_SStructVectorSetBoxValues(&b, &part, &bc_ilower[0], 
                                         &bc_iupper[0], &var, &bvalues[0]);
#else
         HYPRE_SStructVectorSetBoxValues(b, part, bc_ilower, bc_iupper,
                                         var, bvalues);
#endif
      }

      /* Processors at x = 1 */
      if (pi == N-1)
      {
         bc_ilower[0] = pi*n + n-1;
         bc_ilower[1] = pj*n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n-1;

         /* Modify the matrix */
#ifdef HYPRE_FORTRAN
         HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                         &var, &nentries,
                                         &stencil_indices[0], &values[0]);
#else
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, nentries,
                                         stencil_indices, values);
#endif

         /* Put the boundary conditions in b */
         for (j = 0; j < n; j++)
            bvalues[j] = bcEval(U0,0,j);

#ifdef HYPRE_FORTRAN
         HYPRE_SStructVectorSetBoxValues(&b, &part, &bc_ilower[0], &bc_iupper[0],
                                         &var, &bvalues[0]);
#else
         HYPRE_SStructVectorSetBoxValues(b, part, bc_ilower, bc_iupper,
                                         var, bvalues);
#endif
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
#ifdef HYPRE_FORTRAN
            HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &bvalues[0]);
#else
            HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                            var, 1,
                                            stencil_indices, bvalues);
#endif

         /* Eliminate the boundary conditions in b */
         for (i = 0; i < n; i++)
            bvalues[i] = bcEval(U0,i,-1) * (bcEval(K,i,-0.5)+bcEval(B2,i,-0.5));

         if (pi == 0)
            bvalues[0] = 0.0;

         if (pi == N-1)
            bvalues[n-1] = 0.0;

         /* Note the use of AddToBoxValues (because we have already set values
            at these nodes) */
#ifdef HYPRE_FORTRAN
         HYPRE_SStructVectorAddToBoxValues(&b, &part, &bc_ilower[0], &bc_iupper[0],
                                           &var, &bvalues[0]);
#else
         HYPRE_SStructVectorAddToBoxValues(b, part, bc_ilower, bc_iupper,
                                           var, bvalues);
#endif
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
#ifdef HYPRE_FORTRAN
            HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &bvalues[0]);
#else
            HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                            var, 1,
                                            stencil_indices, bvalues);
#endif

         /* Eliminate the boundary conditions in b */
         for (j = 0; j < n; j++)
            bvalues[j] = bcEval(U0,-1,j) * (bcEval(K,-0.5,j)+bcEval(B1,-0.5,j));

         if (pj == 0)
            bvalues[0] = 0.0;

         if (pj == N-1)
            bvalues[n-1] = 0.0;

#ifdef HYPRE_FORTRAN
         HYPRE_SStructVectorAddToBoxValues(&b, &part, &bc_ilower[0], &bc_iupper[0],
                                           &var, &bvalues[0]);
#else
         HYPRE_SStructVectorAddToBoxValues(b, part, bc_ilower, bc_iupper, var, bvalues);
#endif
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

#ifdef HYPRE_FORTRAN
         HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0], 
                                         &var, &one,
                                         &stencil_indices[0], &bvalues[0]);
#else
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper, var, 1,
                                         stencil_indices, bvalues);
#endif

         /* Eliminate the boundary conditions in b */
         for (i = 0; i < n; i++)
            bvalues[i] = bcEval(U0,i,1) * (bcEval(K,i,0.5)+bcEval(B2,i,0.5));

         if (pi == 0)
            bvalues[0] = 0.0;

         if (pi == N-1)
            bvalues[n-1] = 0.0;

#ifdef HYPRE_FORTRAN
         HYPRE_SStructVectorAddToBoxValues(&b, &part, &bc_ilower[0], &bc_iupper[0],
                                           &var, &bvalues[0]);
#else
         HYPRE_SStructVectorAddToBoxValues(b, part, bc_ilower, bc_iupper,
                                           var, bvalues);
#endif
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

#ifdef HYPRE_FORTRAN
         HYPRE_SStructMatrixSetBoxValues(&A, &part, &bc_ilower[0], &bc_iupper[0],
                                         &var, &one,
                                         &stencil_indices[0], &bvalues[0]);
#else
         HYPRE_SStructMatrixSetBoxValues(A, part, bc_ilower, bc_iupper,
                                         var, 1,
                                         stencil_indices, bvalues);
#endif

         /* Eliminate the boundary conditions in b */
         for (j = 0; j < n; j++)
            bvalues[j] = bcEval(U0,1,j) * (bcEval(K,0.5,j)+bcEval(B1,0.5,j));

         if (pj == 0)
            bvalues[0] = 0.0;

         if (pj == N-1)
            bvalues[n-1] = 0.0;

#ifdef HYPRE_FORTRAN
         HYPRE_SStructVectorAddToBoxValues(&b, &part, &bc_ilower[0], &bc_iupper[0],
                                           &var, &bvalues[0]);
#else
         HYPRE_SStructVectorAddToBoxValues(b, part, bc_ilower, bc_iupper, var, bvalues);
#endif
      }

      free(values);
      free(bvalues);
   }

   /* Finalize the vector and matrix assembly */
#ifdef HYPRE_FORTRAN
   HYPRE_SStructMatrixAssemble(&A);
   HYPRE_SStructVectorAssemble(&b);
   HYPRE_SStructVectorAssemble(&x);
#else
   HYPRE_SStructMatrixAssemble(A);
   HYPRE_SStructVectorAssemble(b);
   HYPRE_SStructVectorAssemble(x);
#endif

   /* 6. Set up and use a solver */
   {
#ifdef HYPRE_FORTRAN
      hypre_F90_Obj sA;
      hypre_F90_Obj sb;
      hypre_F90_Obj sx;
#else
      HYPRE_StructMatrix    sA;
      HYPRE_StructVector    sb;
      HYPRE_StructVector    sx;
#endif

      /* Because we are using a struct solver, we need to get the
         object of the matrix and vectors to pass in to the struct solvers */

#ifdef HYPRE_FORTRAN
      HYPRE_SStructMatrixGetObject(&A, &sA);
      HYPRE_SStructVectorGetObject(&b, &sb);
      HYPRE_SStructVectorGetObject(&x, &sx);
#else
      HYPRE_SStructMatrixGetObject(A, (void **) &sA);
      HYPRE_SStructVectorGetObject(b, (void **) &sb);
      HYPRE_SStructVectorGetObject(x, (void **) &sx);
#endif

      if (solver_id == 0) /* SMG */
      {
         /* Start timing */
         time_index = hypre_InitializeTiming("SMG Setup");
         hypre_BeginTiming(time_index);

         /* Options and setup */
#ifdef HYPRE_FORTRAN
         HYPRE_StructSMGCreate(&temp_COMM, &solver);
         HYPRE_StructSMGSetMemoryUse(&solver, &zero);
         HYPRE_StructSMGSetMaxIter(&solver, &fifty);
         HYPRE_StructSMGSetTol(&solver, &tol);
         HYPRE_StructSMGSetRelChange(&solver, &zero);
         HYPRE_StructSMGSetNumPreRelax(&solver, &n_pre);
         HYPRE_StructSMGSetNumPostRelax(&solver, &n_post);
         HYPRE_StructSMGSetPrintLevel(&solver, &one);
         HYPRE_StructSMGSetLogging(&solver, &one);
         HYPRE_StructSMGSetup(&solver, &sA, &sb, &sx);
#else
         HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &solver);
         HYPRE_StructSMGSetMemoryUse(solver, 0);
         HYPRE_StructSMGSetMaxIter(solver, 50);
         HYPRE_StructSMGSetTol(solver, 1.0e-06);
         HYPRE_StructSMGSetRelChange(solver, 0);
         HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
         HYPRE_StructSMGSetNumPostRelax(solver, n_post);
         HYPRE_StructSMGSetPrintLevel(solver, 1);
         HYPRE_StructSMGSetLogging(solver, 1);
         HYPRE_StructSMGSetup(solver, sA, sb, sx);
#endif

         /* Finalize current timing */
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         /* Start timing again */
         time_index = hypre_InitializeTiming("SMG Solve");
         hypre_BeginTiming(time_index);

         /* Solve */
#ifdef HYPRE_FORTRAN
         HYPRE_StructSMGSolve(&solver, &sA, &sb, &sx);
#else
         HYPRE_StructSMGSolve(solver, sA, sb, sx);
#endif
         hypre_EndTiming(time_index);
         /* Finalize current timing */

         hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         /* Get info and release memory */
#ifdef HYPRE_FORTRAN
         HYPRE_StructSMGGetNumIterations(&solver, &num_iterations);
         HYPRE_StructSMGGetFinalRelativeResidualNorm(&solver, &final_res_norm);
         HYPRE_StructSMGDestroy(&solver);
#else
         HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
         HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         HYPRE_StructSMGDestroy(solver);
#endif
      }

      if (solver_id == 1) /* PFMG */
      {
         /* Start timing */
         time_index = hypre_InitializeTiming("PFMG Setup");
         hypre_BeginTiming(time_index);

         /* Options and setup */
#ifdef HYPRE_FORTRAN
         HYPRE_StructPFMGCreate(&temp_COMM, &solver);
         HYPRE_StructPFMGSetMaxIter(&solver, &fifty);
         HYPRE_StructPFMGSetTol(&solver, &tol);
         HYPRE_StructPFMGSetRelChange(&solver, &zero);
         HYPRE_StructPFMGSetRAPType(&solver, &rap);
         HYPRE_StructPFMGSetRelaxType(&solver, &relax);
         HYPRE_StructPFMGSetNumPreRelax(&solver, &n_pre);
         HYPRE_StructPFMGSetNumPostRelax(&solver, &n_post);
         HYPRE_StructPFMGSetSkipRelax(&solver, &skip);
         HYPRE_StructPFMGSetPrintLevel(&solver, &one);
         HYPRE_StructPFMGSetLogging(&solver, &one);
         HYPRE_StructPFMGSetup(&solver, &sA, &sb, &sx);
#else
         HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &solver);
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
#endif

         /* Finalize current timing */
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         /* Start timing again */
         time_index = hypre_InitializeTiming("PFMG Solve");
         hypre_BeginTiming(time_index);

         /* Solve */
#ifdef HYPRE_FORTRAN
         HYPRE_StructPFMGSolve(&solver, &sA, &sb, &sx);
#else
         HYPRE_StructPFMGSolve(solver, sA, sb, sx);
#endif

         /* Finalize current timing */
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         /* Get info and release memory */
#ifdef HYPRE_FORTRAN
         HYPRE_StructPFMGGetNumIterations(&solver, &num_iterations);
         HYPRE_StructPFMGGetFinalRelativeResidualNorm(&solver, &final_res_norm);
         HYPRE_StructPFMGDestroy(&solver);
#else
         HYPRE_StructPFMGGetNumIterations(solver, &num_iterations);
         HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         HYPRE_StructPFMGDestroy(solver);
#endif
      }

      /* Preconditioned CG */
      if ((solver_id > 9) && (solver_id < 20))
      {
         time_index = hypre_InitializeTiming("PCG Setup");
         hypre_BeginTiming(time_index);

#ifdef HYPRE_FORTRAN
         HYPRE_StructPCGCreate(&temp_COMM, &solver);
         HYPRE_StructPCGSetMaxIter(&solver, &twohundred );
         HYPRE_StructPCGSetTol(&solver, &tol );
         HYPRE_StructPCGSetTwoNorm(&solver, &one );
         HYPRE_StructPCGSetRelChange(&solver, &zero );
         HYPRE_StructPCGSetPrintLevel(&solver, &two );
#else
         HYPRE_StructPCGCreate(hypre_MPI_COMM_WORLD, &solver);
         HYPRE_StructPCGSetMaxIter(solver, 200 );
         HYPRE_StructPCGSetTol(solver, 1.0e-06 );
         HYPRE_StructPCGSetTwoNorm(solver, 1 );
         HYPRE_StructPCGSetRelChange(solver, 0 );
         HYPRE_StructPCGSetPrintLevel(solver, 2 );
#endif

         if (solver_id == 10)
         {
            /* use symmetric SMG as preconditioner */
#ifdef HYPRE_FORTRAN
            HYPRE_StructSMGCreate(&temp_COMM, &precond);
            HYPRE_StructSMGSetMemoryUse(&precond, &zero);
            HYPRE_StructSMGSetMaxIter(&precond, &one);
            HYPRE_StructSMGSetTol(&precond, &zerodot);
            HYPRE_StructSMGSetZeroGuess(&precond);
            HYPRE_StructSMGSetNumPreRelax(&precond, &n_pre);
            HYPRE_StructSMGSetNumPostRelax(&precond, &n_post);
            HYPRE_StructSMGSetPrintLevel(&precond, &zero);
            HYPRE_StructSMGSetLogging(&precond, &zero);
            precond_id = 0;
            HYPRE_StructPCGSetPrecond(&solver, &precond_id, &precond);
#else
            HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &precond);
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
#endif
         }

         else if (solver_id == 11)
         {
            /* use symmetric PFMG as preconditioner */
#ifdef HYPRE_FORTRAN
            HYPRE_StructPFMGCreate(&temp_COMM, &precond);
            HYPRE_StructPFMGSetMaxIter(&precond, &one);
            HYPRE_StructPFMGSetTol(&precond, &zerodot);
            HYPRE_StructPFMGSetZeroGuess(&precond);
            HYPRE_StructPFMGSetRAPType(&precond, &rap);
            HYPRE_StructPFMGSetRelaxType(&precond, &relax);
            HYPRE_StructPFMGSetNumPreRelax(&precond, &n_pre);
            HYPRE_StructPFMGSetNumPostRelax(&precond, &n_post);
            HYPRE_StructPFMGSetSkipRelax(&precond, &skip);
            HYPRE_StructPFMGSetPrintLevel(&precond, &zero);
            HYPRE_StructPFMGSetLogging(&precond, &zero);
            precond_id = 1;
            HYPRE_StructPCGSetPrecond(&solver, &precond_id, &precond);
#else
            HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &precond);
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
#endif
         }

         else if (solver_id == 17)
         {
            /* use two-step Jacobi as preconditioner */
#ifdef HYPRE_FORTRAN
            HYPRE_StructJacobiCreate(&temp_COMM, &precond);
            HYPRE_StructJacobiSetMaxIter(&precond, &two);
            HYPRE_StructJacobiSetTol(&precond, &zerodot);
            HYPRE_StructJacobiSetZeroGuess(&precond);
            precond_id = 7;
            HYPRE_StructPCGSetPrecond(&solver, &precond_id, &precond);
#else
            HYPRE_StructJacobiCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructJacobiSetMaxIter(precond, 2);
            HYPRE_StructJacobiSetTol(precond, 0.0);
            HYPRE_StructJacobiSetZeroGuess(precond);
            HYPRE_StructPCGSetPrecond( solver,
                                       HYPRE_StructJacobiSolve,
                                       HYPRE_StructJacobiSetup,
                                       precond);
#endif
         }

         else if (solver_id == 18)
         {
            /* use diagonal scaling as preconditioner */
#ifdef HYPRE_FORTRAN
            precond_id = 8;
            HYPRE_StructPCGSetPrecond(&solver, &precond_id, &precond);
#else
            precond = NULL;
            HYPRE_StructPCGSetPrecond(solver,
                                      HYPRE_StructDiagScale,
                                      HYPRE_StructDiagScaleSetup,
                                      precond);
#endif
         }

         /* PCG Setup */
#ifdef HYPRE_FORTRAN
         HYPRE_StructPCGSetup(&solver, &sA, &sb, &sx );
#else
         HYPRE_StructPCGSetup(solver, sA, sb, sx );
#endif

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("PCG Solve");
         hypre_BeginTiming(time_index);

         /* PCG Solve */
#ifdef HYPRE_FORTRAN
         HYPRE_StructPCGSolve(&solver, &sA, &sb, &sx );
#else
         HYPRE_StructPCGSolve(solver, sA, sb, sx);
#endif

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         /* Get info and release memory */
#ifdef HYPRE_FORTRAN
         HYPRE_StructPCGGetNumIterations(&solver, &num_iterations );
         HYPRE_StructPCGGetFinalRelativeResidualNorm(&solver, &final_res_norm );
         HYPRE_StructPCGDestroy(&solver);
#else
         HYPRE_StructPCGGetNumIterations( solver, &num_iterations );
         HYPRE_StructPCGGetFinalRelativeResidualNorm( solver, &final_res_norm );
         HYPRE_StructPCGDestroy(solver);
#endif

         if (solver_id == 10)
         {
#ifdef HYPRE_FORTRAN
            HYPRE_StructSMGDestroy(&precond);
#else
            HYPRE_StructSMGDestroy(precond);
#endif
         }
         else if (solver_id == 11 )
         {
#ifdef HYPRE_FORTRAN
            HYPRE_StructPFMGDestroy(&precond);
#else
            HYPRE_StructPFMGDestroy(precond);
#endif
         }
         else if (solver_id == 17)
         {
#ifdef HYPRE_FORTRAN
            HYPRE_StructJacobiDestroy(&precond);
#else
            HYPRE_StructJacobiDestroy(precond);
#endif
         }
      }

      /* Preconditioned GMRES */
      if ((solver_id > 29) && (solver_id < 40))
      {
         time_index = hypre_InitializeTiming("GMRES Setup");
         hypre_BeginTiming(time_index);

#ifdef HYPRE_FORTRAN
         HYPRE_StructGMRESCreate(&temp_COMM, &solver);
#else
         HYPRE_StructGMRESCreate(hypre_MPI_COMM_WORLD, &solver);
#endif

         /* Note that GMRES can be used with all the interfaces - not
            just the struct.  So here we demonstrate the
            more generic GMRES interface functions. Since we have chosen
            a struct solver then we must type cast to the more generic
            HYPRE_Solver when setting options with these generic functions.
            Note that one could declare the solver to be
            type HYPRE_Solver, and then the casting would not be necessary.*/

         /*  Using struct GMRES routines to test FORTRAN Interface --3/3/2006  */

#ifdef HYPRE_FORTRAN
         HYPRE_StructGMRESSetMaxIter(&solver, &fivehundred );
         HYPRE_StructGMRESSetTol(&solver, &tol );
         HYPRE_StructGMRESSetPrintLevel(&solver, &two );
         HYPRE_StructGMRESSetLogging(&solver, &one );
#else
         HYPRE_StructGMRESSetMaxIter(solver, 500 );
         HYPRE_StructGMRESSetTol(solver, 1.0e-6 );
         HYPRE_StructGMRESSetPrintLevel(solver, 2 );
         HYPRE_StructGMRESSetLogging(solver, 1 );
#endif

         if (solver_id == 30)
         {
            /* use symmetric SMG as preconditioner */
#ifdef HYPRE_FORTRAN
            HYPRE_StructSMGCreate(&temp_COMM, &precond);
            HYPRE_StructSMGSetMemoryUse(&precond, &zero);
            HYPRE_StructSMGSetMaxIter(&precond, &one);
            HYPRE_StructSMGSetTol(&precond, &zerodot);
            HYPRE_StructSMGSetZeroGuess(&precond);
            HYPRE_StructSMGSetNumPreRelax(&precond, &n_pre);
            HYPRE_StructSMGSetNumPostRelax(&precond, &n_post);
            HYPRE_StructSMGSetPrintLevel(&precond, &zero);
            HYPRE_StructSMGSetLogging(&precond, &zero);
            precond_id = 0;
            HYPRE_StructGMRESSetPrecond(&solver, &precond_id, &precond);
#else
            HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &precond);
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
#endif
         }

         else if (solver_id == 31)
         {
            /* use symmetric PFMG as preconditioner */
#ifdef HYPRE_FORTRAN
            HYPRE_StructPFMGCreate(&temp_COMM, &precond);
            HYPRE_StructPFMGSetMaxIter(&precond, &one);
            HYPRE_StructPFMGSetTol(&precond, &zerodot);
            HYPRE_StructPFMGSetZeroGuess(&precond);
            HYPRE_StructPFMGSetRAPType(&precond, &rap);
            HYPRE_StructPFMGSetRelaxType(&precond, &relax);
            HYPRE_StructPFMGSetNumPreRelax(&precond, &n_pre);
            HYPRE_StructPFMGSetNumPostRelax(&precond, &n_post);
            HYPRE_StructPFMGSetSkipRelax(&precond, &skip);
            HYPRE_StructPFMGSetPrintLevel(&precond, &zero);
            HYPRE_StructPFMGSetLogging(&precond, &zero);
            precond_id = 1;
            HYPRE_StructGMRESSetPrecond(&solver, &precond_id, &precond);
#else
            HYPRE_StructPFMGCreate(hypre_MPI_COMM_WORLD, &precond);
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
#endif
         }

         else if (solver_id == 37)
         {
            /* use two-step Jacobi as preconditioner */
#ifdef HYPRE_FORTRAN
            HYPRE_StructJacobiCreate(&temp_COMM, &precond);
            HYPRE_StructJacobiSetMaxIter(&precond, &two);
            HYPRE_StructJacobiSetTol(&precond, &zerodot);
            HYPRE_StructJacobiSetZeroGuess(&precond);
            precond_id = 7;
            HYPRE_StructGMRESSetPrecond(&solver, &precond_id, &precond);
#else
            HYPRE_StructJacobiCreate(hypre_MPI_COMM_WORLD, &precond);
            HYPRE_StructJacobiSetMaxIter(precond, 2);
            HYPRE_StructJacobiSetTol(precond, 0.0);
            HYPRE_StructJacobiSetZeroGuess(precond);
            HYPRE_StructGMRESSetPrecond( solver,
                                         HYPRE_StructJacobiSolve,
                                         HYPRE_StructJacobiSetup,
                                         precond);
#endif
         }

         else if (solver_id == 38)
         {
            /* use diagonal scaling as preconditioner */
#ifdef HYPRE_FORTRAN
            precond_id = 8;
            HYPRE_StructGMRESSetPrecond(&solver, &precond_id, &precond);
#else
            precond = NULL;
            HYPRE_StructGMRESSetPrecond( solver,
                                         HYPRE_StructDiagScale,
                                         HYPRE_StructDiagScaleSetup,
                                         precond);
#endif
         }

         /* GMRES Setup */
#ifdef HYPRE_FORTRAN
         HYPRE_StructGMRESSetup(&solver, &sA, &sb, &sx );
#else
         HYPRE_StructGMRESSetup(solver, sA, sb, sx );
#endif

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("GMRES Solve");
         hypre_BeginTiming(time_index);

         /* GMRES Solve */
#ifdef HYPRE_FORTRAN
         HYPRE_StructGMRESSolve(&solver, &sA, &sb, &sx );
#else
         HYPRE_StructGMRESSolve(solver, sA, sb, sx);
#endif

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         /* Get info and release memory */
#ifdef HYPRE_FORTRAN
         HYPRE_StructGMRESGetNumIterations(&solver, &num_iterations);
         HYPRE_StructGMRESGetFinalRelativeResidualNorm(&solver, &final_res_norm);
         HYPRE_StructGMRESDestroy(&solver);
#else
         HYPRE_StructGMRESGetNumIterations(solver, &num_iterations);
         HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         HYPRE_StructGMRESDestroy(solver);
#endif

         if (solver_id == 30)
         {
#ifdef HYPRE_FORTRAN
            HYPRE_StructSMGDestroy(&precond);
#else
            HYPRE_StructSMGDestroy(precond);
#endif
         }
         else if (solver_id == 31)
         {
#ifdef HYPRE_FORTRAN
            HYPRE_StructPFMGDestroy(&precond);
#else
            HYPRE_StructPFMGDestroy(precond);
#endif
         }
         else if (solver_id == 37)
         {
#ifdef HYPRE_FORTRAN
            HYPRE_StructJacobiDestroy(&precond);
#else
            HYPRE_StructJacobiDestroy(precond);
#endif
         }
      }

   }

   /* Print the solution and other info */
   if (print_solution)
#ifdef HYPRE_FORTRAN
      HYPRE_SStructVectorPrint("sstruct.out.x", &x, &zero);
#else
      HYPRE_SStructVectorPrint("sstruct.out.x", x, 0);
#endif

   if (myid == 0)
   {
      hypre_printf("\n");
      hypre_printf("Iterations = %d\n", num_iterations);
      hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
      hypre_printf("\n");
   }

   /* Free memory */
#ifdef HYPRE_FORTRAN
   HYPRE_SStructGridDestroy(&grid);
   HYPRE_SStructStencilDestroy(&stencil);
   HYPRE_SStructGraphDestroy(&graph);
   HYPRE_SStructMatrixDestroy(&A);
   HYPRE_SStructVectorDestroy(&b);
   HYPRE_SStructVectorDestroy(&x);
#else
   HYPRE_SStructGridDestroy(grid);
   HYPRE_SStructStencilDestroy(stencil);
   HYPRE_SStructGraphDestroy(graph);
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);
#endif

   /* Finalize MPI */
   hypre_MPI_Finalize();

   return (0);
}

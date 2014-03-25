/*
   Example 12

   Interface:    Semi-Structured interface (SStruct)

   Compile with: make ex12 (may need to edit HYPRE_DIR in Makefile)

   Sample runs:  mpirun -np 2 ex12 -pfmg
                 mpirun -np 2 ex12 -boomeramg

   Description:  The grid layout is the same as ex1, but with nodal unknowns. The
                 solver is PCG preconditioned with either PFMG or BoomerAMG,
                 selected on the command line.

                 We recommend viewing the Struct examples before viewing this
                 and the other SStruct examples.  This is one of the simplest
                 SStruct examples, used primarily to demonstrate how to set up
                 non-cell-centered problems, and to demonstrate how easy it is
                 to switch between structured solvers (PFMG) and solvers
                 designed for more general settings (AMG).
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "HYPRE_sstruct_ls.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

int main (int argc, char *argv[])
{
   int i, j, myid, num_procs;

   HYPRE_SStructGrid     grid;
   HYPRE_SStructGraph    graph;
   HYPRE_SStructStencil  stencil;
   HYPRE_SStructMatrix   A;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   x;

   /* We only have one part and one variable */
   int nparts = 1;
   int nvars  = 1;
   int part   = 0;
   int var    = 0;

   int precond_id, object_type;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   if (num_procs != 2)
   {
      if (myid == 0) printf("Must run with 2 processors!\n");
      exit(1);
   }

   /* Parse the command line to determine the solver */
   if (argc != 2)
   {
      if (myid == 0) printf("Must specify a solver!\n");
      exit(1);
   }
   if ( strcmp(argv[1], "-pfmg") == 0 )
   {
      precond_id = 1;
      object_type = HYPRE_STRUCT;
   }
   else if ( strcmp(argv[1], "-boomeramg") == 0 )
   {
      precond_id = 2;
      object_type = HYPRE_PARCSR;
   }
   else
   {
      if (myid == 0) printf("Invalid solver!\n");
      exit(1);
   }

   /* 1. Set up the grid.  Here we use only one part.  Each processor describes
      the piece of the grid that it owns.  */
   {
      /* Create an empty 2D grid object */
      HYPRE_SStructGridCreate(MPI_COMM_WORLD, 2, nparts, &grid);

      /* Add boxes to the grid */
      if (myid == 0)
      {
         int ilower[2]={-3,1}, iupper[2]={-1,2};
         HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
      }
      else if (myid == 1)
      {
         int ilower[2]={0,1}, iupper[2]={2,4};
         HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
      }

      /* Set the variable type and number of variables on each part. */
      {
         HYPRE_SStructVariable vartypes[1] = {HYPRE_SSTRUCT_VARIABLE_NODE};

         HYPRE_SStructGridSetVariables(grid, part, nvars, vartypes);
      }

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      HYPRE_SStructGridAssemble(grid);
   }

   /* 2. Define the discretization stencil */
   {
      /* Create an empty 2D, 5-pt stencil object */
      HYPRE_SStructStencilCreate(2, 5, &stencil);

      /* Define the geometry of the stencil. Each represents a relative offset
         (in the index space). */
      {
         int entry;
         int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};

         /* Assign numerical values to the offsets so that we can easily refer
            to them - the last argument indicates the variable for which we are
            assigning this stencil */
         for (entry = 0; entry < 5; entry++)
            HYPRE_SStructStencilSetEntry(stencil, entry, offsets[entry], var);
      }
   }

   /* 3. Set up the Graph - this determines the non-zero structure of the matrix
      and allows non-stencil relationships between the parts */
   {
      /* Create the graph object */
      HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);

      /* See MatrixSetObjectType below */
      HYPRE_SStructGraphSetObjectType(graph, object_type);

      /* Now we need to tell the graph which stencil to use for each variable on
         each part (we only have one variable and one part) */
      HYPRE_SStructGraphSetStencil(graph, part, var, stencil);

      /* Here we could establish connections between parts if we had more than
         one part using the graph. For example, we could use
         HYPRE_GraphAddEntries() routine or HYPRE_GridSetNeighborPart() */

      /* Assemble the graph */
      HYPRE_SStructGraphAssemble(graph);
   }

   /* 4. Set up a SStruct Matrix */
   {
      /* Create an empty matrix object */
      HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);

      /* Set the object type (by default HYPRE_SSTRUCT). This determines the
         data structure used to store the matrix.  For PFMG we need to use
         HYPRE_STRUCT, and for BoomerAMG we need HYPRE_PARCSR (set above). */
      HYPRE_SStructMatrixSetObjectType(A, object_type);

      /* Get ready to set values */
      HYPRE_SStructMatrixInitialize(A);

      /* Set the matrix coefficients.  Each processor assigns coefficients for
         the boxes in the grid that it owns.  Note that the coefficients
         associated with each stencil entry may vary from grid point to grid
         point if desired.  Here, we first set the same stencil entries for each
         grid point.  Then we make modifications to grid points near the
         boundary.  Note that the ilower values are different from those used in
         ex1 because of the way nodal variables are referenced.  Also note that
         some of the stencil values are set on both processor 0 and processor 1.
         See the User and Reference manuals for more details. */
      if (myid == 0)
      {
         int ilower[2]={-4,0}, iupper[2]={-1,2};
         int stencil_indices[5] = {0,1,2,3,4}; /* labels for the stencil entries -
                                                  these correspond to the offsets
                                                  defined above */
         int nentries = 5;
         int nvalues  = 60; /* 12 grid points, each with 5 stencil entries */
         double values[60];

         for (i = 0; i < nvalues; i += nentries)
         {
            values[i] = 4.0;
            for (j = 1; j < nentries; j++)
               values[i+j] = -1.0;
         }

         HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper, var, nentries,
                                         stencil_indices, values);
      }
      else if (myid == 1)
      {
         int ilower[2]={-1,0}, iupper[2]={2,4};
         int stencil_indices[5] = {0,1,2,3,4};
         int nentries = 5;
         int nvalues  = 100; /* 20 grid points, each with 5 stencil entries */
         double values[100];

         for (i = 0; i < nvalues; i += nentries)
         {
            values[i] = 4.0;
            for (j = 1; j < nentries; j++)
               values[i+j] = -1.0;
         }

         HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper, var, nentries,
                                         stencil_indices, values);
      }

      /* Set the coefficients reaching outside of the boundary to 0.  Note that
       * both ilower *and* iupper may be different from those in ex1. */
      if (myid == 0)
      {
         double values[4];
         for (i = 0; i < 4; i++)
            values[i] = 0.0;
         {
            /* values below our box */
            int ilower[2]={-4,0}, iupper[2]={-1,0};
            int stencil_indices[1] = {3};
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper, var, 1,
                                            stencil_indices, values);
         }
         {
            /* values to the left of our box */
            int ilower[2]={-4,0}, iupper[2]={-4,2};
            int stencil_indices[1] = {1};
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper, var, 1,
                                            stencil_indices, values);
         }
         {
            /* values above our box */
            int ilower[2]={-4,2}, iupper[2]={-2,2};
            int stencil_indices[1] = {4};
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper, var, 1,
                                            stencil_indices, values);
         }
      }
      else if (myid == 1)
      {
         double values[5];
         for (i = 0; i < 5; i++)
            values[i] = 0.0;
         {
            /* values below our box */
            int ilower[2]={-1,0}, iupper[2]={2,0};
            int stencil_indices[1] = {3};
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper, var, 1,
                                            stencil_indices, values);
         }
         {
            /* values to the right of our box */
            int ilower[2]={2,0}, iupper[2]={2,4};
            int stencil_indices[1] = {2};
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper, var, 1,
                                            stencil_indices, values);
         }
         {
            /* values above our box */
            int ilower[2]={-1,4}, iupper[2]={2,4};
            int stencil_indices[1] = {4};
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper, var, 1,
                                            stencil_indices, values);
         }
         {
            /* values to the left of our box
               (that do not border the other box on proc. 0) */
            int ilower[2]={-1,3}, iupper[2]={-1,4};
            int stencil_indices[1] = {1};
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper, var, 1,
                                            stencil_indices, values);
         }
      }

      /* This is a collective call finalizing the matrix assembly.
         The matrix is now ``ready to be used'' */
      HYPRE_SStructMatrixAssemble(A);
   }

   /* 5. Set up SStruct Vectors for b and x. */
   {
      /* Create an empty vector object */
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
      HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);

      /* As with the matrix, set the appropriate object type for the vectors */
      HYPRE_SStructVectorSetObjectType(b, object_type);
      HYPRE_SStructVectorSetObjectType(x, object_type);

      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_SStructVectorInitialize(b);
      HYPRE_SStructVectorInitialize(x);

      /* Set the vector coefficients.  Again, note that the ilower values are
         different from those used in ex1, and some of the values are set on
         both processors. */
      if (myid == 0)
      {
         int ilower[2]={-4,0}, iupper[2]={-1,2};
         double values[12]; /* 12 grid points */

         for (i = 0; i < 12; i ++)
            values[i] = 1.0;
         HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);

         for (i = 0; i < 12; i ++)
            values[i] = 0.0;
         HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
      }
      else if (myid == 1)
      {
         int ilower[2]={0,1}, iupper[2]={2,4};
         double values[20]; /* 20 grid points */

         for (i = 0; i < 20; i ++)
            values[i] = 1.0;
         HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);

         for (i = 0; i < 20; i ++)
            values[i] = 0.0;
         HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
      }

      /* This is a collective call finalizing the vector assembly.
         The vectors are now ``ready to be used'' */
      HYPRE_SStructVectorAssemble(b);
      HYPRE_SStructVectorAssemble(x);
   }

   /* 6. Set up and use a solver (See the Reference Manual for descriptions
      of all of the options.) */
   if (precond_id == 1) /* PFMG */
   {
      HYPRE_StructMatrix sA;
      HYPRE_StructVector sb;
      HYPRE_StructVector sx;

      HYPRE_StructSolver solver;
      HYPRE_StructSolver precond;

      /* Because we are using a struct solver, we need to get the
         object of the matrix and vectors to pass in to the struct solvers */
      HYPRE_SStructMatrixGetObject(A, (void **) &sA);
      HYPRE_SStructVectorGetObject(b, (void **) &sb);
      HYPRE_SStructVectorGetObject(x, (void **) &sx);

      /* Create an empty PCG Struct solver */
      HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set PCG parameters */
      HYPRE_StructPCGSetTol(solver, 1.0e-06);
      HYPRE_StructPCGSetPrintLevel(solver, 2);
      HYPRE_StructPCGSetMaxIter(solver, 50);

      /* Create the Struct PFMG solver for use as a preconditioner */
      HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);

      /* Set PFMG parameters */
      HYPRE_StructPFMGSetMaxIter(precond, 1);
      HYPRE_StructPFMGSetTol(precond, 0.0);
      HYPRE_StructPFMGSetZeroGuess(precond);
      HYPRE_StructPFMGSetNumPreRelax(precond, 2);
      HYPRE_StructPFMGSetNumPostRelax(precond, 2);
      /* non-Galerkin coarse grid (more efficient for this problem) */
      HYPRE_StructPFMGSetRAPType(precond, 1);
      /* R/B Gauss-Seidel */
      HYPRE_StructPFMGSetRelaxType(precond, 2);
      /* skip relaxation on some levels (more efficient for this problem) */
      HYPRE_StructPFMGSetSkipRelax(precond, 1);


      /* Set preconditioner and solve */
      HYPRE_StructPCGSetPrecond(solver, HYPRE_StructPFMGSolve,
                          HYPRE_StructPFMGSetup, precond);
      HYPRE_StructPCGSetup(solver, sA, sb, sx);
      HYPRE_StructPCGSolve(solver, sA, sb, sx);

      /* Free memory */
      HYPRE_StructPCGDestroy(solver);
      HYPRE_StructPFMGDestroy(precond);
   }
   else if (precond_id == 2) /* BoomerAMG */
   {
      HYPRE_ParCSRMatrix parA;
      HYPRE_ParVector    parb;
      HYPRE_ParVector    parx;

      HYPRE_Solver       solver;
      HYPRE_Solver       precond;

      /* Because we are using a struct solver, we need to get the
         object of the matrix and vectors to pass in to the struct solvers */
      HYPRE_SStructMatrixGetObject(A, (void **) &parA);
      HYPRE_SStructVectorGetObject(b, (void **) &parb);
      HYPRE_SStructVectorGetObject(x, (void **) &parx);

      /* Create an empty PCG Struct solver */
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set PCG parameters */
      HYPRE_ParCSRPCGSetTol(solver, 1.0e-06);
      HYPRE_ParCSRPCGSetPrintLevel(solver, 2);
      HYPRE_ParCSRPCGSetMaxIter(solver, 50);

      /* Create the BoomerAMG solver for use as a preconditioner */
      HYPRE_BoomerAMGCreate(&precond);

      /* Set BoomerAMG parameters */
      HYPRE_BoomerAMGSetMaxIter(precond, 1);
      HYPRE_BoomerAMGSetTol(precond, 0.0);
      HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info */
      HYPRE_BoomerAMGSetCoarsenType(precond, 6);
      HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
      HYPRE_BoomerAMGSetNumSweeps(precond, 1);

      /* Set preconditioner and solve */
      HYPRE_ParCSRPCGSetPrecond(solver, HYPRE_BoomerAMGSolve,
                                HYPRE_BoomerAMGSetup, precond);
      HYPRE_ParCSRPCGSetup(solver, parA, parb, parx);
      HYPRE_ParCSRPCGSolve(solver, parA, parb, parx);

      /* Free memory */
      HYPRE_ParCSRPCGDestroy(solver);
      HYPRE_BoomerAMGDestroy(precond);
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

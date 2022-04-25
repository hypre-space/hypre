/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 6

   Interface:    Semi-Structured interface (SStruct)

   Compile with: make ex6

   Sample run:   mpirun -np 2 ex6

   Description:  This is a two processor example and is the same problem
                 as is solved with the structured interface in Example 2.
                 (The grid boxes are exactly those in the example
                 diagram in the struct interface chapter of the User's Manual.
                 Processor 0 owns two boxes and processor 1 owns one box.)

                 This is the simplest sstruct example, and it demonstrates how
                 the semi-structured interface can be used for structured problems.
                 There is one part and one variable.  The solver is PCG with SMG
                 preconditioner. We use a structured solver for this example.
*/

#include <stdio.h>

/* SStruct linear solvers headers */
#include "HYPRE_sstruct_ls.h"

/*     include fortran headers       */
#ifdef HYPRE_FORTRAN
#include "fortran.h"
#include "hypre_struct_fortran_test.h"
#include "hypre_sstruct_fortran_test.h"
#endif

HYPRE_Int main (HYPRE_Int argc, char *argv[])
{
   HYPRE_Int myid, num_procs;

   /* We are using struct solvers for this example */
#ifdef HYPRE_FORTRAN
   hypre_F90_Obj     grid;
   hypre_F90_Obj     graph;
   hypre_F90_Obj     stencil;
   hypre_F90_Obj     A;
   hypre_F90_Obj     b;
   hypre_F90_Obj     x;

   hypre_F90_Obj     solver;
   hypre_F90_Obj     precond;

   hypre_F90_Obj     long_temp_COMM;
   HYPRE_Int          temp_COMM;

   HYPRE_Int          precond_id;

   HYPRE_Int          hypre_var_cell = HYPRE_SSTRUCT_VARIABLE_CELL;

   HYPRE_Int          one = 1;
   HYPRE_Int          two = 2;
   HYPRE_Int          five = 5;
   HYPRE_Int          fifty = 50;

   HYPRE_Real   tol = 1.e-6;
   HYPRE_Real   zerodot = 0.;
#else
   HYPRE_SStructGrid     grid;
   HYPRE_SStructGraph    graph;
   HYPRE_SStructStencil  stencil;
   HYPRE_SStructMatrix   A;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   x;

   HYPRE_StructSolver solver;
   HYPRE_StructSolver precond;
#endif

   HYPRE_Int object_type;

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   if (num_procs != 2)
   {
      if (myid == 0) { hypre_printf("Must run with 2 processors!\n"); }
      hypre_MPI_Finalize();

      return (0);
   }
#ifdef HYPRE_FORTRAN
   temp_COMM = (HYPRE_Int) hypre_MPI_COMM_WORLD;
   long_temp_COMM = (hypre_F90_Obj) hypre_MPI_COMM_WORLD;
#endif

   /* 1. Set up the 2D grid.  This gives the index space in each part.
      Here we only use one part and one variable. (So the part id is 0
      and the variable id is 0) */
   {
      HYPRE_Int ndim = 2;
      HYPRE_Int nparts = 1;
      HYPRE_Int part = 0;

      /* Create an empty 2D grid object */
#ifdef HYPRE_FORTRAN
      HYPRE_SStructGridCreate(&temp_COMM, &ndim, &nparts, &grid);
#else
      HYPRE_SStructGridCreate(hypre_MPI_COMM_WORLD, ndim, nparts, &grid);
#endif

      /* Set the extents of the grid - each processor sets its grid
         boxes.  Each part has its own relative index space numbering,
         but in this example all boxes belong to the same part. */

      /* Processor 0 owns two boxes in the grid. */
      if (myid == 0)
      {
         /* Add a new box to the grid */
         {
            HYPRE_Int ilower[2] = {-3, 1};
            HYPRE_Int iupper[2] = {-1, 2};

#ifdef HYPRE_FORTRAN
            HYPRE_SStructGridSetExtents(&grid, &part, &ilower[0], &iupper[0]);
#else
            HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
#endif
         }

         /* Add a new box to the grid */
         {
            HYPRE_Int ilower[2] = {0, 1};
            HYPRE_Int iupper[2] = {2, 4};


#ifdef HYPRE_FORTRAN
            HYPRE_SStructGridSetExtents(&grid, &part, &ilower[0], &iupper[0]);
#else
            HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
#endif
         }
      }

      /* Processor 1 owns one box in the grid. */
      else if (myid == 1)
      {
         /* Add a new box to the grid */
         {
            HYPRE_Int ilower[2] = {3, 1};
            HYPRE_Int iupper[2] = {6, 4};


#ifdef HYPRE_FORTRAN
            HYPRE_SStructGridSetExtents(&grid, &part, &ilower[0], &iupper[0]);
#else
            HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);
#endif
         }
      }

      /* Set the variable type and number of variables on each part. */
      {
         HYPRE_Int i;
         HYPRE_Int nvars = 1;

#ifdef HYPRE_FORTRAN
         hypre_F90_Obj vartypes[1] = {HYPRE_SSTRUCT_VARIABLE_CELL};
#else
         HYPRE_SStructVariable vartypes[1] = {HYPRE_SSTRUCT_VARIABLE_CELL};
#endif

         for (i = 0; i < nparts; i++)

#ifdef HYPRE_FORTRAN
            HYPRE_SStructGridSetVariables(&grid, &i, &nvars, &vartypes[0]);
#else
            HYPRE_SStructGridSetVariables(grid, i, nvars, vartypes);
#endif
      }

      /* Now the grid is ready to use */

#ifdef HYPRE_FORTRAN
      HYPRE_SStructGridAssemble(&grid);
#else
      HYPRE_SStructGridAssemble(grid);
#endif
   }

   /* 2. Define the discretization stencil(s) */
   {
      /* Create an empty 2D, 5-pt stencil object */

#ifdef HYPRE_FORTRAN
      HYPRE_SStructStencilCreate(&two, &five, &stencil);
#else
      HYPRE_SStructStencilCreate(2, 5, &stencil);
#endif

      /* Define the geometry of the stencil. Each represents a
         relative offset (in the index space). */
      {
         HYPRE_Int entry;
         HYPRE_Int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
         HYPRE_Int var = 0;

         /* Assign numerical values to the offsets so that we can
            easily refer to them  - the last argument indicates the
            variable for which we are assigning this stencil - we are
            just using one variable in this example so it is the first one (0) */
         for (entry = 0; entry < 5; entry++)

#ifdef HYPRE_FORTRAN
            HYPRE_SStructStencilSetEntry(&stencil, &entry, &offsets[entry][0], &var);
#else
            HYPRE_SStructStencilSetEntry(stencil, entry, offsets[entry], var);
#endif
      }
   }

   /* 3. Set up the Graph  - this determines the non-zero structure
      of the matrix and allows non-stencil relationships between the parts */
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
         variable on each part (we only have one variable and one part) */

#ifdef HYPRE_FORTRAN
      HYPRE_SStructGraphSetStencil(&graph, &part, &var, &stencil);
#else
      HYPRE_SStructGraphSetStencil(graph, part, var, stencil);
#endif

      /* Here we could establish connections between parts if we
         had more than one part using the graph. For example, we could
         use HYPRE_GraphAddEntries() routine or HYPRE_GridSetNeighborBox() */

      /* Assemble the graph */

#ifdef HYPRE_FORTRAN
      HYPRE_SStructGraphAssemble(&graph);
#else
      HYPRE_SStructGraphAssemble(graph);
#endif
   }

   /* 4. Set up a SStruct Matrix */
   {
      HYPRE_Int i, j;
      HYPRE_Int part = 0;
      HYPRE_Int var = 0;

      /* Create the empty matrix object */

#ifdef HYPRE_FORTRAN
      HYPRE_SStructMatrixCreate(&temp_COMM, &graph, &A);
#else
      HYPRE_SStructMatrixCreate(hypre_MPI_COMM_WORLD, graph, &A);
#endif

      /* Set the object type (by default HYPRE_SSTRUCT). This determines the
         data structure used to store the matrix.  If you want to use unstructured
         solvers, e.g. BoomerAMG, the object type should be HYPRE_PARCSR.
         If the problem is purely structured (with one part), you may want to use
         HYPRE_STRUCT to access the structured solvers. Here we have a purely
         structured example. */
      object_type = HYPRE_STRUCT;

#ifdef HYPRE_FORTRAN
      HYPRE_SStructMatrixSetObjectType(&A, &object_type);
#else
      HYPRE_SStructMatrixSetObjectType(A, object_type);
#endif

      /* Get ready to set values */

#ifdef HYPRE_FORTRAN
      HYPRE_SStructMatrixInitialize(&A);
#else
      HYPRE_SStructMatrixInitialize(A);
#endif

      /* Each processor must set the stencil values for their boxes on each part.
         In this example, we only set stencil entries and therefore use
         HYPRE_SStructMatrixSetBoxValues.  If we need to set non-stencil entries,
         we have to use HYPRE_SStructMatrixSetValues (shown in a later example). */

      if (myid == 0)
      {
         /* Set the matrix coefficients for some set of stencil entries
            over all the gridpoints in my first box (account for boundary
            grid points later) */
         {
            HYPRE_Int ilower[2] = {-3, 1};
            HYPRE_Int iupper[2] = {-1, 2};

            HYPRE_Int nentries = 5;
            HYPRE_Int nvalues  = 30; /* 6 grid points, each with 5 stencil entries */
            HYPRE_Real values[30];

            HYPRE_Int stencil_indices[5];
            for (j = 0; j < nentries; j++) /* label the stencil indices -
                                              these correspond to the offsets
                                              defined above */
            {
               stencil_indices[j] = j;
            }

            for (i = 0; i < nvalues; i += nentries)
            {
               values[i] = 4.0;
               for (j = 1; j < nentries; j++)
               {
                  values[i + j] = -1.0;
               }
            }


#ifdef HYPRE_FORTRAN
            HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &nentries,
                                            &stencil_indices[0], &values[0]);
#else
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, nentries,
                                            stencil_indices, values);
#endif
         }

         /* Set the matrix coefficients for some set of stencil entries
            over the gridpoints in my second box */
         {
            HYPRE_Int ilower[2] = {0, 1};
            HYPRE_Int iupper[2] = {2, 4};

            HYPRE_Int nentries = 5;
            HYPRE_Int nvalues  = 60; /* 12 grid points, each with 5 stencil entries */
            HYPRE_Real values[60];

            HYPRE_Int stencil_indices[5];
            for (j = 0; j < nentries; j++)
            {
               stencil_indices[j] = j;
            }

            for (i = 0; i < nvalues; i += nentries)
            {
               values[i] = 4.0;
               for (j = 1; j < nentries; j++)
               {
                  values[i + j] = -1.0;
               }
            }


#ifdef HYPRE_FORTRAN
            HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &nentries,
                                            &stencil_indices[0], &values[0]);
#else
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, nentries,
                                            stencil_indices, values);
#endif
         }
      }
      else if (myid == 1)
      {
         /* Set the matrix coefficients for some set of stencil entries
            over the gridpoints in my box */
         {
            HYPRE_Int ilower[2] = {3, 1};
            HYPRE_Int iupper[2] = {6, 4};

            HYPRE_Int nentries = 5;
            HYPRE_Int nvalues  = 80; /* 16 grid points, each with 5 stencil entries */
            HYPRE_Real values[80];

            HYPRE_Int stencil_indices[5];
            for (j = 0; j < nentries; j++)
            {
               stencil_indices[j] = j;
            }

            for (i = 0; i < nvalues; i += nentries)
            {
               values[i] = 4.0;
               for (j = 1; j < nentries; j++)
               {
                  values[i + j] = -1.0;
               }
            }


#ifdef HYPRE_FORTRAN
            HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &nentries,
                                            &stencil_indices[0], &values[0]);
#else
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, nentries,
                                            stencil_indices, values);
#endif
         }
      }

      /* For each box, set any coefficients that reach ouside of the
         boundary to 0 */
      if (myid == 0)
      {
         HYPRE_Int maxnvalues = 6;
         HYPRE_Real values[6];

         for (i = 0; i < maxnvalues; i++)
         {
            values[i] = 0.0;
         }

         {
            /* Values below our first AND second box */
            HYPRE_Int ilower[2] = {-3, 1};
            HYPRE_Int iupper[2] = { 2, 1};

            HYPRE_Int stencil_indices[1] = {3};


#ifdef HYPRE_FORTRAN
            HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }

         {
            /* Values to the left of our first box */
            HYPRE_Int ilower[2] = {-3, 1};
            HYPRE_Int iupper[2] = {-3, 2};

            HYPRE_Int stencil_indices[1] = {1};


#ifdef HYPRE_FORTRAN
            HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }

         {
            /* Values above our first box */
            HYPRE_Int ilower[2] = {-3, 2};
            HYPRE_Int iupper[2] = {-1, 2};

            HYPRE_Int stencil_indices[1] = {4};


#ifdef HYPRE_FORTRAN
            HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }

         {
            /* Values to the left of our second box (that do not border the
               first box). */
            HYPRE_Int ilower[2] = { 0, 3};
            HYPRE_Int iupper[2] = { 0, 4};

            HYPRE_Int stencil_indices[1] = {1};


#ifdef HYPRE_FORTRAN
            HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }

         {
            /* Values above our second box */
            HYPRE_Int ilower[2] = { 0, 4};
            HYPRE_Int iupper[2] = { 2, 4};

            HYPRE_Int stencil_indices[1] = {4};


#ifdef HYPRE_FORTRAN
            HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }
      }
      else if (myid == 1)
      {
         HYPRE_Int maxnvalues = 4;
         HYPRE_Real values[4];
         for (i = 0; i < maxnvalues; i++)
         {
            values[i] = 0.0;
         }

         {
            /* Values below our box */
            HYPRE_Int ilower[2] = { 3, 1};
            HYPRE_Int iupper[2] = { 6, 1};

            HYPRE_Int stencil_indices[1] = {3};


#ifdef HYPRE_FORTRAN
            HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }

         {
            /* Values to the right of our box */
            HYPRE_Int ilower[2] = { 6, 1};
            HYPRE_Int iupper[2] = { 6, 4};

            HYPRE_Int stencil_indices[1] = {2};


#ifdef HYPRE_FORTRAN
            HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }

         {
            /* Values above our box */
            HYPRE_Int ilower[2] = { 3, 4};
            HYPRE_Int iupper[2] = { 6, 4};

            HYPRE_Int stencil_indices[1] = {4};


#ifdef HYPRE_FORTRAN
            HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                            &var, &one,
                                            &stencil_indices[0], &values[0]);
#else
            HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                            var, 1,
                                            stencil_indices, values);
#endif
         }
      }

      /* This is a collective call finalizing the matrix assembly.
         The matrix is now ``ready to be used'' */

#ifdef HYPRE_FORTRAN
      HYPRE_SStructMatrixAssemble(&A);
#else
      HYPRE_SStructMatrixAssemble(A);
#endif
   }


   /* 5. Set up SStruct Vectors for b and x */
   {
      HYPRE_Int i;

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

      /* As with the matrix,  set the object type for the vectors
         to be the struct type */
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

      if (myid == 0)
      {
         /* Set the vector coefficients over the gridpoints in my first box */
         {
            HYPRE_Int ilower[2] = {-3, 1};
            HYPRE_Int iupper[2] = {-1, 2};

            HYPRE_Int nvalues = 6;  /* 6 grid points */
            HYPRE_Real values[6];

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 1.0;
            }

#ifdef HYPRE_FORTRAN
            HYPRE_SStructVectorSetBoxValues(&b, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
            HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);
#endif

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 0.0;
            }

#ifdef HYPRE_FORTRAN
            HYPRE_SStructVectorSetBoxValues(&x, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
            HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
#endif
         }

         /* Set the vector coefficients over the gridpoints in my second box */
         {
            HYPRE_Int ilower[2] = { 0, 1};
            HYPRE_Int iupper[2] = { 2, 4};

            HYPRE_Int nvalues = 12; /* 12 grid points */
            HYPRE_Real values[12];

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 1.0;
            }

#ifdef HYPRE_FORTRAN
            HYPRE_SStructVectorSetBoxValues(&b, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
            HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);
#endif

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 0.0;
            }

#ifdef HYPRE_FORTRAN
            HYPRE_SStructVectorSetBoxValues(&x, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
            HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
#endif
         }
      }
      else if (myid == 1)
      {
         /* Set the vector coefficients over the gridpoints in my box */
         {
            HYPRE_Int ilower[2] = { 3, 1};
            HYPRE_Int iupper[2] = { 6, 4};

            HYPRE_Int nvalues = 16; /* 16 grid points */
            HYPRE_Real values[16];

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 1.0;
            }

#ifdef HYPRE_FORTRAN
            HYPRE_SStructVectorSetBoxValues(&b, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
            HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);
#endif

            for (i = 0; i < nvalues; i ++)
            {
               values[i] = 0.0;
            }

#ifdef HYPRE_FORTRAN
            HYPRE_SStructVectorSetBoxValues(&x, &part, &ilower[0], &iupper[0], &var, &values[0]);
#else
            HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
#endif
         }
      }

      /* This is a collective call finalizing the vector assembly.
         The vectors are now ``ready to be used'' */

#ifdef HYPRE_FORTRAN
      HYPRE_SStructVectorAssemble(&b);
      HYPRE_SStructVectorAssemble(&x);
#else
      HYPRE_SStructVectorAssemble(b);
      HYPRE_SStructVectorAssemble(x);
#endif
   }


   /* 6. Set up and use a solver (See the Reference Manual for descriptions
      of all of the options.) */
   {

#ifdef HYPRE_FORTRAN
      hypre_F90_Obj sA;
      hypre_F90_Obj sb;
      hypre_F90_Obj sx;
#else
      HYPRE_StructMatrix sA;
      HYPRE_StructVector sb;
      HYPRE_StructVector sx;
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

      /* Create an empty PCG Struct solver */

#ifdef HYPRE_FORTRAN
      HYPRE_StructPCGCreate(&temp_COMM, &solver);
#else
      HYPRE_StructPCGCreate(hypre_MPI_COMM_WORLD, &solver);
#endif

      /* Set PCG parameters */

#ifdef HYPRE_FORTRAN
      HYPRE_StructPCGSetTol(&solver, &tol);
      HYPRE_StructPCGSetPrintLevel(&solver, &two);
      HYPRE_StructPCGSetMaxIter(&solver, &fifty);
#else
      HYPRE_StructPCGSetTol(solver, 1.0e-06);
      HYPRE_StructPCGSetPrintLevel(solver, 2);
      HYPRE_StructPCGSetMaxIter(solver, 50);
#endif

      /* Create the Struct SMG solver for use as a preconditioner */

#ifdef HYPRE_FORTRAN
      HYPRE_StructSMGCreate(&temp_COMM, &precond);
#else
      HYPRE_StructSMGCreate(hypre_MPI_COMM_WORLD, &precond);
#endif

      /* Set SMG parameters */

#ifdef HYPRE_FORTRAN
      HYPRE_StructSMGSetMaxIter(&precond, &one);
      HYPRE_StructSMGSetTol(&precond, &zerodot);
      HYPRE_StructSMGSetZeroGuess(&precond);
      HYPRE_StructSMGSetNumPreRelax(&precond, &one);
      HYPRE_StructSMGSetNumPostRelax(&precond, &one);
#else
      HYPRE_StructSMGSetMaxIter(precond, 1);
      HYPRE_StructSMGSetTol(precond, 0.0);
      HYPRE_StructSMGSetZeroGuess(precond);
      HYPRE_StructSMGSetNumPreRelax(precond, 1);
      HYPRE_StructSMGSetNumPostRelax(precond, 1);
#endif

      /* Set preconditioner and solve */

#ifdef HYPRE_FORTRAN
      precond_id = 0;
      HYPRE_StructPCGSetPrecond(&solver, &precond_id, &precond);
      HYPRE_StructPCGSetup(&solver, &sA, &sb, &sx);
      HYPRE_StructPCGSolve(&solver, &sA, &sb, &sx);
#else
      HYPRE_StructPCGSetPrecond(solver, HYPRE_StructSMGSolve,
                                HYPRE_StructSMGSetup, precond);
      HYPRE_StructPCGSetup(solver, sA, sb, sx);
      HYPRE_StructPCGSolve(solver, sA, sb, sx);
#endif
   }

   /* Free memory */

#ifdef HYPRE_FORTRAN
   HYPRE_SStructGridDestroy(&grid);
   HYPRE_SStructStencilDestroy(&stencil);
   HYPRE_SStructGraphDestroy(&graph);
   HYPRE_SStructMatrixDestroy(&A);
   HYPRE_SStructVectorDestroy(&b);
   HYPRE_SStructVectorDestroy(&x);

   HYPRE_StructPCGDestroy(&solver);
   HYPRE_StructSMGDestroy(&precond);
#else
   HYPRE_SStructGridDestroy(grid);
   HYPRE_SStructStencilDestroy(stencil);
   HYPRE_SStructGraphDestroy(graph);
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);

   HYPRE_StructPCGDestroy(solver);
   HYPRE_StructSMGDestroy(precond);
#endif

   /* Finalize MPI */
   hypre_MPI_Finalize();

   return (0);
}

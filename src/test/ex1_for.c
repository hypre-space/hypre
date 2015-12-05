/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.4 $
 ***********************************************************************EHEADER*/

/*
   Example 1

   Interface:    Structured interface (Struct)

   Compile with: make ex1 (may need to edit HYPRE_DIR in Makefile)

   Sample run:   mpirun -np 2 ex1

   Description:  This is a two processor example.  Each processor owns one
                 box in the grid.  For reference, the two grid boxes are those
                 in the example diagram in the struct interface chapter
                 of the User's Manual. Note that in this example code, we have
                 used the two boxes shown in the diagram as belonging
                 to processor 0 (and given one box to each processor). The
                 solver is PCG with no preconditioner.

                 We recommend viewing examples 1-4 sequentially for
                 a nice overview/tutorial of the struct interface.
*/

/* Struct linear solvers header */
#include "HYPRE_struct_ls.h"

#ifdef HYPRE_FORTRAN
#include "fortran.h"
#include "hypre_struct_fortran_test.h"
#endif

HYPRE_Int main (HYPRE_Int argc, char *argv[])
{
   HYPRE_Int i, j, myid;

#ifdef HYPRE_FORTRAN
   hypre_F90_Obj grid;
   hypre_F90_Obj stencil;
   hypre_F90_Obj A;
   hypre_F90_Obj b;
   hypre_F90_Obj x;
   hypre_F90_Obj solver;
        HYPRE_Int temp_COMM;
        HYPRE_Int one = 1;
        HYPRE_Int two = 2;
        HYPRE_Int five = 5;
     double tol = 1.e-6;
#else
   HYPRE_StructGrid     grid;
   HYPRE_StructStencil  stencil;
   HYPRE_StructMatrix   A;
   HYPRE_StructVector   b;
   HYPRE_StructVector   x;
   HYPRE_StructSolver   solver;
#endif

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   /* 1. Set up a grid. Each processor describes the piece
      of the grid that it owns. */
   {
      /* Create an empty 2D grid object */
#ifdef HYPRE_FORTRAN
      temp_COMM = (HYPRE_Int) hypre_MPI_COMM_WORLD;
      HYPRE_StructGridCreate(&temp_COMM, &two, &grid);
#else
      HYPRE_StructGridCreate(hypre_MPI_COMM_WORLD, 2, &grid);
#endif

      /* Add boxes to the grid */
      if (myid == 0)
      {
         HYPRE_Int ilower[2]={-3,1}, iupper[2]={-1,2};
#ifdef HYPRE_FORTRAN
         HYPRE_StructGridSetExtents(&grid, &ilower[0], &iupper[0]);
#else
         HYPRE_StructGridSetExtents(grid, ilower, iupper);
#endif
      }
      else if (myid == 1)
      {
         HYPRE_Int ilower[2]={0,1}, iupper[2]={2,4};
#ifdef HYPRE_FORTRAN
         HYPRE_StructGridSetExtents(&grid, &ilower[0], &iupper[0]);
#else
         HYPRE_StructGridSetExtents(grid, ilower, iupper);
#endif
      }

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
#ifdef HYPRE_FORTRAN
      HYPRE_StructGridAssemble(&grid);
#else
      HYPRE_StructGridAssemble(grid);
#endif
   }

   /* 2. Define the discretization stencil */
   {
      /* Create an empty 2D, 5-pt stencil object */
#ifdef HYPRE_FORTRAN
      HYPRE_StructStencilCreate(&two, &five, &stencil);
#else
      HYPRE_StructStencilCreate(2, 5, &stencil);
#endif

      /* Define the geometry of the stencil. Each represents a
         relative offset (in the index space). */
      {
         HYPRE_Int entry;
         HYPRE_Int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};

         /* Assign each of the 5 stencil entries */
#ifdef HYPRE_FORTRAN
         for (entry = 0; entry < 5; entry++)
            HYPRE_StructStencilSetElement(&stencil, &entry, offsets[entry]);
#else
         for (entry = 0; entry < 5; entry++)
            HYPRE_StructStencilSetElement(stencil, entry, offsets[entry]);
#endif
      }
   }

   /* 3. Set up a Struct Matrix */
   {
      /* Create an empty matrix object */
#ifdef HYPRE_FORTRAN
      HYPRE_StructMatrixCreate(&temp_COMM, &grid, &stencil, &A);
#else
      HYPRE_StructMatrixCreate(hypre_MPI_COMM_WORLD, grid, stencil, &A);
#endif

      /* Indicate that the matrix coefficients are ready to be set */
#ifdef HYPRE_FORTRAN
      HYPRE_StructMatrixInitialize(&A);
#else
      HYPRE_StructMatrixInitialize(A);
#endif

      /* Set the matrix coefficients.  Each processor assigns coefficients
         for the boxes in the grid that it owns. Note that the coefficients
         associated with each stencil entry may vary from grid point to grid
         point if desired.  Here, we first set the same stencil entries for
         each grid point.  Then we make modifications to grid points near
         the boundary. */
      if (myid == 0)
      {
         HYPRE_Int ilower[2]={-3,1}, iupper[2]={-1,2};
         HYPRE_Int stencil_indices[5] = {0,1,2,3,4}; /* labels for the stencil entries -
                                                  these correspond to the offsets
                                                  defined above */
         HYPRE_Int nentries = 5;
         HYPRE_Int nvalues  = 30; /* 6 grid points, each with 5 stencil entries */
         double values[30];

         /* We have 6 grid points, each with 5 stencil entries */
         for (i = 0; i < nvalues; i += nentries)
         {
            values[i] = 4.0;
            for (j = 1; j < nentries; j++)
               values[i+j] = -1.0;
         }

#ifdef HYPRE_FORTRAN
         HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &nentries,
                                        &stencil_indices[0], &values[0]);
#else
         HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
                                        stencil_indices, values);
#endif
      }
      else if (myid == 1)
      {
         HYPRE_Int ilower[2]={0,1}, iupper[2]={2,4};
         HYPRE_Int stencil_indices[5] = {0,1,2,3,4};
         HYPRE_Int nentries = 5;
         HYPRE_Int nvalues  = 60; /* 12 grid points, each with 5 stencil entries */
         double values[60];

         for (i = 0; i < nvalues; i += nentries)
         {
            values[i] = 4.0;
            for (j = 1; j < nentries; j++)
               values[i+j] = -1.0;
         }

#ifdef HYPRE_FORTRAN
         HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &nentries,
                                        &stencil_indices[0], &values[0]);
#else
         HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
                                        stencil_indices, values);
#endif
      }

      /* Set the coefficients reaching outside of the boundary to 0 */
      if (myid == 0)
      {
         double values[3];
         for (i = 0; i < 3; i++)
            values[i] = 0.0;
         {
            /* values below our box */
            HYPRE_Int ilower[2]={-3,1}, iupper[2]={-1,1};
            HYPRE_Int stencil_indices[1] = {3};
#ifdef HYPRE_FORTRAN
            HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                        &stencil_indices[0], &values[0]);
#else
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
         {
            /* values to the left of our box */
            HYPRE_Int ilower[2]={-3,1}, iupper[2]={-3,2};
            HYPRE_Int stencil_indices[1] = {1};
#ifdef HYPRE_FORTRAN
            HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                        &stencil_indices[0], &values[0]);
#else
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
         {
            /* values above our box */
            HYPRE_Int ilower[2]={-3,2}, iupper[2]={-1,2};
            HYPRE_Int stencil_indices[1] = {4};
#ifdef HYPRE_FORTRAN
            HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                        &stencil_indices[0], &values[0]);
#else
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
      }
      else if (myid == 1)
      {
         double values[4];
         for (i = 0; i < 4; i++)
            values[i] = 0.0;
         {
            /* values below our box */
            HYPRE_Int ilower[2]={0,1}, iupper[2]={2,1};
            HYPRE_Int stencil_indices[1] = {3};
#ifdef HYPRE_FORTRAN
            HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                        &stencil_indices[0], &values[0]);
#else
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
         {
            /* values to the right of our box
               (that do not border the other box on proc. 0) */
            HYPRE_Int ilower[2]={2,1}, iupper[2]={2,4};
            HYPRE_Int stencil_indices[1] = {2};
#ifdef HYPRE_FORTRAN
            HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                        &stencil_indices[0], &values[0]);
#else
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
         {
            /* values above our box */
            HYPRE_Int ilower[2]={0,4}, iupper[2]={2,4};
            HYPRE_Int stencil_indices[1] = {4};
#ifdef HYPRE_FORTRAN
            HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                        &stencil_indices[0], &values[0]);
#else
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
         {
            /* values to the left of our box */
            HYPRE_Int ilower[2]={0,3}, iupper[2]={0,4};
            HYPRE_Int stencil_indices[1] = {1};
#ifdef HYPRE_FORTRAN
            HYPRE_StructMatrixSetBoxValues(&A, &ilower[0], &iupper[0], &one,
                                        &stencil_indices[0], &values[0]);
#else
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, values);
#endif
         }
      }

      /* This is a collective call finalizing the matrix assembly.
         The matrix is now ``ready to be used'' */
#ifdef HYPRE_FORTRAN
      HYPRE_StructMatrixAssemble(&A);
#else
      HYPRE_StructMatrixAssemble(A);
#endif
   }

   /* 4. Set up Struct Vectors for b and x.  Each processor sets the vectors
      corresponding to its boxes. */
   {
      /* Create an empty vector object */
#ifdef HYPRE_FORTRAN
      HYPRE_StructVectorCreate(&temp_COMM, &grid, &b);
      HYPRE_StructVectorCreate(&temp_COMM, &grid, &x);
#else
      HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, grid, &b);
      HYPRE_StructVectorCreate(hypre_MPI_COMM_WORLD, grid, &x);
#endif

      /* Indicate that the vector coefficients are ready to be set */
#ifdef HYPRE_FORTRAN
      HYPRE_StructVectorInitialize(&b);
      HYPRE_StructVectorInitialize(&x);
#else
      HYPRE_StructVectorInitialize(b);
      HYPRE_StructVectorInitialize(x);
#endif

      /* Set the vector coefficients */
      if (myid == 0)
      {
         HYPRE_Int ilower[2]={-3,1}, iupper[2]={-1,2};
         double values[6]; /* 6 grid points */

         for (i = 0; i < 6; i ++)
            values[i] = 1.0;
#ifdef HYPRE_FORTRAN
         HYPRE_StructVectorSetBoxValues(&b, &ilower[0], &iupper[0], &values[0]);
#else
         HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);
#endif

         for (i = 0; i < 6; i ++)
            values[i] = 0.0;
#ifdef HYPRE_FORTRAN
         HYPRE_StructVectorSetBoxValues(&x, &ilower[0], &iupper[0], &values[0]);
#else
         HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
#endif
      }
      else if (myid == 1)
      {
         HYPRE_Int ilower[2]={0,1}, iupper[2]={2,4};
         double values[12]; /* 12 grid points */

         for (i = 0; i < 12; i ++)
            values[i] = 1.0;
#ifdef HYPRE_FORTRAN
         HYPRE_StructVectorSetBoxValues(&b, &ilower[0], &iupper[0], &values[0]);
#else
         HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);
#endif

         for (i = 0; i < 12; i ++)
            values[i] = 0.0;
#ifdef HYPRE_FORTRAN
         HYPRE_StructVectorSetBoxValues(&x, &ilower[0], &iupper[0], &values[0]);
#else
         HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
#endif
      }

      /* This is a collective call finalizing the vector assembly.
         The vectors are now ``ready to be used'' */
#ifdef HYPRE_FORTRAN
      HYPRE_StructVectorAssemble(&b);
      HYPRE_StructVectorAssemble(&x);
#else
      HYPRE_StructVectorAssemble(b);
      HYPRE_StructVectorAssemble(x);
#endif
   }

   /* 5. Set up and use a solver (See the Reference Manual for descriptions
      of all of the options.) */
   {
      /* Create an empty PCG Struct solver */
#ifdef HYPRE_FORTRAN
      HYPRE_StructPCGCreate(&temp_COMM, &solver);
#else
      HYPRE_StructPCGCreate(hypre_MPI_COMM_WORLD, &solver);
#endif

      /* Set some parameters */
#ifdef HYPRE_FORTRAN
      HYPRE_StructPCGSetTol(&solver, &tol); /* convergence tolerance */
      HYPRE_StructPCGSetPrintLevel(&solver, &two); /* amount of info. printed */
#else
      HYPRE_StructPCGSetTol(solver, 1.0e-06); /* convergence tolerance */
      HYPRE_StructPCGSetPrintLevel(solver, 2); /* amount of info. printed */
#endif

      /* Setup and solve */
#ifdef HYPRE_FORTRAN
      HYPRE_StructPCGSetup(&solver, &A, &b, &x);
      HYPRE_StructPCGSolve(&solver, &A, &b, &x);
#else
      HYPRE_StructPCGSetup(solver, A, b, x);
      HYPRE_StructPCGSolve(solver, A, b, x);
#endif
   }

   /* Free memory */
#ifdef HYPRE_FORTRAN
   HYPRE_StructGridDestroy(&grid);
   HYPRE_StructStencilDestroy(&stencil);
   HYPRE_StructMatrixDestroy(&A);
   HYPRE_StructVectorDestroy(&b);
   HYPRE_StructVectorDestroy(&x);
   HYPRE_StructPCGDestroy(&solver);
#else
   HYPRE_StructGridDestroy(grid);
   HYPRE_StructStencilDestroy(stencil);
   HYPRE_StructMatrixDestroy(A);
   HYPRE_StructVectorDestroy(b);
   HYPRE_StructVectorDestroy(x);
   HYPRE_StructPCGDestroy(solver);
#endif

   /* Finalize MPI */
   hypre_MPI_Finalize();

   return (0);
}

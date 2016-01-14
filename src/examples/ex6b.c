/*
   Example 6

   Interface:    Semi-Structured interface (SStruct).  Babel version.

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

#include <mpi.h>
#include "bHYPRE.h"
#include "HYPRE.h"

int main (int argc, char *argv[])
{
   int myid, num_procs;

   bHYPRE_SStructGrid     grid;
   bHYPRE_SStructGraph    graph;
   bHYPRE_SStructStencil  stencil;
   bHYPRE_SStructMatrix   A;
   bHYPRE_SStructVector   b;
   bHYPRE_SStructVector   x;

   /* We are using struct solvers for this example */
   bHYPRE_PCG PCGsolver;
   bHYPRE_StructSMG SMGprecond;
   bHYPRE_Solver precond;

   sidl_BaseInterface _ex = NULL;
   MPI_Comm mpicommworld = MPI_COMM_WORLD;
   bHYPRE_MPICommunicator mpi_comm;
   int object_type;
   int ndim = 2;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   mpi_comm = bHYPRE_MPICommunicator_CreateC( &mpicommworld, &_ex );

   if (num_procs != 2)
   {
      if (myid ==0) printf("Must run with 2 processors!\n");
      MPI_Finalize();

      return(0);
   }

   /* 1. Set up the 2D grid.  This gives the index space in each part.
      Here we only use one part and one variable. (So the part id is 0
      and the variable id is 0) */
   {
      int nparts = 1;
      int part = 0;

      /* Create an empty 2D grid object */
      grid = bHYPRE_SStructGrid_Create( mpi_comm, ndim, nparts, &_ex );

      /* Set the extents of the grid - each processor sets its grid
         boxes.  Each part has its own relative index space numbering,
         but in this example all boxes belong to the same part. */

      /* Processor 0 owns two boxes in the grid. */
      if (myid == 0)
      {
         /* Add a new box to the grid */
         {
            int ilower[2] = {-3, 1};
            int iupper[2] = {-1, 2};

            bHYPRE_SStructGrid_SetExtents( grid, part, ilower, iupper, ndim, &_ex );
         }

         /* Add a new box to the grid */
         {
            int ilower[2] = {0, 1};
            int iupper[2] = {2, 4};

            bHYPRE_SStructGrid_SetExtents( grid, part, ilower, iupper, ndim, &_ex );
         }
      }

      /* Processor 1 owns one box in the grid. */
      else if (myid == 1)
      {
         /* Add a new box to the grid */
         {
            int ilower[2] = {3, 1};
            int iupper[2] = {6, 4};

            bHYPRE_SStructGrid_SetExtents( grid, part, ilower, iupper, ndim, &_ex );
         }
      }

      /* Set the variable type and number of variables on each part. */
      {
         int i;
         int nvars = 1;
         int var = 0;
         enum bHYPRE_SStructVariable__enum vartypes[1] = {bHYPRE_SStructVariable_CELL};

         for (i = 0; i< nparts; i++)
            bHYPRE_SStructGrid_SetVariable( grid, i, var, nvars, vartypes[1], &_ex );
         /* ... if nvars>1 we would need a number of AddVariable calls */
      }

      /* Now the grid is ready to use */
      bHYPRE_SStructGrid_Assemble( grid, &_ex );
   }

   /* 2. Define the discretization stencil(s) */
   {
      /* Create an empty 2D, 5-pt stencil object */
      stencil = bHYPRE_SStructStencil_Create( 2, 5, &_ex );

      /* Define the geometry of the stencil. Each represents a
         relative offset (in the index space). */
      {
         int entry;
         int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};
         int var = 0;

         /* Assign numerical values to the offsets so that we can
            easily refer to them  - the last argument indicates the
            variable for which we are assigning this stencil - we are
            just using one variable in this example so it is the first one (0) */
         for (entry = 0; entry < 5; entry++)
            bHYPRE_SStructStencil_SetEntry( stencil, entry, offsets[entry], ndim, var, &_ex );
      }
   }

   /* 3. Set up the Graph  - this determines the non-zero structure
      of the matrix and allows non-stencil relationships between the parts */
   {
      int var = 0;
      int part = 0;

      /* Create the graph object */
      graph = bHYPRE_SStructGraph_Create( mpi_comm, grid, &_ex );

      /* Now we need to tell the graph which stencil to use for each
         variable on each part (we only have one variable and one part) */
      bHYPRE_SStructGraph_SetStencil( graph, part, var, stencil, &_ex );

      /* Here we could establish connections between parts if we
         had more than one part using the graph. For example, we could
         use HYPRE_GraphAddEntries() routine or HYPRE_GridSetNeighborBox() */

      /* Assemble the graph */
      bHYPRE_SStructGraph_Assemble( graph, &_ex );
   }

   /* 4. Set up a SStruct Matrix */
   {
      int i,j;
      int part = 0;
      int var = 0;

      /* Create the empty matrix object */
      A = bHYPRE_SStructMatrix_Create( mpi_comm, graph, &_ex );

      /* Set the object type (by default HYPRE_SSTRUCT). This determines the
         data structure used to store the matrix.  If you want to use unstructured
         solvers, e.g. BoomerAMG, the object type should be HYPRE_PARCSR.
         If the problem is purely structured (with one part), you may want to use
         HYPRE_STRUCT to access the structured solvers. Here we have a purely
         structured example. */
      object_type = HYPRE_STRUCT;
      bHYPRE_SStructMatrix_SetObjectType( A, object_type, &_ex );

      /* Get ready to set values */
      bHYPRE_SStructMatrix_Initialize( A, &_ex );

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
            int ilower[2] = {-3, 1};
            int iupper[2] = {-1, 2};

            int nentries = 5;
            int nvalues  = 30; /* 6 grid points, each with 5 stencil entries */
            double values[30];

            int stencil_indices[5];
            for (j = 0; j < nentries; j++) /* label the stencil indices -
                                              these correspond to the offsets
                                              defined above */
               stencil_indices[j] = j;

            for (i = 0; i < nvalues; i += nentries)
            {
               values[i] = 4.0;
               for (j = 1; j < nentries; j++)
                  values[i+j] = -1.0;
            }

            bHYPRE_SStructMatrix_SetBoxValues( A, part, ilower, iupper, ndim,
                                               var, nentries, stencil_indices,
                                               values, nvalues, &_ex );
         }

         /* Set the matrix coefficients for some set of stencil entries
            over the gridpoints in my second box */
         {
            int ilower[2] = {0, 1};
            int iupper[2] = {2, 4};

            int nentries = 5;
            int nvalues  = 60; /* 12 grid points, each with 5 stencil entries */
            double values[60];

            int stencil_indices[5];
            for (j = 0; j < nentries; j++)
               stencil_indices[j] = j;

            for (i = 0; i < nvalues; i += nentries)
            {
               values[i] = 4.0;
               for (j = 1; j < nentries; j++)
                  values[i+j] = -1.0;
            }

            bHYPRE_SStructMatrix_SetBoxValues( A, part, ilower, iupper, ndim,
                                               var, nentries, stencil_indices,
                                               values, nvalues, &_ex );
         }
      }
      else if (myid == 1)
      {
         /* Set the matrix coefficients for some set of stencil entries
            over the gridpoints in my box */
         {
            int ilower[2] = {3, 1};
            int iupper[2] = {6, 4};

            int nentries = 5;
            int nvalues  = 80; /* 16 grid points, each with 5 stencil entries */
            double values[80];

            int stencil_indices[5];
            for (j = 0; j < nentries; j++)
               stencil_indices[j] = j;

            for (i = 0; i < nvalues; i += nentries)
            {
               values[i] = 4.0;
               for (j = 1; j < nentries; j++)
                  values[i+j] = -1.0;
            }

            bHYPRE_SStructMatrix_SetBoxValues( A, part, ilower, iupper, ndim,
                                               var, nentries, stencil_indices,
                                               values, nvalues, &_ex );
         }
      }

      /* For each box, set any coefficients that reach ouside of the
         boundary to 0 */
      if (myid == 0)
      {
         int maxnvalues = 6;
         double values[6];

         for (i = 0; i < maxnvalues; i++)
            values[i] = 0.0;

         {
            /* Values below our first AND second box */
            int ilower[2] = {-3, 1};
            int iupper[2] = { 2, 1};
            int nvalues = 6;
            int stencil_indices[1] = {3};

            bHYPRE_SStructMatrix_SetBoxValues( A, part, ilower, iupper, ndim,
                                               var, 1, stencil_indices,
                                               values, nvalues, &_ex );
         }

         {
            /* Values to the left of our first box */
            int ilower[2] = {-3, 1};
            int iupper[2] = {-3, 2};
            int nvalues = 2;
            int stencil_indices[1] = {1};

            bHYPRE_SStructMatrix_SetBoxValues( A, part, ilower, iupper, ndim,
                                               var, 1, stencil_indices,
                                               values, nvalues, &_ex );
         }

         {
            /* Values above our first box */
            int ilower[2] = {-3, 2};
            int iupper[2] = {-1, 2};
            int nvalues = 3;
            int stencil_indices[1] = {4};

            bHYPRE_SStructMatrix_SetBoxValues( A, part, ilower, iupper, ndim,
                                               var, 1, stencil_indices,
                                               values, nvalues, &_ex );
         }

         {
            /* Values to the left of our second box (that do not border the
               first box). */
            int ilower[2] = { 0, 3};
            int iupper[2] = { 0, 4};
            int nvalues = 2;
            int stencil_indices[1] = {1};

            bHYPRE_SStructMatrix_SetBoxValues( A, part, ilower, iupper, ndim,
                                               var, 1, stencil_indices,
                                               values, nvalues, &_ex );
         }

         {
            /* Values above our second box */
            int ilower[2] = { 0, 4};
            int iupper[2] = { 2, 4};
            int nvalues = 3;
            int stencil_indices[1] = {4};

            bHYPRE_SStructMatrix_SetBoxValues( A, part, ilower, iupper, ndim,
                                               var, 1, stencil_indices,
                                               values, nvalues, &_ex );
         }
      }
      else if (myid == 1)
      {
         int maxnvalues = 4;
         double values[4];
         for (i = 0; i < maxnvalues; i++)
            values[i] = 0.0;

         {
            /* Values below our box */
            int ilower[2] = { 3, 1};
            int iupper[2] = { 6, 1};
            int nvalues = 4;
            int stencil_indices[1] = {3};

            bHYPRE_SStructMatrix_SetBoxValues( A, part, ilower, iupper, ndim,
                                               var, 1, stencil_indices,
                                               values, nvalues, &_ex );
         }

         {
            /* Values to the right of our box */
            int ilower[2] = { 6, 1};
            int iupper[2] = { 6, 4};
            int nvalues = 4;
            int stencil_indices[1] = {2};

            bHYPRE_SStructMatrix_SetBoxValues( A, part, ilower, iupper, ndim,
                                               var, 1, stencil_indices,
                                               values, nvalues, &_ex );
         }

         {
            /* Values above our box */
            int ilower[2] = { 3, 4};
            int iupper[2] = { 6, 4};
            int nvalues = 4;
            int stencil_indices[1] = {4};

            bHYPRE_SStructMatrix_SetBoxValues( A, part, ilower, iupper, ndim,
                                               var, 1, stencil_indices,
                                               values, nvalues, &_ex );
         }
      }

      /* This is a collective call finalizing the matrix assembly.
         The matrix is now ``ready to be used'' */
      bHYPRE_SStructMatrix_Assemble( A, &_ex );
   }


   /* 5. Set up SStruct Vectors for b and x */
   {
      int i;

      /* We have one part and one variable. */
      int part = 0;
      int var = 0;

      /* Create an empty vector object */
      b = bHYPRE_SStructVector_Create( mpi_comm, grid, &_ex );
      x = bHYPRE_SStructVector_Create( mpi_comm, grid, &_ex );

      /* As with the matrix,  set the object type for the vectors
         to be the struct type */
      object_type = HYPRE_STRUCT;
      bHYPRE_SStructVector_SetObjectType( b, object_type, &_ex );
      bHYPRE_SStructVector_SetObjectType( x, object_type, &_ex );

      /* Indicate that the vector coefficients are ready to be set */
      bHYPRE_SStructVector_Initialize( b, &_ex );
      bHYPRE_SStructVector_Initialize( x, &_ex );

      if (myid == 0)
      {
         /* Set the vector coefficients over the gridpoints in my first box */
         {
            int ilower[2] = {-3, 1};
            int iupper[2] = {-1, 2};

            int nvalues = 6;  /* 6 grid points */
            double values[6];

            for (i = 0; i < nvalues; i ++)
               values[i] = 1.0;
            bHYPRE_SStructVector_SetBoxValues( b, part, ilower, iupper, ndim,
                                               var, values, nvalues, &_ex );

            for (i = 0; i < nvalues; i ++)
               values[i] = 0.0;
            bHYPRE_SStructVector_SetBoxValues( x, part, ilower, iupper, ndim,
                                               var, values, nvalues, &_ex );
         }

         /* Set the vector coefficients over the gridpoints in my second box */
         {
            int ilower[2] = { 0, 1};
            int iupper[2] = { 2, 4};

            int nvalues = 12; /* 12 grid points */
            double values[12];

            for (i = 0; i < nvalues; i ++)
               values[i] = 1.0;
            bHYPRE_SStructVector_SetBoxValues( b, part, ilower, iupper, ndim,
                                               var, values, nvalues, &_ex );

            for (i = 0; i < nvalues; i ++)
               values[i] = 0.0;
            bHYPRE_SStructVector_SetBoxValues( x, part, ilower, iupper, ndim,
                                               var, values, nvalues, &_ex );
         }
      }
      else if (myid == 1)
      {
         /* Set the vector coefficients over the gridpoints in my box */
         {
            int ilower[2] = { 3, 1};
            int iupper[2] = { 6, 4};

            int nvalues = 16; /* 16 grid points */
            double values[16];

            for (i = 0; i < nvalues; i ++)
               values[i] = 1.0;
            bHYPRE_SStructVector_SetBoxValues( b, part, ilower, iupper, ndim,
                                               var, values, nvalues, &_ex );

            for (i = 0; i < nvalues; i ++)
               values[i] = 0.0;
            bHYPRE_SStructVector_SetBoxValues( x, part, ilower, iupper, ndim,
                                               var, values, nvalues, &_ex );
         }
      }

      /* This is a collective call finalizing the vector assembly.
         The vectors are now ``ready to be used'' */
      bHYPRE_SStructVector_Assemble( b, &_ex );
      bHYPRE_SStructVector_Assemble( x, &_ex );
   }


   /* 6. Set up and use a solver (See the Reference Manual for descriptions
      of all of the options.) */
   {
      bHYPRE_StructMatrix sA;
      bHYPRE_Vector vb;
      bHYPRE_Vector vx;
      sidl_BaseInterface dummy;
      bHYPRE_Operator opA;

      /* Because we are using a struct solver, we need to get the
         object of the matrix and vectors to pass in to the struct solvers */
      bHYPRE_SStructMatrix_GetObject( A, &dummy, &_ex );
      sA = bHYPRE_StructMatrix__cast( dummy, &_ex );
      sidl_BaseInterface_deleteRef( dummy, &_ex );
      bHYPRE_SStructVector_GetObject( b, &dummy, &_ex );
      vb = bHYPRE_Vector__cast( dummy, &_ex );
      sidl_BaseInterface_deleteRef( dummy, &_ex );
      bHYPRE_SStructVector_GetObject( x, &dummy, &_ex );
      vx = bHYPRE_Vector__cast( dummy, &_ex );
      sidl_BaseInterface_deleteRef( dummy, &_ex );

      /* Create an empty PCG Struct solver */
      opA = bHYPRE_Operator__cast( sA, &_ex );
      PCGsolver = bHYPRE_PCG_Create( mpi_comm, opA, &_ex );

      /* Set PCG parameters */
      bHYPRE_PCG_SetDoubleParameter( PCGsolver, "Tolerance", 1.0e-6, &_ex );
      bHYPRE_PCG_SetIntParameter( PCGsolver, "PrintLevel", 2, &_ex );
      bHYPRE_PCG_SetIntParameter( PCGsolver, "MaxIter", 50, &_ex );

      /* Create the Struct SMG solver for use as a preconditioner */
      SMGprecond = bHYPRE_StructSMG_Create( mpi_comm, sA, &_ex );

      /* Set SMG parameters */
      bHYPRE_StructSMG_SetIntParameter( SMGprecond, "MaxIter", 1, &_ex );
      bHYPRE_StructSMG_SetDoubleParameter( SMGprecond, "Tolerance", 0.0, &_ex );
      bHYPRE_StructSMG_SetIntParameter( SMGprecond, "ZeroGuess", 1, &_ex );
      bHYPRE_StructSMG_SetIntParameter( SMGprecond, "NumPreRelax", 1, &_ex );
      bHYPRE_StructSMG_SetIntParameter( SMGprecond, "NumPostRelax", 1, &_ex );

      /* Set preconditioner and solve */
      precond = bHYPRE_Solver__cast( SMGprecond, &_ex );
      bHYPRE_PCG_SetPreconditioner( PCGsolver, precond, &_ex );

      bHYPRE_PCG_Setup( PCGsolver, vb, vx, &_ex );
      bHYPRE_PCG_Apply( PCGsolver, vb, &vx, &_ex );

      bHYPRE_Operator_deleteRef( opA, &_ex );
      bHYPRE_Vector_deleteRef( vx, &_ex );
      bHYPRE_Vector_deleteRef( vb, &_ex );
      bHYPRE_StructMatrix_deleteRef( sA, &_ex );
   }

   /* Free memory */
   bHYPRE_Solver_deleteRef( precond, &_ex );
   bHYPRE_StructSMG_deleteRef( SMGprecond, &_ex );
   bHYPRE_PCG_deleteRef( PCGsolver, &_ex );
   bHYPRE_SStructVector_deleteRef( x, &_ex );
   bHYPRE_SStructVector_deleteRef( b, &_ex );
   bHYPRE_SStructMatrix_deleteRef( A, &_ex );
   bHYPRE_SStructGraph_deleteRef( graph, &_ex );
   bHYPRE_SStructStencil_deleteRef( stencil, &_ex );
   bHYPRE_SStructGrid_deleteRef( grid, &_ex );
   bHYPRE_MPICommunicator_deleteRef( mpi_comm, &_ex );

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}

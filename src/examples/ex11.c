/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Example 11

   Interface:    Linear-Algebraic (IJ)

   Compile with: make ex11

   Sample run:   mpirun -np 4 ex11

   Description:  This example solves the 2-D Laplacian eigenvalue
                 problem with zero boundary conditions on an nxn grid.
                 The number of unknowns is N=n^2. The standard 5-point
                 stencil is used, and we solve for the interior nodes
                 only.

                 We use the same matrix as in Examples 3 and 5.
                 The eigensolver is LOBPCG with AMG preconditioner.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#include "ex.h"

/* lobpcg stuff */
#include "HYPRE_lobpcg.h"

#ifdef HYPRE_EXVIS
#include "_hypre_utilities.h"
#include "vis.c"
#endif

#define my_min(a,b)  (((a)<(b)) ? (a) : (b))

int main (int argc, char *argv[])
{
   int i;
   int myid, num_procs;
   int N, n;
   int blockSize;

   int ilower, iupper;
   int local_size, extra;

   int vis;

   HYPRE_IJMatrix A;
   HYPRE_ParCSRMatrix parcsr_A;
   HYPRE_IJVector b;
   HYPRE_ParVector par_b;
   HYPRE_IJVector x;
   HYPRE_ParVector par_x;

   HYPRE_Solver precond, lobpcg_solver;
   mv_InterfaceInterpreter* interpreter;
   mv_MultiVectorPtr eigenvectors = NULL;
   mv_MultiVectorPtr constraints = NULL;
   HYPRE_MatvecFunctions matvec_fn;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Initialize HYPRE */
   HYPRE_Init();

   /* Print GPU info */
   /* HYPRE_PrintDeviceInfo(); */

   /* Default problem parameters */
   n = 33;
   blockSize = 10;
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
         else if ( strcmp(argv[arg_index], "-blockSize") == 0 )
         {
            arg_index++;
            blockSize = atoi(argv[arg_index++]);
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
         printf("  -n <n>              : problem size in each direction (default: 33)\n");
         printf("  -blockSize <n>      : eigenproblem block size (default: 10)\n");
         printf("  -vis                : save the solution for GLVis visualization\n");
         printf("\n");
      }

      if (print_usage)
      {
         MPI_Finalize();
         return (0);
      }
   }

   /* Preliminaries: want at least one processor per row */
   if (n*n < num_procs) n = sqrt(num_procs) + 1;
   N = n*n; /* global number of rows */

   /* Each processor knows only of its own rows - the range is denoted by ilower
      and iupper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   local_size = N/num_procs;
   extra = N - local_size*num_procs;

   ilower = local_size*myid;
   ilower += my_min(myid, extra);

   iupper = local_size*(myid+1);
   iupper += my_min(myid+1, extra);
   iupper = iupper - 1;

   /* How many rows do I have? */
   local_size = iupper - ilower + 1;

   /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);

   /* Choose a parallel csr format storage (see the User's Manual) */
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);

   /* Initialize before setting coefficients */
   HYPRE_IJMatrixInitialize(A);

   /* Now go through my local rows and set the matrix entries.
      Each row has at most 5 entries. For example, if n=3:

      A = [M -I 0; -I M -I; 0 -I M]
      M = [4 -1 0; -1 4 -1; 0 -1 4]

      Note that here we are setting one row at a time, though
      one could set all the rows together (see the User's Manual).
   */
   {
      int nnz;
      /* double values[5];
       * int cols[5]; OK to use constant-length arrays for CPUs */
      double *values = (double *) malloc(5*sizeof(double));
      int *cols = (int *) malloc(5*sizeof(int));

      for (i = ilower; i <= iupper; i++)
      {
         nnz = 0;

         /* The left identity block:position i-n */
         if ((i-n)>=0)
         {
            cols[nnz] = i-n;
            values[nnz] = -1.0;
            nnz++;
         }

         /* The left -1: position i-1 */
         if (i%n)
         {
            cols[nnz] = i-1;
            values[nnz] = -1.0;
            nnz++;
         }

         /* Set the diagonal: position i */
         cols[nnz] = i;
         values[nnz] = 4.0;
         nnz++;

         /* The right -1: position i+1 */
         if ((i+1)%n)
         {
            cols[nnz] = i+1;
            values[nnz] = -1.0;
            nnz++;
         }

         /* The right identity block:position i+n */
         if ((i+n)< N)
         {
            cols[nnz] = i+n;
            values[nnz] = -1.0;
            nnz++;
         }

         /* Set the values for row i */
         HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, cols, values);
      }

      free(values);
      free(cols);
   }

   /* Assemble after setting the coefficients */
   HYPRE_IJMatrixAssemble(A);
   /* Get the parcsr matrix object to use */
   HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);

   /* Create sample rhs and solution vectors */
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);
   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorGetObject(b, (void **) &par_b);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);
   HYPRE_IJVectorAssemble(x);
   HYPRE_IJVectorGetObject(x, (void **) &par_x);

   /* Create a preconditioner and solve the eigenproblem */

   /* AMG preconditioner */
   {
      HYPRE_BoomerAMGCreate(&precond);
      HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info */
      HYPRE_BoomerAMGSetNumSweeps(precond, 2); /* 2 sweeps of smoothing */
      HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
      HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */
   }

   /* LOBPCG eigensolver */
   {
      double mytime = 0.0;
      double walltime = 0.0;

      int maxIterations = 100; /* maximum number of iterations */
      int pcgMode = 1;         /* use rhs as initial guess for inner pcg iterations */
      int verbosity = 1;       /* print iterations info */
      double tol = 1.e-8;      /* absolute tolerance (all eigenvalues) */
      int lobpcgSeed = 775;    /* random seed */

      double *eigenvalues = NULL;

      if (myid != 0)
         verbosity = 0;

      /* define an interpreter for the ParCSR interface */
      interpreter = (mv_InterfaceInterpreter *) calloc(1, sizeof(mv_InterfaceInterpreter));
      HYPRE_ParCSRSetupInterpreter(interpreter);
      HYPRE_ParCSRSetupMatvec(&matvec_fn);

      /* eigenvectors - create a multivector */
      eigenvectors =
         mv_MultiVectorCreateFromSampleVector(interpreter, blockSize, par_x);
      mv_MultiVectorSetRandom (eigenvectors, lobpcgSeed);

      /* eigenvalues - allocate space */
      eigenvalues = (double*) calloc( blockSize, sizeof(double) );

      HYPRE_LOBPCGCreate(interpreter, &matvec_fn, &lobpcg_solver);
      HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
      HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
      HYPRE_LOBPCGSetTol(lobpcg_solver, tol);
      HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);

      /* use a preconditioner */
      HYPRE_LOBPCGSetPrecond(lobpcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                             precond);

      HYPRE_LOBPCGSetup(lobpcg_solver, (HYPRE_Matrix)parcsr_A,
                        (HYPRE_Vector)par_b, (HYPRE_Vector)par_x);

      mytime -= MPI_Wtime();

      HYPRE_LOBPCGSolve(lobpcg_solver, constraints, eigenvectors, eigenvalues );

      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nLOBPCG Solve time = %f seconds\n\n", walltime);
      }

      /* clean-up */
      HYPRE_BoomerAMGDestroy(precond);
      HYPRE_LOBPCGDestroy(lobpcg_solver);
      free(eigenvalues);
      free(interpreter);
   }

   /* Save the solution for GLVis visualization, see vis/glvis-ex11.sh */
   if (vis)
   {
#ifdef HYPRE_EXVIS
      FILE *file;
      char filename[255];

      int nvalues = local_size;
      double *values;

      /* eigenvectors - get a pointer */
      mv_TempMultiVector* tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors);
      HYPRE_ParVector*    pvx = (HYPRE_ParVector*)(tmp -> vector);

      /* get the local solution */
      values = hypre_VectorData(hypre_ParVectorLocalVector(
                                   (hypre_ParVector*)pvx[blockSize-1]));

      sprintf(filename, "%s.%06d", "vis/ex11.sol", myid);
      if ((file = fopen(filename, "w")) == NULL)
      {
         printf("Error: can't open output file %s\n", filename);
         MPI_Finalize();
         exit(1);
      }

      /* save solution */
      for (i = 0; i < nvalues; i++)
         fprintf(file, "%.14e\n", values[i]);

      fflush(file);
      fclose(file);

      /* save global finite element mesh */
      if (myid == 0)
         GLVis_PrintGlobalSquareMesh("vis/ex11.mesh", n-1);
#endif
   }

   /* Clean up */
   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);

   /* Finalize HYPRE */
   HYPRE_Finalize();

   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}

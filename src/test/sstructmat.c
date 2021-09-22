/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "sstruct_helpers.h"
#include "_hypre_sstruct_mv.h"

/* Tests prototypes */
HYPRE_Int test_SStructMatmult( HYPRE_Int nmatrices , HYPRE_SStructMatrix *ss_A , HYPRE_IJMatrix *ij_A , HYPRE_Int nterms , HYPRE_Int *terms , HYPRE_Int *trans , HYPRE_SStructMatrix *ss_M_ptr);

/*--------------------------------------------------------------------------
 * Print usage info
 *--------------------------------------------------------------------------*/

HYPRE_Int
PrintUsage( char      *progname,
            HYPRE_Int  myid )
{
   if ( myid == 0 )
   {
      hypre_printf("\n");
      hypre_printf("Usage: %s [-in <filename>] [<options>]\n", progname);
      hypre_printf("\n");
      hypre_printf("  -in <nfiles> <filenames> : input file(s) \n");
      hypre_printf("\n");
      hypre_printf("  -pt <pt1> <pt2> ...  : set part(s) for subsequent options\n");
      hypre_printf("  -pooldist <p> ...    : pool distribution to use\n");
      hypre_printf("  -P <Px> <Py> ...     : refine and distribute part(s)\n");
      hypre_printf("  -r <rx> <ry> ...     : refine part(s)\n");
      hypre_printf("  -b <bx> <by> ...     : refine and block part(s)\n");
      hypre_printf("\n");
      hypre_printf("  -mat-vec <A> <x> <y> : compute A*x + y\n");
      hypre_printf("  -matTvec <A> <x> <y> : compute A^T*x + y\n");
      hypre_printf("  -ab      <a> <b>     : alpha/beta values for matvec (default = 1)\n");
      hypre_printf("\n");
      hypre_printf("  -mat-mat <n> <A>[T] <B>[T] ... : compute A*B*... or A^T*B*..., etc. \n");
      hypre_printf("                                 : for n possibly transposed matrices \n");
      hypre_printf("                                 : example P^T*A*P: -mat-mat 3 1T 0 1 \n");
      hypre_printf("\n");
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Test driver for semi-structured matrix interface
 *--------------------------------------------------------------------------*/

hypre_int
main( hypre_int  argc,
      char      *argv[] )
{
   MPI_Comm                comm = hypre_MPI_COMM_WORLD;

   char                  **infile;
   ProblemData            *global_data;
   ProblemData            *data;
   HYPRE_Int               nmatrices;

   HYPRE_SStructGrid      *grid;
   HYPRE_SStructStencil  **stencils;
   HYPRE_SStructGraph     *graph;
   HYPRE_SStructMatrix    *ss_A;
   HYPRE_IJMatrix         *ij_A;
   HYPRE_SStructMatrix     ss_M;

   /* Driver options */
   HYPRE_Int               nparts;
   HYPRE_Int               pooldist;
   HYPRE_Int              *parts;
   Index                  *refine;
   Index                  *distribute;
   Index                  *block;
   HYPRE_Int               print;
   HYPRE_Int               nterms;
   HYPRE_Int              *terms = NULL;
   HYPRE_Int              *trans = NULL;

   /* Local variables */
   HYPRE_Int               myid, nprocs;
   HYPRE_Int               arg_index;
   HYPRE_Int               time_index;
   HYPRE_Int               i, j, k, s, part;
   HYPRE_Int               do_matmult;
   HYPRE_Int               ierr;
   char                    infile_default[50] = "sstruct.in.cubes4_7pt";
   char                    matname[64];
   char                    heading[64];
   char                    transposechar;

   /*-----------------------------------------------------------
    * Initialize libraries
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_size(comm, &nprocs);
   hypre_MPI_Comm_rank(comm, &myid);

   /* GPU Device binding - Must be done before HYPRE_Init() */
   hypre_bind_device(myid, nprocs, comm);

   /* Initialize hypre */
   HYPRE_Init();

   /* Initialize some input parameters */
   nmatrices = 1;
   infile = hypre_TAlloc(char *, nmatrices, HYPRE_MEMORY_HOST);
   infile[0] = infile_default;

   /*-----------------------------------------------------------
    * Read input file
    *-----------------------------------------------------------*/
   arg_index = 1;
   if (argc > 1)
   {
      if ( strcmp(argv[arg_index], "-in") == 0 )
      {
         arg_index++;
         nmatrices = atoi(argv[arg_index++]);
         infile = hypre_TReAlloc(infile, char *, nmatrices, HYPRE_MEMORY_HOST);

         for (i = 0; i < nmatrices; i++)
         {
            infile[i] = argv[arg_index++];
         }
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 ||
                strcmp(argv[arg_index], "-h") == 0 )
      {
         PrintUsage(argv[0], myid);
         exit(1);
      }
      else
      {
         if (!myid)
         {
            hypre_printf("First command line argument must be the input file(s).\n");
            exit(1);
         }
      }
   }

   /*-----------------------------------------------------------
    * Read data from input file
    *-----------------------------------------------------------*/
   global_data = hypre_TAlloc(ProblemData, nmatrices, HYPRE_MEMORY_HOST);
   for (i = 0; i < nmatrices; i++)
   {
      ReadData(infile[i], &global_data[i]);
   }
   nparts = global_data[0].nparts;

   /* Allocate and initialize data */
   data = hypre_TAlloc(ProblemData, nmatrices, HYPRE_MEMORY_HOST);

   grid = hypre_TAlloc(HYPRE_SStructGrid, nmatrices, HYPRE_MEMORY_HOST);
   graph = hypre_TAlloc(HYPRE_SStructGraph, nmatrices, HYPRE_MEMORY_HOST);
   stencils = hypre_TAlloc(HYPRE_SStructStencil *, nmatrices, HYPRE_MEMORY_HOST);
   ss_A = hypre_TAlloc(HYPRE_SStructMatrix, nmatrices, HYPRE_MEMORY_HOST);
   ij_A = hypre_TAlloc(HYPRE_IJMatrix, nmatrices, HYPRE_MEMORY_HOST);

   parts = hypre_TAlloc(HYPRE_Int, nparts, HYPRE_MEMORY_HOST);
   block = hypre_TAlloc(Index, nparts, HYPRE_MEMORY_HOST);
   refine = hypre_TAlloc(Index, nparts, HYPRE_MEMORY_HOST);
   distribute = hypre_TAlloc(Index, nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      parts[part] = part;
      for (j = 0; j < HYPRE_MAXDIM; j++)
      {
         block[part][j] = 1;
         refine[part][j] = 1;
         distribute[part][j] = 1;
      }
   }

   /* Initialize input arguments */
   pooldist = 0;
   print = 0;
   do_matmult = 0;

   /*-----------------------------------------------------------
    * Parse other command line options
    *-----------------------------------------------------------*/
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-pt") == 0 )
      {
         arg_index++;
         nparts = 0;
         while ( strncmp(argv[arg_index], "-", 1) != 0 )
         {
            parts[nparts++] = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-pooldist") == 0 )
      {
         arg_index++;
         pooldist = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-r") == 0 )
      {
         arg_index++;
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            k = arg_index;
            for (j = 0; j < HYPRE_MAXDIM; j++)
            {
               refine[part][j] = atoi(argv[k++]);
            }
         }
         arg_index += HYPRE_MAXDIM;
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            k = arg_index;
            for (j = 0; j < HYPRE_MAXDIM; j++)
            {
               distribute[part][j] = atoi(argv[k++]);
            }
         }
         arg_index += HYPRE_MAXDIM;
      }
      else if ( strcmp(argv[arg_index], "-b") == 0 )
      {
         arg_index++;
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            k = arg_index;
            for (j = 0; j < HYPRE_MAXDIM; j++)
            {
               block[part][j] = atoi(argv[k++]);
            }
         }
         arg_index += HYPRE_MAXDIM;
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print = 1;
      }
      else if ( strcmp(argv[arg_index], "-mat-mat") == 0 )
      {
         arg_index++;
         do_matmult = 1;
         nterms = atoi(argv[arg_index++]);
         terms = hypre_CTAlloc(HYPRE_Int, nterms, HYPRE_MEMORY_HOST);
         trans = hypre_CTAlloc(HYPRE_Int, nterms, HYPRE_MEMORY_HOST);
         for (i = 0; i < nterms; i++)
         {
            transposechar = ' ';
            hypre_sscanf(argv[arg_index++], "%d%c", &terms[i], &transposechar);
            if (transposechar == 'T')
            {
               trans[i] = 1;
            }
         }
      }
   } /* while (arg_index < argc) */

   /*-----------------------------------------------------------
    * Distribute data
    *-----------------------------------------------------------*/
   for (i = 0; i < nmatrices; i++)
   {
      DistributeData(global_data[i], pooldist, refine, distribute, block,
                     nprocs, myid, &data[i]);
   }

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   hypre_MPI_Barrier(comm);

   /*-----------------------------------------------------------
    * Set up semi-structured matrices
    *-----------------------------------------------------------*/
   for (i = 0; i < nmatrices; i++)
   {
      hypre_sprintf(heading, "Build matrix #%d", i);
      time_index = hypre_InitializeTiming(heading);
      hypre_BeginTiming(time_index);

      BuildGrid(comm, data[i], &grid[i]);
      BuildStencils(data[i], grid[i], &stencils[i]);
      BuildGraph(comm, data[i], grid[i], HYPRE_SSTRUCT, stencils[i], &graph[i]);
      BuildMatrix(comm, data[i], grid[i], stencils[i], graph[i], &ss_A[i]);

      hypre_EndTiming(time_index);
      hypre_PrintTiming(heading, hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Convert to IJMatrix */
      HYPRE_SStructMatrixToIJMatrix(ss_A[i], 0, &ij_A[i]);

      if (print)
      {
         hypre_sprintf(matname, "sstruct.out.mat.m%02d", i);
         HYPRE_SStructMatrixPrint(matname, ss_A[i], 0);

         hypre_sprintf(matname, "ij.out.mat.m%02d", i);
         HYPRE_IJMatrixPrint(ij_A[i], matname);
      }
   }

   /*-----------------------------------------------------------
    * Matrix-matrix multiply
    *-----------------------------------------------------------*/

   if (do_matmult)
   {
      /* Sanity check */
      if (nterms < 2)
      {
         if (!myid)
         {
            hypre_printf("Need at least 2 terms for performing matrix/matrix product!\n");
         }
      }
      else
      {
         hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
         time_index = hypre_InitializeTiming("Matrix-matrix multiply");
         hypre_BeginTiming(time_index);

         ierr = test_SStructMatmult(nmatrices, ss_A, ij_A, nterms, terms, trans, &ss_M);
         if (ierr)
         {
            hypre_printf("test_SStructMatmult failed!\n\n");
         }
         else
         {
            hypre_printf("test_SStructMatmult passed!\n\n");
         }

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Matrix-matrix multiply", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         if (print)
         {
            HYPRE_SStructMatrixPrint("sstruct.out.M", ss_M, 0);
         }

         HYPRE_SStructMatrixDestroy(ss_M);
      }
   }


   /*-----------------------------------------------------------
    * Free memory
    *-----------------------------------------------------------*/
   for (i = 0; i < nmatrices; i++)
   {
      for (s = 0; s < data[i].nstencils; s++)
      {
         HYPRE_SStructStencilDestroy(stencils[i][s]);
      }
      hypre_TFree(stencils[i], HYPRE_MEMORY_HOST);
      HYPRE_SStructGridDestroy(grid[i]);
      HYPRE_SStructGraphDestroy(graph[i]);
      HYPRE_SStructMatrixDestroy(ss_A[i]);
      HYPRE_IJMatrixDestroy(ij_A[i]);
   }
   hypre_TFree(stencils, HYPRE_MEMORY_HOST);
   hypre_TFree(grid, HYPRE_MEMORY_HOST);
   hypre_TFree(graph, HYPRE_MEMORY_HOST);
   hypre_TFree(ss_A, HYPRE_MEMORY_HOST);
   hypre_TFree(ij_A, HYPRE_MEMORY_HOST);

   for (i = 0; i < nmatrices; i++)
   {
      DestroyData(data[i]);
   }
   hypre_TFree(data, HYPRE_MEMORY_HOST);
   hypre_TFree(global_data, HYPRE_MEMORY_HOST);
   hypre_TFree(parts, HYPRE_MEMORY_HOST);
   hypre_TFree(block, HYPRE_MEMORY_HOST);
   hypre_TFree(refine, HYPRE_MEMORY_HOST);
   hypre_TFree(distribute, HYPRE_MEMORY_HOST);
   hypre_TFree(terms, HYPRE_MEMORY_HOST);
   hypre_TFree(trans, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------
    * Finalize libraries
    *-----------------------------------------------------------*/
   HYPRE_Finalize();
   hypre_MPI_Finalize();

   return 0;
}

HYPRE_Int
test_SStructMatmult( HYPRE_Int            nmatrices,
                     HYPRE_SStructMatrix *ss_A,
                     HYPRE_IJMatrix      *ij_A,
                     HYPRE_Int            nterms,
                     HYPRE_Int           *terms,
                     HYPRE_Int           *trans,
                     HYPRE_SStructMatrix *ss_M_ptr )
{
   MPI_Comm             comm = hypre_SStructMatrixComm(ss_A[0]);

   HYPRE_SStructMatrix  ss_M;
   HYPRE_IJMatrix       ij_M;
   HYPRE_ParCSRMatrix   par_A[3];
   HYPRE_ParCSRMatrix   par_E;
   HYPRE_ParCSRMatrix   par_M;

   HYPRE_Int            m, t;
   HYPRE_Complex        alpha = 1.0;
   HYPRE_Complex        beta = -1.0;
   HYPRE_Int            myid;
   HYPRE_Real           norm_M, norm_E;
   HYPRE_Int            ierr = 0;

   hypre_MPI_Comm_rank(comm, &myid);

   /* Compute semi-structured matrices product */
   hypre_SStructMatmult(nmatrices, ss_A, nterms, terms, trans, &ss_M);
   HYPRE_SStructMatrixToIJMatrix(ss_M, 0, &ij_M);

   /* Compute IJ matrices product */
   m = terms[nterms-1];
   HYPRE_IJMatrixGetObject(ij_A[m], (void **) &par_A[0]);
   for (t = (nterms - 2); t >= 0; t--)
   {
      m = terms[t];
      HYPRE_IJMatrixGetObject(ij_A[m], (void **) &par_A[1]);

      if (trans[t])
      {
         par_A[2] = hypre_ParTMatmul(par_A[1], par_A[0]);
      }
      else
      {
         par_A[2] = hypre_ParMatmul(par_A[1], par_A[0]);
      }

      /* Free temporary work matrix */
      if (t < (nterms - 2))
      {
         HYPRE_ParCSRMatrixDestroy(par_A[0]);
      }

      /* Update pointers */
      par_A[0] = par_A[2];
   }

   /* Move diagonal coefficients to first positions */
   hypre_ParCSRMatrixReorder(par_A[0]);

   /* Compute error */
   HYPRE_IJMatrixGetObject(ij_M, (void **) &par_M);
   hypre_ParCSRMatrixAdd(alpha, par_A[0], beta, par_M, &par_E);
   hypre_ParCSRMatrixInfNorm(par_M, &norm_M);
   hypre_ParCSRMatrixInfNorm(par_E, &norm_E);

   if (!myid)
   {
      hypre_printf("[test_SStructMatmult]: Error norm = %e\n", norm_E);
      if (norm_E > HYPRE_REAL_EPSILON*norm_M)
      {
         ierr = 1;
         hypre_ParCSRMatrixPrintIJ(par_M, 0, 0, "par_Mss");
         hypre_ParCSRMatrixPrintIJ(par_A[0], 0, 0, "par_Mij");
         hypre_ParCSRMatrixPrintIJ(par_E, 0, 0, "par_E");
      }
   }

   /* Free memory */
   HYPRE_ParCSRMatrixDestroy(par_A[0]);
   HYPRE_ParCSRMatrixDestroy(par_E);
   HYPRE_IJMatrixDestroy(ij_M);

   *ss_M_ptr = ss_M;

   return ierr;
}

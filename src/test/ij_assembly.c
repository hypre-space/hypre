/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (IJ).
 *
 * It tests the assembly phase of an IJ matrix in both CPU and GPU.
 *--------------------------------------------------------------------------*/

#include "HYPRE.h"
#include "_hypre_IJ_mv.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.h"

HYPRE_Int hypre_GeneratePartitioning(HYPRE_BigInt length, HYPRE_Int num_procs, HYPRE_BigInt **part_ptr);
HYPRE_Int buildMatrixEntries(MPI_Comm comm,
                             HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz,
                             HYPRE_Int Px, HYPRE_Int Py, HYPRE_Int Pz,
                             HYPRE_Int cx, HYPRE_Int cy, HYPRE_Int cz,
                             HYPRE_BigInt *ilower, HYPRE_BigInt *iupper,
                             HYPRE_Int *nrows, HYPRE_BigInt *num_nonzeros,
                             HYPRE_Int **nnzrow_ptr, HYPRE_BigInt **rows_ptr,
                             HYPRE_BigInt **cols_ptr, HYPRE_Real **coefs_ptr);
HYPRE_Int buildRefMatrix(MPI_Comm comm,
                         HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                         HYPRE_BigInt num_nonzeros, HYPRE_Int *nnzrow,
                         HYPRE_BigInt *rows, HYPRE_BigInt *cols,
                         HYPRE_Real *coefs, HYPRE_IJMatrix *ij_ref_ptr);
HYPRE_Int checkMatrix(HYPRE_IJMatrix ij_ref, HYPRE_IJMatrix ij_A);
HYPRE_Int test_SetSet(MPI_Comm comm, HYPRE_Int memory_loc,
                      HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                      HYPRE_Int nrows, HYPRE_BigInt num_nonzeros,
                      HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
                      HYPRE_BigInt *rows, HYPRE_BigInt *cols,
                      HYPRE_Real *coefs, HYPRE_IJMatrix *ij_A);
HYPRE_Int test_AddSet(MPI_Comm comm, HYPRE_Int memory_loc,
                      HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                      HYPRE_Int nrows, HYPRE_BigInt num_nonzeros,
                      HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
                      HYPRE_BigInt *rows, HYPRE_BigInt *cols,
                      HYPRE_Real *coefs, HYPRE_IJMatrix *ij_A);
HYPRE_Int test_SetAddSet(MPI_Comm comm, HYPRE_Int memory_loc,
                         HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                         HYPRE_Int nrows, HYPRE_BigInt num_nonzeros,
                         HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
                         HYPRE_BigInt *rows, HYPRE_BigInt *cols,
                         HYPRE_Real *coefs, HYPRE_IJMatrix *ij_A);

HYPRE_Int
main( HYPRE_Int  argc,
      char      *argv[] )
{
   MPI_Comm                  comm = hypre_MPI_COMM_WORLD;
   HYPRE_Int                 num_procs;
   HYPRE_Int                 myid;
   HYPRE_Int                 arg_index;
   HYPRE_Int                 time_index;
   HYPRE_Int                 print_usage;
   HYPRE_Int                 memory_loc;
   char                      memory_loc_name[32];

   HYPRE_Int                 nrows;
   HYPRE_BigInt              num_nonzeros;
   HYPRE_BigInt              ilower, iupper;
   HYPRE_Int                *nnzrow, *h_nnzrow, *d_nnzrow;
   HYPRE_BigInt             *rows,   *h_rows,   *d_rows;
   HYPRE_BigInt             *cols,   *h_cols,   *d_cols;
   HYPRE_Real               *coefs,  *h_coefs,  *d_coefs;
   HYPRE_IJMatrix            ij_A;
   HYPRE_IJMatrix            ij_ref;

   // Driver input parameters
   HYPRE_Int                 Px, Py, Pz;
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Real                cx, cy, cz;
   HYPRE_Int                 nchunks;

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

   /* Initialize Hypre */
   HYPRE_Init(argc, argv);
   hypre_SetExecPolicy(HYPRE_EXEC_DEVICE);
/* #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP) */
/*    memory_loc = hypre_handle->no_cuda_um ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_SHARED; */
/* #else */
/*    memory_loc = HYPRE_MEMORY_SHARED; */
/* #endif */
   memory_loc = HYPRE_MEMORY_SHARED;

   switch (memory_loc)
   {
      case HYPRE_MEMORY_UNSET:
         hypre_sprintf(memory_loc_name, "Unset"); break;

      case HYPRE_MEMORY_DEVICE:
         hypre_sprintf(memory_loc_name, "Device"); break;

      case HYPRE_MEMORY_HOST:
         hypre_sprintf(memory_loc_name, "Host"); break;

      case HYPRE_MEMORY_SHARED:
         hypre_sprintf(memory_loc_name, "Shared"); break;

      case HYPRE_MEMORY_HOST_PINNED:
         hypre_sprintf(memory_loc_name, "Host pinned"); break;
   }

   /* Initialize Hypre: must be the first Hypre function to call */
   time_index = hypre_InitializeTiming("Hypre init");
   hypre_BeginTiming(time_index);
   HYPRE_Init(argc, argv);
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Hypre init times", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Set default parameters
    *-----------------------------------------------------------*/
   Px = num_procs; Py = 1;   Pz = 1;
   nx = 10;        ny = 10;  nz = 10;
   cx = 1.0;       cy = 1.0; cz = 1.0;
   nchunks = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   print_usage = 0;
   arg_index = 1;
   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         Px = atoi(argv[arg_index++]);
         Py = atoi(argv[arg_index++]);
         Pz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx  = atoi(argv[arg_index++]);
         ny  = atoi(argv[arg_index++]);
         nz  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nchunks") == 0 )
      {
         arg_index++;
         nchunks = atoi(argv[arg_index++]);
      }
      else
      {
         print_usage = 1;
      }
   }

   /*-----------------------------------------------------------
    * Safety checks
    *-----------------------------------------------------------*/
   if (Px*Py*Pz != num_procs)
   {
      hypre_printf("Px x Py x Pz is different than the number of MPI processes");
      return (-1);
   }

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/
   if ( print_usage )
   {
      if ( myid == 0 )
      {
         hypre_printf("\n");
         hypre_printf("Usage: %s [<options>]\n", argv[0]);
         hypre_printf("\n");
         hypre_printf("      -n <nx> <ny> <nz>    : total problem size \n");
         hypre_printf("      -P <Px> <Py> <Pz>    : processor topology\n");
         hypre_printf("      -c <cx> <cy> <cz>    : diffusion coefficients\n");
         hypre_printf("\n");
      }

      return (0);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
   if (myid == 0)
   {
      hypre_printf("  Memory location: %s\n", memory_loc_name);
      hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", Px, Py, Pz);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf("\n");
   }

   /*-----------------------------------------------------------
    * Build matrix entries
    *-----------------------------------------------------------*/
   buildMatrixEntries(comm, nx, ny, nz, Px, Py, Pz, cx, cy, cz,
                      &ilower, &iupper, &nrows, &num_nonzeros,
                      &h_nnzrow, &h_rows, &h_cols, &h_coefs);
   switch (hypre_GetActualMemLocation(memory_loc))
   {
      case HYPRE_MEMORY_DEVICE:
         d_nnzrow = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_DEVICE);
         d_rows   = hypre_TAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_DEVICE);
         d_cols   = hypre_TAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_DEVICE);
         d_coefs  = hypre_TAlloc(HYPRE_Real, num_nonzeros, HYPRE_MEMORY_DEVICE);

         hypre_TMemcpy(d_nnzrow, h_nnzrow, HYPRE_Int, nrows, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_rows, h_rows, HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_cols, h_cols, HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_coefs, h_coefs, HYPRE_Real, num_nonzeros, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

         nnzrow = d_nnzrow;
         rows   = d_rows;
         cols   = d_cols;
         coefs  = d_coefs;
         break;

      case HYPRE_MEMORY_SHARED:
         d_nnzrow = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_SHARED);
         d_rows   = hypre_TAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_SHARED);
         d_cols   = hypre_TAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_SHARED);
         d_coefs  = hypre_TAlloc(HYPRE_Real, num_nonzeros, HYPRE_MEMORY_SHARED);

         hypre_TMemcpy(d_nnzrow, h_nnzrow, HYPRE_Int, nrows, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_rows, h_rows, HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_cols, h_cols, HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_coefs, h_coefs, HYPRE_Real, num_nonzeros, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);

         nnzrow = d_nnzrow;
         rows   = d_rows;
         cols   = d_cols;
         coefs  = d_coefs;
         break;

      case HYPRE_MEMORY_HOST:
         nnzrow = h_nnzrow;
         rows   = h_rows;
         cols   = h_cols;
         coefs  = h_coefs;
         break;
   }

   /* Build Reference matrix */
   buildRefMatrix(comm, ilower, iupper, num_nonzeros, nnzrow,
                  rows, cols, coefs, &ij_ref);

   /*-----------------------------------------------------------
    * Test different Set/Add combinations
    *-----------------------------------------------------------*/
   time_index = hypre_InitializeTiming("Test Set/Set");
   hypre_BeginTiming(time_index);
   test_SetSet(comm, memory_loc, ilower, iupper, nrows, num_nonzeros,
               nchunks, h_nnzrow, nnzrow, rows, cols, coefs, &ij_A);
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Test Set/Set", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   checkMatrix(ij_A, ij_ref);
   HYPRE_IJMatrixPrint(ij_A, "SetSet");

   /* Test Add/Set */
   time_index = hypre_InitializeTiming("Test Add/Set");
   hypre_BeginTiming(time_index);
   test_AddSet(comm, memory_loc, ilower, iupper, nrows, num_nonzeros,
               nchunks, h_nnzrow, nnzrow, rows, cols, coefs, &ij_A);
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Test Add/Set", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   checkMatrix(ij_A, ij_ref);
   //HYPRE_IJMatrixPrint(ij_A, "AddSet");

   /* Test Set/Add/Set */
   time_index = hypre_InitializeTiming("Test Set/Add/Set");
   hypre_BeginTiming(time_index);
   test_AddSet(comm, memory_loc, ilower, iupper, nrows, num_nonzeros,
               nchunks, h_nnzrow, nnzrow, rows, cols, coefs, &ij_A);
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Test Set/Add/Set", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   checkMatrix(ij_A, ij_ref);
   //HYPRE_IJMatrixPrint(ij_A, "SetAddSet");

   /*-----------------------------------------------------------
    * Free memory
    *-----------------------------------------------------------*/
   hypre_TFree(nnzrow, memory_loc);
   hypre_TFree(rows, memory_loc);
   hypre_TFree(cols, memory_loc);
   hypre_TFree(coefs, memory_loc);
   HYPRE_IJMatrixDestroy(ij_A);
   HYPRE_IJMatrixDestroy(ij_ref);

   /* Finalize Hypre */
   HYPRE_Finalize();

   /* Finalize MPI */
   hypre_MPI_Finalize();

   /* when using cuda-memcheck --leak-check full, uncomment this */
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   cudaDeviceReset();
#endif

   return (0);
}

HYPRE_Int
buildMatrixEntries(MPI_Comm comm,
                   HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz,
                   HYPRE_Int Px, HYPRE_Int Py, HYPRE_Int Pz,
                   HYPRE_Int cx, HYPRE_Int cy, HYPRE_Int cz,
                   HYPRE_BigInt *ilower, HYPRE_BigInt *iupper,
                   HYPRE_Int *nrows, HYPRE_BigInt *num_nonzeros,
                   HYPRE_Int **nnzrow_ptr, HYPRE_BigInt **rows_ptr,
                   HYPRE_BigInt **cols_ptr, HYPRE_Real **coefs_ptr)
{
   HYPRE_Int        num_procs;
   HYPRE_Int        myid;

   HYPRE_Int       *nnzrow;
   HYPRE_BigInt    *rows;
   HYPRE_BigInt    *cols;
   HYPRE_Real      *coefs;

   HYPRE_BigInt    *nx_part;
   HYPRE_BigInt    *ny_part;
   HYPRE_BigInt    *nz_part;

   HYPRE_BigInt     nx_local;
   HYPRE_BigInt     ny_local;
   HYPRE_BigInt     nz_local;
   HYPRE_BigInt     nxy_local;
   HYPRE_BigInt     nyz_local;

   HYPRE_BigInt     big_row_index;
   HYPRE_Int        row_index_local;
   HYPRE_Int        cnt;

   HYPRE_Int        nxy;
   HYPRE_Int        ip, iq, ir;
   HYPRE_BigInt     ix, iy, iz;
   HYPRE_BigInt     big_nx, big_ny, big_nxy;

   HYPRE_Int        stencil_size;
   HYPRE_Real       values[4];

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

   // Generate partitioning
   hypre_GeneratePartitioning(nx, Px, &nx_part);
   hypre_GeneratePartitioning(ny, Py, &ny_part);
   hypre_GeneratePartitioning(nz, Pz, &nz_part);

   /* compute p, q, r from Px, Py, Pz and myid */
   ip = myid % Px;
   iq = (( myid - ip)/Px) % Py;
   ir = ( myid - ip - Px*iq)/( Px*Py );

   nx_local  = nx_part[ip+1] - nx_part[ip];
   ny_local  = ny_part[iq+1] - ny_part[iq];
   nz_local  = nz_part[ir+1] - nz_part[ir];
   nxy_local = nx_local*ny_local;
   nyz_local = ny_local*nz_local;
   nxy     = nx*ny;
   big_nx  = (HYPRE_BigInt) nx;
   big_ny  = (HYPRE_BigInt) ny;
   big_nxy = (HYPRE_BigInt) nxy;

   values[0] = 0;   values[1] = -cx;
   values[2] = -cy; values[3] = -cz;
   stencil_size = 7;
   if (nx == 0)
   {
      stencil_size -= 2;
   }
   else
   {
      values[0] += 2.0*cx;
   }
   if (ny == 0)
   {
      stencil_size -= 2;
   }
   else
   {
      values[0] += 2.0*cy;
   }
   if (nz == 0)
   {
      stencil_size -= 2;
   }
   else
   {
      values[0] += 2.0*cz;
   }

   *nrows  = nx_local*ny_local*nz_local;
   *ilower = nz_part[ir]*big_nx*big_ny + ny_part[iq]*big_nx*nz_local + nx_part[ip]*nyz_local;
   *iupper = *ilower + *nrows - 1;

   // Test Set/Add values before assembly
   *num_nonzeros = (*nrows)*stencil_size;
   nnzrow = hypre_CTAlloc(HYPRE_Int, *nrows, HYPRE_MEMORY_HOST);
   rows   = hypre_CTAlloc(HYPRE_BigInt, *num_nonzeros, HYPRE_MEMORY_HOST);
   cols   = hypre_CTAlloc(HYPRE_BigInt, *num_nonzeros, HYPRE_MEMORY_HOST);
   coefs  = hypre_CTAlloc(HYPRE_Real, *num_nonzeros, HYPRE_MEMORY_HOST);

   cnt = 0;
   for (iz = nz_part[ir]; iz < nz_part[ir+1]; iz++)
   {
      for (iy = ny_part[iq]; iy < ny_part[iq+1]; iy++)
      {
         for (ix = nx_part[ip]; ix < nx_part[ip+1]; ix++)
         {
            row_index_local = (iz - nz_part[ir])*nxy_local +
                              (iy - ny_part[iq])*nx_local +
                              (ix - nx_part[ip]);
            big_row_index   = *ilower + (HYPRE_BigInt) row_index_local;
            rows[row_index_local] = big_row_index;

            /* Center coefficient */
            cols[cnt]  = big_row_index;
            coefs[cnt] = values[0];
            nnzrow[row_index_local]++;
            cnt++;

            /* Bottom coefficient */
            if (iz > 0)
            {
               cols[cnt]  = big_row_index - big_nxy;
               coefs[cnt] = values[3];
               nnzrow[row_index_local]++;
               cnt++;
            }

            /* Top coefficient */
            if (iz < (nz - 1))
            {
               rows[cnt]  = big_row_index;
               cols[cnt]  = big_row_index + big_nxy;
               coefs[cnt] = values[3];
               nnzrow[row_index_local]++;
               cnt++;
            }

            /* South coefficient */
            if (iy > 0)
            {
               rows[cnt]  = big_row_index;
               cols[cnt]  = big_row_index - big_nx;
               coefs[cnt] = values[2];
               nnzrow[row_index_local]++;
               cnt++;
            }

            /* North coefficient */
            if (iy < (ny - 1))
            {
               rows[cnt]  = big_row_index;
               cols[cnt]  = big_row_index + big_nx;
               coefs[cnt] = values[2];
               nnzrow[row_index_local]++;
               cnt++;
            }

            /* West coefficient */
            if (ix > 0)
            {
               rows[cnt]  = big_row_index;
               cols[cnt]  = big_row_index - 1;
               coefs[cnt] = values[1];
               nnzrow[row_index_local]++;
               cnt++;
            }

            /* East coefficient */
            if (ix < (nx - 1))
            {
               rows[cnt]  = big_row_index;
               cols[cnt]  = big_row_index + 1;
               coefs[cnt] = values[1];
               nnzrow[row_index_local]++;
               cnt++;
            }
         }
      }
   }

   // Set pointers
   *num_nonzeros = cnt;
   *nnzrow_ptr   = nnzrow;
   *rows_ptr     = rows;
   *cols_ptr     = cols;
   *coefs_ptr    = coefs;

   /* Free memory */
   hypre_TFree(nx_part, HYPRE_MEMORY_HOST);
   hypre_TFree(ny_part, HYPRE_MEMORY_HOST);
   hypre_TFree(nz_part, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

HYPRE_Int buildRefMatrix(MPI_Comm comm,
                         HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                         HYPRE_BigInt num_nonzeros, HYPRE_Int *nnzrow,
                         HYPRE_BigInt *rows, HYPRE_BigInt *cols,
                         HYPRE_Real *coefs, HYPRE_IJMatrix *ij_ref_ptr)
{
   HYPRE_IJMatrix  ij_ref;
   HYPRE_Int       nrows;

   nrows = (HYPRE_Int) (iupper - ilower) + 1;

   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ij_ref);
   HYPRE_IJMatrixSetObjectType(ij_ref, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize_v2(ij_ref, HYPRE_MEMORY_HOST);
   HYPRE_IJMatrixSetValues(ij_ref, nrows, nnzrow, rows, cols, coefs);
   HYPRE_IJMatrixAssemble(ij_ref);

   // Set pointer to matrix
   *ij_ref_ptr = ij_ref;

   return hypre_error_flag;
}

HYPRE_Int
checkMatrix(HYPRE_IJMatrix ij_ref, HYPRE_IJMatrix ij_A)
{
   MPI_Comm              comm = hypre_IJMatrixComm(ij_ref);
   HYPRE_ParCSRMatrix    parcsr_A = (HYPRE_ParCSRMatrix) hypre_IJMatrixObject(ij_A);
   HYPRE_ParCSRMatrix    h_parcsr_ref = (HYPRE_ParCSRMatrix) hypre_IJMatrixObject(ij_ref);
   HYPRE_ParCSRMatrix    h_parcsr_A;
   HYPRE_ParCSRMatrix    parcsr_error;
   HYPRE_Int             memory_loc;
   HYPRE_Int             myid;
   HYPRE_Real            fnorm;

   hypre_MPI_Comm_rank(comm, &myid );
   memory_loc = hypre_GetActualMemLocation(hypre_IJMatrixMemoryLocation(ij_A));
   if (memory_loc == HYPRE_MEMORY_DEVICE)
   {
      h_parcsr_A = hypre_ParCSRMatrixClone_v2(parcsr_A, 1, HYPRE_MEMORY_HOST);
   }
   else
   {
      h_parcsr_A = parcsr_A;
   }

   // Check norm of (parcsr_ref - parcsr_A)
   hypre_ParcsrAdd(1.0, h_parcsr_ref, -1.0, h_parcsr_A, &parcsr_error);
   fnorm = hypre_ParCSRMatrixFnorm(parcsr_error);
   if (myid == 0)
   {
      hypre_printf("Frobenius norm of (A_ref - A): %f\n", fnorm);
   }

   return hypre_error_flag;
}

HYPRE_Int
test_SetSet(MPI_Comm comm, HYPRE_Int memory_loc,
            HYPRE_BigInt ilower, HYPRE_BigInt iupper,
            HYPRE_Int nrows, HYPRE_BigInt num_nonzeros,
            HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
            HYPRE_BigInt *rows, HYPRE_BigInt *cols,
            HYPRE_Real *coefs, HYPRE_IJMatrix *ij_A_ptr)
{
   HYPRE_IJMatrix  ij_A;
   HYPRE_Int       i, chunk, chunk_size;
   HYPRE_Int       row_cnt, nnz_cnt;
   HYPRE_Int       nrows_left;
   HYPRE_Real     *new_coefs;

   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ij_A);
   HYPRE_IJMatrixSetObjectType(ij_A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize_v2(ij_A, memory_loc);

   // First Set
   chunk_size = nrows/nchunks;
   new_coefs = hypre_TAlloc(HYPRE_Real, num_nonzeros, memory_loc);
   for (i = 0; i < num_nonzeros; i++)
   {
      new_coefs[i] = 2.0*coefs[i];
   }
   row_cnt = 0; nnz_cnt = 0;
   for (chunk = 0; chunk < nchunks; chunk++)
   {
      HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[row_cnt], &rows[row_cnt],
                              &cols[nnz_cnt], &new_coefs[nnz_cnt]);

      for (i = row_cnt; i < (row_cnt + chunk_size); i++)
      {
         nnz_cnt += h_nnzrow[i];
      }
      row_cnt += chunk_size;
   }

   nrows_left = nrows - row_cnt;
   if (nrows_left > 0)
   {
      HYPRE_IJMatrixSetValues(ij_A, nrows_left, &nnzrow[row_cnt], &rows[row_cnt],
                              &cols[nnz_cnt], &new_coefs[nnz_cnt]);
   }

   // Second set
   row_cnt = 0; nnz_cnt = 0;
   for (chunk = 0; chunk < nchunks; chunk++)
   {
      HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[row_cnt], &rows[row_cnt],
                              &cols[nnz_cnt], &coefs[nnz_cnt]);

      for (i = row_cnt; i < (row_cnt + chunk_size); i++)
      {
         nnz_cnt += h_nnzrow[i];
      }
      row_cnt += chunk_size;
   }

   nrows_left = nrows - row_cnt;
   if (nrows_left > 0)
   {
      HYPRE_IJMatrixSetValues(ij_A, nrows_left, &nnzrow[row_cnt], &rows[row_cnt],
                              &cols[nnz_cnt], &coefs[nnz_cnt]);
   }

   // Assemble matrix
   HYPRE_IJMatrixAssemble(ij_A);

   // Free memory
   hypre_TFree(new_coefs, memory_loc);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return hypre_error_flag;
}

HYPRE_Int
test_AddSet(MPI_Comm comm, HYPRE_Int memory_loc,
            HYPRE_BigInt ilower, HYPRE_BigInt iupper,
            HYPRE_Int nrows, HYPRE_BigInt num_nonzeros,
            HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
            HYPRE_BigInt *rows, HYPRE_BigInt *cols,
            HYPRE_Real *coefs, HYPRE_IJMatrix *ij_A_ptr)
{
   HYPRE_IJMatrix  ij_A;
   HYPRE_Int       i, chunk, chunk_size;
   HYPRE_Int       row_cnt, nnz_cnt;
   HYPRE_Int       nrows_left;

   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ij_A);
   HYPRE_IJMatrixSetObjectType(ij_A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize_v2(ij_A, memory_loc);

   // First Add
   chunk_size = nrows/nchunks;
   row_cnt = 0; nnz_cnt = 0;
   for (chunk = 0; chunk < nchunks; chunk++)
   {
      HYPRE_IJMatrixAddToValues(ij_A, chunk_size, &nnzrow[row_cnt], &rows[row_cnt],
                                &cols[nnz_cnt], &coefs[nnz_cnt]);

      for (i = row_cnt; i < (row_cnt + chunk_size); i++)
      {
         nnz_cnt += h_nnzrow[i];
      }
      row_cnt += chunk_size;
   }

   nrows_left = nrows - row_cnt;
   if (nrows_left > 0)
   {
      HYPRE_IJMatrixAddToValues(ij_A, nrows_left, &nnzrow[row_cnt], &rows[row_cnt],
                                &cols[nnz_cnt], &coefs[nnz_cnt]);
   }

   // Then Set
   row_cnt = 0; nnz_cnt = 0;
   for (chunk = 0; chunk < nchunks; chunk++)
   {
      HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[row_cnt], &rows[row_cnt],
                              &cols[nnz_cnt], &coefs[nnz_cnt]);

      for (i = row_cnt; i < (row_cnt + chunk_size); i++)
      {
         nnz_cnt += h_nnzrow[i];
      }
      row_cnt += chunk_size;
   }

   nrows_left = nrows - row_cnt;
   if (nrows_left > 0)
   {
      HYPRE_IJMatrixSetValues(ij_A, nrows_left, &nnzrow[row_cnt], &rows[row_cnt],
                              &cols[nnz_cnt], &coefs[nnz_cnt]);
   }

   // Assemble matrix
   HYPRE_IJMatrixAssemble(ij_A);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return hypre_error_flag;
}

HYPRE_Int
test_SetAddSet(MPI_Comm comm, HYPRE_Int memory_loc,
               HYPRE_BigInt ilower, HYPRE_BigInt iupper,
               HYPRE_Int nrows, HYPRE_BigInt num_nonzeros,
               HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
               HYPRE_BigInt *rows, HYPRE_BigInt *cols,
               HYPRE_Real *coefs, HYPRE_IJMatrix *ij_A_ptr)
{
   HYPRE_IJMatrix  ij_A;
   HYPRE_Int       i, chunk, chunk_size;
   HYPRE_Int       row_cnt, nnz_cnt;
   HYPRE_Int       nrows_left;
   HYPRE_Real     *new_coefs;

   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ij_A);
   HYPRE_IJMatrixSetObjectType(ij_A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize_v2(ij_A, memory_loc);

   // First Set
   new_coefs = hypre_TAlloc(HYPRE_Real, num_nonzeros, memory_loc);
   for (i = 0; i < num_nonzeros; i++)
   {
      new_coefs[i] = 2.0*coefs[i];
   }
   chunk_size = nrows/nchunks;
   row_cnt = 0; nnz_cnt = 0;
   for (chunk = 0; chunk < nchunks; chunk++)
   {
      HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[row_cnt], &rows[row_cnt],
                              &cols[nnz_cnt], &new_coefs[nnz_cnt]);

      for (i = row_cnt; i < (row_cnt + chunk_size); i++)
      {
         nnz_cnt += h_nnzrow[i];
      }
      row_cnt += chunk_size;
   }

   nrows_left = nrows - row_cnt;
   if (nrows_left > 0)
   {
      HYPRE_IJMatrixSetValues(ij_A, nrows_left, &nnzrow[row_cnt], &rows[row_cnt],
                              &cols[nnz_cnt], &new_coefs[nnz_cnt]);
   }

   // Then Add ...
   row_cnt = 0; nnz_cnt = 0;
   for (chunk = 0; chunk < nchunks; chunk++)
   {
      HYPRE_IJMatrixAddToValues(ij_A, chunk_size, &nnzrow[row_cnt], &rows[row_cnt],
                                &cols[nnz_cnt], &coefs[nnz_cnt]);

      for (i = row_cnt; i < (row_cnt + chunk_size); i++)
      {
         nnz_cnt += h_nnzrow[i];
      }
      row_cnt += chunk_size;
   }

   nrows_left = nrows - row_cnt;
   if (nrows_left > 0)
   {
      HYPRE_IJMatrixAddToValues(ij_A, nrows_left, &nnzrow[row_cnt], &rows[row_cnt],
                                &cols[nnz_cnt], &coefs[nnz_cnt]);
   }

   // Final Set with original coefficients
   chunk_size = nrows/nchunks;
   row_cnt = 0; nnz_cnt = 0;
   for (chunk = 0; chunk < nchunks; chunk++)
   {
      HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[row_cnt], &rows[row_cnt],
                              &cols[nnz_cnt], &coefs[nnz_cnt]);

      for (i = row_cnt; i < (row_cnt + chunk_size); i++)
      {
         nnz_cnt += h_nnzrow[i];
      }
      row_cnt += chunk_size;
   }

   nrows_left = nrows - row_cnt;
   if (nrows_left > 0)
   {
      HYPRE_IJMatrixSetValues(ij_A, nrows_left, &nnzrow[row_cnt], &rows[row_cnt],
                              &cols[nnz_cnt], &coefs[nnz_cnt]);
   }

   // Assemble matrix
   HYPRE_IJMatrixAssemble(ij_A);

   // Free memory
   hypre_TFree(new_coefs, memory_loc);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return hypre_error_flag;
}

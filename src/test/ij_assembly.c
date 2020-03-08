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
#include "HYPRE_utilities.h"
#include "_hypre_IJ_mv.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"

HYPRE_Int buildMatrixEntries(MPI_Comm comm,
                             HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz,
                             HYPRE_Int Px, HYPRE_Int Py, HYPRE_Int Pz,
                             HYPRE_Int cx, HYPRE_Int cy, HYPRE_Int cz,
                             HYPRE_BigInt *ilower, HYPRE_BigInt *iupper,
                             HYPRE_BigInt *jlower, HYPRE_BigInt *jupper,
                             HYPRE_Int *nrows, HYPRE_BigInt *num_nonzeros,
                             HYPRE_Int **nnzrow_ptr, HYPRE_BigInt **rows_ptr, HYPRE_BigInt **rows2_ptr,
                             HYPRE_BigInt **cols_ptr, HYPRE_Real **coefs_ptr, HYPRE_Int stencil);

HYPRE_Int checkMatrix(HYPRE_IJMatrix ij_ref, HYPRE_IJMatrix ij_A);

HYPRE_Int test_Set(MPI_Comm comm, HYPRE_MemoryLocation memory_location, HYPRE_Int option,
                   HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                   HYPRE_Int nrows, HYPRE_BigInt num_nonzeros,
                   HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
                   HYPRE_BigInt *rows, HYPRE_BigInt *cols,
                   HYPRE_Real *coefs, HYPRE_IJMatrix *ij_A);

HYPRE_Int test_SetSet(MPI_Comm comm, HYPRE_MemoryLocation memory_location,
                      HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                      HYPRE_Int nrows, HYPRE_BigInt num_nonzeros,
                      HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
                      HYPRE_BigInt *rows, HYPRE_BigInt *cols,
                      HYPRE_Real *coefs, HYPRE_IJMatrix *ij_A);

HYPRE_Int test_AddSet(MPI_Comm comm, HYPRE_MemoryLocation memory_location,
                      HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                      HYPRE_Int nrows, HYPRE_BigInt num_nonzeros,
                      HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
                      HYPRE_BigInt *rows, HYPRE_BigInt *cols,
                      HYPRE_Real *coefs, HYPRE_IJMatrix *ij_A);

HYPRE_Int test_SetAddSet(MPI_Comm comm, HYPRE_MemoryLocation memory_location,
                         HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                         HYPRE_Int nrows, HYPRE_BigInt num_nonzeros,
                         HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
                         HYPRE_BigInt *rows, HYPRE_BigInt *cols,
                         HYPRE_Real *coefs, HYPRE_IJMatrix *ij_A);

hypre_int
main( HYPRE_Int  argc,
      char      *argv[] )
{
   MPI_Comm                  comm = hypre_MPI_COMM_WORLD;
   HYPRE_Int                 num_procs;
   HYPRE_Int                 myid;
   HYPRE_Int                 arg_index;
   HYPRE_Int                 time_index;
   HYPRE_Int                 print_usage;
   HYPRE_MemoryLocation      memory_location;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_ExecuctionPolicy    default_exec_policy;
#endif
   char                      memory_location_name[8];

   HYPRE_Int                 nrows;
   HYPRE_BigInt              num_nonzeros;
   HYPRE_BigInt              ilower, iupper;
   HYPRE_BigInt              jlower, jupper;
   HYPRE_Int                *nnzrow, *h_nnzrow, *d_nnzrow;
   HYPRE_BigInt             *rows,   *h_rows,   *d_rows;
   HYPRE_BigInt             *rows2,  *h_rows2,  *d_rows2;
   HYPRE_BigInt             *cols,   *h_cols,   *d_cols;
   HYPRE_Real               *coefs,  *h_coefs,  *d_coefs;
   HYPRE_IJMatrix            ij_A;
   HYPRE_IJMatrix            ij_ref;

   // Driver input parameters
   HYPRE_Int                 Px, Py, Pz;
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Real                cx, cy, cz;
   HYPRE_Int                 nchunks;
   HYPRE_Int                 mode;
   HYPRE_Int                 option;
   HYPRE_Int                 stencil;
   HYPRE_Int                 print_matrix;

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

   /* Initialize Hypre */
   /* Initialize Hypre: must be the first Hypre function to call */
   time_index = hypre_InitializeTiming("Hypre init");
   hypre_BeginTiming(time_index);
   HYPRE_Init();
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Hypre init times", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Set default parameters
    *-----------------------------------------------------------*/
   Px = num_procs;
   Py = 1;
   Pz = 1;

   nx = 10;
   ny = 10;
   nz = 10;

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   default_exec_policy = HYPRE_EXEC_DEVICE;
#endif
   memory_location     = HYPRE_MEMORY_DEVICE;
   mode                = 1;
   option              = 1;
   nchunks             = 1;
   print_matrix        = 0;
   stencil             = 7;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   print_usage = 0;
   arg_index = 1;
   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-memory_location") == 0 )
      {
         arg_index++;
         memory_location = (HYPRE_MemoryLocation) atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
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
      else if ( strcmp(argv[arg_index], "-mode") == 0 )
      {
         arg_index++;
         mode = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-option") == 0 )
      {
         arg_index++;
         option = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-9pt") == 0 )
      {
         arg_index++;
         stencil = 9;
      }
      else if ( strcmp(argv[arg_index], "-27pt") == 0 )
      {
         arg_index++;
         stencil = 27;
      }
      else if ( strcmp(argv[arg_index], "-nchunks") == 0 )
      {
         arg_index++;
         nchunks = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_matrix = 1;
      }
      else
      {
         print_usage = 1; break;
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
         hypre_printf("      -n <nx> <ny> <nz>  : total problem size \n");
         hypre_printf("      -P <Px> <Py> <Pz>  : processor topology\n");
         hypre_printf("      -c <cx> <cy> <cz>  : diffusion coefficients\n");
         hypre_printf("      -nchunks <val>     : number of chunks passed to Set/AddValues\n");
         hypre_printf("      -mode <val>        : tests to be performed\n");
         hypre_printf("             1 = Set (default)\n");
         hypre_printf("             2 = SetSet\n");
         hypre_printf("             4 = AddSet\n");
         hypre_printf("             8 = SetAddSet\n");
         hypre_printf("      -option <val>      : interface option of Set/AddToValues\n");
         hypre_printf("             1 = CSR-like (default)\n");
         hypre_printf("             2 = COO-like\n");
         hypre_printf("      -print             : print matrices\n");
         hypre_printf("\n");
      }

      return (0);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
   switch (memory_location)
   {
      case HYPRE_MEMORY_UNDEFINED:
         return -1;

      case HYPRE_MEMORY_DEVICE:
         hypre_sprintf(memory_location_name, "Device"); break;

      case HYPRE_MEMORY_HOST:
         hypre_sprintf(memory_location_name, "Host"); break;
   }

   if (myid == 0)
   {
      hypre_printf("  Memory location: %s\n", memory_location_name);
      hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", Px, Py, Pz);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf("\n");
   }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   hypre_HandleDefaultExecPolicy(hypre_handle) = default_exec_policy;
#endif

   /*-----------------------------------------------------------
    * Build matrix entries
    *-----------------------------------------------------------*/
   buildMatrixEntries(comm, nx, ny, nz, Px, Py, Pz, cx, cy, cz,
                      &ilower, &iupper, &jlower, &jupper, &nrows, &num_nonzeros,
                      &h_nnzrow, &h_rows, &h_rows2, &h_cols, &h_coefs, stencil);

   switch (memory_location)
   {
      case HYPRE_MEMORY_DEVICE:
         d_nnzrow = hypre_TAlloc(HYPRE_Int,    nrows,        HYPRE_MEMORY_DEVICE);
         d_rows   = hypre_TAlloc(HYPRE_BigInt, nrows,        HYPRE_MEMORY_DEVICE);
         d_rows2  = hypre_TAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_DEVICE);
         d_cols   = hypre_TAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_DEVICE);
         d_coefs  = hypre_TAlloc(HYPRE_Real,   num_nonzeros, HYPRE_MEMORY_DEVICE);

         hypre_TMemcpy(d_nnzrow, h_nnzrow, HYPRE_Int,    nrows,        HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_rows,   h_rows,   HYPRE_BigInt, nrows,        HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_rows2,  h_rows2,  HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_cols,   h_cols,   HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_coefs,  h_coefs,  HYPRE_Real,   num_nonzeros, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

         nnzrow = d_nnzrow;
         rows   = d_rows;
         rows2  = d_rows2;
         cols   = d_cols;
         coefs  = d_coefs;
         break;

      case HYPRE_MEMORY_HOST:
         nnzrow = h_nnzrow;
         rows   = h_rows;
         rows2  = h_rows2;
         cols   = h_cols;
         coefs  = h_coefs;
         break;

      case HYPRE_MEMORY_UNDEFINED:
         return -1;
   }

   /* Build Reference matrix */
   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &ij_ref);
   HYPRE_IJMatrixSetObjectType(ij_ref, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize_v2(ij_ref, HYPRE_MEMORY_HOST);
   HYPRE_IJMatrixSetValues(ij_ref, nrows, h_nnzrow, h_rows, h_cols, h_coefs);
   HYPRE_IJMatrixAssemble(ij_ref);

   /*-----------------------------------------------------------
    * Test different Set/Add combinations
    *-----------------------------------------------------------*/
   /* Test Set */
   if (mode & 1)
   {
      test_Set(comm, memory_location, option, ilower, iupper, nrows, num_nonzeros,
               nchunks, h_nnzrow, nnzrow, option == 1 ? rows : rows2, cols, coefs, &ij_A);

      checkMatrix(ij_ref, ij_A);
      if (print_matrix)
      {
         HYPRE_IJMatrixPrint(ij_A, "Set");
      }
      HYPRE_IJMatrixDestroy(ij_A);
   }

#if 0
   /* Test Set/Set */
   if (mode & 2)
   {
      time_index = hypre_InitializeTiming("Test Set/Set");
      hypre_BeginTiming(time_index);
      test_SetSet(comm, memory_location, ilower, iupper, nrows, num_nonzeros,
                  nchunks, h_nnzrow, nnzrow, rows, cols, coefs, &ij_A);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Test Set/Set", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      checkMatrix(ij_A, ij_ref);
      if (print_matrix)
      {
         HYPRE_IJMatrixPrint(ij_A, "SetSet");
      }
      HYPRE_IJMatrixDestroy(ij_A);
   }

   /* Test Add/Set */
   if (mode & 4)
   {
      time_index = hypre_InitializeTiming("Test Add/Set");
      hypre_BeginTiming(time_index);
      test_AddSet(comm, memory_location, ilower, iupper, nrows, num_nonzeros,
                  nchunks, h_nnzrow, nnzrow, rows, cols, coefs, &ij_A);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Test Add/Set", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      checkMatrix(ij_A, ij_ref);
      if (print_matrix)
      {
         HYPRE_IJMatrixPrint(ij_A, "AddSet");
      }
      HYPRE_IJMatrixDestroy(ij_A);
   }

   /* Test Set/Add/Set */
   if (mode & 8)
   {
      time_index = hypre_InitializeTiming("Test Set/Add/Set");
      hypre_BeginTiming(time_index);
      test_AddSet(comm, memory_location, ilower, iupper, nrows, num_nonzeros,
                  nchunks, h_nnzrow, nnzrow, rows, cols, coefs, &ij_A);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Test Set/Add/Set", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      checkMatrix(ij_A, ij_ref);
      if (print_matrix)
      {
         HYPRE_IJMatrixPrint(ij_A, "SetAddSet");
      }
      HYPRE_IJMatrixDestroy(ij_A);
   }
#endif

   /*-----------------------------------------------------------
    * Free memory
    *-----------------------------------------------------------*/
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      hypre_TFree(d_nnzrow, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_rows,   HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_rows2,  HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_cols,   HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_coefs,  HYPRE_MEMORY_DEVICE);
   }
   hypre_TFree(h_nnzrow, HYPRE_MEMORY_HOST);
   hypre_TFree(h_rows,   HYPRE_MEMORY_HOST);
   hypre_TFree(h_rows2,  HYPRE_MEMORY_HOST);
   hypre_TFree(h_cols,   HYPRE_MEMORY_HOST);
   hypre_TFree(h_coefs,  HYPRE_MEMORY_HOST);

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
buildMatrixEntries(MPI_Comm       comm,
                   HYPRE_Int      nx,
                   HYPRE_Int      ny,
                   HYPRE_Int      nz,
                   HYPRE_Int      Px,
                   HYPRE_Int      Py,
                   HYPRE_Int      Pz,
                   HYPRE_Int      cx,
                   HYPRE_Int      cy,
                   HYPRE_Int      cz,
                   HYPRE_BigInt  *ilower_ptr,
                   HYPRE_BigInt  *iupper_ptr,
                   HYPRE_BigInt  *jlower_ptr,
                   HYPRE_BigInt  *jupper_ptr,
                   HYPRE_Int     *nrows_ptr,
                   HYPRE_BigInt  *num_nonzeros_ptr,
                   HYPRE_Int    **nnzrow_ptr,
                   HYPRE_BigInt **rows_ptr,   /* row indices of length nrows */
                   HYPRE_BigInt **rows2_ptr,  /* row indices of length nnz */
                   HYPRE_BigInt **cols_ptr,   /* col indices of length nnz */
                   HYPRE_Real   **coefs_ptr,  /* values of length nnz */
                   HYPRE_Int      stencil)
{
   HYPRE_Int        num_procs;
   HYPRE_Int        myid;
   HYPRE_BigInt     ilower, iupper;
   HYPRE_BigInt     jlower, jupper;
   HYPRE_Int        nrows;
   HYPRE_BigInt     num_nonzeros;
   HYPRE_Int       *nnzrow;
   HYPRE_BigInt    *rows;
   HYPRE_BigInt    *rows2;
   HYPRE_BigInt    *cols;
   HYPRE_Real      *coefs;
   HYPRE_Real       values[4];
   HYPRE_Int        i, j, k;
   HYPRE_ParCSRMatrix A;

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

   HYPRE_Int ip = myid % Px;
   HYPRE_Int iq = (( myid - ip)/Px) % Py;
   HYPRE_Int ir = ( myid - ip - Px*iq)/( Px*Py );

   values[0] = 0;
   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   if (stencil == 7)
   {
      A = (HYPRE_ParCSRMatrix) GenerateLaplacian(comm, nx, ny, nz, Px, Py, Pz, ip, iq, ir, values);
   }
   else if (stencil == 9)
   {
      A = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(comm, nx, ny, Px, Py, ip, iq, values);
   }
   else if (stencil == 27)
   {
      A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(comm, nx, ny, nz, Px, Py, Pz, ip, iq, ir, values);
   }
   else
   {
      hypre_assert(0);
   }

   ilower = hypre_ParCSRMatrixFirstRowIndex(A);
   iupper = hypre_ParCSRMatrixLastRowIndex(A);
   jlower = hypre_ParCSRMatrixFirstColDiag(A);
   jupper = hypre_ParCSRMatrixLastColDiag(A);
   nrows  = hypre_ParCSRMatrixNumRows(A);
   num_nonzeros = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(A)) + hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(A));

   nnzrow = hypre_CTAlloc(HYPRE_Int,    nrows,        HYPRE_MEMORY_HOST);
   rows   = hypre_CTAlloc(HYPRE_BigInt, nrows,        HYPRE_MEMORY_HOST);
   rows2  = hypre_CTAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_HOST);
   cols   = hypre_CTAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_HOST);
   coefs  = hypre_CTAlloc(HYPRE_Real,   num_nonzeros, HYPRE_MEMORY_HOST);

   k = 0;
#if 0
   for (i = 0; i < nrows; i++)
   {
      nnzrow[i] = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A))[i+1] - hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A))[i] +
                  hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A))[i+1] - hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A))[i];
      rows[i]   = ilower + i;

      for (j = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A))[i]; j < hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A))[i+1]; j++)
      {
         rows2[k]   = ilower + i;
         cols[k]    = jlower + hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A))[j];
         coefs[k++] = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A))[j];
      }
      for (j = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A))[i]; j < hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A))[i+1]; j++)
      {
         rows2[k]   = ilower + i;
         cols[k]    = hypre_ParCSRMatrixColMapOffd(A)[hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(A))[j]];
         coefs[k++] = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A))[j];
      }
   }
#else
   for (i = nrows-1; i >= 0; i--)
   {
      nnzrow[nrows-1-i] = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A))[i+1] - hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A))[i] +
                          hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A))[i+1] - hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A))[i];
      rows[nrows-1-i]   = ilower + i;

      for (j = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A))[i]; j < hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A))[i+1]; j++)
      {
         rows2[k]   = ilower + i;
         cols[k]    = jlower + hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A))[j];
         coefs[k++] = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A))[j];
      }
      for (j = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A))[i]; j < hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A))[i+1]; j++)
      {
         rows2[k]   = ilower + i;
         cols[k]    = hypre_ParCSRMatrixColMapOffd(A)[hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(A))[j]];
         coefs[k++] = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A))[j];
      }
   }
#endif

   hypre_assert(k == num_nonzeros);

   // Set pointers
   *ilower_ptr       = ilower;
   *iupper_ptr       = iupper;
   *jlower_ptr       = jlower;
   *jupper_ptr       = jupper;
   *nrows_ptr        = nrows;
   *num_nonzeros_ptr = num_nonzeros;
   *nnzrow_ptr       = nnzrow;
   *rows_ptr         = rows;
   *rows2_ptr        = rows2;
   *cols_ptr         = cols;
   *coefs_ptr        = coefs;

   HYPRE_ParCSRMatrixDestroy(A);

   return hypre_error_flag;
}

HYPRE_Int
checkMatrix(HYPRE_IJMatrix ij_ref, HYPRE_IJMatrix ij_A)
{
   MPI_Comm            comm         = hypre_IJMatrixComm(ij_ref);
   HYPRE_ParCSRMatrix  parcsr_A     = (HYPRE_ParCSRMatrix) hypre_IJMatrixObject(ij_A);
   HYPRE_ParCSRMatrix  h_parcsr_ref = (HYPRE_ParCSRMatrix) hypre_IJMatrixObject(ij_ref);
   HYPRE_ParCSRMatrix  h_parcsr_A;
   HYPRE_ParCSRMatrix  parcsr_error;
   HYPRE_Int           myid;
   HYPRE_Real          fnorm;

   hypre_MPI_Comm_rank(comm, &myid);

   if (hypre_GetActualMemLocation(hypre_ParCSRMatrixMemoryLocation(parcsr_A)) != hypre_MEMORY_HOST)
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
      hypre_printf("Frobenius norm of (A_ref - A): %e\n", fnorm);
   }

   HYPRE_ParCSRMatrixDestroy(parcsr_error);

   return hypre_error_flag;
}

HYPRE_Int
test_Set(MPI_Comm             comm,
         HYPRE_MemoryLocation memory_location,
         HYPRE_Int            option,           /* 1 or 2 */
         HYPRE_BigInt         ilower,
         HYPRE_BigInt         iupper,
         HYPRE_Int            nrows,
         HYPRE_BigInt         num_nonzeros,
         HYPRE_Int            nchunks,
         HYPRE_Int           *h_nnzrow,
         HYPRE_Int           *nnzrow,
         HYPRE_BigInt        *rows,             /* option = 1: length of nrows, = 2: length of num_nonzeros */
         HYPRE_BigInt        *cols,
         HYPRE_Real          *coefs,
         HYPRE_IJMatrix      *ij_A_ptr)
{
   HYPRE_IJMatrix  ij_A;
   HYPRE_Int       i, chunk, chunk_size;
   HYPRE_Int      *h_rowptr = hypre_TAlloc(HYPRE_Int, nrows+1, HYPRE_MEMORY_HOST);

   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ij_A);
   HYPRE_IJMatrixSetObjectType(ij_A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize_v2(ij_A, memory_location);

   h_rowptr[0] = 0;
   for (i = 1; i < nrows + 1; i++)
   {
      h_rowptr[i] = h_rowptr[i-1] + h_nnzrow[i-1];
   }
   hypre_assert(h_rowptr[nrows] == num_nonzeros);

   chunk_size = nrows / nchunks;

   HYPRE_Int time_index = hypre_InitializeTiming("Test SetValues");
   hypre_BeginTiming(time_index);

   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = hypre_min(chunk_size, nrows-chunk);

      if (1 == option)
      {
         HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk+chunk_size]-h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Test SetValues", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   // Assemble matrix
   HYPRE_IJMatrixAssemble(ij_A);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   hypre_TFree(h_rowptr, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

HYPRE_Int
test_SetSet(MPI_Comm comm, HYPRE_MemoryLocation memory_location,
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
   HYPRE_IJMatrixInitialize_v2(ij_A, memory_location);

   // First Set
   chunk_size = nrows/nchunks;
   new_coefs = hypre_TAlloc(HYPRE_Real, num_nonzeros, memory_location);
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
   hypre_TFree(new_coefs, memory_location);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return hypre_error_flag;
}

HYPRE_Int
test_AddSet(MPI_Comm comm, HYPRE_MemoryLocation memory_location,
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
   HYPRE_IJMatrixInitialize_v2(ij_A, memory_location);

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
test_SetAddSet(MPI_Comm comm, HYPRE_MemoryLocation memory_location,
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
   HYPRE_IJMatrixInitialize_v2(ij_A, memory_location);

   // First Set
   new_coefs = hypre_TAlloc(HYPRE_Real, num_nonzeros, memory_location);
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
   hypre_TFree(new_coefs, memory_location);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return hypre_error_flag;
}

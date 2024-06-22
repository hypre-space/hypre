/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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
//#include "_hypre_utilities.hpp"

HYPRE_Int buildMatrixEntries(MPI_Comm comm,
                             HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz,
                             HYPRE_Int Px, HYPRE_Int Py, HYPRE_Int Pz,
                             HYPRE_Real cx, HYPRE_Real cy, HYPRE_Real cz, HYPRE_Int base,
                             HYPRE_BigInt *ilower, HYPRE_BigInt *iupper,
                             HYPRE_BigInt *jlower, HYPRE_BigInt *jupper,
                             HYPRE_Int *nrows, HYPRE_BigInt *num_nonzeros,
                             HYPRE_Int **nnzrow_ptr, HYPRE_BigInt **rows_ptr,
                             HYPRE_BigInt **rows_coo_ptr, HYPRE_BigInt **cols_ptr,
                             HYPRE_Real **coefs_ptr, HYPRE_Int stencil, HYPRE_ParCSRMatrix *parcsr_ptr);

HYPRE_Int getParCSRMatrixData(HYPRE_ParCSRMatrix  A, HYPRE_Int base, HYPRE_Int *nrows_ptr,
                              HYPRE_BigInt *num_nonzeros_ptr,
                              HYPRE_Int **nnzrow_ptr, HYPRE_BigInt **rows_ptr, HYPRE_BigInt **rows_coo_ptr,
                              HYPRE_BigInt **cols_ptr, HYPRE_Real **coefs_ptr);

HYPRE_Real checkMatrix(HYPRE_ParCSRMatrix parcsr_ref, HYPRE_IJMatrix ij_A);

HYPRE_Int test_Set(MPI_Comm comm, HYPRE_MemoryLocation memory_location, HYPRE_Int option,
                   HYPRE_BigInt ilower,
                   HYPRE_BigInt iupper, HYPRE_BigInt jlower, HYPRE_BigInt jupper, HYPRE_Int nrows,
                   HYPRE_BigInt num_nonzeros, HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
                   HYPRE_BigInt *rows, HYPRE_BigInt *cols, HYPRE_Real *coefs, HYPRE_IJMatrix *ij_A_ptr);

HYPRE_Int test_AddTranspose(MPI_Comm comm, HYPRE_MemoryLocation memory_location, HYPRE_Int option,
                            HYPRE_BigInt ilower,
                            HYPRE_BigInt iupper, HYPRE_BigInt jlower, HYPRE_BigInt jupper, HYPRE_Int nrows,
                            HYPRE_BigInt num_nonzeros, HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
                            HYPRE_BigInt *rows, HYPRE_BigInt *cols, HYPRE_Real *coefs, HYPRE_IJMatrix *ij_AT_ptr);

HYPRE_Int test_SetSet(MPI_Comm comm, HYPRE_MemoryLocation memory_location, HYPRE_Int option,
                      HYPRE_BigInt ilower,
                      HYPRE_BigInt iupper, HYPRE_BigInt jlower, HYPRE_BigInt jupper, HYPRE_Int nrows,
                      HYPRE_BigInt num_nonzeros, HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
                      HYPRE_BigInt *rows, HYPRE_BigInt *cols, HYPRE_Real *coefs, HYPRE_IJMatrix *ij_AT_ptr);

HYPRE_Int test_AddSet(MPI_Comm comm, HYPRE_MemoryLocation memory_location, HYPRE_Int option,
                      HYPRE_BigInt ilower,
                      HYPRE_BigInt iupper, HYPRE_BigInt jlower, HYPRE_BigInt jupper, HYPRE_Int nrows,
                      HYPRE_BigInt num_nonzeros, HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
                      HYPRE_BigInt *rows, HYPRE_BigInt *cols, HYPRE_Real *coefs, HYPRE_IJMatrix *ij_AT_ptr);

HYPRE_Int test_SetAdd(MPI_Comm comm, HYPRE_MemoryLocation memory_location, HYPRE_Int option,
                      HYPRE_BigInt ilower,
                      HYPRE_BigInt iupper, HYPRE_BigInt jlower, HYPRE_BigInt jupper, HYPRE_Int nrows,
                      HYPRE_BigInt num_nonzeros, HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
                      HYPRE_BigInt *rows, HYPRE_BigInt *cols, HYPRE_Real *coefs, HYPRE_IJMatrix *ij_AT_ptr);

HYPRE_Int test_SetAddSet(MPI_Comm comm, HYPRE_MemoryLocation memory_location, HYPRE_Int option,
                         HYPRE_BigInt ilower,
                         HYPRE_BigInt iupper, HYPRE_BigInt jlower, HYPRE_BigInt jupper, HYPRE_Int nrows,
                         HYPRE_BigInt num_nonzeros, HYPRE_Int nchunks, HYPRE_Int *h_nnzrow, HYPRE_Int *nnzrow,
                         HYPRE_BigInt *rows, HYPRE_BigInt *cols, HYPRE_Real *coefs, HYPRE_IJMatrix *ij_AT_ptr);


hypre_int
main( hypre_int  argc,
      char      *argv[] )
{
   MPI_Comm                  comm = hypre_MPI_COMM_WORLD;
   HYPRE_Int                 num_procs;
   HYPRE_Int                 myid;
   HYPRE_Int                 arg_index;
   HYPRE_Int                 time_index;
   HYPRE_Int                 print_usage;
   HYPRE_MemoryLocation      memory_location;
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_ExecutionPolicy     default_exec_policy;
#endif
   char                      memory_location_name[8];
   HYPRE_Int                 nrows;
   HYPRE_BigInt              num_nonzeros;
   HYPRE_BigInt              ilower, iupper;
   HYPRE_BigInt              jlower, jupper;
   HYPRE_Int                *nnzrow = NULL, *h_nnzrow, *d_nnzrow = NULL;
   HYPRE_BigInt             *rows = NULL,   *h_rows,   *d_rows = NULL;
   HYPRE_BigInt             *rows_coo = NULL,  *h_rows_coo,  *d_rows_coo = NULL;
   HYPRE_BigInt             *cols = NULL,   *h_cols,   *d_cols = NULL;
   HYPRE_Real               *coefs = NULL,  *h_coefs,  *d_coefs = NULL;
   HYPRE_IJMatrix            ij_A, ij_AT;
   HYPRE_ParCSRMatrix        parcsr_ref = NULL, parcsr_trans = NULL;
   // Driver input parameters
   HYPRE_Int                 Px, Py, Pz;
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Real                cx, cy, cz;
   HYPRE_Int                 nchunks;
   HYPRE_Int                 mode, ierr = 0;
   HYPRE_Real                tol = 0.;
   HYPRE_Int                 option, base;
   HYPRE_Int                 stencil;
   HYPRE_Int                 print_matrix;

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before HYPRE_Initialize() and should not be changed after
    *-----------------------------------------------------------------*/
   hypre_bind_device_id(-1, myid, num_procs, hypre_MPI_COMM_WORLD);

   /* Initialize Hypre */
   /* Initialize Hypre: must be the first Hypre function to call */
   time_index = hypre_InitializeTiming("Hypre init");
   hypre_BeginTiming(time_index);
   HYPRE_Initialize();
   HYPRE_DeviceInitialize();
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

   nx = 100;
   ny = 101;
   nz = 102;

   cx = 1.0;
   cy = 2.0;
   cz = 3.0;

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   default_exec_policy = HYPRE_EXEC_DEVICE;
#endif
   memory_location     = HYPRE_MEMORY_DEVICE;
   mode                = (1 << 6) - 1;
   option              = 1;
   nchunks             = 3;
   base                = 0;
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
         cx = (HYPRE_Real)atof(argv[arg_index++]);
         cy = (HYPRE_Real)atof(argv[arg_index++]);
         cz = (HYPRE_Real)atof(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-base") == 0 )
      {
         arg_index++;
         base = atoi(argv[arg_index++]);
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
   if (Px * Py * Pz != num_procs)
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
         hypre_printf("      -n <nx> <ny> <nz>      : total problem size \n");
         hypre_printf("      -P <Px> <Py> <Pz>      : processor topology\n");
         hypre_printf("      -c <cx> <cy> <cz>      : diffusion coefficients\n");
         hypre_printf("      -memory_location <val> : memory location of the assembled matrix\n");
         hypre_printf("             0 = HOST\n");
         hypre_printf("             1 = DEVICE (default)\n");
         hypre_printf("      -nchunks <val>         : number of chunks passed to Set/AddValues\n");
         hypre_printf("      -base <val>            : matrix index base\n");
         hypre_printf("      -mode <val>            : tests to be performed\n");
         hypre_printf("             1 = Set (default)\n");
         hypre_printf("             2 = SetOffProc\n");
         hypre_printf("             4 = SetSet\n");
         hypre_printf("             8 = AddSet\n");
         hypre_printf("            16 = SetAddSet\n");
         hypre_printf("      -option <val>          : interface option of Set/AddToValues\n");
         hypre_printf("             1 = CSR-like (default)\n");
         hypre_printf("             2 = COO-like\n");
         hypre_printf("      -print                 : print matrices\n");
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

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   hypre_HandleDefaultExecPolicy(hypre_handle()) = default_exec_policy;
#endif

#if defined(HYPRE_USING_OPENMP)
   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_HOST)
   {
      mode = mode & ~2; /* skip AddTranspose with OMP */
   }
#endif

   /*-----------------------------------------------------------
    * Build matrix entries
    *-----------------------------------------------------------*/
   buildMatrixEntries(comm, nx, ny, nz, Px, Py, Pz, cx, cy, cz, base,
                      &ilower, &iupper, &jlower, &jupper, &nrows, &num_nonzeros,
                      &h_nnzrow, &h_rows, &h_rows_coo, &h_cols, &h_coefs, stencil, &parcsr_ref);

   switch (memory_location)
   {
      case HYPRE_MEMORY_DEVICE:
         d_nnzrow = hypre_TAlloc(HYPRE_Int,    nrows,        HYPRE_MEMORY_DEVICE);
         d_rows   = hypre_TAlloc(HYPRE_BigInt, nrows,        HYPRE_MEMORY_DEVICE);
         d_rows_coo  = hypre_TAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_DEVICE);
         d_cols   = hypre_TAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_DEVICE);
         d_coefs  = hypre_TAlloc(HYPRE_Real,   num_nonzeros, HYPRE_MEMORY_DEVICE);

         hypre_TMemcpy(d_nnzrow, h_nnzrow, HYPRE_Int,    nrows,        HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_rows,   h_rows,   HYPRE_BigInt, nrows,        HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_rows_coo,  h_rows_coo,  HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_cols,   h_cols,   HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_coefs,  h_coefs,  HYPRE_Real,   num_nonzeros, HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_HOST);

         nnzrow = d_nnzrow;
         rows   = d_rows;
         rows_coo  = d_rows_coo;
         cols   = d_cols;
         coefs  = d_coefs;
         break;

      case HYPRE_MEMORY_HOST:
         nnzrow = h_nnzrow;
         rows   = h_rows;
         rows_coo  = h_rows_coo;
         cols   = h_cols;
         coefs  = h_coefs;
         break;

      case HYPRE_MEMORY_UNDEFINED:
         return -1;
   }

   /*-----------------------------------------------------------
    * Test different Set/Add combinations
    *-----------------------------------------------------------*/
   /* Test Set */
   if (mode & 1)
   {
      test_Set(comm, memory_location, option, ilower, iupper, jlower, jupper, nrows, num_nonzeros,
               nchunks, h_nnzrow, nnzrow, option == 1 ? rows : rows_coo, cols, coefs, &ij_A);

      ierr += checkMatrix(parcsr_ref, ij_A) > tol;
      if (print_matrix)
      {
         HYPRE_IJMatrixPrint(ij_A, "ij_Set");
      }
      HYPRE_IJMatrixDestroy(ij_A);
   }

   /* Test AddTranspose */
   if (mode & 2)
   {
      test_AddTranspose(comm, memory_location, 2, ilower, iupper, jlower, jupper, nrows, num_nonzeros,
                        nchunks, h_nnzrow, nnzrow, rows_coo, cols, coefs, &ij_AT);

      hypre_ParCSRMatrixTranspose(parcsr_ref, &parcsr_trans, 1);

      ierr += checkMatrix(parcsr_trans, ij_AT) > tol;
      if (print_matrix)
      {
         HYPRE_IJMatrixPrint(ij_AT, "ij_AddTrans");
      }
      HYPRE_IJMatrixDestroy(ij_AT);
      HYPRE_ParCSRMatrixDestroy(parcsr_trans);
   }

   /* Test Set/Set */
   if (mode & 4)
   {
      test_SetSet(comm, memory_location, option, ilower, iupper, jlower, jupper, nrows, num_nonzeros,
                  nchunks, h_nnzrow, nnzrow, option == 1 ? rows : rows_coo, cols, coefs, &ij_A);

      ierr += checkMatrix(parcsr_ref, ij_A) > tol;
      if (print_matrix)
      {
         HYPRE_IJMatrixPrint(ij_A, "ij_SetSet");
      }
      HYPRE_IJMatrixDestroy(ij_A);
   }

   /* Test Add/Set */
   if (mode & 8)
   {
      test_AddSet(comm, memory_location, option, ilower, iupper, jlower, jupper, nrows, num_nonzeros,
                  nchunks, h_nnzrow, nnzrow, option == 1 ? rows : rows_coo, cols, coefs, &ij_A);

      ierr += checkMatrix(parcsr_ref, ij_A) > tol;
      if (print_matrix)
      {
         HYPRE_IJMatrixPrint(ij_A, "ij_AddSet");
      }
      HYPRE_IJMatrixDestroy(ij_A);
   }

   /* Test Set/Add */
   if (mode & 16)
   {
      test_SetAdd(comm, memory_location, option, ilower, iupper, jlower, jupper, nrows, num_nonzeros,
                  nchunks, h_nnzrow, nnzrow, option == 1 ? rows : rows_coo, cols, coefs, &ij_A);

      hypre_ParCSRMatrix *parcsr_ref2 = hypre_ParCSRMatrixClone(parcsr_ref, 1);
      hypre_ParCSRMatrixScale(parcsr_ref2, 2.0);

      ierr += checkMatrix(parcsr_ref2, ij_A) > tol;
      if (print_matrix)
      {
         HYPRE_IJMatrixPrint(ij_A, "ij_SetAdd");
      }
      HYPRE_IJMatrixDestroy(ij_A);
      HYPRE_ParCSRMatrixDestroy(parcsr_ref2);
   }

   /* Test Set/Add/Set */
   if (mode & 32)
   {
      test_SetAddSet(comm, memory_location, option, ilower, iupper, jlower, jupper, nrows, num_nonzeros,
                     nchunks, h_nnzrow, nnzrow, option == 1 ? rows : rows_coo, cols, coefs, &ij_A);

      ierr += checkMatrix(parcsr_ref, ij_A) > tol;
      if (print_matrix)
      {
         HYPRE_IJMatrixPrint(ij_A, "ij_SetAddSet");
      }
      HYPRE_IJMatrixDestroy(ij_A);
   }

   /* Print the error code */
   hypre_ParPrintf(comm, "Test error code = %d\n", ierr);

   /*-----------------------------------------------------------
    * Free memory
    *-----------------------------------------------------------*/
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      hypre_TFree(d_nnzrow, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_rows,   HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_rows_coo,  HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_cols,   HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_coefs,  HYPRE_MEMORY_DEVICE);
   }
   hypre_TFree(h_nnzrow, HYPRE_MEMORY_HOST);
   hypre_TFree(h_rows,   HYPRE_MEMORY_HOST);
   hypre_TFree(h_rows_coo,  HYPRE_MEMORY_HOST);
   hypre_TFree(h_cols,   HYPRE_MEMORY_HOST);
   hypre_TFree(h_coefs,  HYPRE_MEMORY_HOST);

   HYPRE_ParCSRMatrixDestroy(parcsr_ref);

   /* Finalize Hypre */
   HYPRE_Finalize();

   /* Finalize MPI */
   hypre_MPI_Finalize();

   /* when using cuda-memcheck --leak-check full, uncomment this */
#if defined(HYPRE_USING_GPU)
   hypre_ResetCudaDevice(NULL);
#endif

   return (0);
}

HYPRE_Int
buildMatrixEntries(MPI_Comm            comm,
                   HYPRE_Int           nx,
                   HYPRE_Int           ny,
                   HYPRE_Int           nz,
                   HYPRE_Int           Px,
                   HYPRE_Int           Py,
                   HYPRE_Int           Pz,
                   HYPRE_Real          cx,
                   HYPRE_Real          cy,
                   HYPRE_Real          cz,
                   HYPRE_Int           base,
                   HYPRE_BigInt       *ilower_ptr,
                   HYPRE_BigInt       *iupper_ptr,
                   HYPRE_BigInt       *jlower_ptr,
                   HYPRE_BigInt       *jupper_ptr,
                   HYPRE_Int          *nrows_ptr,
                   HYPRE_BigInt       *num_nonzeros_ptr,
                   HYPRE_Int         **nnzrow_ptr,
                   HYPRE_BigInt      **rows_ptr,   /* row indices of length nrows */
                   HYPRE_BigInt      **rows_coo_ptr,  /* row indices of length nnz */
                   HYPRE_BigInt      **cols_ptr,   /* col indices of length nnz */
                   HYPRE_Real        **coefs_ptr,  /* values of length nnz */
                   HYPRE_Int           stencil,
                   HYPRE_ParCSRMatrix *parcsr_ptr)
{
   HYPRE_Int          num_procs;
   HYPRE_Int          myid;
   HYPRE_Real         values[4];
   HYPRE_ParCSRMatrix A = NULL;

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

   HYPRE_Int ip = myid % Px;
   HYPRE_Int iq = (( myid - ip) / Px) % Py;
   HYPRE_Int ir = ( myid - ip - Px * iq) / ( Px * Py );

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

   hypre_ParCSRMatrixMigrate(A, HYPRE_MEMORY_HOST);

   /* modify the upper triangular part to A nonsymmetric */
   HYPRE_Int           i, j;
   HYPRE_Int           nrows    = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix    *A_diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix    *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int          *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int          *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int          *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int          *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_BigInt       *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_BigInt       ilower = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_BigInt       jlower = hypre_ParCSRMatrixFirstColDiag(A);

   for (i = 0; i < nrows; i++)
   {
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         HYPRE_BigInt row = ilower + (HYPRE_BigInt) i;
         HYPRE_BigInt col = jlower + (HYPRE_BigInt) A_diag_j[j];
         if (row < col)
         {
            hypre_CSRMatrixData(A_diag)[j] += myid + .89;
         }
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         HYPRE_BigInt row = ilower + (HYPRE_BigInt) i;
         HYPRE_BigInt col = col_map_offd_A[A_offd_j[j]];
         if (row < col)
         {
            hypre_CSRMatrixData(A_offd)[j] += myid + .64;
         }
      }
   }

   /* get I, J, data from A */
   getParCSRMatrixData(A, base, nrows_ptr, num_nonzeros_ptr, nnzrow_ptr, rows_ptr, rows_coo_ptr,
                       cols_ptr,
                       coefs_ptr);

   // Set pointers
   *ilower_ptr = hypre_ParCSRMatrixFirstRowIndex(A) + base;
   *iupper_ptr = hypre_ParCSRMatrixLastRowIndex(A) + base;
   *jlower_ptr = hypre_ParCSRMatrixFirstColDiag(A) + base;
   *jupper_ptr = hypre_ParCSRMatrixLastColDiag(A) + base;
   *parcsr_ptr = A;

   return hypre_error_flag;
}

HYPRE_Int
getParCSRMatrixData(HYPRE_ParCSRMatrix  A,
                    HYPRE_Int           base,
                    HYPRE_Int          *nrows_ptr,
                    HYPRE_BigInt       *num_nonzeros_ptr,
                    HYPRE_Int         **nnzrow_ptr,
                    HYPRE_BigInt      **rows_ptr,
                    HYPRE_BigInt      **rows_coo_ptr,
                    HYPRE_BigInt      **cols_ptr,
                    HYPRE_Real        **coefs_ptr)
{
   hypre_CSRMatrix    *A_diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix    *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int          *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int          *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int          *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int          *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_BigInt       *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);

   HYPRE_BigInt       ilower = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_BigInt       jlower = hypre_ParCSRMatrixFirstColDiag(A);

   HYPRE_Int          nrows;
   HYPRE_BigInt       num_nonzeros;
   HYPRE_Int         *nnzrow;
   HYPRE_BigInt      *rows;
   HYPRE_BigInt      *rows_coo;
   HYPRE_BigInt      *cols;
   HYPRE_Real        *coefs;
   HYPRE_Int          i, j, k;

   nrows  = hypre_ParCSRMatrixNumRows(A);
   num_nonzeros = hypre_CSRMatrixNumNonzeros(A_diag) + hypre_CSRMatrixNumNonzeros(A_offd);
   nnzrow = hypre_CTAlloc(HYPRE_Int,    nrows,        HYPRE_MEMORY_HOST);
   rows   = hypre_CTAlloc(HYPRE_BigInt, nrows,        HYPRE_MEMORY_HOST);
   rows_coo  = hypre_CTAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_HOST);
   cols   = hypre_CTAlloc(HYPRE_BigInt, num_nonzeros, HYPRE_MEMORY_HOST);
   coefs  = hypre_CTAlloc(HYPRE_Real,   num_nonzeros, HYPRE_MEMORY_HOST);

   k = 0;
#if 0
   for (i = 0; i < nrows; i++)
   {
      nnzrow[i] = A_diag_i[i + 1] - A_diag_i[i] +
                  A_offd_i[i + 1] - A_offd_i[i];
      rows[i]   = ilower + i;

      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         rows_coo[k]   = ilower + (HYPRE_BigInt) i;
         cols[k]    = jlower + (HYPRE_BigInt) A_diag_j[j];
         coefs[k++] = hypre_CSRMatrixData(A_diag)[j];
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         rows_coo[k]   = ilower + (HYPRE_BigInt) i;
         cols[k]    = hypre_ParCSRMatrixColMapOffd(A)[A_offd_j[j]];
         coefs[k++] = hypre_CSRMatrixData(A_offd)[j];
      }
   }
#else
   for (i = nrows - 1; i >= 0; i--)
   {
      nnzrow[nrows - 1 - i] = A_diag_i[i + 1] - A_diag_i[i] +
                              A_offd_i[i + 1] - A_offd_i[i];
      rows[nrows - 1 - i]   = ilower + i + base;

      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         rows_coo[k]   = ilower + (HYPRE_BigInt) i + base;
         cols[k]    = jlower + (HYPRE_BigInt) A_diag_j[j] + base;
         coefs[k++] = hypre_CSRMatrixData(A_diag)[j];
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         rows_coo[k]   = ilower + (HYPRE_BigInt) i + base;
         cols[k]    = col_map_offd_A[A_offd_j[j]] + base;
         coefs[k++] = hypre_CSRMatrixData(A_offd)[j];
      }
   }
#endif

   hypre_assert(k == num_nonzeros);

   // Set pointers
   *nrows_ptr        = nrows;
   *num_nonzeros_ptr = num_nonzeros;
   *nnzrow_ptr       = nnzrow;
   *rows_ptr         = rows;
   *rows_coo_ptr     = rows_coo;
   *cols_ptr         = cols;
   *coefs_ptr        = coefs;

   return hypre_error_flag;
}


HYPRE_Real
checkMatrix(HYPRE_ParCSRMatrix h_parcsr_ref, HYPRE_IJMatrix ij_A)
{
   MPI_Comm            comm         = hypre_IJMatrixComm(ij_A);
   HYPRE_ParCSRMatrix  parcsr_A     = (HYPRE_ParCSRMatrix) hypre_IJMatrixObject(ij_A);
   HYPRE_ParCSRMatrix  h_parcsr_A;
   HYPRE_ParCSRMatrix  parcsr_error;
   HYPRE_Real          fnorm;

   h_parcsr_A = hypre_ParCSRMatrixClone_v2(parcsr_A, 1, HYPRE_MEMORY_HOST);

   // Check norm of (parcsr_ref - parcsr_A)
   hypre_ParCSRMatrixAdd(1.0, h_parcsr_ref, -1.0, h_parcsr_A, &parcsr_error);
   fnorm = hypre_ParCSRMatrixFnorm(parcsr_error);

   hypre_ParPrintf(comm, "Frobenius norm of (A_ref - A): %e\n", fnorm);

   HYPRE_ParCSRMatrixDestroy(h_parcsr_A);
   HYPRE_ParCSRMatrixDestroy(parcsr_error);

   return fnorm;
}

/* ---------------------------------- *
 * All test functions for assembly    *
 * option = 1: length of nrows,       *
 *        = 2: length of num_nonzeros *
 * ---------------------------------- */

/* set values */
HYPRE_Int
test_Set(MPI_Comm             comm,
         HYPRE_MemoryLocation memory_location,
         HYPRE_Int            option,
         HYPRE_BigInt         ilower,
         HYPRE_BigInt         iupper,
         HYPRE_BigInt         jlower,
         HYPRE_BigInt         jupper,
         HYPRE_Int            nrows,
         HYPRE_BigInt         num_nonzeros,
         HYPRE_Int            nchunks,
         HYPRE_Int           *h_nnzrow,
         HYPRE_Int           *nnzrow,
         HYPRE_BigInt        *rows,
         HYPRE_BigInt        *cols,
         HYPRE_Real          *coefs,
         HYPRE_IJMatrix      *ij_A_ptr)
{
   HYPRE_IJMatrix  ij_A;
   HYPRE_Int       i, chunk, chunk_size;
   HYPRE_Int       time_index;
   HYPRE_Int      *h_rowptr = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_HOST);

   for (i = 1; i < nrows + 1; i++)
   {
      h_rowptr[i] = h_rowptr[i - 1] + h_nnzrow[i - 1];
   }
   hypre_assert(h_rowptr[nrows] == num_nonzeros);

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &ij_A);
   HYPRE_IJMatrixSetObjectType(ij_A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize_v2(ij_A, memory_location);
   HYPRE_IJMatrixSetOMPFlag(ij_A, 1);

   chunk_size = nrows / nchunks;

#if defined(HYPRE_USING_GPU)
   hypre_SyncCudaDevice(hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStart();
#endif
#endif

   time_index = hypre_InitializeTiming("Test SetValues");
   hypre_BeginTiming(time_index);
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Assemble matrix
   HYPRE_IJMatrixAssemble(ij_A);

#if defined(HYPRE_USING_GPU)
   hypre_SyncCudaDevice(hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStop();
#endif
#endif

   // Finalize timer
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Test SetValues", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   // Free memory
   hypre_TFree(h_rowptr, HYPRE_MEMORY_HOST);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return hypre_error_flag;
}

/* set values with (row, col) reversed, i.e., the transpose of A
 * in this way, we can test off-proc set values */
HYPRE_Int
test_AddTranspose(MPI_Comm             comm,
                  HYPRE_MemoryLocation memory_location,
                  HYPRE_Int            option,
                  HYPRE_BigInt         ilower,
                  HYPRE_BigInt         iupper,
                  HYPRE_BigInt         jlower,
                  HYPRE_BigInt         jupper,
                  HYPRE_Int            nrows,
                  HYPRE_BigInt         num_nonzeros,
                  HYPRE_Int            nchunks,
                  HYPRE_Int           *h_nnzrow,
                  HYPRE_Int           *nnzrow,
                  HYPRE_BigInt        *rows,
                  HYPRE_BigInt        *cols,
                  HYPRE_Real          *coefs,
                  HYPRE_IJMatrix      *ij_AT_ptr)


{
   hypre_assert(option == 2);

   HYPRE_IJMatrix  ij_AT;
   HYPRE_Int       i, chunk, chunk_size;
   HYPRE_Int       time_index;
   HYPRE_Int      *h_rowptr = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_HOST);

   for (i = 1; i < nrows + 1; i++)
   {
      h_rowptr[i] = h_rowptr[i - 1] + h_nnzrow[i - 1];
   }
   hypre_assert(h_rowptr[nrows] == num_nonzeros);

   HYPRE_IJMatrixCreate(comm, jlower, jupper, ilower, iupper, &ij_AT);
   HYPRE_IJMatrixSetObjectType(ij_AT, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize_v2(ij_AT, memory_location);
   HYPRE_IJMatrixSetOMPFlag(ij_AT, 1);

   chunk_size = nrows / nchunks;

#if defined(HYPRE_USING_GPU)
   hypre_SyncCudaDevice(hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStart();
#endif
#endif

   time_index = hypre_InitializeTiming("Test SetValuesTranspose");
   hypre_BeginTiming(time_index);

   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = hypre_min(chunk_size, nrows - chunk);

      HYPRE_IJMatrixAddToValues(ij_AT, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                NULL, &cols[h_rowptr[chunk]],
                                &rows[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
   }

   // Assemble matrix
   HYPRE_IJMatrixAssemble(ij_AT);

#if defined(HYPRE_USING_GPU)
   hypre_SyncCudaDevice(hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStop();
#endif
#endif

   // Finalize timer
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Test SetValuesTranspose", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   // Free memory
   hypre_TFree(h_rowptr, HYPRE_MEMORY_HOST);

   // Set pointer to output
   *ij_AT_ptr = ij_AT;

   return hypre_error_flag;
}

HYPRE_Int
test_SetSet(MPI_Comm             comm,
            HYPRE_MemoryLocation memory_location,
            HYPRE_Int            option,
            HYPRE_BigInt         ilower,
            HYPRE_BigInt         iupper,
            HYPRE_BigInt         jlower,
            HYPRE_BigInt         jupper,
            HYPRE_Int            nrows,
            HYPRE_BigInt         num_nonzeros,
            HYPRE_Int            nchunks,
            HYPRE_Int           *h_nnzrow,
            HYPRE_Int           *nnzrow,
            HYPRE_BigInt        *rows,
            HYPRE_BigInt        *cols,
            HYPRE_Real          *coefs,
            HYPRE_IJMatrix      *ij_A_ptr)
{
   HYPRE_IJMatrix  ij_A;
   HYPRE_Int       i, chunk, chunk_size;
   HYPRE_Int       time_index;
   HYPRE_Int      *h_rowptr;
   HYPRE_Real     *new_coefs;

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &ij_A);
   HYPRE_IJMatrixSetObjectType(ij_A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize_v2(ij_A, memory_location);
   HYPRE_IJMatrixSetOMPFlag(ij_A, 1);

   h_rowptr = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_HOST);
   for (i = 1; i < nrows + 1; i++)
   {
      h_rowptr[i] = h_rowptr[i - 1] + h_nnzrow[i - 1];
   }
   hypre_assert(h_rowptr[nrows] == num_nonzeros);

   chunk_size = nrows / nchunks;
   new_coefs = hypre_TAlloc(HYPRE_Real, num_nonzeros, memory_location);

   if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_HOST)
   {
      for (i = 0; i < num_nonzeros; i++)
      {
         new_coefs[i] = 2.0 * coefs[i];
      }
   }
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   else
   {
      hypre_TMemcpy(new_coefs, coefs, HYPRE_Real, num_nonzeros, memory_location, memory_location);
      hypreDevice_ComplexScalen(new_coefs, num_nonzeros, new_coefs, 2.0);
   }
#endif

#if defined(HYPRE_USING_GPU)
   hypre_SyncCudaDevice(hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStart();
#endif
#endif

   // First Set
   time_index = hypre_InitializeTiming("Test Set/Set");
   hypre_BeginTiming(time_index);
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &new_coefs[h_rowptr[chunk]]);
      }
      else
      {
         HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &new_coefs[h_rowptr[chunk]]);
      }
   }

   // Assemble matrix
   HYPRE_IJMatrixAssemble(ij_A);

   // Second set
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Assemble matrix
   HYPRE_IJMatrixAssemble(ij_A);

#if defined(HYPRE_USING_GPU)
   hypre_SyncCudaDevice(hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStop();
#endif
#endif

   // Finalize timer
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Test Set/Set", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   // Free memory
   hypre_TFree(h_rowptr, HYPRE_MEMORY_HOST);
   hypre_TFree(new_coefs, memory_location);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return hypre_error_flag;
}

HYPRE_Int
test_AddSet(MPI_Comm             comm,
            HYPRE_MemoryLocation memory_location,
            HYPRE_Int            option,
            HYPRE_BigInt         ilower,
            HYPRE_BigInt         iupper,
            HYPRE_BigInt         jlower,
            HYPRE_BigInt         jupper,
            HYPRE_Int            nrows,
            HYPRE_BigInt         num_nonzeros,
            HYPRE_Int            nchunks,
            HYPRE_Int           *h_nnzrow,
            HYPRE_Int           *nnzrow,
            HYPRE_BigInt        *rows,
            HYPRE_BigInt        *cols,
            HYPRE_Real          *coefs,
            HYPRE_IJMatrix      *ij_A_ptr)
{
   HYPRE_IJMatrix  ij_A;
   HYPRE_Int       i, chunk, chunk_size;
   HYPRE_Int       time_index;
   HYPRE_Int      *h_rowptr;
   HYPRE_Real     *new_coefs;

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &ij_A);
   HYPRE_IJMatrixSetObjectType(ij_A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize_v2(ij_A, memory_location);
   HYPRE_IJMatrixSetOMPFlag(ij_A, 1);

   h_rowptr = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_HOST);
   for (i = 1; i < nrows + 1; i++)
   {
      h_rowptr[i] = h_rowptr[i - 1] + h_nnzrow[i - 1];
   }
   hypre_assert(h_rowptr[nrows] == num_nonzeros);

   chunk_size = nrows / nchunks;
   new_coefs = hypre_TAlloc(HYPRE_Real, num_nonzeros, memory_location);

   if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_HOST)
   {
      for (i = 0; i < num_nonzeros; i++)
      {
         new_coefs[i] = 2.0 * coefs[i];
      }
   }
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   else
   {
      hypre_TMemcpy(new_coefs, coefs, HYPRE_Real, num_nonzeros, memory_location, memory_location);
      hypreDevice_ComplexScalen(new_coefs, num_nonzeros, new_coefs, 2.0);
   }
#endif

#if defined(HYPRE_USING_GPU)
   hypre_SyncCudaDevice(hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStart();
#endif
#endif

   // First Add
   time_index = hypre_InitializeTiming("Test Add/Set");
   hypre_BeginTiming(time_index);
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         HYPRE_IJMatrixAddToValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                   &cols[h_rowptr[chunk]], &new_coefs[h_rowptr[chunk]]);
      }
      else
      {
         HYPRE_IJMatrixAddToValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                   NULL, &rows[h_rowptr[chunk]],
                                   &cols[h_rowptr[chunk]], &new_coefs[h_rowptr[chunk]]);
      }
   }

   // Then Set
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Assemble matrix
   HYPRE_IJMatrixAssemble(ij_A);

#if defined(HYPRE_USING_GPU)
   hypre_SyncCudaDevice(hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStop();
#endif
#endif

   // Finalize timer
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Test Add/Set", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   // Free memory
   hypre_TFree(h_rowptr, HYPRE_MEMORY_HOST);
   hypre_TFree(new_coefs, memory_location);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return hypre_error_flag;
}

HYPRE_Int
test_SetAdd(MPI_Comm             comm,
            HYPRE_MemoryLocation memory_location,
            HYPRE_Int            option,
            HYPRE_BigInt         ilower,
            HYPRE_BigInt         iupper,
            HYPRE_BigInt         jlower,
            HYPRE_BigInt         jupper,
            HYPRE_Int            nrows,
            HYPRE_BigInt         num_nonzeros,
            HYPRE_Int            nchunks,
            HYPRE_Int           *h_nnzrow,
            HYPRE_Int           *nnzrow,
            HYPRE_BigInt        *rows,
            HYPRE_BigInt        *cols,
            HYPRE_Real          *coefs,
            HYPRE_IJMatrix      *ij_A_ptr)
{
   HYPRE_IJMatrix  ij_A;
   HYPRE_Int       i, chunk, chunk_size;
   HYPRE_Int       time_index;
   HYPRE_Int      *h_rowptr;

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &ij_A);
   HYPRE_IJMatrixSetObjectType(ij_A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize_v2(ij_A, memory_location);
   HYPRE_IJMatrixSetOMPFlag(ij_A, 1);

   h_rowptr = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_HOST);
   for (i = 1; i < nrows + 1; i++)
   {
      h_rowptr[i] = h_rowptr[i - 1] + h_nnzrow[i - 1];
   }
   hypre_assert(h_rowptr[nrows] == num_nonzeros);

   chunk_size = nrows / nchunks;

#if defined(HYPRE_USING_GPU)
   hypre_SyncCudaDevice(hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStart();
#endif
#endif

   // First Set
   time_index = hypre_InitializeTiming("Test Set/Add");
   hypre_BeginTiming(time_index);
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Second Add
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         HYPRE_IJMatrixAddToValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                   &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         HYPRE_IJMatrixAddToValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                   NULL, &rows[h_rowptr[chunk]],
                                   &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Assemble matrix
   HYPRE_IJMatrixAssemble(ij_A);

#if defined(HYPRE_USING_GPU)
   hypre_SyncCudaDevice(hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStop();
#endif
#endif

   // Finalize timer
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Test Set/Add", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   // Free memory
   hypre_TFree(h_rowptr, HYPRE_MEMORY_HOST);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return hypre_error_flag;
}

HYPRE_Int
test_SetAddSet(MPI_Comm             comm,
               HYPRE_MemoryLocation memory_location,
               HYPRE_Int            option,           /* 1 or 2 */
               HYPRE_BigInt         ilower,
               HYPRE_BigInt         iupper,
               HYPRE_BigInt         jlower,
               HYPRE_BigInt         jupper,
               HYPRE_Int            nrows,
               HYPRE_BigInt         num_nonzeros,
               HYPRE_Int            nchunks,
               HYPRE_Int           *h_nnzrow,
               HYPRE_Int           *nnzrow,
               HYPRE_BigInt
               *rows,             /* option = 1: length of nrows, = 2: length of num_nonzeros */
               HYPRE_BigInt        *cols,
               HYPRE_Real          *coefs,
               HYPRE_IJMatrix      *ij_A_ptr)
{
   HYPRE_IJMatrix  ij_A;
   HYPRE_Int       i, chunk, chunk_size;
   HYPRE_Int       time_index;
   HYPRE_Int      *h_rowptr;

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &ij_A);
   HYPRE_IJMatrixSetObjectType(ij_A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize_v2(ij_A, memory_location);
   HYPRE_IJMatrixSetOMPFlag(ij_A, 1);

   h_rowptr = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_HOST);
   for (i = 1; i < nrows + 1; i++)
   {
      h_rowptr[i] = h_rowptr[i - 1] + h_nnzrow[i - 1];
   }
   hypre_assert(h_rowptr[nrows] == num_nonzeros);
   chunk_size = nrows / nchunks;

#if defined(HYPRE_USING_GPU)
   hypre_SyncCudaDevice(hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStart();
#endif
#endif

   // First Set
   time_index = hypre_InitializeTiming("Test Set/Add/Set");
   hypre_BeginTiming(time_index);
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Then Add
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         HYPRE_IJMatrixAddToValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                   &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         HYPRE_IJMatrixAddToValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                   NULL, &rows[h_rowptr[chunk]],
                                   &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Then Set
   for (chunk = 0; chunk < nrows; chunk += chunk_size)
   {
      chunk_size = hypre_min(chunk_size, nrows - chunk);

      if (1 == option)
      {
         HYPRE_IJMatrixSetValues(ij_A, chunk_size, &nnzrow[chunk], &rows[chunk],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
      else
      {
         HYPRE_IJMatrixSetValues(ij_A, h_rowptr[chunk + chunk_size] - h_rowptr[chunk],
                                 NULL, &rows[h_rowptr[chunk]],
                                 &cols[h_rowptr[chunk]], &coefs[h_rowptr[chunk]]);
      }
   }

   // Assemble matrix
   HYPRE_IJMatrixAssemble(ij_A);

#if defined(HYPRE_USING_GPU)
   hypre_SyncCudaDevice(hypre_handle());
#if defined(CUDA_PROFILER)
   cudaProfilerStop();
#endif
#endif

   // Finalize timer
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Test Set/Add/Set", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   // Free memory
   hypre_TFree(h_rowptr, HYPRE_MEMORY_HOST);

   // Set pointer to matrix
   *ij_A_ptr = ij_A;

   return hypre_error_flag;
}

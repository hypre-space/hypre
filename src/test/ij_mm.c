/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (IJ_matrix interface).
 * Do `driver -help' for usage info.
 * This driver started from the driver for parcsr_linear_solvers, and it
 * works by first building a parcsr matrix as before and then "copying"
 * that matrix row-by-row into the IJMatrix interface. AJC 7/99.
 *--------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_krylov.h"

#ifdef HYPRE_USING_CUDA
#include "cuda_profiler_api.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

HYPRE_Int BuildParFromFile (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                            HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParRhsFromFile (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                               HYPRE_ParVector *b_ptr );

HYPRE_Int BuildParLaplacian (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                             HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParSysLaplacian (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParDifConv (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                           HYPRE_ParCSRMatrix *A_ptr);
HYPRE_Int BuildFuncsFromFiles (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                               HYPRE_ParCSRMatrix A, HYPRE_Int **dof_func_ptr );
HYPRE_Int BuildFuncsFromOneFile (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                 HYPRE_ParCSRMatrix A, HYPRE_Int **dof_func_ptr );
HYPRE_Int BuildParLaplacian9pt (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian27pt (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                 HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParRotate7pt (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                             HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParVarDifConv (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                              HYPRE_ParCSRMatrix *A_ptr, HYPRE_ParVector *rhs_ptr );
HYPRE_Int SetSysVcoefValues(HYPRE_Int num_fun, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                            HYPRE_Real vcx, HYPRE_Real vcy, HYPRE_Real vcz, HYPRE_Int mtx_entry, HYPRE_Real *values);

HYPRE_Int BuildParCoordinates (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                               HYPRE_Int *coorddim_ptr, float **coord_ptr );

#ifdef __cplusplus
}
#endif

char *gpu_ptr = NULL;
size_t total_size = 0, current_size = 0;

void gpu_alloc(void **ptr, size_t size)
{
   size = ((size + 128 - 1) / 128) * 128;
   *ptr = (void *) &gpu_ptr[current_size];
   current_size += size;
   if (current_size > total_size)
   {
      hypre_printf("Out of GPU memory\n");
      exit(0);
   }
}

void gpu_free(void *ptr)
{
   HYPRE_UNUSED_VAR(ptr);
   return;
}

char file_dir[] = "./";

void runjob1( HYPRE_ParCSRMatrix parcsr_A,
              HYPRE_Int          print_system,
              HYPRE_Int          rep,
              HYPRE_Int          verify)
{
   HYPRE_Int          i;
   HYPRE_ParCSRMatrix parcsr_A_host  = NULL;
   HYPRE_ParCSRMatrix parcsr_B_host  = NULL;
   HYPRE_ParCSRMatrix parcsr_B_host2 = NULL;
   HYPRE_ParCSRMatrix parcsr_B       = NULL;
   HYPRE_ParCSRMatrix parcsr_error_host = NULL;
   HYPRE_Real         fnorm, rfnorm, fnorm0;
   HYPRE_Int          time_index;
   char               fname[1024];

   MPI_Comm comm = hypre_ParCSRMatrixComm(parcsr_A);

   hypre_ParPrintf(comm, "A %d x %d, NNZ %d, RNZ %d\n", hypre_ParCSRMatrixGlobalNumRows(parcsr_A),
                   hypre_ParCSRMatrixGlobalNumCols(parcsr_A),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_A),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_A) / hypre_ParCSRMatrixGlobalNumRows(parcsr_A));

   hypre_assert(hypre_ParCSRMatrixMemoryLocation(parcsr_A) == HYPRE_MEMORY_DEVICE);

   hypre_MatvecCommPkgCreate(parcsr_A);
   hypre_ParCSRMatrixCopyColMapOffdToDevice(parcsr_A);

   if (verify)
   {
      parcsr_A_host = hypre_ParCSRMatrixClone_v2(parcsr_A, 1, HYPRE_MEMORY_HOST);
      hypre_MatvecCommPkgCreate(parcsr_A_host);

      time_index = hypre_InitializeTiming("Host Parcsr Matrix-by-Matrix, A*A");

      hypre_BeginTiming(time_index);

      parcsr_B_host = hypre_ParCSRMatMat(parcsr_A_host, parcsr_A_host);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Host Parcsr Matrix-by-Matrix, A*A", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      hypre_ParCSRMatrixSetNumNonzeros(parcsr_B_host);
   }

   if (print_system)
   {
      sprintf(fname, "%s/%s", file_dir, "IJ.out.A");
      hypre_ParCSRMatrixPrintIJ(parcsr_A, 0, 0, fname);

      //hypre_CSRMatrixPrintMM(hypre_ParCSRMatrixDiag(parcsr_A_host), 1, 1, 0, "A.mtx");
   }

   for (i = 0 ; i < rep; i++)
   {
      hypre_ParPrintf(comm, "--- rep %d (out of %d) ---\n", i, rep);

      if (i == rep - 1)
      {
         time_index = hypre_InitializeTiming("Device Parcsr Matrix-by-Matrix, A*A");
         hypre_BeginTiming(time_index);
         //cudaProfilerStart();
      }

      parcsr_B = hypre_ParCSRMatMat(parcsr_A, parcsr_A);

      if (i == rep - 1)
      {
#if defined(HYPRE_USING_GPU)
         hypre_SyncCudaDevice(hypre_handle());
#endif
         //cudaProfilerStop();
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Device Parcsr Matrix-by-Matrix, A*A", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
      }

      if (i < rep - 1)
      {
         hypre_ParCSRMatrixDestroy(parcsr_B);
      }
   }

   if (verify)
   {
      parcsr_B_host2 = hypre_ParCSRMatrixClone_v2(parcsr_B, 1, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixSetNumNonzeros(parcsr_B_host2);

      hypre_ParCSRMatrixAdd(1.0, parcsr_B_host, -1.0, parcsr_B_host2, &parcsr_error_host);
      fnorm = hypre_ParCSRMatrixFnorm(parcsr_error_host);
      fnorm0 = hypre_ParCSRMatrixFnorm(parcsr_B_host);
      rfnorm = fnorm0 > 0 ? fnorm / fnorm0 : fnorm;

      hypre_ParPrintf(comm, "A^2: %d x %d, nnz [CPU %d, GPU %d], CPU-GPU err %e\n",
                      hypre_ParCSRMatrixGlobalNumRows(parcsr_B_host2),
                      hypre_ParCSRMatrixGlobalNumCols(parcsr_B_host2),
                      hypre_ParCSRMatrixNumNonzeros(parcsr_B_host),
                      hypre_ParCSRMatrixNumNonzeros(parcsr_B_host2),
                      rfnorm);
   }

   if (print_system)
   {
      if (!parcsr_B_host2)
      {
         parcsr_B_host2 = hypre_ParCSRMatrixClone_v2(parcsr_B, 1, HYPRE_MEMORY_HOST);
      }
      sprintf(fname, "%s/%s", file_dir, "IJ.out.B");
      hypre_ParCSRMatrixPrintIJ(parcsr_B_host2, 0, 0, fname);
      sprintf(fname, "%s/%s", file_dir, "IJ.out.B.CPU");
      hypre_ParCSRMatrixPrintIJ(parcsr_B_host, 0, 0, fname);
   }

   hypre_ParCSRMatrixSetNumNonzeros(parcsr_B);

   hypre_ParPrintf(comm, "B %d x %d, NNZ %d, RNZ %d\n", hypre_ParCSRMatrixGlobalNumRows(parcsr_B),
                   hypre_ParCSRMatrixGlobalNumCols(parcsr_B),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_B),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_B) / hypre_ParCSRMatrixGlobalNumRows(parcsr_B));

   hypre_ParCSRMatrixDestroy(parcsr_B);
   hypre_ParCSRMatrixDestroy(parcsr_A_host);
   hypre_ParCSRMatrixDestroy(parcsr_B_host);
   hypre_ParCSRMatrixDestroy(parcsr_B_host2);
   hypre_ParCSRMatrixDestroy(parcsr_error_host);
}

void runjob2( HYPRE_ParCSRMatrix parcsr_A,
              HYPRE_Int          print_system,
              HYPRE_Int          rep,
              HYPRE_Int          verify)
{
   HYPRE_Int          i;
   HYPRE_ParCSRMatrix parcsr_A_host  = NULL;
   HYPRE_ParCSRMatrix parcsr_B_host  = NULL;
   HYPRE_ParCSRMatrix parcsr_B_host2 = NULL;
   HYPRE_ParCSRMatrix parcsr_B       = NULL;
   HYPRE_ParCSRMatrix parcsr_error_host = NULL;
   HYPRE_Real         fnorm, rfnorm, fnorm0;
   HYPRE_Int          time_index;
   char               fname[1024];

   MPI_Comm comm = hypre_ParCSRMatrixComm(parcsr_A);

   hypre_ParPrintf(comm, "A %d x %d, NNZ %d, RNZ %d\n", hypre_ParCSRMatrixGlobalNumRows(parcsr_A),
                   hypre_ParCSRMatrixGlobalNumCols(parcsr_A),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_A),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_A) / hypre_ParCSRMatrixGlobalNumRows(parcsr_A));

   hypre_assert(hypre_ParCSRMatrixMemoryLocation(parcsr_A) == HYPRE_MEMORY_DEVICE);

   hypre_MatvecCommPkgCreate(parcsr_A);
   hypre_ParCSRMatrixCopyColMapOffdToDevice(parcsr_A);

   if (verify)
   {
      parcsr_A_host = hypre_ParCSRMatrixClone_v2(parcsr_A, 1, HYPRE_MEMORY_HOST);
      hypre_MatvecCommPkgCreate(parcsr_A_host);

      time_index = hypre_InitializeTiming("Host Parcsr Matrix-by-Matrix, AT*A");

      hypre_BeginTiming(time_index);

      parcsr_B_host = hypre_ParCSRTMatMat(parcsr_A_host, parcsr_A_host);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Host Parcsr Matrix-by-Matrix, AT*A", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      hypre_ParCSRMatrixSetNumNonzeros(parcsr_B_host);
   }

   if (print_system)
   {
      if (!parcsr_A_host)
      {
         parcsr_A_host = hypre_ParCSRMatrixClone_v2(parcsr_A, 1, HYPRE_MEMORY_HOST);
      }
      sprintf(fname, "%s/%s", file_dir, "IJ.out.A");
      hypre_ParCSRMatrixPrintIJ(parcsr_A_host, 0, 0, fname);
   }

   for (i = 0 ; i < rep; i++)
   {
      hypre_ParPrintf(comm, "--- rep %d (out of %d) ---\n", i, rep);

      if (i == rep - 1)
      {
         time_index = hypre_InitializeTiming("Device Parcsr Matrix-by-Matrix, AT*A");
         hypre_BeginTiming(time_index);
         //cudaProfilerStart();
      }

      parcsr_B = hypre_ParCSRTMatMatKT(parcsr_A, parcsr_A, 0);

      if (i == rep - 1)
      {
#if defined(HYPRE_USING_GPU)
         hypre_SyncCudaDevice(hypre_handle());
#endif
         //cudaProfilerStop();
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Device Parcsr Matrix-by-Matrix, AT*A", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
      }

      if (i < rep - 1)
      {
         hypre_ParCSRMatrixDestroy(parcsr_B);
      }
   }

   if (verify)
   {
      parcsr_B_host2 = hypre_ParCSRMatrixClone_v2(parcsr_B, 1, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixSetNumNonzeros(parcsr_B_host2);

      hypre_ParCSRMatrixAdd(1.0, parcsr_B_host, -1.0, parcsr_B_host2, &parcsr_error_host);
      fnorm = hypre_ParCSRMatrixFnorm(parcsr_error_host);
      fnorm0 = hypre_ParCSRMatrixFnorm(parcsr_B_host);
      rfnorm = fnorm0 > 0 ? fnorm / fnorm0 : fnorm;

      hypre_ParPrintf(comm, "A^T*A: %d x %d, nnz [CPU %d, GPU %d], CPU-GPU err %e\n",
                      hypre_ParCSRMatrixGlobalNumRows(parcsr_B_host2),
                      hypre_ParCSRMatrixGlobalNumCols(parcsr_B_host2),
                      hypre_ParCSRMatrixNumNonzeros(parcsr_B_host),
                      hypre_ParCSRMatrixNumNonzeros(parcsr_B_host2),
                      rfnorm);
   }

   if (print_system)
   {
      sprintf(fname, "%s/%s", file_dir, "IJ.out.B");
      hypre_ParCSRMatrixPrintIJ(parcsr_B, 0, 0, fname);
      sprintf(fname, "%s/%s", file_dir, "IJ.out.B.CPU");
      hypre_ParCSRMatrixPrintIJ(parcsr_B_host, 0, 0, fname);
   }

   hypre_ParCSRMatrixSetNumNonzeros(parcsr_B);

   hypre_ParPrintf(comm, "B %d x %d, NNZ %d, RNZ %d\n", hypre_ParCSRMatrixGlobalNumRows(parcsr_B),
                   hypre_ParCSRMatrixGlobalNumCols(parcsr_B),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_B),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_B) / hypre_ParCSRMatrixGlobalNumRows(parcsr_B));

   hypre_ParCSRMatrixDestroy(parcsr_B);
   hypre_ParCSRMatrixDestroy(parcsr_A_host);
   hypre_ParCSRMatrixDestroy(parcsr_B_host);
   hypre_ParCSRMatrixDestroy(parcsr_B_host2);
   hypre_ParCSRMatrixDestroy(parcsr_error_host);
}

void runjob3( HYPRE_ParCSRMatrix parcsr_A,
              HYPRE_ParCSRMatrix parcsr_P,
              HYPRE_Int          boolean_P,
              HYPRE_Int          print_system,
              HYPRE_Int          rap2,
              HYPRE_Int          mult_order,
              HYPRE_Int          rep,
              HYPRE_Int          verify)
{
   HYPRE_Int          i;
   char               fname[1024];
   HYPRE_Int          time_index;
   HYPRE_Int          keepTranspose = 0;
   HYPRE_Real         fnorm, rfnorm, fnorm0;

   HYPRE_ParCSRMatrix parcsr_Q          = NULL;
   HYPRE_ParCSRMatrix parcsr_AH         = NULL;
   HYPRE_ParCSRMatrix parcsr_A_host     = NULL;
   HYPRE_ParCSRMatrix parcsr_P_host     = NULL;
   HYPRE_ParCSRMatrix parcsr_Q_host     = NULL;
   HYPRE_ParCSRMatrix parcsr_AH_host    = NULL;
   HYPRE_ParCSRMatrix parcsr_AH_host_2  = NULL;
   HYPRE_ParCSRMatrix parcsr_error_host = NULL;

   MPI_Comm comm = hypre_ParCSRMatrixComm(parcsr_A);

   hypre_ParPrintf(comm, "A %d x %d, NNZ %d\n", hypre_ParCSRMatrixGlobalNumRows(parcsr_A),
                   hypre_ParCSRMatrixGlobalNumCols(parcsr_A), hypre_ParCSRMatrixNumNonzeros(parcsr_A));
   hypre_ParPrintf(comm, "P %d x %d, NNZ %d\n", hypre_ParCSRMatrixGlobalNumRows(parcsr_P),
                   hypre_ParCSRMatrixGlobalNumCols(parcsr_P), hypre_ParCSRMatrixNumNonzeros(parcsr_P));

   /* !!! */
   hypre_assert(hypre_ParCSRMatrixMemoryLocation(parcsr_A) == HYPRE_MEMORY_DEVICE);
   hypre_assert(hypre_ParCSRMatrixMemoryLocation(parcsr_P) == HYPRE_MEMORY_DEVICE);

   if (!hypre_ParCSRMatrixCommPkg(parcsr_A))
   {
      hypre_MatvecCommPkgCreate(parcsr_A);
   }

   if (!hypre_ParCSRMatrixCommPkg(parcsr_P))
   {
      hypre_MatvecCommPkgCreate(parcsr_P);
   }

   hypre_ParCSRMatrixSetPatternOnly(parcsr_P, boolean_P);

   /*-----------------------------------------------------------
    * Matrix-by-Matrix on host
    *-----------------------------------------------------------*/
   if (verify)
   {
      //hypre_ParPrintf("Clone matrices to the host\n");
      parcsr_A_host = hypre_ParCSRMatrixClone_v2(parcsr_A, 1, HYPRE_MEMORY_HOST);
      parcsr_P_host = hypre_ParCSRMatrixClone_v2(parcsr_P, 1, HYPRE_MEMORY_HOST);

      hypre_MatvecCommPkgCreate(parcsr_A_host);
      hypre_MatvecCommPkgCreate(parcsr_P_host);

      time_index = hypre_InitializeTiming("Host Parcsr Matrix-by-Matrix, RAP");
      hypre_BeginTiming(time_index);

      if (mult_order == 0)
      {
         parcsr_Q_host  = hypre_ParCSRMatMat(parcsr_A_host, parcsr_P_host);
         parcsr_AH_host = hypre_ParCSRTMatMatKT(parcsr_P_host, parcsr_Q_host, keepTranspose);
      }
      else
      {
         parcsr_Q_host  = hypre_ParCSRTMatMatKT(parcsr_P_host, parcsr_A_host, keepTranspose);
         parcsr_AH_host = hypre_ParCSRMatMat(parcsr_Q_host, parcsr_P_host);
      }
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Host Parcsr Matrix-by-Matrix, RAP", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      hypre_ParCSRMatrixSetNumNonzeros(parcsr_AH_host);
   }

   /*-----------------------------------------------------------
    * Print out the matrices
    *-----------------------------------------------------------*/
   if (print_system)
   {
      sprintf(fname, "%s/%s", file_dir, "IJ.out.A");
      hypre_ParCSRMatrixPrintIJ(parcsr_A_host, 0, 0, fname);
      sprintf(fname, "%s/%s", file_dir, "IJ.out.P");
      hypre_ParCSRMatrixPrintIJ(parcsr_P_host, 0, 0, fname);
   }

   /*-----------------------------------------------------------
    * Matrix-by-Matrix on device
    *-----------------------------------------------------------*/

   /* run for the first time without timing [some allocation is done] */
   /* run for a second time for timing */
   for (i = 0 ; i < rep; i++)
   {
      hypre_ParPrintf(comm, "--- rep %d (out of %d) ---\n", i, rep);

      if (i == rep - 1)
      {
         time_index = hypre_InitializeTiming("Device Parcsr Matrix-by-Matrix, RAP");
         hypre_BeginTiming(time_index);
         //cudaProfilerStart();
      }

      if (rap2)
      {
         if (mult_order == 0)
         {
            parcsr_Q  = hypre_ParCSRMatMat(parcsr_A, parcsr_P);
            parcsr_AH = hypre_ParCSRTMatMatKT(parcsr_P, parcsr_Q, keepTranspose);
         }
         else
         {
            parcsr_Q  = hypre_ParCSRTMatMatKT(parcsr_P, parcsr_A, keepTranspose);
            parcsr_AH = hypre_ParCSRMatMat(parcsr_Q, parcsr_P);
         }
      }
      else
      {
         parcsr_AH = hypre_ParCSRMatrixRAPKT(parcsr_P, parcsr_A, parcsr_P, keepTranspose);
      }

      if (i == rep - 1)
      {
#if defined(HYPRE_USING_GPU)
         hypre_SyncCudaDevice(hypre_handle());
#endif
         //cudaProfilerStop();
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Device Parcsr Matrix-by-Matrix, RAP", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
      }

      if (i < rep - 1)
      {
         hypre_ParCSRMatrixDestroy(parcsr_Q);
         hypre_ParCSRMatrixDestroy(parcsr_AH);
      }
   }

   if (verify)
   {
      /*-----------------------------------------------------------
       * Verify results
       *-----------------------------------------------------------*/
      parcsr_AH_host_2 = hypre_ParCSRMatrixClone_v2(parcsr_AH, 1, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixSetNumNonzeros(parcsr_AH_host_2);

      hypre_ParCSRMatrixAdd(1.0, parcsr_AH_host, -1.0, parcsr_AH_host_2, &parcsr_error_host);
      fnorm = hypre_ParCSRMatrixFnorm(parcsr_error_host);
      fnorm0 = hypre_ParCSRMatrixFnorm(parcsr_AH_host);
      rfnorm = fnorm0 > 0 ? fnorm / fnorm0 : fnorm;

      hypre_ParPrintf(comm, "AH: %d x %d, nnz [CPU %d, GPU %d], CPU-GPU err %e\n",
                      hypre_ParCSRMatrixGlobalNumRows(parcsr_AH_host_2),
                      hypre_ParCSRMatrixGlobalNumCols(parcsr_AH_host_2),
                      hypre_ParCSRMatrixNumNonzeros(parcsr_AH_host),
                      hypre_ParCSRMatrixNumNonzeros(parcsr_AH_host_2),
                      rfnorm);
   }

   hypre_ParCSRMatrixSetNumNonzeros(parcsr_AH);

   hypre_ParPrintf(comm, "AH %d x %d, NNZ %d, RNZ %d\n", hypre_ParCSRMatrixGlobalNumRows(parcsr_AH),
                   hypre_ParCSRMatrixGlobalNumCols(parcsr_AH),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_AH),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_AH) / hypre_ParCSRMatrixGlobalNumRows(parcsr_AH));

   hypre_ParCSRMatrixDestroy(parcsr_Q);
   hypre_ParCSRMatrixDestroy(parcsr_AH);
   hypre_ParCSRMatrixDestroy(parcsr_A_host);
   hypre_ParCSRMatrixDestroy(parcsr_P_host);
   hypre_ParCSRMatrixDestroy(parcsr_Q_host);
   hypre_ParCSRMatrixDestroy(parcsr_AH_host);
   hypre_ParCSRMatrixDestroy(parcsr_AH_host_2);
   hypre_ParCSRMatrixDestroy(parcsr_error_host);
}

void runjob4( HYPRE_ParCSRMatrix parcsr_A,
              HYPRE_Int          print_system,
              HYPRE_Int          rap2,
              HYPRE_Int          mult_order,
              HYPRE_Int          rep,
              HYPRE_Int          verify)
{
   HYPRE_Int    measure_type = 0;
   HYPRE_Real   trunc_factor = 0.0;
   HYPRE_Int    P_max_elmts = 8;
   HYPRE_Int    debug_flag = 0;
   HYPRE_Int    num_functions = 1;
   HYPRE_Real   strong_threshold = 0.25;
   HYPRE_Real   max_row_sum = 1.0;
   HYPRE_Int    local_num_vars;
   HYPRE_BigInt coarse_pnts_global[2];

   hypre_IntArray    *CF_marker         = NULL;
   HYPRE_ParCSRMatrix parcsr_S          = NULL;
   HYPRE_ParCSRMatrix parcsr_P          = NULL;

   /* coarsening */
   hypre_BoomerAMGCreateS(parcsr_A, strong_threshold, max_row_sum, num_functions, NULL, &parcsr_S);

   hypre_BoomerAMGCoarsenPMIS(parcsr_S, parcsr_A, measure_type, debug_flag, &CF_marker);

   local_num_vars = hypre_ParCSRMatrixNumRows(parcsr_A);

   hypre_BoomerAMGCoarseParms(hypre_ParCSRMatrixComm(parcsr_A), local_num_vars, num_functions, NULL,
                              CF_marker, NULL, coarse_pnts_global);

   /* generate P */
   hypre_BoomerAMGBuildExtPIInterp(parcsr_A, hypre_IntArrayData(CF_marker), parcsr_S,
                                   coarse_pnts_global,
                                   num_functions, NULL, debug_flag, trunc_factor, P_max_elmts,
                                   &parcsr_P);

   runjob3(parcsr_A, parcsr_P, 0, print_system, rap2, mult_order, rep, verify);

   hypre_IntArrayDestroy(CF_marker);
   hypre_ParCSRMatrixDestroy(parcsr_S);
   hypre_ParCSRMatrixDestroy(parcsr_P);
}

void runjob5( HYPRE_ParCSRMatrix parcsr_A,
              HYPRE_Int          print_system,
              HYPRE_Int          rep,
              HYPRE_Int          verify)
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_Int          i;
   HYPRE_ParCSRMatrix parcsr_B          = NULL;
   HYPRE_ParCSRMatrix parcsr_A_host     = NULL;
   HYPRE_ParCSRMatrix parcsr_B_host     = NULL;
   HYPRE_ParCSRMatrix parcsr_error_host = NULL;
   HYPRE_Real         fnorm, rfnorm, fnorm0;
   HYPRE_Int          time_index;
   char               fname[1024];

   MPI_Comm comm = hypre_ParCSRMatrixComm(parcsr_A);

   hypre_ParPrintf(comm, "A %d x %d, NNZ %d, RNZ %d\n", hypre_ParCSRMatrixGlobalNumRows(parcsr_A),
                   hypre_ParCSRMatrixGlobalNumCols(parcsr_A),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_A),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_A) / hypre_ParCSRMatrixGlobalNumRows(parcsr_A));

   hypre_assert(hypre_ParCSRMatrixMemoryLocation(parcsr_A) == HYPRE_MEMORY_DEVICE);

   // D = diag(diag(A))
   HYPRE_ParCSRMatrix parcsr_D = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(parcsr_A),
                                                          hypre_ParCSRMatrixGlobalNumRows(parcsr_A),
                                                          hypre_ParCSRMatrixGlobalNumCols(parcsr_A),
                                                          hypre_ParCSRMatrixRowStarts(parcsr_A),
                                                          hypre_ParCSRMatrixColStarts(parcsr_A),
                                                          0,
                                                          hypre_ParCSRMatrixNumRows(parcsr_A),
                                                          0);
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(parcsr_D));
   hypre_ParCSRMatrixDiag(parcsr_D) = hypre_CSRMatrixDiagMatrixFromMatrixDevice(hypre_ParCSRMatrixDiag(
                                                                                   parcsr_A), 0);

   // Diag Scale on host
   if (verify)
   {
      parcsr_A_host = hypre_ParCSRMatrixClone_v2(parcsr_A, 1, HYPRE_MEMORY_HOST);
      hypre_Vector *vec_d_host = hypre_SeqVectorCreate(hypre_ParCSRMatrixNumRows(parcsr_A_host));
      hypre_SeqVectorInitialize_v2(vec_d_host, HYPRE_MEMORY_HOST);

      hypre_CSRMatrixExtractDiagonal(hypre_ParCSRMatrixDiag(parcsr_A_host),
                                     hypre_VectorData(vec_d_host), 0);

      time_index = hypre_InitializeTiming("Host Parcsr DiagScale Matrix, Diag(A)*A");
      hypre_BeginTiming(time_index);

      hypre_CSRMatrixDiagScale(hypre_ParCSRMatrixDiag(parcsr_A_host), vec_d_host, NULL);
      hypre_CSRMatrixDiagScale(hypre_ParCSRMatrixOffd(parcsr_A_host), vec_d_host, NULL);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Host Parcsr DiagScale Matrix, Diag(A)*A", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      hypre_SeqVectorDestroy(vec_d_host);
   }

   // [FAST] diag scale using vector
   {
      HYPRE_ParCSRMatrix parcsr_C = hypre_ParCSRMatrixClone_v2(parcsr_A, 1, HYPRE_MEMORY_DEVICE);
      hypre_Vector *vec_d = hypre_SeqVectorCreate(hypre_ParCSRMatrixNumRows(parcsr_C));
      hypre_SeqVectorInitialize_v2(vec_d, HYPRE_MEMORY_DEVICE);

      hypre_CSRMatrixExtractDiagonal(hypre_ParCSRMatrixDiag(parcsr_C),
                                     hypre_VectorData(vec_d), 0);

      for (i = 0 ; i < rep; i++)
      {
         if (i == rep - 1)
         {
            time_index = hypre_InitializeTiming("Device Parcsr DiagScale Matrix");
            hypre_BeginTiming(time_index);
         }

         hypre_CSRMatrixDiagScale(hypre_ParCSRMatrixDiag(parcsr_C), vec_d, NULL);
         hypre_CSRMatrixDiagScale(hypre_ParCSRMatrixOffd(parcsr_C), vec_d, NULL);

         if (i == rep - 1)
         {
#if defined(HYPRE_USING_GPU)
            hypre_SyncCudaDevice(hypre_handle());
#endif
            hypre_EndTiming(time_index);
            hypre_PrintTiming("Device Parcsr DiagScale Matrix", hypre_MPI_COMM_WORLD);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();
         }
      }

      hypre_ParCSRMatrixDestroy(parcsr_C);
      hypre_SeqVectorDestroy(vec_d);
   }

   // [SLOW] Diag-by-Matrix
   for (i = 0 ; i < rep; i++)
   {
      hypre_ParPrintf(comm, "--- rep %d (out of %d) ---\n", i, rep);

      if (i == rep - 1)
      {
         time_index = hypre_InitializeTiming("Device Parcsr Matrix-by-Matrix, Diag(A)*A");
         hypre_BeginTiming(time_index);
      }

      parcsr_B = hypre_ParCSRMatMat(parcsr_D, parcsr_A);

      if (i == rep - 1)
      {
#if defined(HYPRE_USING_GPU)
         hypre_SyncCudaDevice(hypre_handle());
#endif
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Device Parcsr Matrix-by-Matrix, Diag(A)*A", hypre_MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
      }

      if (i < rep - 1)
      {
         hypre_ParCSRMatrixDestroy(parcsr_B);
      }
   }

   if (verify)
   {
      parcsr_B_host = hypre_ParCSRMatrixClone_v2(parcsr_B, 1, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixSetNumNonzeros(parcsr_B_host);

      hypre_ParCSRMatrixAdd(1.0, parcsr_A_host, -1.0, parcsr_B_host, &parcsr_error_host);
      fnorm = hypre_ParCSRMatrixFnorm(parcsr_error_host);
      fnorm0 = hypre_ParCSRMatrixFnorm(parcsr_B_host);
      rfnorm = fnorm0 > 0 ? fnorm / fnorm0 : fnorm;

      hypre_ParPrintf(comm, "Diag(A)*A: %d x %d, nnz [CPU %d, GPU %d], CPU-GPU err %e\n",
                      hypre_ParCSRMatrixGlobalNumRows(parcsr_B_host),
                      hypre_ParCSRMatrixGlobalNumCols(parcsr_B_host),
                      hypre_ParCSRMatrixNumNonzeros(parcsr_A_host),
                      hypre_ParCSRMatrixNumNonzeros(parcsr_B_host),
                      rfnorm);
   }

   if (print_system)
   {
      if (!parcsr_B_host)
      {
         parcsr_B_host = hypre_ParCSRMatrixClone_v2(parcsr_B, 1, HYPRE_MEMORY_HOST);
      }
      sprintf(fname, "%s/%s", file_dir, "IJ.out.B");
      hypre_ParCSRMatrixPrintIJ(parcsr_B_host, 0, 0, fname);
   }

   hypre_ParCSRMatrixSetNumNonzeros(parcsr_B);

   hypre_ParPrintf(comm, "B %d x %d, NNZ %d, RNZ %d\n", hypre_ParCSRMatrixGlobalNumRows(parcsr_B),
                   hypre_ParCSRMatrixGlobalNumCols(parcsr_B),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_B),
                   hypre_ParCSRMatrixNumNonzeros(parcsr_B) / hypre_ParCSRMatrixGlobalNumRows(parcsr_B));

   hypre_ParCSRMatrixDestroy(parcsr_B);
   hypre_ParCSRMatrixDestroy(parcsr_A_host);
   hypre_ParCSRMatrixDestroy(parcsr_B_host);
   hypre_ParCSRMatrixDestroy(parcsr_D);
   hypre_ParCSRMatrixDestroy(parcsr_error_host);
#else
   HYPRE_UNUSED_VAR(parcsr_A);
   HYPRE_UNUSED_VAR(print_system);
   HYPRE_UNUSED_VAR(rep);
   HYPRE_UNUSED_VAR(verify);
#endif
}

hypre_int
main( hypre_int argc,
      char *argv[] )
{
   HYPRE_Int           arg_index;
   HYPRE_Int           print_usage;
   HYPRE_Int           build_matrix_type;
   HYPRE_Int           build_matrix_arg_index;
   HYPRE_Int           ierr = 0;
   void               *object;
   HYPRE_IJMatrix      ij_A = NULL;
   HYPRE_IJMatrix      ij_P = NULL;
   HYPRE_ParCSRMatrix  parcsr_A  = NULL;
   HYPRE_ParCSRMatrix  parcsr_P  = NULL;
   HYPRE_Int           errcode, job = 1;
   HYPRE_Int           num_procs, myid;
   HYPRE_Int           time_index;
   MPI_Comm            comm = hypre_MPI_COMM_WORLD;
   HYPRE_BigInt        first_local_row, last_local_row;
   HYPRE_BigInt        first_local_col, last_local_col;
   HYPRE_Int           rap2 = 0;
   HYPRE_Int           mult_order = 0;
   HYPRE_Int           print_system = 0;
   HYPRE_Int           verify       = 0;
   HYPRE_Int           use_vendor = 0;
   HYPRE_Int           spgemm_alg = 1;
   HYPRE_Int           spgemm_binned = 0;
   HYPRE_Int           rowest_mtd = 3;
   HYPRE_Int           rowest_nsamples = -1; /* default */
   HYPRE_Real          rowest_mult = -1.0; /* default */
   HYPRE_Int           zero_mem_cost = 0;
   HYPRE_Int           boolean_P = 1;
   HYPRE_Int           rep = 10;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before HYPRE_Initialize() and should not be changed after
    *-----------------------------------------------------------------*/
   hypre_bind_device_id(-1, myid, num_procs, hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Initialize : must be the first HYPRE function to call
    *-----------------------------------------------------------*/
   HYPRE_Initialize();
   HYPRE_DeviceInitialize();

   if (myid == 0)
   {
      HYPRE_PrintDeviceInfo();
   }

   /* for timing, sync after kernels */
#if defined(HYPRE_USING_GPU)
   hypre_SetSyncCudaCompute(1);
#endif

   HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);

   //hypre_SetNumThreads(20);
   if (myid == 0)
   {
      hypre_printf("CPU #OMP THREADS %d\n", hypre_NumThreads());
   }

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
   build_matrix_type = 2;
   build_matrix_arg_index = argc;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-fromMMfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = -2;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = -1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromparcsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 0;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-from2files") == 0 )
      {
         arg_index++;
         build_matrix_type      = 1;
         build_matrix_arg_index = arg_index;
         arg_index++;
      }
      else if ( strcmp(argv[arg_index], "-laplacian") == 0 )
      {
         arg_index++;
         build_matrix_type      = 2;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-9pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 3;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-27pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 4;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-difconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 5;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-vardifconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 6;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rotate") == 0 )
      {
         arg_index++;
         build_matrix_type      = 7;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-concrete_parcsr") == 0 )
      {
         arg_index++;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-verify") == 0 )
      {
         arg_index++;
         verify = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rap2") == 0 )
      {
         arg_index++;
         rap2 = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-order") == 0 )
      {
         arg_index++;
         mult_order = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vendor") == 0 )
      {
         arg_index++;
         use_vendor = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-spgemmalg") == 0 )
      {
         arg_index++;
         spgemm_alg = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-spgemm_binned") == 0 )
      {
         arg_index++;
         spgemm_binned  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rowest") == 0 )
      {
         arg_index++;
         rowest_mtd = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rowestmult") == 0 )
      {
         arg_index++;
         rowest_mult = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rowestnsamples") == 0 )
      {
         arg_index++;
         rowest_nsamples = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-job") == 0 )
      {
         arg_index++;
         job  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rep") == 0 )
      {
         arg_index++;
         rep  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-zeromemcost") == 0 )
      {
         arg_index++;
         zero_mem_cost = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
      else
      {
         arg_index++;
      }
   }

   if (zero_mem_cost)
   {
#ifdef HYPRE_USING_CUDA
      total_size = 15LL * 1024 * 1024 * 1024;
#elif defined(HYPRE_USING_HIP)
      total_size = 20LL * 1024 * 1024 * 1024;
#endif
      gpu_ptr = hypre_TAlloc(char, total_size, HYPRE_MEMORY_DEVICE);
      hypre_SetUserDeviceMalloc(gpu_alloc);
      hypre_SetUserDeviceMfree(gpu_free);
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
         hypre_printf("  -fromfile <filename>       : ");
         hypre_printf("matrix read from multiple files (IJ format)\n");
         hypre_printf("  -fromparcsrfile <filename> : ");
         hypre_printf("matrix read from multiple files (ParCSR format)\n");
         hypre_printf("  -fromonecsrfile <filename> : ");
         hypre_printf("matrix read from a single file (CSR format)\n");
         hypre_printf("\n");
         hypre_printf("  -laplacian [<options>] : build 5pt 2D laplacian problem (default) \n");
         hypre_printf("  -sysL <num functions>  : build SYSTEMS laplacian 7pt operator\n");
         hypre_printf("  -9pt [<opts>]          : build 9pt 2D laplacian problem\n");
         hypre_printf("  -27pt [<opts>]         : build 27pt 3D laplacian problem\n");
         hypre_printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
         hypre_printf("    -n <nx> <ny> <nz>    : total problem size \n");
         hypre_printf("    -P <Px> <Py> <Pz>    : processor topology\n");
         hypre_printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
         hypre_printf("    -a <ax> <ay> <az>    : convection coefficients\n");
         hypre_printf("    -atype <type>        : FD scheme for convection \n");
         hypre_printf("           0=Forward (default)       1=Backward\n");
         hypre_printf("           2=Centered                3=Upwind\n");
         hypre_printf("\n");
         hypre_printf("  -concrete_parcsr       : use parcsr matrix type as concrete type\n");
         hypre_printf("\n");
         hypre_printf("  -job                   : 1. A^2  2. A^T*A  3. P^T*A*P\n");
         hypre_printf("                           4. P^T*A*P (P is AMG Interp from A)\n");
         hypre_printf("                           5. Diag(A) * A\n");
      }
      goto final;
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
   errcode = hypre_SetSpGemmRownnzEstimateMethod(rowest_mtd);
   hypre_assert(errcode == 0);
   if (rowest_nsamples > 0)
   {
      errcode = hypre_SetSpGemmRownnzEstimateNSamples(rowest_nsamples);
      hypre_assert(errcode == 0);
   }
   if (rowest_mult > 0.0)
   {
      errcode = hypre_SetSpGemmRownnzEstimateMultFactor(rowest_mult);
      hypre_assert(errcode == 0);
   }
   errcode = HYPRE_SetSpGemmUseVendor(use_vendor);
   hypre_assert(errcode == 0);
   errcode = hypre_SetSpGemmAlgorithm(spgemm_alg);
   hypre_assert(errcode == 0);
   ierr = hypre_SetSpGemmBinned(spgemm_binned);
   hypre_assert(ierr == 0);

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/
   time_index = hypre_InitializeTiming("Generate Matrices");
   hypre_BeginTiming(time_index);
   if ( build_matrix_type == -2 )
   {
      ierr = HYPRE_IJMatrixReadMM( argv[build_matrix_arg_index], comm,
                                   HYPRE_PARCSR, &ij_A );
      if (ierr)
      {
         hypre_printf("ERROR: Problem reading in the system matrix in MM format!\n");
         exit(1);
      }
   }
   else if ( build_matrix_type == -1 )
   {
      ierr = HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                                 HYPRE_PARCSR, &ij_A );
      if (ierr)
      {
         hypre_printf("ERROR: Problem reading in the system matrix!\n");
         exit(1);
      }
   }
   else if ( build_matrix_type == 0 )
   {
      BuildParFromFile(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 1 )
   {
      HYPRE_Int ierr1 = HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                                            HYPRE_PARCSR, &ij_A );
      HYPRE_Int ierr2 = HYPRE_IJMatrixRead( argv[build_matrix_arg_index + 1], comm,
                                            HYPRE_PARCSR, &ij_P );
      if (ierr1 || ierr2)
      {
         hypre_printf("ERROR: Problem reading in the system matrix!\n");
         exit(1);
      }
   }
   else if ( build_matrix_type == 2 )
   {
      BuildParLaplacian(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 3 )
   {
      BuildParLaplacian9pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 4 )
   {
      BuildParLaplacian27pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 5 )
   {
      BuildParDifConv(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 7 )
   {
      BuildParRotate7pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else
   {
      hypre_printf("You have asked for an unsupported problem with\n");
      hypre_printf("build_matrix_type = %d.\n", build_matrix_type);
      return (-1);
   }

   /* first matrix */
   if (ij_A)
   {
      ierr = HYPRE_IJMatrixGetLocalRange( ij_A,
                                          &first_local_row, &last_local_row,
                                          &first_local_col, &last_local_col );

      ierr += HYPRE_IJMatrixGetObject( ij_A, &object);
      parcsr_A = (HYPRE_ParCSRMatrix) object;
   }
   else
   {
      /*-----------------------------------------------------------
       * Copy the parcsr matrix into the IJMatrix through interface calls
       *-----------------------------------------------------------*/
      ierr = HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                              &first_local_row, &last_local_row,
                                              &first_local_col, &last_local_col );
   }

   hypre_ParCSRMatrixSetNumNonzeros(parcsr_A);

   /* [optional] second matrix */
   if (ij_P)
   {
      ierr += HYPRE_IJMatrixGetObject( ij_P, &object);
      parcsr_P = (HYPRE_ParCSRMatrix) object;
   }

   if (parcsr_P)
   {
      hypre_ParCSRMatrixSetNumNonzeros(parcsr_P);
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Generate Matrices", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   hypre_ParCSRMatrixMigrate(parcsr_A, hypre_HandleMemoryLocation(hypre_handle()));
   if (parcsr_P)
   {
      hypre_ParCSRMatrixMigrate(parcsr_P, hypre_HandleMemoryLocation(hypre_handle()));
   }

   if (job == 1)
   {
      runjob1(parcsr_A, print_system, rep, verify);
   }
   else if (job == 2)
   {
      runjob2(parcsr_A, print_system, rep, verify);
   }
   else if (job == 3)
   {
      runjob3(parcsr_A, parcsr_P, boolean_P, print_system, rap2, mult_order, rep, verify);
   }
   else if (job == 4)
   {
      runjob4(parcsr_A, print_system, rap2, mult_order, rep, verify);
   }
   else if (job == 5)
   {
      runjob5(parcsr_A, print_system, rep, verify);
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/
   if (ij_A)
   {
      HYPRE_IJMatrixDestroy(ij_A);
   }
   else
   {
      HYPRE_ParCSRMatrixDestroy(parcsr_A);
   }

   if (ij_P)
   {
      HYPRE_IJMatrixDestroy(ij_P);
   }

final:

   hypre_TFree(gpu_ptr, HYPRE_MEMORY_DEVICE);

   /* Finalize Hypre */
   HYPRE_Finalize();

   /* Finalize MPI */
   hypre_MPI_Finalize();

   return (0);
}

/*----------------------------------------------------------------------
 * Build matrix from file. Expects three files on each processor.
 * filename.D.n contains the diagonal part, filename.O.n contains
 * the offdiagonal part and filename.INFO.n contains global row
 * and column numbers, number of columns of offdiagonal matrix
 * and the mapping of offdiagonal column numbers to global column numbers.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParFromFile( HYPRE_Int            argc,
                  char                *argv[],
                  HYPRE_Int            arg_index,
                  HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   HYPRE_ParCSRMatrix A;

   HYPRE_Int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  FromFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   HYPRE_ParCSRMatrixRead(hypre_MPI_COMM_WORLD, filename, &A);

   *A_ptr = A;

   return (0);
}


/*----------------------------------------------------------------------
 * Build rhs from file. Expects two files on each processor.
 * filename.n contains the data and
 * and filename.INFO.n contains global row
 * numbers
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParRhsFromFile( HYPRE_Int            argc,
                     char                *argv[],
                     HYPRE_Int            arg_index,
                     HYPRE_ParVector      *b_ptr     )
{
   char               *filename;

   HYPRE_ParVector b;

   HYPRE_Int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  RhsFromParFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   HYPRE_ParVectorRead(hypre_MPI_COMM_WORLD, filename, &b);

   *b_ptr = b;

   return (0);
}




/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian( HYPRE_Int            argc,
                   char                *argv[],
                   HYPRE_Int            arg_index,
                   HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;
   HYPRE_Real          cx, cy, cz;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Int                 num_fun = 1;
   HYPRE_Real         *values;
   HYPRE_Real         *mtrx;

   HYPRE_Real          ep = .1;

   HYPRE_Int                 system_vcoef = 0;
   HYPRE_Int                 sys_opt = 0;
   HYPRE_Int                 vcoef_opt = 0;


   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = (HYPRE_Real)atof(argv[arg_index++]);
         cy = (HYPRE_Real)atof(argv[arg_index++]);
         cz = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sysL") == 0 )
      {
         arg_index++;
         num_fun = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sysL_opt") == 0 )
      {
         arg_index++;
         sys_opt = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sys_vcoef") == 0 )
      {
         /* have to use -sysL for this to */
         arg_index++;
         system_vcoef = 1;
      }
      else if ( strcmp(argv[arg_index], "-sys_vcoef_opt") == 0 )
      {
         arg_index++;
         vcoef_opt = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ep") == 0 )
      {
         arg_index++;
         ep = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Laplacian:   num_fun = %d\n", num_fun);
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(HYPRE_Real,  4, HYPRE_MEMORY_HOST);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0 * cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0 * cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0 * cz;
   }

   if (num_fun == 1)
      A = (HYPRE_ParCSRMatrix) GenerateLaplacian(hypre_MPI_COMM_WORLD,
                                                 nx, ny, nz, P, Q, R, p, q, r, values);
   else
   {
      mtrx = hypre_CTAlloc(HYPRE_Real,  num_fun * num_fun, HYPRE_MEMORY_HOST);

      if (num_fun == 2)
      {
         if (sys_opt == 1) /* identity  */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 2)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 20.0;
         }
         else if (sys_opt == 3) /* similar to barry's talk - ex1 */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 2.0;
            mtrx[2] = 2.0;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 4) /* can use with vcoef to get barry's ex*/
         {
            mtrx[0] = 1.0;
            mtrx[1] = 1.0;
            mtrx[2] = 1.0;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 5) /* barry's talk - ex1 */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 1.1;
            mtrx[2] = 1.1;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 6) /*  */
         {
            mtrx[0] = 1.1;
            mtrx[1] = 1.0;
            mtrx[2] = 1.0;
            mtrx[3] = 1.1;
         }

         else /* == 0 */
         {
            mtrx[0] = 2;
            mtrx[1] = 1;
            mtrx[2] = 1;
            mtrx[3] = 2;
         }
      }
      else if (num_fun == 3)
      {
         if (sys_opt == 1)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 0.0;
            mtrx[4] = 1.0;
            mtrx[5] = 0.0;
            mtrx[6] = 0.0;
            mtrx[7] = 0.0;
            mtrx[8] = 1.0;
         }
         else if (sys_opt == 2)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 0.0;
            mtrx[4] = 20.0;
            mtrx[5] = 0.0;
            mtrx[6] = 0.0;
            mtrx[7] = 0.0;
            mtrx[8] = .01;
         }
         else if (sys_opt == 3)
         {
            mtrx[0] = 1.01;
            mtrx[1] = 1;
            mtrx[2] = 0.0;
            mtrx[3] = 1;
            mtrx[4] = 2;
            mtrx[5] = 1;
            mtrx[6] = 0.0;
            mtrx[7] = 1;
            mtrx[8] = 1.01;
         }
         else if (sys_opt == 4) /* barry ex4 */
         {
            mtrx[0] = 3;
            mtrx[1] = 1;
            mtrx[2] = 0.0;
            mtrx[3] = 1;
            mtrx[4] = 4;
            mtrx[5] = 2;
            mtrx[6] = 0.0;
            mtrx[7] = 2;
            mtrx[8] = .25;
         }
         else /* == 0 */
         {
            mtrx[0] = 2.0;
            mtrx[1] = 1.0;
            mtrx[2] = 0.0;
            mtrx[3] = 1.0;
            mtrx[4] = 2.0;
            mtrx[5] = 1.0;
            mtrx[6] = 0.0;
            mtrx[7] = 1.0;
            mtrx[8] = 2.0;
         }

      }
      else if (num_fun == 4)
      {
         mtrx[0] = 1.01;
         mtrx[1] = 1;
         mtrx[2] = 0.0;
         mtrx[3] = 0.0;
         mtrx[4] = 1;
         mtrx[5] = 2;
         mtrx[6] = 1;
         mtrx[7] = 0.0;
         mtrx[8] = 0.0;
         mtrx[9] = 1;
         mtrx[10] = 1.01;
         mtrx[11] = 0.0;
         mtrx[12] = 2;
         mtrx[13] = 1;
         mtrx[14] = 0.0;
         mtrx[15] = 1;
      }




      if (!system_vcoef)
      {
         A = (HYPRE_ParCSRMatrix) GenerateSysLaplacian(hypre_MPI_COMM_WORLD,
                                                       nx, ny, nz, P, Q,
                                                       R, p, q, r, num_fun, mtrx, values);
      }
      else
      {


         HYPRE_Real *mtrx_values;

         mtrx_values = hypre_CTAlloc(HYPRE_Real,  num_fun * num_fun * 4, HYPRE_MEMORY_HOST);

         if (num_fun == 2)
         {
            if (vcoef_opt == 1)
            {
               /* Barry's talk * - must also have sys_opt = 4, all fail */
               mtrx[0] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, .10, 1.0, 0, mtrx_values);

               mtrx[1]  = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, .1, 1.0, 1.0, 1, mtrx_values);

               mtrx[2] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, .01, 1.0, 1.0, 2, mtrx_values);

               mtrx[3] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 2.0, .02, 1.0, 3, mtrx_values);

            }
            else if (vcoef_opt == 2)
            {
               /* Barry's talk * - ex2 - if have sys-opt = 4*/
               mtrx[0] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, .010, 1.0, 0, mtrx_values);

               mtrx[1]  = 200.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 1, mtrx_values);

               mtrx[2] = 200.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 2, mtrx_values);

               mtrx[3] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 2.0, .02, 1.0, 3, mtrx_values);

            }
            else if (vcoef_opt == 3) /* use with default sys_opt  - ulrike ex 3*/
            {

               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 4) /* use with default sys_opt  - ulrike ex 4*/
            {
               HYPRE_Real ep2 = ep;

               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, ep * 1.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, ep2 * 1.0, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 5) /* use with default sys_opt  - */
            {
               HYPRE_Real  alp, beta;
               alp = .001;
               beta = 10;

               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, alp * 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, beta * 1.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, alp * 1.0, 1.0, 1.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, beta * 1.0, 1.0, 3, mtrx_values);
            }
            else  /* = 0 */
            {
               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 2.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 2.0, 1.0, 0.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 3.0, 1.0, 3, mtrx_values);
            }

         }
         else if (num_fun == 3)
         {
            mtrx[0] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, .01, 1, 0, mtrx_values);

            mtrx[1] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 1, mtrx_values);

            mtrx[2] = 0.0;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 2, mtrx_values);

            mtrx[3] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 3, mtrx_values);

            mtrx[4] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 2, .02, 1, 4, mtrx_values);

            mtrx[5] = 2;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 5, mtrx_values);

            mtrx[6] = 0.0;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 6, mtrx_values);

            mtrx[7] = 2;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 7, mtrx_values);

            mtrx[8] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1.5, .04, 1, 8, mtrx_values);

         }

         A = (HYPRE_ParCSRMatrix) GenerateSysLaplacianVCoef(hypre_MPI_COMM_WORLD,
                                                            nx, ny, nz, P, Q,
                                                            R, p, q, r, num_fun, mtrx, mtrx_values);





         hypre_TFree(mtrx_values, HYPRE_MEMORY_HOST);
      }

      hypre_TFree(mtrx, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * returns the sign of a real number
 *  1 : positive
 *  0 : zero
 * -1 : negative
 *----------------------------------------------------------------------*/
static inline HYPRE_Int sign_double(HYPRE_Real a)
{
   return ( (0.0 < a) - (0.0 > a) );
}

/*----------------------------------------------------------------------
 * Build standard 7-point convection-diffusion operator
 * Parameters given in command line.
 * Operator:
 *
 *  -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f
 *
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParDifConv( HYPRE_Int            argc,
                 char                *argv[],
                 HYPRE_Int            arg_index,
                 HYPRE_ParCSRMatrix  *A_ptr)
{
   HYPRE_Int           nx, ny, nz;
   HYPRE_Int           P, Q, R;
   HYPRE_Real          cx, cy, cz;
   HYPRE_Real          ax, ay, az, atype;
   HYPRE_Real          hinx, hiny, hinz;
   HYPRE_Int           sign_prod;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           p, q, r;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   ax = 1.;
   ay = 1.;
   az = 1.;

   atype = 0;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = (HYPRE_Real)atof(argv[arg_index++]);
         cy = (HYPRE_Real)atof(argv[arg_index++]);
         cz = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-a") == 0 )
      {
         arg_index++;
         ax = (HYPRE_Real)atof(argv[arg_index++]);
         ay = (HYPRE_Real)atof(argv[arg_index++]);
         az = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-atype") == 0 )
      {
         arg_index++;
         atype = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Convection-Diffusion: \n");
      hypre_printf("    -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf("    (ax, ay, az) = (%f, %f, %f)\n\n", ax, ay, az);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   hinx = 1. / (nx + 1);
   hiny = 1. / (ny + 1);
   hinz = 1. / (nz + 1);

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/
   /* values[7]:
    *    [0]: center
    *    [1]: X-
    *    [2]: Y-
    *    [3]: Z-
    *    [4]: X+
    *    [5]: Y+
    *    [6]: Z+
    */
   values = hypre_CTAlloc(HYPRE_Real,  7, HYPRE_MEMORY_HOST);

   values[0] = 0.;

   if (0 == atype) /* forward scheme for conv */
   {
      values[1] = -cx / (hinx * hinx);
      values[2] = -cy / (hiny * hiny);
      values[3] = -cz / (hinz * hinz);
      values[4] = -cx / (hinx * hinx) + ax / hinx;
      values[5] = -cy / (hiny * hiny) + ay / hiny;
      values[6] = -cz / (hinz * hinz) + az / hinz;

      if (nx > 1)
      {
         values[0] += 2.0 * cx / (hinx * hinx) - 1.*ax / hinx;
      }
      if (ny > 1)
      {
         values[0] += 2.0 * cy / (hiny * hiny) - 1.*ay / hiny;
      }
      if (nz > 1)
      {
         values[0] += 2.0 * cz / (hinz * hinz) - 1.*az / hinz;
      }
   }
   else if (1 == atype) /* backward scheme for conv */
   {
      values[1] = -cx / (hinx * hinx) - ax / hinx;
      values[2] = -cy / (hiny * hiny) - ay / hiny;
      values[3] = -cz / (hinz * hinz) - az / hinz;
      values[4] = -cx / (hinx * hinx);
      values[5] = -cy / (hiny * hiny);
      values[6] = -cz / (hinz * hinz);

      if (nx > 1)
      {
         values[0] += 2.0 * cx / (hinx * hinx) + 1.*ax / hinx;
      }
      if (ny > 1)
      {
         values[0] += 2.0 * cy / (hiny * hiny) + 1.*ay / hiny;
      }
      if (nz > 1)
      {
         values[0] += 2.0 * cz / (hinz * hinz) + 1.*az / hinz;
      }
   }
   else if (3 == atype) /* upwind scheme */
   {
      sign_prod = sign_double(cx) * sign_double(ax);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[1] = -cx / (hinx * hinx) - ax / hinx;
         values[4] = -cx / (hinx * hinx);
         if (nx > 1)
         {
            values[0] += 2.0 * cx / (hinx * hinx) + 1.*ax / hinx;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[1] = -cx / (hinx * hinx);
         values[4] = -cx / (hinx * hinx) + ax / hinx;
         if (nx > 1)
         {
            values[0] += 2.0 * cx / (hinx * hinx) - 1.*ax / hinx;
         }
      }

      sign_prod = sign_double(cy) * sign_double(ay);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[2] = -cy / (hiny * hiny) - ay / hiny;
         values[5] = -cy / (hiny * hiny);
         if (ny > 1)
         {
            values[0] += 2.0 * cy / (hiny * hiny) + 1.*ay / hiny;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[2] = -cy / (hiny * hiny);
         values[5] = -cy / (hiny * hiny) + ay / hiny;
         if (ny > 1)
         {
            values[0] += 2.0 * cy / (hiny * hiny) - 1.*ay / hiny;
         }
      }

      sign_prod = sign_double(cz) * sign_double(az);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[3] = -cz / (hinz * hinz) - az / hinz;
         values[6] = -cz / (hinz * hinz);
         if (nz > 1)
         {
            values[0] += 2.0 * cz / (hinz * hinz) + 1.*az / hinz;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[3] = -cz / (hinz * hinz);
         values[6] = -cz / (hinz * hinz) + az / hinz;
         if (nz > 1)
         {
            values[0] += 2.0 * cz / (hinz * hinz) - 1.*az / hinz;
         }
      }
   }
   else /* centered difference scheme */
   {
      values[1] = -cx / (hinx * hinx) - ax / (2.*hinx);
      values[2] = -cy / (hiny * hiny) - ay / (2.*hiny);
      values[3] = -cz / (hinz * hinz) - az / (2.*hinz);
      values[4] = -cx / (hinx * hinx) + ax / (2.*hinx);
      values[5] = -cy / (hiny * hiny) + ay / (2.*hiny);
      values[6] = -cz / (hinz * hinz) + az / (2.*hinz);

      if (nx > 1)
      {
         values[0] += 2.0 * cx / (hinx * hinx);
      }
      if (ny > 1)
      {
         values[0] += 2.0 * cy / (hiny * hiny);
      }
      if (nz > 1)
      {
         values[0] += 2.0 * cz / (hinz * hinz);
      }
   }

   A = (HYPRE_ParCSRMatrix) GenerateDifConv(hypre_MPI_COMM_WORLD,
                                            nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian9pt( HYPRE_Int                  argc,
                      char                *argv[],
                      HYPRE_Int                  arg_index,
                      HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny;
   HYPRE_Int                 P, Q;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;

   P  = 1;
   Q  = num_procs;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Laplacian 9pt:\n");
      hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      hypre_printf("    (Px, Py) = (%d, %d)\n\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid % P;
   q = ( myid - p) / P;

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(HYPRE_Real,  2, HYPRE_MEMORY_HOST);

   values[1] = -1.;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0;
   }
   if (ny > 1)
   {
      values[0] += 2.0;
   }
   if (nx > 1 && ny > 1)
   {
      values[0] += 4.0;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(hypre_MPI_COMM_WORLD,
                                                 nx, ny, P, Q, p, q, values);

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}
/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian27pt( HYPRE_Int                  argc,
                       char                *argv[],
                       HYPRE_Int                  arg_index,
                       HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Laplacian_27pt:\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(HYPRE_Real,  2, HYPRE_MEMORY_HOST);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
   {
      values[0] = 8.0;
   }
   if (nx * ny == 1 || nx * nz == 1 || ny * nz == 1)
   {
      values[0] = 2.0;
   }
   values[1] = -1.;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}


/*----------------------------------------------------------------------
 * Build 7-point in 2D
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParRotate7pt( HYPRE_Int            argc,
                   char                *argv[],
                   HYPRE_Int            arg_index,
                   HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny;
   HYPRE_Int                 P, Q;

   HYPRE_ParCSRMatrix        A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q;
   HYPRE_Real                eps = 0.0, alpha = 1.0;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;

   P  = 1;
   Q  = num_procs;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-alpha") == 0 )
      {
         arg_index++;
         alpha  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-eps") == 0 )
      {
         arg_index++;
         eps  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Rotate 7pt:\n");
      hypre_printf("    alpha = %f, eps = %f\n", alpha, eps);
      hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      hypre_printf("    (Px, Py) = (%d, %d)\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid % P;
   q = ( myid - p) / P;

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   A = (HYPRE_ParCSRMatrix) GenerateRotate7pt(hypre_MPI_COMM_WORLD,
                                              nx, ny, P, Q, p, q, alpha, eps);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point difference operator using centered differences
 *
 *  eps*(a(x,y,z) ux)x + (b(x,y,z) uy)y + (c(x,y,z) uz)z
 *  d(x,y,z) ux + e(x,y,z) uy + f(x,y,z) uz + g(x,y,z) u
 *
 *  functions a,b,c,d,e,f,g need to be defined inside par_vardifconv.c
 *
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParVarDifConv( HYPRE_Int                  argc,
                    char                *argv[],
                    HYPRE_Int                  arg_index,
                    HYPRE_ParCSRMatrix  *A_ptr,
                    HYPRE_ParVector  *rhs_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;
   HYPRE_ParVector  rhs;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           p, q, r;
   HYPRE_Int           type;
   HYPRE_Real          eps;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;
   P  = 1;
   Q  = num_procs;
   R  = 1;
   eps = 1.0;

   /* type: 0   : default FD;
    *       1-3 : FD and examples 1-3 in Ruge-Stuben paper */
   type = 0;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-eps") == 0 )
      {
         arg_index++;
         eps  = (HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vardifconvRS") == 0 )
      {
         arg_index++;
         type = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  ell PDE: eps = %f\n", eps);
      hypre_printf("    Dx(aDxu) + Dy(bDyu) + Dz(cDzu) + d Dxu + e Dyu + f Dzu  + g u= f\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
   }
   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   if (0 == type)
   {
      A = (HYPRE_ParCSRMatrix) GenerateVarDifConv(hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, eps, &rhs);
   }
   else
   {
      A = (HYPRE_ParCSRMatrix) GenerateRSVarDifConv(hypre_MPI_COMM_WORLD,
                                                    nx, ny, nz, P, Q, R, p, q, r, eps, &rhs,
                                                    type);
   }

   *A_ptr = A;
   *rhs_ptr = rhs;

   return (0);
}

/**************************************************************************/

HYPRE_Int SetSysVcoefValues(HYPRE_Int num_fun, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                            HYPRE_Real vcx,
                            HYPRE_Real vcy, HYPRE_Real vcz, HYPRE_Int mtx_entry, HYPRE_Real *values)
{


   HYPRE_Int sz = num_fun * num_fun;

   values[1 * sz + mtx_entry] = -vcx;
   values[2 * sz + mtx_entry] = -vcy;
   values[3 * sz + mtx_entry] = -vcz;
   values[0 * sz + mtx_entry] = 0.0;

   if (nx > 1)
   {
      values[0 * sz + mtx_entry] += 2.0 * vcx;
   }
   if (ny > 1)
   {
      values[0 * sz + mtx_entry] += 2.0 * vcy;
   }
   if (nz > 1)
   {
      values[0 * sz + mtx_entry] += 2.0 * vcz;
   }

   return 0;

}

/*----------------------------------------------------------------------
 * Build coordinates for 1D/2D/3D
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParCoordinates( HYPRE_Int                  argc,
                     char                      *argv[],
                     HYPRE_Int                  arg_index,
                     HYPRE_Int                 *coorddim_ptr,
                     float                    **coord_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;

   HYPRE_Int                 coorddim;
   float               *coordinates;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the coordinates
    *-----------------------------------------------------------*/

   coorddim = 3;
   if (nx < 2) { coorddim--; }
   if (ny < 2) { coorddim--; }
   if (nz < 2) { coorddim--; }

   if (coorddim > 0)
   {
      coordinates = hypre_GenerateCoordinates(hypre_MPI_COMM_WORLD,
                                              nx, ny, nz, P, Q, R, p, q, r, coorddim);
   }
   else
   {
      coordinates = NULL;
   }

   *coorddim_ptr = coorddim;
   *coord_ptr = coordinates;
   return (0);
}

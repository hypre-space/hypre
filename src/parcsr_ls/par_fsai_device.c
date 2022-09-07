/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"
#include "par_fsai.h"

//#define DEBUG_FSAI

#if defined(HYPRE_USING_GPU)

#define mat_(ldim, k, i, j) mat_data[ldim * (ldim * k + i) + j]
#define rhs_(ldim, i, j)    rhs_data[ldim * i + j]
#define sol_(ldim, i, j)    sol_data[ldim * i + j]

#define HYPRE_THRUST_ZIP3(A, B, C) thrust::make_zip_iterator(thrust::make_tuple(A, B, C))

/*--------------------------------------------------------------------------
 * hypreGPUKernel_ComplexArrayToArrayOfPtrs
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_ComplexArrayToArrayOfPtrs( hypre_DeviceItem  &item,
                                          HYPRE_Int          num_rows,
                                          HYPRE_Int          ldim,
                                          HYPRE_Complex     *data,
                                          HYPRE_Complex    **data_aop )
{
   HYPRE_Int i = threadIdx.x + blockIdx.x * blockDim.x;

   if (i < num_rows)
   {
      data_aop[i] = &data[i * ldim];
   }
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_FSAIExtractSubSystems
 *
 * Output:
 *   1) mat_data: dense matrix coefficients.
 *   2) rhs_data: right hand side coefficients.
 *   3) G_r: number of nonzero coefficients per row of the matrix G.
 *
 * Notes:
 *   1) Naive implementation using one thread per row
 *   2) A_j and P_j are sorted in the intervals (A_i[i], A_i[i+1])
 *      and (P_i[i], P_i[i+1]), respectively
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_FSAIExtractSubSystems( hypre_DeviceItem &item,
                                      HYPRE_Int         num_rows,
                                      HYPRE_Int        *A_i,
                                      HYPRE_Int        *A_j,
                                      HYPRE_Complex    *A_a,
                                      HYPRE_Int        *P_i,
                                      HYPRE_Int        *P_j,
                                      HYPRE_Int         ldim,
                                      HYPRE_Complex    *mat_data,
                                      HYPRE_Complex    *rhs_data,
                                      HYPRE_Int        *G_r )
{
   HYPRE_Int      i, j;
   HYPRE_Int      il, jl;
   HYPRE_Int      row;
   HYPRE_Int      Aptr, Pptr;

   /* Grid-stride loop over matrix rows */
   for (i = hypre_gpu_get_grid_thread_id<1, 1>(item);
        i < num_rows;
        i += hypre_gpu_get_grid_num_threads<1, 1>(item))
   {
      /* Set identity matrix */
      for (j = 0; j < ldim; j++)
      {
         mat_(ldim, i, j, j) = 1.0;
      }

      /* Set right hand side vector */
      il = 0;
      Aptr = A_i[i];
      Pptr = P_i[i];
      while ((Pptr < P_i[i + 1]) && (Aptr < A_i[i + 1]))
      {
         if (P_j[Pptr] >= i)
         {
            break;
         }
         else if (P_j[Pptr] < A_j[Aptr])
         {
            Pptr++;
            il++;
         }
         else if (A_j[Aptr] < P_j[Pptr])
         {
            Aptr++;
         }
         else
         {
            rhs_(ldim, i, il) = - A_a[Aptr];
            Aptr++;
            Pptr++;
            il++;
         }
      } /* Gather RHS entries */

      /* Loop over requested rows */
      il = 0;
      for (j = P_i[i]; j < P_i[i + 1]; j++)
      {
         row = P_j[j];

         /* Skip upper triangular pattern */
         if (row < i)
         {
            /* Gather matrix entries */
            Aptr = A_i[row];
            Pptr = P_i[i];
            jl = 0;
            while ((Pptr < P_i[i + 1]) && (Aptr < A_i[row + 1]))
            {
               if (P_j[Pptr] >= i)
               {
                  break;
               }
               else if (P_j[Pptr] < A_j[Aptr])
               {
                  Pptr++;
                  jl++;
               }
               else if (A_j[Aptr] < P_j[Pptr])
               {
                  Aptr++;
               }
               else
               {
                  mat_(ldim, i, il, jl) = A_a[Aptr];
                  Aptr++;
                  Pptr++;
                  jl++;
               }
            } /* Gather entries */

            il++;
         } /* Skip upper triangular pattern */
      } /* Loop over requested rows */

      /* Set number of nonzero coefficients per row of G */
      G_r[i] = il + 1;
   } /* Grid-stride loop over matrix rows */
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_FSAIScaling
 *
 * TODO: unroll inner loop
 *       Use fma?
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_FSAIScaling( hypre_DeviceItem &item,
                            HYPRE_Int         num_rows,
                            HYPRE_Int         ldim,
                            HYPRE_Complex    *sol_data,
                            HYPRE_Complex    *rhs_data,
                            HYPRE_Complex    *scaling,
                            HYPRE_Int        *info )
{
   HYPRE_Int      i, j;
   HYPRE_Complex  val;

   /* Grid-stride loop over matrix rows */
   for (i = hypre_gpu_get_grid_thread_id<1, 1>(item);
        i < num_rows;
        i += hypre_gpu_get_grid_num_threads<1, 1>(item))
   {
      val = scaling[i];
      for (j = 0; j < ldim; j++)
      {
         val += sol_(ldim, i, j) * rhs_(ldim, i, j);
      }

      if (val > 0)
      {
         scaling[i] = 1.0 / sqrt(val);
      }
      else
      {
         scaling[i] = 1.0 / sqrt(scaling[i]);
         info[i] = 1;
      }
   }
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_FSAIGatherEntries
 *
 * Output:
 *   1) G_j: column indices of G_diag
 *   2) G_a: coefficients of G_diag
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_FSAIGatherEntries( hypre_DeviceItem &item,
                                  HYPRE_Int         num_rows,
                                  HYPRE_Int         ldim,
                                  HYPRE_Complex    *sol_data,
                                  HYPRE_Complex    *scaling,
                                  HYPRE_Int        *K_i,
                                  HYPRE_Int        *K_j,
                                  HYPRE_Int        *G_i,
                                  HYPRE_Int        *G_j,
                                  HYPRE_Complex    *G_a )
{
   HYPRE_Int      i, j;
   HYPRE_Int      cnt, il;
   HYPRE_Int      col;
   HYPRE_Complex  val;

   /* Grid-stride loop over matrix rows */
   for (i = hypre_gpu_get_grid_thread_id<1, 1>(item);
        i < num_rows;
        i += hypre_gpu_get_grid_num_threads<1, 1>(item))
   {
      /* Set scaling factor */
      val = scaling[i];

      /* Set diagonal coefficient */
      cnt = G_i[i];
      G_j[cnt] = i;
      G_a[cnt] = val;
      cnt++;

      /* Set off-diagonal coefficients */
      il = 0;
      for (j = K_i[i]; j < K_i[i + 1]; j++)
      {
         col = K_j[j];

         /* Skip upper triangular pattern */
         if (col < i)
         {
            G_j[cnt + il] = col;
            G_a[cnt + il] = sol_(ldim, i, il) * val;
            il++;
         }
      }
   }
}

/*--------------------------------------------------------------------
 * hypreGPUKernel_FSAITruncateCandidate
 *
 * Truncates the candidate pattern matrix (K). This function extracts
 * lower triangular portion of the matrix up to the largest
 * "max_nonzeros_row" coefficients in absolute value.
 *
 * Assumptions:
 *    1) columns are ordered with descreasing absolute coef. values
 *    2) max_nonzeros_row < warp_size.
 *
 * TODO:
 *    1) Perform truncation with COO matrix
 *    2) Use less than one warp per row when possible
 *--------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_FSAITruncateCandidate( hypre_DeviceItem &item,
                                      HYPRE_Int         max_nonzeros_row,
                                      HYPRE_Int         num_rows,
                                      HYPRE_Int        *K_i,
                                      HYPRE_Int        *K_j,
                                      HYPRE_Complex    *K_a )
{
   HYPRE_Int      lane = (blockDim.x * blockIdx.x + threadIdx.x) & (HYPRE_WARP_SIZE - 1);
   HYPRE_Int      p = 0;
   HYPRE_Int      q = 0;
   HYPRE_Int      i, j, k, kk, cnt;
   HYPRE_Int      col;
   hypre_int      bitmask;
   HYPRE_Complex  val;
   HYPRE_Int      max_lane;
   HYPRE_Int      max_idx;
   HYPRE_Complex  max_val;
   HYPRE_Complex  warp_max_val;

   /* Grid-stride loop over matrix rows */
   for (i = (blockDim.x * blockIdx.x + threadIdx.x) / HYPRE_WARP_SIZE;
        i < num_rows;
        i += (gridDim.x * blockDim.x) / HYPRE_WARP_SIZE )
   {
      if (lane < 2)
      {
         p = read_only_load(K_i + i + lane);
      }
      q = __shfl_sync(0xFFFFFFFFU, p, 1);
      p = __shfl_sync(0xFFFFFFFFU, p, 0);

      k = 0;
      while (k < max_nonzeros_row)
      {
         /* Initialize variables */
         j = p + k + lane;
         max_val = 0.0;
         max_idx = -1;

         /* Find maximum val/col pair in each lane */
         if (j < q)
         {
            if (K_j[j] < i)
            {
               max_val = abs(K_a[j]);
               max_idx = j;
            }
         }

         for (j += HYPRE_WARP_SIZE; j < q; j += HYPRE_WARP_SIZE)
         {
            if (K_j[j] < i)
            {
               val = abs(K_a[j]);
               if (val > max_val)
               {
                  max_val = val;
                  max_idx = j;
               }
            }
         }

         /* Find maximum coefficient in absolute value in the warp */
         warp_max_val = max_val;
         #pragma unroll
         for (hypre_int d = HYPRE_WARP_SIZE >> 1; d > 0; d >>= 1)
         {
            warp_max_val = max(warp_max_val, __shfl_xor_sync(0xFFFFFFFFU, warp_max_val, d));
         }

         /* Reorder col/val entries associated with warp_max_val */
         bitmask = __ballot_sync(0xFFFFFFFFU, warp_max_val == max_val);
         if (warp_max_val > 0.0)
         {
            cnt = min((HYPRE_Int) __popc(bitmask), max_nonzeros_row - k);

            for (kk = 0; kk < cnt; kk++)
            {
               __syncwarp();
               max_lane = __ffs(bitmask) - 1;
               if (lane == max_lane)
               {
                  col = K_j[p + k + kk];
                  val = K_a[p + k + kk];

                  K_j[p + k + kk] = K_j[max_idx];
                  K_a[p + k + kk] = max_val;

                  K_j[max_idx] = col;
                  K_a[max_idx] = val;
               }

               /* Update bitmask */
               bitmask ^= (1 << max_lane);
            }

            /* Update number of nonzeros per row */
            k += cnt;
         }
         else
         {
            break;
         }
      }

      /* Exclude remaining columns */
      for (j = p + k + lane; j < q; j += HYPRE_WARP_SIZE)
      {
         K_j[j] = -1;
      }
   }
}

/*--------------------------------------------------------------------------
 * hypreDevice_FSAIExtractSubSystems
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_FSAIExtractSubSystems( HYPRE_Int       num_rows,
                                   HYPRE_Int      *A_i,
                                   HYPRE_Int      *A_j,
                                   HYPRE_Complex  *A_a,
                                   HYPRE_Int      *P_i,
                                   HYPRE_Int      *P_j,
                                   HYPRE_Int       ldim,
                                   HYPRE_Complex  *mat_data,
                                   HYPRE_Complex  *rhs_data,
                                   HYPRE_Int      *G_r )
{
   /* trivial case */
   if (num_rows <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);
   /* dim3 bDim = {1, 1, 1}; */
   /* dim3 gDim = {1, 1, 1}; */

   HYPRE_GPU_LAUNCH( hypreGPUKernel_FSAIExtractSubSystems, gDim, bDim, num_rows,
                     A_i, A_j, A_a, P_i, P_j, ldim, mat_data, rhs_data, G_r );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreDevice_FSAIScaling
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_FSAIScaling( HYPRE_Int       num_rows,
                         HYPRE_Int       ldim,
                         HYPRE_Complex  *sol_data,
                         HYPRE_Complex  *rhs_data,
                         HYPRE_Complex  *scaling,
                         HYPRE_Int      *info )
{
   /* trivial case */
   if (num_rows <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);
   /* dim3 bDim = {1, 1, 1}; */
   /* dim3 gDim = {1, 1, 1}; */

   HYPRE_GPU_LAUNCH( hypreGPUKernel_FSAIScaling, gDim, bDim,
                     num_rows, ldim, sol_data, rhs_data, scaling, info );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreDevice_FSAIGatherEntries
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypreDevice_FSAIGatherEntries( HYPRE_Int       num_rows,
                               HYPRE_Int       ldim,
                               HYPRE_Complex  *sol_data,
                               HYPRE_Complex  *scaling,
                               HYPRE_Int      *K_i,
                               HYPRE_Int      *K_j,
                               HYPRE_Int      *G_i,
                               HYPRE_Int      *G_j,
                               HYPRE_Complex  *G_a )
{
   /* trivial case */
   if (num_rows <= 0)
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);
   /* dim3 bDim = {1, 1, 1}; */
   /* dim3 gDim = {1, 1, 1}; */

   HYPRE_GPU_LAUNCH( hypreGPUKernel_FSAIGatherEntries, gDim, bDim,
                     num_rows, ldim, sol_data, scaling, K_i, K_j, G_i, G_j, G_a );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAITruncateCandidateDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAITruncateCandidateDevice( hypre_CSRMatrix *matrix,
                                   HYPRE_Int        max_nonzeros_row,
                                   HYPRE_Int        trunc_option,
                                   HYPRE_Real       trunc_factor )
{
   HYPRE_Int      num_rows     = hypre_CSRMatrixNumRows(matrix);
   HYPRE_Int      nnz          = hypre_CSRMatrixNumNonzeros(matrix);
   HYPRE_Int     *mat_i        = hypre_CSRMatrixI(matrix);
   HYPRE_Int     *mat_j        = hypre_CSRMatrixJ(matrix);
   HYPRE_Complex *mat_a        = hypre_CSRMatrixData(matrix);

   HYPRE_Int      nnz_new;
   HYPRE_Int     *mat_ii;
   HYPRE_Int     *mat_ii_new;
   HYPRE_Int     *mat_j_new;
   HYPRE_Complex *mat_a_new;

   if (trunc_factor > 0)
   {
      /* TODO */
   }
   else
   {
      /*-----------------------------------------------------
       * Keep only the largest coefficients in absolute value
       *-----------------------------------------------------*/

      /* Allocate memory for row indices array*/
      hypre_GpuProfilingPushRange("Storage1");
      //nnz = max_nonzeros_row * num_rows;
      mat_ii     = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);
      mat_ii_new = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);
      mat_j_new  = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);
      mat_a_new  = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_DEVICE);
      hypre_GpuProfilingPopRange();

      /* Build row indices array */
      hypreDevice_CsrRowPtrsToIndices_v2(num_rows, nnz, mat_i, mat_ii);

      /* Mark unwanted entries with -1 */
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "warp", bDim);

      hypre_GpuProfilingPushRange("TruncCand");
      HYPRE_GPU_LAUNCH(hypreGPUKernel_FSAITruncateCandidate, gDim, bDim,
                       max_nonzeros_row, num_rows, mat_i, mat_j, mat_a );
      hypre_SyncComputeStream(hypre_handle());
      hypre_GetDeviceLastError();
      hypre_GpuProfilingPopRange();

      /* Remove unwanted entries */
      hypre_GpuProfilingPushRange("CopyIf");
      auto new_end = HYPRE_THRUST_CALL(
                        copy_if,
                        HYPRE_THRUST_ZIP3(mat_ii,       mat_j,       mat_a),
                        HYPRE_THRUST_ZIP3(mat_ii + nnz, mat_j + nnz, mat_a + nnz),
                        mat_j,
                        HYPRE_THRUST_ZIP3(mat_ii_new,   mat_j_new,   mat_a_new),
                        is_nonnegative<HYPRE_Int>() );

      nnz_new = thrust::get<0>(new_end.get_iterator_tuple()) - mat_ii_new;
      hypre_GpuProfilingPopRange();

      /* Recompute row pointer */
      hypreDevice_CsrRowIndicesToPtrs_v2(num_rows, nnz_new, mat_ii_new, mat_i);

      /* TODO: Reallocate mat_j_new and mat_a_new? */

      /* Update matrix info */
      hypre_CSRMatrixJ(matrix)           = mat_j_new;
      hypre_CSRMatrixData(matrix)        = mat_a_new;
      hypre_CSRMatrixNumNonzeros(matrix) = nnz_new;

      /* Free memory */
      hypre_TFree(mat_ii, HYPRE_MEMORY_DEVICE);
      hypre_TFree(mat_ii_new, HYPRE_MEMORY_DEVICE);
      hypre_TFree(mat_j, HYPRE_MEMORY_DEVICE);
      hypre_TFree(mat_a, HYPRE_MEMORY_DEVICE);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAISetupStaticPowerDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISetupStaticPowerDevice( void               *fsai_vdata,
                                  hypre_ParCSRMatrix *A,
                                  hypre_ParVector    *f,
                                  hypre_ParVector    *u )
{
   hypre_ParFSAIData   *fsai_data = (hypre_ParFSAIData*) fsai_vdata;
   hypre_ParCSRMatrix  *G         = hypre_ParFSAIDataGmat(fsai_data);
   hypre_CSRMatrix     *G_diag    = hypre_ParCSRMatrixDiag(G);

   hypre_CSRMatrix     *A_diag    = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int            num_rows  = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int            num_nonzeros_G;

   hypre_ParCSRMatrix  *Atilde;
   hypre_ParCSRMatrix  *B;
   hypre_ParCSRMatrix  *Ktilde;
   HYPRE_Int            i;

   /* TODO: Move to fsai_data? */
   HYPRE_Complex       *scaling;
   HYPRE_Int           *info;

   /* TODO: these variables belong to hypre_ParFSAIData */
   HYPRE_Int            max_nonzeros_row = 8;
   HYPRE_Int            num_levels = 2;
   HYPRE_Int            filter_option = 2;
   HYPRE_Real           filter_threshold = 0.0;

   /* Error code array for FSAI */
   info = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);

   /*-----------------------------------------------------
    *  Compute candidate pattern
    *-----------------------------------------------------*/

   hypre_GpuProfilingPushRange("CandPat");

   /* Compute filtered version of A */
   Atilde = hypre_ParCSRMatrixClone(A, 1);
   hypre_ParCSRMatrixDropSmallEntriesDevice(Atilde, 5e-2, filter_option);

   /* Compute power pattern */
   switch (num_levels)
   {
      case 1:
         Ktilde = Atilde;
         break;

      case 2:
         Ktilde = hypre_ParCSRMatMatDevice(Atilde, Atilde);
         break;

      case 3:
         /* First pass */
         B = hypre_ParCSRMatMatDevice(Atilde, Atilde);

         /* Second pass */
         Ktilde = hypre_ParCSRMatMatDevice(Atilde, B);
         hypre_ParCSRMatrixDestroy(B);
         break;

      case 4:
         /* First pass */
         B = hypre_ParCSRMatMatDevice(Atilde, Atilde);

         /* Second pass */
         Ktilde = hypre_ParCSRMatMatDevice(B, B);
         hypre_ParCSRMatrixDestroy(B);
         break;

      default:
         Ktilde = hypre_ParCSRMatrixClone(Atilde, 1);
         for (i = 1; i < num_levels; i++)
         {
            /* Compute temporary matrix */
            B = hypre_ParCSRMatMatDevice(Atilde, Ktilde);

            /* Update resulting matrix */
            hypre_ParCSRMatrixDestroy(Ktilde);
            Ktilde = hypre_ParCSRMatrixClone(B, 1);
         }
   }

   hypre_GpuProfilingPopRange();

   /*-----------------------------------------------------
    *  Filter candidate pattern
    *-----------------------------------------------------*/

   hypre_GpuProfilingPushRange("FilterPat");

#if defined (DEBUG_FSAI)
   {
      hypre_ParCSRMatrixPrintIJ(Ktilde, 0, 0, "FSAI.out.H.ij");
   }
#endif

   /* Set pattern matrix diagonal matrix */
   hypre_CSRMatrix  *K_diag = hypre_ParCSRMatrixDiag(Ktilde);

   /* Filter candidate pattern */
   hypre_FSAITruncateCandidateDevice(K_diag, max_nonzeros_row, filter_option, filter_threshold);

   /* Sort candidate pattern matrix */
   hypre_CSRMatrixSortRow(K_diag);

#if defined (DEBUG_FSAI)
   {
      hypre_ParCSRMatrixPrintIJ(Ktilde, 0, 0, "FSAI.out.K.ij");
   }
#endif

   hypre_GpuProfilingPopRange();

   /*-----------------------------------------------------
    *  Preprocess input matrix
    *-----------------------------------------------------*/

   hypre_GpuProfilingPushRange("PreProcessA");

   /* TODO: implement faster diagonal extraction (use "i == A_j[A_i[i]]")*/
   scaling = hypre_TAlloc(HYPRE_Complex, num_rows, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixExtractDiagonalDevice(A_diag, scaling, 0);

   /* Sort matrix A */
   hypre_CSRMatrixSortRow(A_diag);

   hypre_GpuProfilingPopRange();

   /*-----------------------------------------------------
    *  Extract local linear systems
    *-----------------------------------------------------*/

   /* Allocate storage */
   hypre_GpuProfilingPushRange("Storage1");
   HYPRE_Complex  *mat_data = hypre_CTAlloc(HYPRE_Complex,
                                            max_nonzeros_row * max_nonzeros_row * num_rows,
                                            HYPRE_MEMORY_DEVICE);
   HYPRE_Complex  *rhs_data = hypre_CTAlloc(HYPRE_Complex, max_nonzeros_row * num_rows,
                                            HYPRE_MEMORY_DEVICE);
   HYPRE_Complex  *sol_data = hypre_CTAlloc(HYPRE_Complex, max_nonzeros_row * num_rows,
                                            HYPRE_MEMORY_DEVICE);
   hypre_GpuProfilingPopRange();

   /* Gather dense linear subsystems */
   hypre_GpuProfilingPushRange("ExtractLS");
   hypreDevice_FSAIExtractSubSystems(num_rows,
                                     hypre_CSRMatrixI(A_diag),
                                     hypre_CSRMatrixJ(A_diag),
                                     hypre_CSRMatrixData(A_diag),
                                     hypre_CSRMatrixI(K_diag),
                                     hypre_CSRMatrixJ(K_diag),
                                     max_nonzeros_row,
                                     mat_data,
                                     rhs_data,
                                     hypre_CSRMatrixI(G_diag) + 1);
   hypre_GpuProfilingPopRange();

   /* Copy rhs to solution vector */
   hypre_GpuProfilingPushRange("CopyRHS");
   hypre_TMemcpy(sol_data, rhs_data, HYPRE_Complex, max_nonzeros_row * num_rows,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_GpuProfilingPopRange();

   /* Build array of pointers */
   hypre_GpuProfilingPushRange("Storage2");
   HYPRE_Complex **sol_aop = hypre_TAlloc(HYPRE_Complex *, num_rows, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex **mat_aop = hypre_TAlloc(HYPRE_Complex *, num_rows, HYPRE_MEMORY_DEVICE);
   hypre_GpuProfilingPopRange();

   hypre_GpuProfilingPushRange("FormAOP");
   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);

      HYPRE_GPU_LAUNCH( hypreGPUKernel_ComplexArrayToArrayOfPtrs, gDim, bDim,
                        num_rows, max_nonzeros_row * max_nonzeros_row, mat_data, mat_aop );

      HYPRE_GPU_LAUNCH( hypreGPUKernel_ComplexArrayToArrayOfPtrs, gDim, bDim,
                        num_rows, max_nonzeros_row, sol_data, sol_aop );

      hypre_SyncComputeStream(hypre_handle());
      hypre_GetDeviceLastError();
   }
   hypre_GpuProfilingPopRange();

   /*-----------------------------------------------------
    *  Solve local linear systems
    *-----------------------------------------------------*/

   /* TODO */

   /*-----------------------------------------------------
    *  Finalize construction of the triangular factor
    *-----------------------------------------------------*/

   hypre_GpuProfilingPushRange("BuildFSAI");

   /* Update scaling factor */
   hypreDevice_FSAIScaling(num_rows, max_nonzeros_row, sol_data, rhs_data, scaling, info);

   /* Compute the row pointer G_i */
   hypreDevice_IntegerInclusiveScan(num_rows + 1, hypre_CSRMatrixI(G_diag));

   /* Get the actual number of nonzero coefficients of G_diag */
   hypre_TMemcpy(&num_nonzeros_G, hypre_CSRMatrixI(G_diag) + num_rows,
                 HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   /* Update the nonzero count of matrix G */
   hypre_CSRMatrixNumNonzeros(G_diag) = num_nonzeros_G;

   /* Set column indices and coefficients of G */
   hypreDevice_FSAIGatherEntries(num_rows,
                                 max_nonzeros_row,
                                 sol_data,
                                 scaling,
                                 hypre_CSRMatrixI(K_diag),
                                 hypre_CSRMatrixJ(K_diag),
                                 hypre_CSRMatrixI(G_diag),
                                 hypre_CSRMatrixJ(G_diag),
                                 hypre_CSRMatrixData(G_diag));

   hypre_GpuProfilingPopRange();
   /* TODO: Reallocate memory for G_j/G_a? */

   /*-----------------------------------------------------
    *  Free memory
    *-----------------------------------------------------*/

   /* TODO: can we free some of these earlier?*/
   hypre_TFree(rhs_data, HYPRE_MEMORY_DEVICE);
   hypre_TFree(sol_data, HYPRE_MEMORY_DEVICE);
   hypre_TFree(mat_data, HYPRE_MEMORY_DEVICE);
   hypre_TFree(sol_aop, HYPRE_MEMORY_DEVICE);
   hypre_TFree(mat_aop, HYPRE_MEMORY_DEVICE);
   hypre_TFree(scaling, HYPRE_MEMORY_DEVICE);
   hypre_TFree(info, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAISetupDevice
 *
 * Notes:
 *   1) Do we need to sort the matrices?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISetupDevice( void               *fsai_vdata,
                       hypre_ParCSRMatrix *A,
                       hypre_ParVector    *f,
                       hypre_ParVector    *u )
{
   hypre_ParFSAIData       *fsai_data     = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Int                algo_type     = hypre_ParFSAIDataAlgoType(fsai_data);
   hypre_ParCSRMatrix      *G             = hypre_ParFSAIDataGmat(fsai_data);
   hypre_ParCSRMatrix      *h_A;

   hypre_GpuProfilingPushRange("FSAISetup");

   if (algo_type == 1 || algo_type == 2)
   {
      /* Initialize matrix G on host */
      hypre_ParCSRMatrixInitialize_v2(G, HYPRE_MEMORY_HOST);

      /* Clone input matrix on host */
      h_A = hypre_ParCSRMatrixClone_v2(A, 1, HYPRE_MEMORY_HOST);

      /* Compute FSAI factor on host */
      switch (algo_type)
      {
         case 2:
            hypre_FSAISetupOMPDyn(fsai_vdata, h_A, f, u);
            break;

         default:
            hypre_FSAISetupNative(fsai_vdata, h_A, f, u);
            break;
      }

      /* Move FSAI factor G to device */
      hypre_ParCSRMatrixMigrate(G, HYPRE_MEMORY_DEVICE);

      /* Destroy temporary data on host */
      HYPRE_ParCSRMatrixDestroy(h_A);
   }
   else
   {
      /* Initialize matrix G */
      hypre_ParCSRMatrixInitialize(G);

      if (algo_type == 3)
      {
         hypre_FSAISetupStaticPowerDevice(fsai_vdata, A, f, u);
      }
   }

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

#endif /* if defined(HYPRE_USING_GPU) */

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Two-grid system solver
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "seq_mv/protos.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

template<typename T>
struct functor : public thrust::binary_function<T, T, T>
{
   T scale;

   functor(T scale_) { scale = scale_; }

   __host__ __device__
   T operator()(T &x, T &y) const
   {
      return x + scale * (y - hypre_abs(x));
   }
};

void hypreDevice_extendWtoP( HYPRE_Int P_nr_of_rows, HYPRE_Int W_nr_of_rows,
                             HYPRE_Int W_nr_of_cols, HYPRE_Int *CF_marker,
                             HYPRE_Int W_diag_nnz, HYPRE_Int *W_diag_i,
                             HYPRE_Int *W_diag_j, HYPRE_Complex *W_diag_data,
                             HYPRE_Int *P_diag_i, HYPRE_Int *P_diag_j,
                             HYPRE_Complex *P_diag_data, HYPRE_Int *W_offd_i,
                             HYPRE_Int *P_offd_i );

/*--------------------------------------------------------------------------
 * hypre_MGRBuildPDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildPDevice(hypre_ParCSRMatrix  *A,
                      HYPRE_Int           *CF_marker,
                      HYPRE_BigInt        *num_cpts_global,
                      HYPRE_Int            method,
                      hypre_ParCSRMatrix **P_ptr)
{
   MPI_Comm            comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int           num_procs, my_id;
   HYPRE_Int           A_nr_of_rows = hypre_ParCSRMatrixNumRows(A);

   hypre_ParCSRMatrix *A_FF = NULL, *A_FC = NULL, *P = NULL;
   hypre_CSRMatrix    *W_diag = NULL, *W_offd = NULL;
   HYPRE_Int           W_nr_of_rows, P_diag_nnz, nfpoints;
   HYPRE_Int          *P_diag_i = NULL, *P_diag_j = NULL, *P_offd_i = NULL;
   HYPRE_Complex      *P_diag_data = NULL, *diag = NULL, *diag1 = NULL;
   HYPRE_BigInt        nC_global;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   nfpoints = HYPRE_THRUST_CALL(count,
                                CF_marker,
                                CF_marker + A_nr_of_rows,
                                -1);

   if (method > 0)
   {
      hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, num_cpts_global, NULL, &A_FC, &A_FF);
      diag = hypre_CTAlloc(HYPRE_Complex, nfpoints, HYPRE_MEMORY_DEVICE);
      if (method == 1)
      {
         // extract diag inverse sqrt
         // hypre_CSRMatrixExtractDiagonalDevice(hypre_ParCSRMatrixDiag(A_FF), diag, 3);

         // L1-Jacobi-type interpolation
         HYPRE_Complex scal = 1.0;

         diag1 = hypre_CTAlloc(HYPRE_Complex, nfpoints, HYPRE_MEMORY_DEVICE);
         hypre_CSRMatrixExtractDiagonalDevice(hypre_ParCSRMatrixDiag(A_FF), diag, 0);

         hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixDiag(A_FF), NULL, NULL,
                                            diag1, 1, 1.0, "set");
         hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixDiag(A_FC), NULL, NULL,
                                            diag1, 1, 1.0, "add");
         hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixOffd(A_FF), NULL, NULL,
                                            diag1, 1, 1.0, "add");
         hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixOffd(A_FC), NULL, NULL,
                                            diag1, 1, 1.0, "add");

         HYPRE_THRUST_CALL(transform,
                           diag,
                           diag + nfpoints,
                           diag1,
                           diag,
                           functor<HYPRE_Complex>(scal));

         HYPRE_THRUST_CALL(transform,
                           diag,
                           diag + nfpoints,
                           diag,
                           1.0 / _1);

         hypre_TFree(diag1, HYPRE_MEMORY_DEVICE);
      }
      else if (method == 2)
      {
         // extract diag inverse
         hypre_CSRMatrixExtractDiagonalDevice(hypre_ParCSRMatrixDiag(A_FF), diag, 2);
      }

      HYPRE_THRUST_CALL( transform, diag, diag + nfpoints, diag, thrust::negate<HYPRE_Complex>() );

      hypre_Vector *D_FF_inv = hypre_SeqVectorCreate(nfpoints);
      hypre_VectorData(D_FF_inv) = diag;
      hypre_SeqVectorInitialize_v2(D_FF_inv, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixDiagScaleDevice(hypre_ParCSRMatrixDiag(A_FC), D_FF_inv, NULL);
      hypre_CSRMatrixDiagScaleDevice(hypre_ParCSRMatrixOffd(A_FC), D_FF_inv, NULL);
      hypre_SeqVectorDestroy(D_FF_inv);
      W_diag = hypre_ParCSRMatrixDiag(A_FC);
      W_offd = hypre_ParCSRMatrixOffd(A_FC);
      nC_global = hypre_ParCSRMatrixGlobalNumCols(A_FC);
   }
   else
   {
      W_diag = hypre_CSRMatrixCreate(nfpoints, A_nr_of_rows - nfpoints, 0);
      W_offd = hypre_CSRMatrixCreate(nfpoints, 0, 0);
      hypre_CSRMatrixInitialize_v2(W_diag, 0, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixInitialize_v2(W_offd, 0, HYPRE_MEMORY_DEVICE);

      if (my_id == (num_procs - 1))
      {
         nC_global = num_cpts_global[1];
      }
      hypre_MPI_Bcast(&nC_global, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }

   W_nr_of_rows = hypre_CSRMatrixNumRows(W_diag);

   /* Construct P from matrix product W_diag */
   P_diag_nnz  = hypre_CSRMatrixNumNonzeros(W_diag) + hypre_CSRMatrixNumCols(W_diag);
   P_diag_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);
   P_diag_j    = hypre_TAlloc(HYPRE_Int,     P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(HYPRE_Complex, P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_offd_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);

   //hypre_NvtxPushRangeColor("Extend matrix", 4);
   hypreDevice_extendWtoP( A_nr_of_rows,
                           W_nr_of_rows,
                           hypre_CSRMatrixNumCols(W_diag),
                           CF_marker,
                           hypre_CSRMatrixNumNonzeros(W_diag),
                           hypre_CSRMatrixI(W_diag),
                           hypre_CSRMatrixJ(W_diag),
                           hypre_CSRMatrixData(W_diag),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           hypre_CSRMatrixI(W_offd),
                           P_offd_i );
   //hypre_NvtxPopRange();

   // final P
   P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                nC_global,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                hypre_CSRMatrixNumCols(W_offd),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(W_offd) );

   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(P)) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(P)) = HYPRE_MEMORY_DEVICE;

   hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(W_offd);
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(W_offd);
   hypre_CSRMatrixJ(W_offd)    = NULL;
   hypre_CSRMatrixData(W_offd) = NULL;

   if (method > 0)
   {
      hypre_ParCSRMatrixDeviceColMapOffd(P)    = hypre_ParCSRMatrixDeviceColMapOffd(A_FC);
      hypre_ParCSRMatrixColMapOffd(P)          = hypre_ParCSRMatrixColMapOffd(A_FC);
      hypre_ParCSRMatrixDeviceColMapOffd(A_FC) = NULL;
      hypre_ParCSRMatrixColMapOffd(A_FC)       = NULL;
      hypre_ParCSRMatrixNumNonzeros(P)         = hypre_ParCSRMatrixNumNonzeros(A_FC) +
                                                 hypre_ParCSRMatrixGlobalNumCols(A_FC);
   }
   else
   {
      hypre_ParCSRMatrixNumNonzeros(P) = nC_global;
   }
   hypre_ParCSRMatrixDNumNonzeros(P) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);

   hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   if (A_FF)
   {
      hypre_ParCSRMatrixDestroy(A_FF);
   }
   if (A_FC)
   {
      hypre_ParCSRMatrixDestroy(A_FC);
   }

   if (method <= 0)
   {
      hypre_CSRMatrixDestroy(W_diag);
      hypre_CSRMatrixDestroy(W_offd);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRRelaxL1JacobiDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRRelaxL1JacobiDevice( hypre_ParCSRMatrix *A,
                              hypre_ParVector    *f,
                              HYPRE_Int          *CF_marker,
                              HYPRE_Int           relax_points,
                              HYPRE_Real          relax_weight,
                              HYPRE_Real         *l1_norms,
                              hypre_ParVector    *u,
                              hypre_ParVector    *Vtemp )
{
   hypre_BoomerAMGRelax(A, f, CF_marker, 18,
                        relax_points, relax_weight, 1.0,
                        l1_norms, u, Vtemp, NULL);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_RealArrayToArrayOfPtrs
 *
 * TODO:
 *   1) data as template arg.
 *   2) Move this to device_utils?
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_RealArrayToArrayOfPtrs( hypre_DeviceItem  &item,
                                       HYPRE_Int          num_rows,
                                       HYPRE_Int          ldim,
                                       HYPRE_Real        *data,
                                       HYPRE_Real       **data_aop )
{
   HYPRE_Int i = threadIdx.x + blockIdx.x * blockDim.x;

   if (i < num_rows)
   {
      data_aop[i] = &data[i * ldim];
   }
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatrixExtractBlockDiag
 *
 * Fills vector diag with the block diagonals from the input matrix.
 * This function uses column-major storage for diag.
 *
 * TODOs:
 *    1) Move this to csr_matop_device.c
 *    2) Use sub-warps?
 *    3) blk_size as template arg.
 *    4) Choose diag storage between row and column-major?
 *    5) Should we build flat arrays, arrays of pointers, or allow both?
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRMatrixExtractBlockDiag( hypre_DeviceItem  &item,
                                          HYPRE_Int          blk_size,
                                          HYPRE_Int          num_rows,
                                          HYPRE_Int         *A_i,
                                          HYPRE_Int         *A_j,
                                          HYPRE_Complex     *A_a,
                                          HYPRE_Real        *diag )
{
   HYPRE_Int   lane = (blockDim.x * blockIdx.x + threadIdx.x) & (HYPRE_WARP_SIZE - 1);
   //HYPRE_Int   bs2  = blk_size * blk_size;
   HYPRE_Int   bidx;
   HYPRE_Int   lidx;
   HYPRE_Int   i, ii, j, pj, qj, k;
   HYPRE_Int   col;

   /* Grid-stride loop over block matrix rows */
   for (bidx = (blockIdx.x * blockDim.x + threadIdx.x) / HYPRE_WARP_SIZE;
        bidx < num_rows / blk_size;
        bidx += (gridDim.x * blockDim.x) * blk_size / HYPRE_WARP_SIZE)
   {
      /* TODO: unroll this loop */
      for (lidx = 0; lidx < blk_size; lidx++)
      {
         ii = bidx * blk_size;
         i  = ii + lidx;

         if (lane < 2)
         {
            pj = read_only_load(A_i + i + lane);
         }
         qj = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pj, 1);
         pj = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pj, 0);

         /* Loop over columns */
         for (j = pj + lane; j < qj; j += HYPRE_WARP_SIZE)
         {
            k = read_only_load(A_j + j);
            col = A_j[k];

            if ((col >= ii) &&
                (col <  ii + blk_size) &&
                (fabs(A_a[k]) > HYPRE_REAL_MIN))
            {
               /* batch offset + column offset + row offset */
               diag[ii * blk_size + (col - ii) * blk_size + lidx] = A_a[k];
            }
         }
      } /* Local block loop */
   } /* Grid-stride loop */
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatrixExtractBlockDiagMarked
 *
 * Fills vector diag with the block diagonals from the input matrix.
 * This function uses column-major storage for diag.
 *
 * TODOs:
 *    1) Move this to csr_matop_device.c
 *    2) Use sub-warps?
 *    3) blk_size as template arg.
 *    4) Choose diag storage between row and column-major?
 *    5) Should we build flat arrays, arrays of pointers, or allow both?
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRMatrixExtractBlockDiagMarked( hypre_DeviceItem  &item,
                                                HYPRE_Int          blk_size,
                                                HYPRE_Int          num_rows,
                                                HYPRE_Int          marker_val,
                                                HYPRE_Int         *marker,
                                                HYPRE_Int         *marker_indices,
                                                HYPRE_Int         *A_i,
                                                HYPRE_Int         *A_j,
                                                HYPRE_Complex     *A_a,
                                                HYPRE_Real        *diag )
{
   HYPRE_Int   lane = (blockDim.x * blockIdx.x + threadIdx.x) & (HYPRE_WARP_SIZE - 1);
   //HYPRE_Int   bs2  = blk_size * blk_size;
   HYPRE_Int   bidx;
   HYPRE_Int   lidx;
   HYPRE_Int   i, ii, j, pj, qj, k;
   HYPRE_Int   col;

   /* Grid-stride loop over block matrix rows */
   for (bidx = (blockIdx.x * blockDim.x + threadIdx.x) / HYPRE_WARP_SIZE;
        bidx < num_rows / blk_size;
        bidx += (gridDim.x * blockDim.x) * blk_size / HYPRE_WARP_SIZE)
   {
      /* TODO: unroll this loop */
      for (lidx = 0; lidx < blk_size; lidx++)
      {
         ii = bidx * blk_size;
         i  = ii + lidx;

         if (marker[i] == marker_val)
         {
            if (lane < 2)
            {
               pj = read_only_load(A_i + i + lane);
            }
            qj = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pj, 1);
            pj = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pj, 0);

            /* Loop over columns */
            for (j = pj + lane; j < qj; j += HYPRE_WARP_SIZE)
            {
               k = read_only_load(A_j + j);
               col = A_j[k];

               if (marker[col] == marker_val)
               {
                  if ((col >= ii) &&
                      (col <  ii + blk_size) &&
                      (fabs(A_a[k]) > HYPRE_REAL_MIN))
                  {
                     /* batch offset + column offset + row offset */
                     diag[marker_indices[ii] * blk_size + (col - ii) * blk_size + lidx] = A_a[k];
                  }
               }
            }
         } /* row check */
      } /* Local block loop */
   } /* Grid-stride loop */
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixExtractBlockDiagDevice
 *
 * TODOs:
 *   1) Allow other local solver choices. Design an interface for that.
 *   2) Currently assuming that HYPRE_Real == double
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixExtractBlockDiagDevice( hypre_ParCSRMatrix   *A,
                                          HYPRE_Int             blk_size,
                                          HYPRE_Int             point_type,
                                          HYPRE_Int            *CF_marker,
                                          HYPRE_Int            *bdiag_size_ptr,
                                          HYPRE_Real          **diag_ptr,
                                          HYPRE_Int             diag_type )
{
   /* Matrix variables */
   hypre_CSRMatrix      *A_diag       = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int             num_rows     = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int            *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int            *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex        *A_diag_data  = hypre_CSRMatrixData(A_diag);

   /* Local LS variables */
   HYPRE_Int            *pivots;
   HYPRE_Int            *infos;
   HYPRE_Int            *blk_row_indices;
   HYPRE_Real           *diag = *diag_ptr;
   HYPRE_Real           *invdiag;
   HYPRE_Real          **diag_aop;
   HYPRE_Real          **invdiag_aop;

   /* Local variables */
   HYPRE_Int             bs2 = blk_size * blk_size;
   HYPRE_Int             num_points;
   HYPRE_Int             num_blocks;
   HYPRE_Int             bdiag_size;

   /* Additional variables for debugging */
#if HYPRE_DEBUG
   HYPRE_Int            *h_infos;
   HYPRE_Int             k, myid;

   hypre_MPI_Comm_rank(hypre_ParCSRMatrixComm(A), &myid);
#endif

   /* Count the number of points matching point_type in CF_marker */
   if (!CF_marker)
   {
      num_points = num_rows;
      blk_row_indices = NULL;
   }
   else
   {
      num_points = HYPRE_THRUST_CALL( count,
                                      CF_marker,
                                      CF_marker + num_rows,
                                      point_type );

      /* Compute block row indices */
      blk_row_indices = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
      hypreDevice_IntFilln(blk_row_indices, (size_t) num_rows, 1);
      HYPRE_THRUST_CALL(exclusive_scan_by_key,
                        CF_marker,
                        CF_marker + num_rows,
                        blk_row_indices,
                        blk_row_indices);
   }

   /* Compute block info */
   num_blocks = num_points / blk_size + 1;
   bdiag_size = num_blocks * bs2;

   if (!diag)
   {
      hypre_TFree(diag, HYPRE_MEMORY_DEVICE);
      diag = hypre_CTAlloc(HYPRE_Real, bdiag_size, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      diag = hypre_CTAlloc(HYPRE_Real, bdiag_size, HYPRE_MEMORY_DEVICE);
   }

  /*-----------------------------------------------------------------
   * Extract diagonal sub-blocks
   *-----------------------------------------------------------------*/
   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "warp", bDim);

      if (CF_marker)
      {
         HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRMatrixExtractBlockDiagMarked, gDim, bDim,
                           blk_size, num_rows, point_type, CF_marker, blk_row_indices,
                           A_diag_i, A_diag_j, A_diag_data, diag );
      }
      else
      {
         HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRMatrixExtractBlockDiag, gDim, bDim,
                           blk_size, num_rows, A_diag_i, A_diag_j, A_diag_data, diag );
      }
   }

  /*-----------------------------------------------------------------
   * Invert diagonal sub-blocks
   *-----------------------------------------------------------------*/

   if (diag_type == 1)
   {
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "InvertDiagSubBlocks");

      /* Memory allocation */
      diag_aop    = hypre_TAlloc(HYPRE_Real *, num_rows, HYPRE_MEMORY_DEVICE);
      invdiag_aop = hypre_TAlloc(HYPRE_Real *, num_rows, HYPRE_MEMORY_DEVICE);
      invdiag     = hypre_TAlloc(HYPRE_Real, bdiag_size, HYPRE_MEMORY_DEVICE);
      pivots      = hypre_CTAlloc(HYPRE_Int, num_rows * blk_size, HYPRE_MEMORY_DEVICE);
      infos       = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
#if defined (HYPRE_DEBUG)
      h_infos = hypre_TAlloc(HYPRE_Int,  num_rows, HYPRE_MEMORY_HOST);
#endif

      /* Set array of pointers */
      {
         dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
         dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);

         HYPRE_GPU_LAUNCH( hypreGPUKernel_RealArrayToArrayOfPtrs, gDim, bDim,
                           num_rows, bs2, diag, diag_aop );

         HYPRE_GPU_LAUNCH( hypreGPUKernel_RealArrayToArrayOfPtrs, gDim, bDim,
                           num_rows, bs2, invdiag, invdiag_aop );
      }

      /* Compute LU factorization */
      HYPRE_CUBLAS_CALL(cublasDgetrfBatched(hypre_HandleCublasHandle(hypre_handle()),
                                            blk_size,
                                            diag_aop,
                                            blk_size,
                                            pivots,
                                            infos,
                                            num_rows));
#if defined (HYPRE_DEBUG)
     hypre_TMemcpy(h_infos, infos, HYPRE_Int, num_rows, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
     for (k = 0; k < num_rows; k++)
     {
        if (h_infos[k] != 0)
        {
           if (h_infos[k] < 0)
           {
              hypre_printf("[%d]: LU fact. failed at system %d, parameter %d ",
                           myid, k, h_infos[k]);
           }
           else
           {
              hypre_printf("[%d]: Singular U(%d, %d) at system %d",
                           myid, h_infos[k], h_infos[k], k);
           }
        }
     }
#endif

      /* Compute sub-blocks inverses */
      HYPRE_CUBLAS_CALL(cublasDgetriBatched(hypre_HandleCublasHandle(hypre_handle()),
                                            blk_size,
                                            diag_aop,
                                            blk_size,
                                            pivots,
                                            invdiag_aop,
                                            blk_size,
                                            infos,
                                            num_rows));

      /* Free memory */
      hypre_TFree(diag_aop, HYPRE_MEMORY_DEVICE);
      hypre_TFree(invdiag_aop, HYPRE_MEMORY_DEVICE);
      hypre_TFree(infos, HYPRE_MEMORY_DEVICE);
      hypre_TFree(pivots, HYPRE_MEMORY_DEVICE);
#if defined (HYPRE_DEBUG)
      hypre_TFree(h_infos, HYPRE_MEMORY_HOST);
#endif

      HYPRE_ANNOTATE_REGION_END("%s", "InvertDiagSubBlocks");
   }

   /* Set output pointers and free memory */
   *bdiag_size_ptr = bdiag_size;
   if (diag_type == 1)
   {
      *diag_ptr = diag;
   }
   else
   {
      *diag_ptr = invdiag;
      hypre_TFree(diag, HYPRE_MEMORY_DEVICE);
   }
   hypre_TFree(blk_row_indices, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif

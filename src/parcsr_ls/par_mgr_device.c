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

#include "_hypre_onedpl.hpp"
#include "seq_mv/seq_mv.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined (HYPRE_USING_GPU)

template<typename T>
#if defined(HYPRE_USING_SYCL)
struct functor
#else
struct functor : public thrust::binary_function<T, T, T>
#endif
{
   T scale;

   functor(T scale_) { scale = scale_; }

   __host__ __device__
   T operator()(const T &x, const T &y) const
   {
      return x + scale * (y - hypre_abs(x));
   }
};

/*--------------------------------------------------------------------------
 * hypre_MGRBuildPFromWpDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildPFromWpDevice( hypre_ParCSRMatrix   *A,
                             hypre_ParCSRMatrix   *Wp,
                             HYPRE_Int            *CF_marker,
                             hypre_ParCSRMatrix  **P_ptr)
{
   /* Wp info */
   hypre_CSRMatrix     *Wp_diag = hypre_ParCSRMatrixDiag(Wp);
   hypre_CSRMatrix     *Wp_offd = hypre_ParCSRMatrixOffd(Wp);

   /* Local variables */
   hypre_ParCSRMatrix  *P;
   hypre_CSRMatrix     *P_diag;
   hypre_CSRMatrix     *P_offd;
   HYPRE_Int            P_diag_nnz;

   hypre_GpuProfilingPushRange("MGRBuildPFromWp");

   /* Set local variables */
   P_diag_nnz = hypre_CSRMatrixNumNonzeros(Wp_diag) +
                hypre_CSRMatrixNumCols(Wp_diag);

   /* Create interpolation matrix */
   P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixGlobalNumCols(Wp),
                                hypre_ParCSRMatrixRowStarts(A),
                                hypre_ParCSRMatrixColStarts(Wp),
                                hypre_CSRMatrixNumCols(Wp_offd),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(Wp_offd));

   /* Initialize interpolation matrix */
   hypre_ParCSRMatrixInitialize_v2(P, HYPRE_MEMORY_DEVICE);
   hypre_ParCSRMatrixDNumNonzeros(P) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);
   P_diag = hypre_ParCSRMatrixDiag(P);
   P_offd = hypre_ParCSRMatrixOffd(P);

   /* Copy contents from W to P and set identity matrix for the mapping between coarse points */
   hypreDevice_extendWtoP(hypre_ParCSRMatrixNumRows(A),
                          hypre_ParCSRMatrixNumRows(Wp),
                          hypre_CSRMatrixNumCols(Wp_diag),
                          CF_marker,
                          hypre_CSRMatrixNumNonzeros(Wp_diag),
                          hypre_CSRMatrixI(Wp_diag),
                          hypre_CSRMatrixJ(Wp_diag),
                          hypre_CSRMatrixData(Wp_diag),
                          hypre_CSRMatrixI(P_diag),
                          hypre_CSRMatrixJ(P_diag),
                          hypre_CSRMatrixData(P_diag),
                          hypre_CSRMatrixI(Wp_offd),
                          hypre_CSRMatrixI(P_offd));

   /* Swap some pointers to avoid data copies */
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(Wp_offd);
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(Wp_offd);
   hypre_CSRMatrixJ(Wp_offd)    = NULL;
   hypre_CSRMatrixData(Wp_offd) = NULL;
   /* hypre_ParCSRMatrixDeviceColMapOffd(P)    = hypre_ParCSRMatrixDeviceColMapOffd(Wp); */
   /* hypre_ParCSRMatrixColMapOffd(P)          = hypre_ParCSRMatrixColMapOffd(Wp); */
   /* hypre_ParCSRMatrixDeviceColMapOffd(Wp)   = NULL; */
   /* hypre_ParCSRMatrixColMapOffd(Wp)         = NULL; */

   /* Create communication package */
   hypre_MatvecCommPkgCreate(P);

   /* Set output pointer to the interpolation matrix */
   *P_ptr = P;

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildPDevice
 *
 * TODO: make use of hypre_MGRBuildPFromWpDevice
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
   hypre_GpuProfilingPushRange("MGRBuildP");

#if defined(HYPRE_USING_SYCL)
   nfpoints = HYPRE_ONEDPL_CALL(std::count,
                                CF_marker,
                                CF_marker + A_nr_of_rows,
                                -1);
#else
   nfpoints = HYPRE_THRUST_CALL(count,
                                CF_marker,
                                CF_marker + A_nr_of_rows,
                                -1);
#endif

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

#if defined(HYPRE_USING_SYCL)
         HYPRE_ONEDPL_CALL(std::transform,
                           diag,
                           diag + nfpoints,
                           diag1,
                           diag,
                           functor<HYPRE_Complex>(scal));

         HYPRE_ONEDPL_CALL(std::transform,
                           diag,
                           diag + nfpoints,
                           diag,
         [] (auto x) { return 1.0 / x; });
#else
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
#endif

         hypre_TFree(diag1, HYPRE_MEMORY_DEVICE);
      }
      else if (method == 2)
      {
         // extract diag inverse
         hypre_CSRMatrixExtractDiagonalDevice(hypre_ParCSRMatrixDiag(A_FF), diag, 2);
      }

#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( transform, diag, diag + nfpoints, diag, std::negate<HYPRE_Complex>() );
#else
      HYPRE_THRUST_CALL( transform, diag, diag + nfpoints, diag, thrust::negate<HYPRE_Complex>() );
#endif

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

   hypre_GpuProfilingPopRange();

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
                                          HYPRE_Int         *B_i,
                                          HYPRE_Int         *B_j,
                                          HYPRE_Complex     *B_a )
{
   HYPRE_Int   lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int   bs2  = blk_size * blk_size;
   HYPRE_Int   bidx;
   HYPRE_Int   lidx;
   HYPRE_Int   i, ii, j, pj, qj;
   HYPRE_Int   col;

   /* Grid-stride loop over block matrix rows */
   for (bidx = hypre_gpu_get_grid_warp_id<1, 1>(item);
        bidx < num_rows / blk_size;
        bidx += hypre_gpu_get_grid_num_warps<1, 1>(item))
   {
      ii = bidx * blk_size;

      /* Set output row pointer and column indices */
      for (i = lane; i < blk_size; i += HYPRE_WARP_SIZE)
      {
         B_i[ii + i + 1] = (ii + i + 1) * blk_size;
      }

      /* Set output column indices (row major) */
      for (j = lane; j < bs2; j += HYPRE_WARP_SIZE)
      {
         B_j[ii * blk_size + j] = ii + j % blk_size;
      }

      /* TODO: unroll this loop */
      for (lidx = 0; lidx < blk_size; lidx++)
      {
         i = ii + lidx;

         if (lane < 2)
         {
            pj = read_only_load(A_i + i + lane);
         }
         qj = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pj, 1);
         pj = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pj, 0);

         /* Loop over columns */
         for (j = pj + lane; j < qj; j += HYPRE_WARP_SIZE)
         {
            col = read_only_load(A_j + j);

            if ((col >= ii) &&
                (col <  ii + blk_size) &&
                (fabs(A_a[j]) > HYPRE_REAL_MIN))
            {
               /* batch offset + column offset + row offset */
               B_a[ii * blk_size + (col - ii) * blk_size + lidx] = A_a[j];
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
                                                HYPRE_Int         *B_i,
                                                HYPRE_Int         *B_j,
                                                HYPRE_Complex     *B_a )
{
   HYPRE_Int   lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int   bidx;
   HYPRE_Int   lidx;
   HYPRE_Int   i, ii, j, pj, qj, k;
   HYPRE_Int   col;

   /* Grid-stride loop over block matrix rows */
   for (bidx = hypre_gpu_get_grid_warp_id<1, 1>(item);
        bidx < num_rows / blk_size;
        bidx += hypre_gpu_get_grid_num_warps<1, 1>(item))
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
                     B_a[marker_indices[ii] * blk_size + (col - ii) * blk_size + lidx] = A_a[k];
                  }
               }
            }
         } /* row check */
      } /* Local block loop */
   } /* Grid-stride loop */
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_ComplexMatrixBatchedTranspose
 *
 * Transposes a group of dense matrices. Assigns one warp per block (batch).
 * Naive implementation.
 *
 * TODOs (VPM):
 *    1) Move to proper file.
 *    2) Use template argument for other data types
 *    3) Implement in-place transpose.
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_ComplexMatrixBatchedTranspose( hypre_DeviceItem  &item,
                                              HYPRE_Int          num_blocks,
                                              HYPRE_Int          block_size,
                                              HYPRE_Complex     *A_data,
                                              HYPRE_Complex     *B_data )
{
   HYPRE_Int   lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int   bs2  = block_size * block_size;
   HYPRE_Int   bidx, lidx;

   /* Grid-stride loop over block matrix rows */
   for (bidx = hypre_gpu_get_grid_warp_id<1, 1>(item);
        bidx < num_blocks;
        bidx += hypre_gpu_get_grid_num_warps<1, 1>(item))
   {
      for (lidx = lane; lidx < bs2; lidx += HYPRE_WARP_SIZE)
      {
         B_data[bidx * bs2 + lidx] =
            A_data[bidx * bs2 + (lidx / block_size + (lidx % block_size) * block_size)];
      }
   } /* Grid-stride loop */
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixExtractBlockDiagDevice
 *
 * TODOs (VPM):
 *   1) Allow other local solver choices. Design an interface for that.
 *   2) Move this to par_csr_matop_device.c
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixExtractBlockDiagDevice( hypre_ParCSRMatrix   *A,
                                          HYPRE_Int             blk_size,
                                          HYPRE_Int             num_points,
                                          HYPRE_Int             point_type,
                                          HYPRE_Int            *CF_marker,
                                          HYPRE_Int             diag_size,
                                          HYPRE_Int             diag_type,
                                          HYPRE_Int            *B_diag_i,
                                          HYPRE_Int            *B_diag_j,
                                          HYPRE_Complex        *B_diag_data )
{
   /* Matrix variables */
   HYPRE_BigInt          num_rows_A   = hypre_ParCSRMatrixGlobalNumRows(A);
   hypre_CSRMatrix      *A_diag       = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int             num_rows     = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int            *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int            *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex        *A_diag_data  = hypre_CSRMatrixData(A_diag);

   /* Local LS variables */
#if defined(HYPRE_USING_ONEMKLBLAS)
   std::int64_t         *pivots;
   std::int64_t          work_sizes[2];
   std::int64_t          work_size;
   HYPRE_Complex        *scratchpad;
#else
   HYPRE_Int            *pivots;
   HYPRE_Complex       **tmpdiag_aop;
   HYPRE_Int            *info;
#endif
   HYPRE_Int            *blk_row_indices;
   HYPRE_Complex        *tmpdiag;
   HYPRE_Complex       **diag_aop;

   /* Local variables */
   HYPRE_Int             bs2 = blk_size * blk_size;
   HYPRE_Int             num_blocks;
   HYPRE_Int             bdiag_size;

   /* Additional variables for debugging */
#if HYPRE_DEBUG
   HYPRE_Int            *h_info;
   HYPRE_Int             k, myid;

   hypre_MPI_Comm_rank(hypre_ParCSRMatrixComm(A), &myid);
#endif

   /*-----------------------------------------------------------------
    * Sanity checks
    *-----------------------------------------------------------------*/

   if (blk_size < 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid block size!");

      return hypre_error_flag;
   }

   if ((num_rows_A > 0) && (num_rows_A < blk_size))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Input matrix is smaller than block size!");

      return hypre_error_flag;
   }

   /* Return if the local matrix is empty */
   if (!num_rows)
   {
      return hypre_error_flag;
   }

   /*-----------------------------------------------------------------
    * Initial
    *-----------------------------------------------------------------*/

   hypre_GpuProfilingPushRange("ParCSRMatrixExtractBlockDiag");

   /* Count the number of points matching point_type in CF_marker */
   if (CF_marker)
   {
      /* Compute block row indices */
      blk_row_indices = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
      hypreDevice_IntFilln(blk_row_indices, (size_t) num_rows, 1);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL(oneapi::dpl::exclusive_scan_by_segment,
                        CF_marker,
                        CF_marker + num_rows,
                        blk_row_indices,
                        blk_row_indices);
#else
      HYPRE_THRUST_CALL(exclusive_scan_by_key,
                        CF_marker,
                        CF_marker + num_rows,
                        blk_row_indices,
                        blk_row_indices);
#endif
   }
   else
   {
      blk_row_indices = NULL;
   }

   /* Compute block info */
   num_blocks = hypre_ceildiv(num_points, blk_size);
   bdiag_size = num_blocks * bs2;

   if (num_points % blk_size)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "TODO! num_points % blk_size != 0");
      hypre_GpuProfilingPopRange();

      return hypre_error_flag;
   }

   /*-----------------------------------------------------------------
    * Extract diagonal sub-blocks (pattern and coefficients)
    *-----------------------------------------------------------------*/
   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_blocks, "warp", bDim);

      if (CF_marker)
      {
         HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRMatrixExtractBlockDiagMarked, gDim, bDim,
                           blk_size, num_rows, point_type, CF_marker, blk_row_indices,
                           A_diag_i, A_diag_j, A_diag_data,
                           B_diag_i, B_diag_j, B_diag_data );
      }
      else
      {
         HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRMatrixExtractBlockDiag, gDim, bDim,
                           blk_size, num_rows,
                           A_diag_i, A_diag_j, A_diag_data,
                           B_diag_i, B_diag_j, B_diag_data );
      }
   }

   /*-----------------------------------------------------------------
    * Invert diagonal sub-blocks
    *-----------------------------------------------------------------*/

   if (diag_type == 1)
   {
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "InvertDiagSubBlocks");

      /* Memory allocation */
      tmpdiag     = hypre_TAlloc(HYPRE_Complex, bdiag_size, HYPRE_MEMORY_DEVICE);
      diag_aop    = hypre_TAlloc(HYPRE_Complex *, num_blocks, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_ONEMKLBLAS)
      pivots      = hypre_CTAlloc(std::int64_t, num_rows * blk_size, HYPRE_MEMORY_DEVICE);
#else
      pivots      = hypre_CTAlloc(HYPRE_Int, num_rows * blk_size, HYPRE_MEMORY_DEVICE);
      tmpdiag_aop = hypre_TAlloc(HYPRE_Complex *, num_blocks, HYPRE_MEMORY_DEVICE);
      info        = hypre_CTAlloc(HYPRE_Int, num_blocks, HYPRE_MEMORY_DEVICE);
#if defined (HYPRE_DEBUG)
      h_info      = hypre_TAlloc(HYPRE_Int,  num_blocks, HYPRE_MEMORY_HOST);
#endif

      /* Memory copy */
      hypre_TMemcpy(tmpdiag, B_diag_data, HYPRE_Complex, bdiag_size,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

      /* Set work array of pointers */
      hypreDevice_ComplexArrayToArrayOfPtrs(num_blocks, bs2, tmpdiag, tmpdiag_aop);
#endif

      /* Set array of pointers */
      hypreDevice_ComplexArrayToArrayOfPtrs(num_blocks, bs2, B_diag_data, diag_aop);

      /* Compute LU factorization */
#if defined(HYPRE_USING_CUBLAS)
      HYPRE_CUBLAS_CALL(hypre_cublas_getrfBatched(hypre_HandleCublasHandle(hypre_handle()),
                                                  blk_size,
                                                  tmpdiag_aop,
                                                  blk_size,
                                                  pivots,
                                                  info,
                                                  num_blocks));
#elif defined(HYPRE_USING_ROCSOLVER)
      HYPRE_ROCSOLVER_CALL(rocsolver_dgetrf_batched(hypre_HandleVendorSolverHandle(hypre_handle()),
                                                    blk_size,
                                                    blk_size,
                                                    tmpdiag_aop,
                                                    blk_size,
                                                    pivots,
                                                    blk_size,
                                                    info,
                                                    num_blocks));

#elif defined(HYPRE_USING_ONEMKLBLAS)
      HYPRE_ONEMKL_CALL( work_sizes[0] =
                            oneapi::mkl::lapack::getrf_batch_scratchpad_size<HYPRE_Complex>( *hypre_HandleComputeStream(
                                                                                                hypre_handle()),
                                                                                             blk_size, // std::int64_t m,
                                                                                             blk_size, // std::int64_t n,
                                                                                             blk_size, // std::int64_t lda,
                                                                                             bs2, // std::int64_t stride_a,
                                                                                             blk_size, // std::int64_t stride_ipiv,
                                                                                             num_blocks ) ); // std::int64_t batch_size

      HYPRE_ONEMKL_CALL( work_sizes[1] =
                            oneapi::mkl::lapack::getri_batch_scratchpad_size<HYPRE_Complex>( *hypre_HandleComputeStream(
                                                                                                hypre_handle()),
                                                                                             (std::int64_t) blk_size, // std::int64_t n,
                                                                                             (std::int64_t) blk_size, // std::int64_t lda,
                                                                                             (std::int64_t) bs2, // std::int64_t stride_a,
                                                                                             (std::int64_t) blk_size, // std::int64_t stride_ipiv,
                                                                                             (std::int64_t) num_blocks // std::int64_t batch_size
                                                                                           ) );
      work_size  = hypre_max(work_sizes[0], work_sizes[1]);
      scratchpad = hypre_TAlloc(HYPRE_Complex, work_size, HYPRE_MEMORY_DEVICE);

      HYPRE_ONEMKL_CALL( oneapi::mkl::lapack::getrf_batch( *hypre_HandleComputeStream(hypre_handle()),
                                                           (std::int64_t) blk_size, // std::int64_t m,
                                                           (std::int64_t) blk_size, // std::int64_t n,
                                                           *diag_aop, // T *a,
                                                           (std::int64_t) blk_size, // std::int64_t lda,
                                                           (std::int64_t) bs2, // std::int64_t stride_a,
                                                           pivots, // std::int64_t *ipiv,
                                                           (std::int64_t) blk_size, // std::int64_t stride_ipiv,
                                                           (std::int64_t) num_blocks, // std::int64_t batch_size,
                                                           scratchpad, // T *scratchpad,
                                                           (std::int64_t) work_size // std::int64_t scratchpad_size,
                                                         ).wait() ); // const std::vector<cl::sycl::event> &events = {} ) );
#else
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Block inversion not available!");
      return hypre_error_flag;
#endif

#if defined (HYPRE_DEBUG) && !defined(HYPRE_USING_ONEMKLBLAS)
      hypre_TMemcpy(h_info, info, HYPRE_Int, num_blocks, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      for (k = 0; k < num_blocks; k++)
      {
         if (h_info[k] != 0)
         {
            if (h_info[k] < 0)
            {
               hypre_printf("[%d]: LU fact. failed at system %d, parameter %d ",
                            myid, k, h_info[k]);
            }
            else
            {
               hypre_printf("[%d]: Singular U(%d, %d) at system %d",
                            myid, h_info[k], h_info[k], k);
            }
         }
      }
#endif

      /* Compute sub-blocks inverses */
#if defined(HYPRE_USING_CUBLAS)
      HYPRE_CUBLAS_CALL(hypre_cublas_getriBatched(hypre_HandleCublasHandle(hypre_handle()),
                                                  blk_size,
                                                  (const HYPRE_Real **) tmpdiag_aop,
                                                  blk_size,
                                                  pivots,
                                                  diag_aop,
                                                  blk_size,
                                                  info,
                                                  num_blocks));
#elif defined(HYPRE_USING_ROCSOLVER)
      HYPRE_ROCSOLVER_CALL(rocsolver_dgetri_batched(hypre_HandleVendorSolverHandle(hypre_handle()),
                                                    blk_size,
                                                    tmpdiag_aop,
                                                    blk_size,
                                                    pivots,
                                                    blk_size,
                                                    info,
                                                    num_blocks));
#elif defined(HYPRE_USING_ONEMKLBLAS)
      HYPRE_ONEMKL_CALL( oneapi::mkl::lapack::getri_batch( *hypre_HandleComputeStream(hypre_handle()),
                                                           (std::int64_t) blk_size, // std::int64_t n,
                                                           *diag_aop, // T *a,
                                                           (std::int64_t) blk_size, // std::int64_t lda,
                                                           (std::int64_t) bs2, // std::int64_t stride_a,
                                                           pivots, // std::int64_t *ipiv,
                                                           (std::int64_t) blk_size, // std::int64_t stride_ipiv,
                                                           (std::int64_t) num_blocks, // std::int64_t batch_size,
                                                           scratchpad, // T *scratchpad,
                                                           work_size // std::int64_t scratchpad_size
                                                         ).wait() );
#else
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Block inversion not available!");
      return hypre_error_flag;
#endif

      /* Free memory */
      hypre_TFree(diag_aop, HYPRE_MEMORY_DEVICE);
      hypre_TFree(pivots, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_ONEMKLBLAS)
      hypre_TFree(scratchpad, HYPRE_MEMORY_DEVICE);
#else
      hypre_TFree(tmpdiag_aop, HYPRE_MEMORY_DEVICE);
      hypre_TFree(info, HYPRE_MEMORY_DEVICE);
#if defined (HYPRE_DEBUG)
      hypre_TFree(h_info, HYPRE_MEMORY_HOST);
#endif
#endif

      /* Transpose data to row-major format */
      {
         dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
         dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_blocks, "warp", bDim);

         /* Memory copy */
         hypre_TMemcpy(tmpdiag, B_diag_data, HYPRE_Complex, bdiag_size,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

         HYPRE_GPU_LAUNCH( hypreGPUKernel_ComplexMatrixBatchedTranspose, gDim, bDim,
                           num_blocks, blk_size, tmpdiag, B_diag_data );
      }

      /* Free memory */
      hypre_TFree(tmpdiag, HYPRE_MEMORY_DEVICE);

      HYPRE_ANNOTATE_REGION_END("%s", "InvertDiagSubBlocks");
   }

   /* Free memory */
   hypre_TFree(blk_row_indices, HYPRE_MEMORY_DEVICE);
   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixBlockDiagMatrixDevice
 *
 * TODO: Move this to par_csr_matop_device.c (VPM)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixBlockDiagMatrixDevice( hypre_ParCSRMatrix  *A,
                                         HYPRE_Int            blk_size,
                                         HYPRE_Int            point_type,
                                         HYPRE_Int           *CF_marker,
                                         HYPRE_Int            diag_type,
                                         hypre_ParCSRMatrix **B_ptr )
{
   /* Input matrix info */
   MPI_Comm              comm            = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt         *row_starts_A    = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_BigInt          num_rows_A      = hypre_ParCSRMatrixGlobalNumRows(A);
   hypre_CSRMatrix      *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int             A_diag_num_rows = hypre_CSRMatrixNumRows(A_diag);

   /* Global block matrix info */
   hypre_ParCSRMatrix   *par_B;
   HYPRE_BigInt          num_rows_B;
   HYPRE_BigInt          row_starts_B[2];

   /* Diagonal block matrix info */
   hypre_CSRMatrix      *B_diag;
   HYPRE_Int             B_diag_num_rows;
   HYPRE_Int             B_diag_size;
   HYPRE_Int            *B_diag_i;
   HYPRE_Int            *B_diag_j;
   HYPRE_Complex        *B_diag_data;

   /* Local variables */
   HYPRE_BigInt          num_rows_big;
   HYPRE_BigInt          scan_recv;
   HYPRE_Int             num_procs, my_id;
   HYPRE_Int             num_blocks;

   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   /*-----------------------------------------------------------------
    * Count the number of points matching point_type in CF_marker
    *-----------------------------------------------------------------*/

   if (!CF_marker)
   {
      B_diag_num_rows = A_diag_num_rows;
   }
   else
   {
#if defined(HYPRE_USING_SYCL)
      B_diag_num_rows = HYPRE_ONEDPL_CALL( std::count,
                                           CF_marker,
                                           CF_marker + A_diag_num_rows,
                                           point_type );
#else
      B_diag_num_rows = HYPRE_THRUST_CALL( count,
                                           CF_marker,
                                           CF_marker + A_diag_num_rows,
                                           point_type );
#endif
   }
   num_blocks  = hypre_ceildiv(B_diag_num_rows, blk_size);
   B_diag_size = blk_size * (blk_size * num_blocks);

   /*-----------------------------------------------------------------
    * Compute global number of rows and partitionings
    *-----------------------------------------------------------------*/

   if (CF_marker)
   {
      num_rows_big = (HYPRE_BigInt) B_diag_num_rows;
      hypre_MPI_Scan(&num_rows_big, &scan_recv, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

      /* first point in my range */
      row_starts_B[0] = scan_recv - num_rows_big;

      /* first point in next proc's range */
      row_starts_B[1] = scan_recv;
      if (my_id == (num_procs - 1))
      {
         num_rows_B = row_starts_B[1];
      }
      hypre_MPI_Bcast(&num_rows_B, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      row_starts_B[0] = row_starts_A[0];
      row_starts_B[1] = row_starts_A[1];
      num_rows_B = num_rows_A;
   }

   /* Create matrix B */
   par_B = hypre_ParCSRMatrixCreate(comm,
                                    num_rows_B,
                                    num_rows_B,
                                    row_starts_B,
                                    row_starts_B,
                                    0,
                                    B_diag_size,
                                    0);
   hypre_ParCSRMatrixInitialize_v2(par_B, HYPRE_MEMORY_DEVICE);
   B_diag      = hypre_ParCSRMatrixDiag(par_B);
   B_diag_i    = hypre_CSRMatrixI(B_diag);
   B_diag_j    = hypre_CSRMatrixJ(B_diag);
   B_diag_data = hypre_CSRMatrixData(B_diag);

   /*-----------------------------------------------------------------------
    * Extract coefficients
    *-----------------------------------------------------------------------*/

   hypre_ParCSRMatrixExtractBlockDiagDevice(A, blk_size, B_diag_num_rows,
                                            point_type, CF_marker,
                                            B_diag_size, diag_type,
                                            B_diag_i, B_diag_j, B_diag_data);

   /* Set output pointer */
   *B_ptr = par_B;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRComputeNonGalerkinCGDevice
 *
 * See hypre_MGRComputeNonGalerkinCoarseGrid for available methods.
 *
 * TODO (VPM): Can we have a single function that works for host and device?
 *             inv(A_FF)*A_FC might have been computed before. Reuse it!
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRComputeNonGalerkinCGDevice(hypre_ParCSRMatrix    *A_FF,
                                    hypre_ParCSRMatrix    *A_FC,
                                    hypre_ParCSRMatrix    *A_CF,
                                    hypre_ParCSRMatrix    *A_CC,
                                    hypre_ParCSRMatrix    *Wp,
                                    hypre_ParCSRMatrix    *Wr,
                                    HYPRE_Int              blk_size,
                                    HYPRE_Int              method,
                                    HYPRE_Complex          threshold,
                                    hypre_ParCSRMatrix   **A_H_ptr)
{
   /* Local variables */
   hypre_ParCSRMatrix   *A_H;
   hypre_ParCSRMatrix   *A_Hc;
   hypre_ParCSRMatrix   *A_CF_trunc;
   hypre_ParCSRMatrix   *Wp_tmp = Wp;
   HYPRE_Complex         alpha  = -1.0;

   hypre_GpuProfilingPushRange("MGRComputeNonGalerkinCG");

   /* Truncate A_CF according to the method */
   if (method == 2 || method == 3)
   {
      hypre_MGRTruncateAcfCPRDevice(A_CF, &A_CF_trunc);
   }
   else
   {
      A_CF_trunc = A_CF;
   }

   /* Compute Wp/Wr if not passed in */
   if (!Wp && (method == 1 || method == 2))
   {
      hypre_Vector         *D_FF_inv;
      HYPRE_Complex        *data;

      /* Create vector to store A_FF's diagonal inverse  */
      D_FF_inv = hypre_SeqVectorCreate(hypre_ParCSRMatrixNumRows(A_FF));
      hypre_SeqVectorInitialize_v2(D_FF_inv, HYPRE_MEMORY_DEVICE);
      data = hypre_VectorData(D_FF_inv);

      /* Compute the inverse of A_FF and compute its inverse */
      hypre_CSRMatrixExtractDiagonalDevice(hypre_ParCSRMatrixDiag(A_FF), data, 2);

      /* Compute D_FF_inv*A_FC */
      Wp_tmp = hypre_ParCSRMatrixClone(A_FC, 1);
      hypre_CSRMatrixDiagScaleDevice(hypre_ParCSRMatrixDiag(Wp_tmp), D_FF_inv, NULL);
      hypre_CSRMatrixDiagScaleDevice(hypre_ParCSRMatrixOffd(Wp_tmp), D_FF_inv, NULL);

      /* Free memory */
      hypre_SeqVectorDestroy(D_FF_inv);
   }
   else if (!Wp && (method == 3))
   {
      hypre_ParCSRMatrix  *B_FF_inv;

      /* Compute the block diagonal inverse of A_FF */
      hypre_ParCSRMatrixBlockDiagMatrixDevice(A_FF, blk_size, -1, NULL, 1, &B_FF_inv);

      /* Compute Wp = A_FF_inv * A_FC */
      Wp_tmp = hypre_ParCSRMatMat(B_FF_inv, A_FC);

      /* Free memory */
      hypre_ParCSRMatrixDestroy(B_FF_inv);
   }
   else
   {
      if (method != 5)
      {
         /* Use approximate inverse for ideal interploation */
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: feature not implemented yet!");
         hypre_GpuProfilingPopRange();

         return hypre_error_flag;
      }
   }

   /* Compute A_Hc (the correction for A_H) */
   if (method != 5)
   {
      A_Hc = hypre_ParCSRMatMat(A_CF_trunc, Wp_tmp);
   }
   else if (Wr && (method == 5))
   {
      A_Hc = hypre_ParCSRMatMat(Wr, A_FC);
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Wr matrix was not provided!");
      hypre_GpuProfilingPopRange();

      return hypre_error_flag;
   }

   /* Drop small entries from A_Hc */
   hypre_ParCSRMatrixDropSmallEntriesDevice(A_Hc, threshold, -1);

   /* Coarse grid (Schur complement) computation */
   hypre_ParCSRMatrixAdd(1.0, A_CC, alpha, A_Hc, &A_H);

   /* Free memory */
   hypre_ParCSRMatrixDestroy(A_Hc);
   if (Wp_tmp != Wp)
   {
      hypre_ParCSRMatrixDestroy(Wp_tmp);
   }
   if (method == 2 || method == 3)
   {
      hypre_ParCSRMatrixDestroy(A_CF_trunc);
   }

   /* Set output pointer to coarse grid matrix */
   *A_H_ptr = A_H;

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

#endif

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

/* return B = [Adiag, Aoffd] */
#if 1
__global__ void
hypreGPUKernel_ConcatDiagAndOffd( hypre_DeviceItem &item,
                                  HYPRE_Int  nrows,    HYPRE_Int  diag_ncol,
                                  HYPRE_Int *d_diag_i, HYPRE_Int *d_diag_j, HYPRE_Complex *d_diag_a,
                                  HYPRE_Int *d_offd_i, HYPRE_Int *d_offd_j, HYPRE_Complex *d_offd_a,
                                  HYPRE_Int *cols_offd_map,
                                  HYPRE_Int *d_ib,     HYPRE_Int *d_jb,     HYPRE_Complex *d_ab)
{
   const HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   /* lane id inside the warp */
   const HYPRE_Int lane_id = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int i, j = 0, k = 0, p, istart, iend, bstart;

   /* diag part */
   if (lane_id < 2)
   {
      j = read_only_load(d_diag_i + row + lane_id);
   }
   if (lane_id == 0)
   {
      k = read_only_load(d_ib + row);
   }
   istart = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 0);
   iend   = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 1);
   bstart = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, k, 0);

   p = bstart - istart;
   for (i = istart + lane_id; i < iend; i += HYPRE_WARP_SIZE)
   {
      d_jb[p + i] = read_only_load(d_diag_j + i);
      d_ab[p + i] = read_only_load(d_diag_a + i);
   }

   /* offd part */
   if (lane_id < 2)
   {
      j = read_only_load(d_offd_i + row + lane_id);
   }
   bstart += iend - istart;
   istart = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 0);
   iend   = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 1);

   p = bstart - istart;
   for (i = istart + lane_id; i < iend; i += HYPRE_WARP_SIZE)
   {
      const HYPRE_Int t = read_only_load(d_offd_j + i);
      d_jb[p + i] = (cols_offd_map ? read_only_load(&cols_offd_map[t]) : t) + diag_ncol;
      d_ab[p + i] = read_only_load(d_offd_a + i);
   }
}

hypre_CSRMatrix*
hypre_ConcatDiagAndOffdDevice(hypre_ParCSRMatrix *A)
{
   hypre_GpuProfilingPushRange("ConcatDiagAndOffdDevice");

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);

   hypre_CSRMatrix *B = hypre_CSRMatrixCreate( hypre_CSRMatrixNumRows(A_diag),
                                               hypre_CSRMatrixNumCols(A_diag) + hypre_CSRMatrixNumCols(A_offd),
                                               hypre_CSRMatrixNumNonzeros(A_diag) + hypre_CSRMatrixNumNonzeros(A_offd) );

   hypre_CSRMatrixInitialize_v2(B, 0, HYPRE_MEMORY_DEVICE);

   hypreDevice_GetRowNnz(hypre_CSRMatrixNumRows(B), NULL, hypre_CSRMatrixI(A_diag),
                         hypre_CSRMatrixI(A_offd), hypre_CSRMatrixI(B));

   hypreDevice_IntegerExclusiveScan(hypre_CSRMatrixNumRows(B) + 1, hypre_CSRMatrixI(B));

   const dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   const dim3 gDim = hypre_GetDefaultDeviceGridDimension(hypre_CSRMatrixNumRows(A_diag), "warp", bDim);

   HYPRE_Int  nrows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int  diag_ncol = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int *d_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *d_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex *d_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int *d_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int *d_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Complex *d_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int *cols_offd_map = NULL;
   HYPRE_Int *d_ib = hypre_CSRMatrixI(B);
   HYPRE_Int *d_jb = hypre_CSRMatrixJ(B);
   HYPRE_Complex *d_ab = hypre_CSRMatrixData(B);
   HYPRE_GPU_LAUNCH( hypreGPUKernel_ConcatDiagAndOffd,
                     gDim, bDim,
                     nrows,
                     diag_ncol,
                     d_diag_i,
                     d_diag_j,
                     d_diag_a,
                     d_offd_i,
                     d_offd_j,
                     d_offd_a,
                     cols_offd_map,
                     d_ib,
                     d_jb,
                     d_ab );

   hypre_GpuProfilingPopRange();

   return B;
}
#else
hypre_CSRMatrix*
hypre_ConcatDiagAndOffdDevice(hypre_ParCSRMatrix *A)
{
   hypre_CSRMatrix *A_diag     = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int       *A_diag_i   = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j   = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex   *A_diag_a   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int        A_diag_nnz = hypre_CSRMatrixNumNonzeros(A_diag);
   hypre_CSRMatrix *A_offd     = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int       *A_offd_i   = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j   = hypre_CSRMatrixJ(A_offd);
   HYPRE_Complex   *A_offd_a   = hypre_CSRMatrixData(A_offd);
   HYPRE_Int        A_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);

   hypre_CSRMatrix *B;
   HYPRE_Int        B_nrows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int        B_ncols = hypre_CSRMatrixNumCols(A_diag) + hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int        B_nnz   = A_diag_nnz + A_offd_nnz;
   HYPRE_Int       *B_ii = hypre_TAlloc(HYPRE_Int,     B_nnz, HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *B_j  = hypre_TAlloc(HYPRE_Int,     B_nnz, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex   *B_a  = hypre_TAlloc(HYPRE_Complex, B_nnz, HYPRE_MEMORY_DEVICE);

   // Adiag
   HYPRE_Int *A_diag_ii = hypreDevice_CsrRowPtrsToIndices(B_nrows, A_diag_nnz, A_diag_i);
   HYPRE_THRUST_CALL( copy_n,
                      thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                      A_diag_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_ii, B_j, B_a)) );
   hypre_TFree(A_diag_ii, HYPRE_MEMORY_DEVICE);

   // Aoffd
   HYPRE_Int *A_offd_ii = hypreDevice_CsrRowPtrsToIndices(B_nrows, A_offd_nnz, A_offd_i);
   HYPRE_THRUST_CALL( copy_n,
                      thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, A_offd_a)),
                      A_offd_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_ii, B_a)) + A_diag_nnz );
   hypre_TFree(A_offd_ii, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( transform,
                      A_offd_j,
                      A_offd_j + A_offd_nnz,
                      thrust::make_constant_iterator(hypre_CSRMatrixNumCols(A_diag)),
                      B_j + A_diag_nnz,
                      thrust::plus<HYPRE_Int>() );

   // B
   HYPRE_THRUST_CALL( stable_sort_by_key,
                      B_ii,
                      B_ii + B_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_j, B_a)) );

   HYPRE_Int *B_i = hypreDevice_CsrRowIndicesToPtrs(B_nrows, B_nnz, B_ii);
   hypre_TFree(B_ii, HYPRE_MEMORY_DEVICE);

   B = hypre_CSRMatrixCreate(B_nrows, B_ncols, B_nnz);
   hypre_CSRMatrixI(B) = B_i;
   hypre_CSRMatrixJ(B) = B_j;
   hypre_CSRMatrixData(B) = B_a;
   hypre_CSRMatrixMemoryLocation(B) = HYPRE_MEMORY_DEVICE;

   return B;
}
#endif

/* return B = [Adiag, Aoffd; E] */
#if 1
HYPRE_Int
hypre_ConcatDiagOffdAndExtDevice(hypre_ParCSRMatrix *A,
                                 hypre_CSRMatrix    *E,
                                 hypre_CSRMatrix   **B_ptr,
                                 HYPRE_Int          *num_cols_offd_ptr,
                                 HYPRE_BigInt      **cols_map_offd_ptr)
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix *E_diag, *E_offd, *B;
   HYPRE_Int       *cols_offd_map, num_cols_offd;
   HYPRE_BigInt    *cols_map_offd;

   hypre_CSRMatrixSplitDevice(E, hypre_ParCSRMatrixFirstColDiag(A), hypre_ParCSRMatrixLastColDiag(A),
                              hypre_CSRMatrixNumCols(A_offd), hypre_ParCSRMatrixDeviceColMapOffd(A),
                              &cols_offd_map, &num_cols_offd, &cols_map_offd, &E_diag, &E_offd);

   B = hypre_CSRMatrixCreate(hypre_ParCSRMatrixNumRows(A) + hypre_CSRMatrixNumRows(E),
                             hypre_ParCSRMatrixNumCols(A) + num_cols_offd,
                             hypre_CSRMatrixNumNonzeros(A_diag) + hypre_CSRMatrixNumNonzeros(A_offd) +
                             hypre_CSRMatrixNumNonzeros(E));

   hypre_CSRMatrixInitialize_v2(B, 0, HYPRE_MEMORY_DEVICE);

   hypreDevice_GetRowNnz(hypre_ParCSRMatrixNumRows(A), NULL, hypre_CSRMatrixI(A_diag),
                         hypre_CSRMatrixI(A_offd), hypre_CSRMatrixI(B));
   hypreDevice_IntegerExclusiveScan(hypre_ParCSRMatrixNumRows(A) + 1, hypre_CSRMatrixI(B));

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(hypre_ParCSRMatrixNumRows(A), "warp", bDim);

   HYPRE_Int  nrows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int  diag_ncol = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int *d_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *d_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex *d_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int *d_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int *d_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Complex *d_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int *d_ib = hypre_CSRMatrixI(B);
   HYPRE_Int *d_jb = hypre_CSRMatrixJ(B);
   HYPRE_Complex *d_ab = hypre_CSRMatrixData(B);
   HYPRE_GPU_LAUNCH( hypreGPUKernel_ConcatDiagAndOffd,
                     gDim, bDim,
                     nrows,
                     diag_ncol,
                     d_diag_i,
                     d_diag_j,
                     d_diag_a,
                     d_offd_i,
                     d_offd_j,
                     d_offd_a,
                     cols_offd_map,
                     d_ib,
                     d_jb,
                     d_ab );

   hypre_TFree(cols_offd_map, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + 1, hypre_CSRMatrixI(E) + 1,
                 HYPRE_Int, hypre_CSRMatrixNumRows(E),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
#ifdef HYPRE_USING_SYCL
   HYPRE_ONEDPL_CALL( std::transform,
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + 1,
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + hypre_CSRMatrixNumRows(E) + 1,
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + 1,
                      [const_val = hypre_CSRMatrixNumNonzeros(A_diag) + hypre_CSRMatrixNumNonzeros(A_offd)] (
   const auto & x) {return x + const_val;} );
#else
   HYPRE_THRUST_CALL( transform,
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + 1,
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + hypre_CSRMatrixNumRows(E) + 1,
                      thrust::make_constant_iterator(hypre_CSRMatrixNumNonzeros(A_diag) + hypre_CSRMatrixNumNonzeros(
                                                        A_offd)),
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + 1,
                      thrust::plus<HYPRE_Int>() );
#endif

   gDim = hypre_GetDefaultDeviceGridDimension(hypre_CSRMatrixNumRows(E), "warp", bDim);

   hypre_assert(hypre_CSRMatrixNumCols(E_diag) == hypre_CSRMatrixNumCols(A_diag));

   nrows = hypre_CSRMatrixNumRows(E_diag);
   diag_ncol = hypre_CSRMatrixNumCols(E_diag);
   d_diag_i = hypre_CSRMatrixI(E_diag);
   d_diag_j = hypre_CSRMatrixJ(E_diag);
   d_diag_a = hypre_CSRMatrixData(E_diag);
   d_offd_i = hypre_CSRMatrixI(E_offd);
   d_offd_j = hypre_CSRMatrixJ(E_offd);
   d_offd_a = hypre_CSRMatrixData(E_offd);
   cols_offd_map = NULL;
   d_ib = hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A);
   d_jb = hypre_CSRMatrixJ(B);
   d_ab = hypre_CSRMatrixData(B);
   HYPRE_GPU_LAUNCH( hypreGPUKernel_ConcatDiagAndOffd,
                     gDim, bDim,
                     nrows,
                     diag_ncol,
                     d_diag_i,
                     d_diag_j,
                     d_diag_a,
                     d_offd_i,
                     d_offd_j,
                     d_offd_a,
                     cols_offd_map,
                     d_ib,
                     d_jb,
                     d_ab );

   hypre_CSRMatrixDestroy(E_diag);
   hypre_CSRMatrixDestroy(E_offd);

   *B_ptr = B;
   *num_cols_offd_ptr = num_cols_offd;
   *cols_map_offd_ptr = cols_map_offd;

   return hypre_error_flag;
}
#else
HYPRE_Int
hypre_ConcatDiagOffdAndExtDevice(hypre_ParCSRMatrix *A,
                                 hypre_CSRMatrix    *E,
                                 hypre_CSRMatrix   **B_ptr,
                                 HYPRE_Int          *num_cols_offd_ptr,
                                 HYPRE_BigInt      **cols_map_offd_ptr)
{
   hypre_CSRMatrix *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int        A_nrows         = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int        A_ncols         = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int       *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex   *A_diag_a        = hypre_CSRMatrixData(A_diag);
   HYPRE_Int        A_diag_nnz      = hypre_CSRMatrixNumNonzeros(A_diag);
   hypre_CSRMatrix *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int       *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   HYPRE_Complex   *A_offd_a        = hypre_CSRMatrixData(A_offd);
   HYPRE_Int        A_offd_nnz      = hypre_CSRMatrixNumNonzeros(A_offd);
   HYPRE_BigInt     first_col_A     = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_BigInt     last_col_A      = hypre_ParCSRMatrixLastColDiag(A);
   HYPRE_Int        num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt    *col_map_offd_A  = hypre_ParCSRMatrixDeviceColMapOffd(A);

   HYPRE_Int       *E_i     = hypre_CSRMatrixI(E);
   HYPRE_BigInt    *E_bigj  = hypre_CSRMatrixBigJ(E);
   HYPRE_Complex   *E_a     = hypre_CSRMatrixData(E);
   HYPRE_Int        E_nrows = hypre_CSRMatrixNumRows(E);
   HYPRE_Int        E_nnz   = hypre_CSRMatrixNumNonzeros(E);
   HYPRE_Int        E_diag_nnz, E_offd_nnz;

   hypre_CSRMatrix *B;
   HYPRE_Int        B_nnz   = A_diag_nnz + A_offd_nnz + E_nnz;
   HYPRE_Int       *B_ii    = hypre_TAlloc(HYPRE_Int,     B_nnz, HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *B_j     = hypre_TAlloc(HYPRE_Int,     B_nnz, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex   *B_a     = hypre_TAlloc(HYPRE_Complex, B_nnz, HYPRE_MEMORY_DEVICE);

   // E
   hypre_CSRMatrixSplitDevice_core(0, E_nrows, E_nnz, NULL, E_bigj, NULL, NULL, first_col_A,
                                   last_col_A, num_cols_offd_A,
                                   NULL, NULL, NULL, NULL, &E_diag_nnz, NULL, NULL, NULL, NULL, &E_offd_nnz,
                                   NULL, NULL, NULL, NULL);

   HYPRE_Int    *cols_offd_map, num_cols_offd;
   HYPRE_BigInt *cols_map_offd;
   HYPRE_Int *E_ii = hypreDevice_CsrRowPtrsToIndices(E_nrows, E_nnz, E_i);

   hypre_CSRMatrixSplitDevice_core(1,
                                   E_nrows, E_nnz, E_ii, E_bigj, E_a, NULL,
                                   first_col_A, last_col_A, num_cols_offd_A, col_map_offd_A,
                                   &cols_offd_map, &num_cols_offd, &cols_map_offd,
                                   &E_diag_nnz,
                                   B_ii + A_diag_nnz + A_offd_nnz,
                                   B_j  + A_diag_nnz + A_offd_nnz,
                                   B_a  + A_diag_nnz + A_offd_nnz,
                                   NULL,
                                   &E_offd_nnz,
                                   B_ii + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                                   B_j  + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                                   B_a  + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                                   NULL);
   hypre_TFree(E_ii, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( transform,
                      B_ii + A_diag_nnz + A_offd_nnz,
                      B_ii + B_nnz,
                      thrust::make_constant_iterator(A_nrows),
                      B_ii + A_diag_nnz + A_offd_nnz,
                      thrust::plus<HYPRE_Int>() );

   // Adiag
   HYPRE_Int *A_diag_ii = hypreDevice_CsrRowPtrsToIndices(A_nrows, A_diag_nnz, A_diag_i);
   HYPRE_THRUST_CALL( copy_n,
                      thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                      A_diag_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_ii, B_j, B_a)) );
   hypre_TFree(A_diag_ii, HYPRE_MEMORY_DEVICE);

   // Aoffd
   HYPRE_Int *A_offd_ii = hypreDevice_CsrRowPtrsToIndices(A_nrows, A_offd_nnz, A_offd_i);
   HYPRE_THRUST_CALL( copy_n,
                      thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, A_offd_a)),
                      A_offd_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_ii, B_a)) + A_diag_nnz );
   hypre_TFree(A_offd_ii, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( gather,
                      A_offd_j,
                      A_offd_j + A_offd_nnz,
                      cols_offd_map,
                      B_j + A_diag_nnz);

   hypre_TFree(cols_offd_map, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( transform,
                      B_j + A_diag_nnz,
                      B_j + A_diag_nnz + A_offd_nnz,
                      thrust::make_constant_iterator(A_ncols),
                      B_j + A_diag_nnz,
                      thrust::plus<HYPRE_Int>() );

   HYPRE_THRUST_CALL( transform,
                      B_j + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                      B_j + B_nnz,
                      thrust::make_constant_iterator(A_ncols),
                      B_j + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                      thrust::plus<HYPRE_Int>() );

   // B
   HYPRE_THRUST_CALL( stable_sort_by_key,
                      B_ii,
                      B_ii + B_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_j, B_a)) );

   HYPRE_Int *B_i = hypreDevice_CsrRowIndicesToPtrs(A_nrows + E_nrows, B_nnz, B_ii);
   hypre_TFree(B_ii, HYPRE_MEMORY_DEVICE);

   B = hypre_CSRMatrixCreate(A_nrows + E_nrows, A_ncols + num_cols_offd, B_nnz);
   hypre_CSRMatrixI(B) = B_i;
   hypre_CSRMatrixJ(B) = B_j;
   hypre_CSRMatrixData(B) = B_a;
   hypre_CSRMatrixMemoryLocation(B) = HYPRE_MEMORY_DEVICE;

   *B_ptr = B;
   *num_cols_offd_ptr = num_cols_offd;
   *cols_map_offd_ptr = cols_map_offd;

   return hypre_error_flag;
}
#endif

/* The input B_ext is a BigJ matrix, so is the output */
/* RL: TODO FIX the num of columns of the output (from B_ext 'big' num cols) */
HYPRE_Int
hypre_ExchangeExternalRowsDeviceInit( hypre_CSRMatrix      *B_ext,
                                      hypre_ParCSRCommPkg  *comm_pkg_A,
                                      HYPRE_Int             want_data,
                                      void                **request_ptr)
{
   MPI_Comm   comm             = hypre_ParCSRCommPkgComm(comm_pkg_A);
   HYPRE_Int  num_recvs        = hypre_ParCSRCommPkgNumRecvs(comm_pkg_A);
   HYPRE_Int *recv_procs       = hypre_ParCSRCommPkgRecvProcs(comm_pkg_A);
   HYPRE_Int *recv_vec_starts  = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
   HYPRE_Int  num_sends        = hypre_ParCSRCommPkgNumSends(comm_pkg_A);
   HYPRE_Int *send_procs       = hypre_ParCSRCommPkgSendProcs(comm_pkg_A);
   HYPRE_Int *send_map_starts  = hypre_ParCSRCommPkgSendMapStarts(comm_pkg_A);

   HYPRE_Int  num_elmts_send   = send_map_starts[num_sends];
   HYPRE_Int  num_elmts_recv   = recv_vec_starts[num_recvs];

   HYPRE_Int     *B_ext_i_d      = hypre_CSRMatrixI(B_ext);
   HYPRE_BigInt  *B_ext_j_d      = hypre_CSRMatrixBigJ(B_ext);
   HYPRE_Complex *B_ext_a_d      = hypre_CSRMatrixData(B_ext);
   HYPRE_Int      B_ext_ncols    = hypre_CSRMatrixNumCols(B_ext);
   HYPRE_Int      B_ext_nrows    = hypre_CSRMatrixNumRows(B_ext);
   HYPRE_Int      B_ext_nnz      = hypre_CSRMatrixNumNonzeros(B_ext);
   HYPRE_Int     *B_ext_rownnz_d = hypre_TAlloc(HYPRE_Int, B_ext_nrows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *B_ext_rownnz_h = hypre_TAlloc(HYPRE_Int, B_ext_nrows,     HYPRE_MEMORY_HOST);
   HYPRE_Int     *B_ext_i_h      = hypre_TAlloc(HYPRE_Int, B_ext_nrows + 1, HYPRE_MEMORY_HOST);

   hypre_assert(num_elmts_recv == B_ext_nrows);

   /* output matrix */
   hypre_CSRMatrix *B_int_d;
   HYPRE_Int        B_int_nrows = num_elmts_send;
   HYPRE_Int        B_int_ncols = B_ext_ncols;
   HYPRE_Int       *B_int_i_h   = hypre_TAlloc(HYPRE_Int, B_int_nrows + 1, HYPRE_MEMORY_HOST);
   HYPRE_Int       *B_int_i_d   = hypre_TAlloc(HYPRE_Int, B_int_nrows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_BigInt    *B_int_j_d   = NULL;
   HYPRE_Complex   *B_int_a_d   = NULL;
   HYPRE_Int        B_int_nnz;

   hypre_ParCSRCommHandle *comm_handle, *comm_handle_j, *comm_handle_a;
   hypre_ParCSRCommPkg    *comm_pkg_j = NULL;

   HYPRE_Int *jdata_recv_vec_starts;
   HYPRE_Int *jdata_send_map_starts;

   HYPRE_Int i;
   HYPRE_Int num_procs, my_id;
   void    **vrequest;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   jdata_send_map_starts = hypre_TAlloc(HYPRE_Int, num_sends + 1, HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------------
    * B_ext_rownnz contains the number of elements of row j
    * (to be determined through send_map_elmnts on the receiving end)
    *--------------------------------------------------------------------------*/
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL(std::adjacent_difference, B_ext_i_d, B_ext_i_d + B_ext_nrows + 1, B_ext_rownnz_d);
#else
   HYPRE_THRUST_CALL(adjacent_difference, B_ext_i_d, B_ext_i_d + B_ext_nrows + 1, B_ext_rownnz_d);
#endif
   hypre_TMemcpy(B_ext_rownnz_h, B_ext_rownnz_d + 1, HYPRE_Int, B_ext_nrows,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   /*--------------------------------------------------------------------------
    * initialize communication: send/recv the row nnz
    * (note the use of comm_pkg_A, mode 12, as in transpose matvec
    *--------------------------------------------------------------------------*/
   comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg_A, B_ext_rownnz_h, B_int_i_h + 1);

   jdata_recv_vec_starts = hypre_TAlloc(HYPRE_Int, num_recvs + 1, HYPRE_MEMORY_HOST);
   jdata_recv_vec_starts[0] = 0;

   B_ext_i_h[0] = 0;
   hypre_TMemcpy(B_ext_i_h + 1, B_ext_rownnz_h, HYPRE_Int, B_ext_nrows, HYPRE_MEMORY_HOST,
                 HYPRE_MEMORY_HOST);
   for (i = 1; i <= B_ext_nrows; i++)
   {
      B_ext_i_h[i] += B_ext_i_h[i - 1];
   }

   hypre_assert(B_ext_i_h[B_ext_nrows] == B_ext_nnz);

   for (i = 1; i <= num_recvs; i++)
   {
      jdata_recv_vec_starts[i] = B_ext_i_h[recv_vec_starts[i]];
   }

   /* Create the communication package - note the order of send/recv is reversed */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_sends, send_procs, jdata_send_map_starts,
                                    num_recvs, recv_procs, jdata_recv_vec_starts,
                                    NULL,
                                    &comm_pkg_j);

   hypre_ParCSRCommHandleDestroy(comm_handle);

   /*--------------------------------------------------------------------------
    * compute B_int: row nnz to row ptrs
    *--------------------------------------------------------------------------*/
   B_int_i_h[0] = 0;
   for (i = 1; i <= B_int_nrows; i++)
   {
      B_int_i_h[i] += B_int_i_h[i - 1];
   }

   B_int_nnz = B_int_i_h[B_int_nrows];

   B_int_j_d = hypre_TAlloc(HYPRE_BigInt, B_int_nnz, HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      B_int_a_d = hypre_TAlloc(HYPRE_Complex, B_int_nnz, HYPRE_MEMORY_DEVICE);
   }

   for (i = 0; i <= num_sends; i++)
   {
      jdata_send_map_starts[i] = B_int_i_h[send_map_starts[i]];
   }

   /* RL: assume B_ext_a_d and B_ext_j_d are ready at input */
   /* send/recv CSR rows */
   if (want_data)
   {
      comm_handle_a = hypre_ParCSRCommHandleCreate_v2( 1, comm_pkg_j,
                                                       HYPRE_MEMORY_DEVICE, B_ext_a_d,
                                                       HYPRE_MEMORY_DEVICE, B_int_a_d );
   }
   else
   {
      comm_handle_a = NULL;
   }

   comm_handle_j = hypre_ParCSRCommHandleCreate_v2(21, comm_pkg_j,
                                                   HYPRE_MEMORY_DEVICE, B_ext_j_d,
                                                   HYPRE_MEMORY_DEVICE, B_int_j_d );

   hypre_TMemcpy(B_int_i_d, B_int_i_h, HYPRE_Int, B_int_nrows + 1, HYPRE_MEMORY_DEVICE,
                 HYPRE_MEMORY_HOST);

   /* create CSR: on device */
   B_int_d = hypre_CSRMatrixCreate(B_int_nrows, B_int_ncols, B_int_nnz);
   hypre_CSRMatrixI(B_int_d)    = B_int_i_d;
   hypre_CSRMatrixBigJ(B_int_d) = B_int_j_d;
   hypre_CSRMatrixData(B_int_d) = B_int_a_d;
   hypre_CSRMatrixMemoryLocation(B_int_d) = HYPRE_MEMORY_DEVICE;

   /* output */
   vrequest = hypre_TAlloc(void *, 3, HYPRE_MEMORY_HOST);
   vrequest[0] = (void *) comm_handle_j;
   vrequest[1] = (void *) comm_handle_a;
   vrequest[2] = (void *) B_int_d;

   *request_ptr = (void *) vrequest;

   /* free */
   hypre_TFree(B_ext_rownnz_d, HYPRE_MEMORY_DEVICE);
   hypre_TFree(B_ext_rownnz_h, HYPRE_MEMORY_HOST);
   hypre_TFree(B_ext_i_h,      HYPRE_MEMORY_HOST);
   hypre_TFree(B_int_i_h,      HYPRE_MEMORY_HOST);

   hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j), HYPRE_MEMORY_HOST);
   hypre_TFree(comm_pkg_j, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

hypre_CSRMatrix*
hypre_ExchangeExternalRowsDeviceWait(void *vrequest)
{
   void **request = (void **) vrequest;

   hypre_ParCSRCommHandle *comm_handle_j = (hypre_ParCSRCommHandle *) request[0];
   hypre_ParCSRCommHandle *comm_handle_a = (hypre_ParCSRCommHandle *) request[1];
   hypre_CSRMatrix        *B_int_d       = (hypre_CSRMatrix *)        request[2];

   /* communication done */
   hypre_ParCSRCommHandleDestroy(comm_handle_j);
   hypre_ParCSRCommHandleDestroy(comm_handle_a);

   hypre_TFree(request, HYPRE_MEMORY_HOST);

   return B_int_d;
}

HYPRE_Int
hypre_ParCSRMatrixExtractBExtDeviceInit( hypre_ParCSRMatrix  *B,
                                         hypre_ParCSRMatrix  *A,
                                         HYPRE_Int            want_data,
                                         void               **request_ptr)
{
   hypre_assert( hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(B)) ==
                 hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(B)) );

   /*
   hypre_assert( hypre_GetActualMemLocation(
            hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(B))) == HYPRE_MEMORY_DEVICE );
   */

   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }

   hypre_ParcsrGetExternalRowsDeviceInit(B,
                                         hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)),
                                         hypre_ParCSRMatrixColMapOffd(A),
                                         hypre_ParCSRMatrixCommPkg(A),
                                         want_data,
                                         request_ptr);
   return hypre_error_flag;
}

hypre_CSRMatrix*
hypre_ParCSRMatrixExtractBExtDeviceWait(void *request)
{
   return hypre_ParcsrGetExternalRowsDeviceWait(request);
}

hypre_CSRMatrix*
hypre_ParCSRMatrixExtractBExtDevice( hypre_ParCSRMatrix *B,
                                     hypre_ParCSRMatrix *A,
                                     HYPRE_Int want_data )
{
   void *request;

   hypre_ParCSRMatrixExtractBExtDeviceInit(B, A, want_data, &request);
   return hypre_ParCSRMatrixExtractBExtDeviceWait(request);
}

HYPRE_Int
hypre_ParcsrGetExternalRowsDeviceInit( hypre_ParCSRMatrix   *A,
                                       HYPRE_Int             indices_len,
                                       HYPRE_BigInt         *indices,
                                       hypre_ParCSRCommPkg  *comm_pkg,
                                       HYPRE_Int             want_data,
                                       void                **request_ptr)
{
   HYPRE_Int      i, j;
   HYPRE_Int      num_sends, num_rows_send, num_nnz_send, num_recvs, num_rows_recv, num_nnz_recv;
   HYPRE_Int     *d_send_i, *send_i, *d_send_map, *d_recv_i, *recv_i;
   HYPRE_BigInt  *d_send_j, *d_recv_j;
   HYPRE_Int     *send_jstarts, *recv_jstarts;
   HYPRE_Complex *d_send_a = NULL, *d_recv_a = NULL;
   hypre_ParCSRCommPkg     *comm_pkg_j = NULL;
   hypre_ParCSRCommHandle  *comm_handle, *comm_handle_j, *comm_handle_a;
   /* HYPRE_Int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A); */
   /* diag part of A */
   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex   *A_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);
   /* HYPRE_Int local_num_rows  = hypre_CSRMatrixNumRows(A_diag); */
   /* off-diag part of A */
   hypre_CSRMatrix *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex   *A_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   /* HYPRE_Int       *row_starts      = hypre_ParCSRMatrixRowStarts(A); */
   /* HYPRE_Int        first_row       = hypre_ParCSRMatrixFirstRowIndex(A); */
   HYPRE_BigInt     first_col        = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_BigInt    *col_map_offd_A   = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int        num_cols_A_offd  = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt    *d_col_map_offd_A = hypre_ParCSRMatrixDeviceColMapOffd(A);

   MPI_Comm         comm  = hypre_ParCSRMatrixComm(A);

   HYPRE_Int        num_procs;
   HYPRE_Int        my_id;
   void           **vrequest;

   hypre_CSRMatrix *A_ext;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* number of sends (#procs) */
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   /* number of rows to send */
   num_rows_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   /* number of recvs (#procs) */
   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   /* number of rows to recv */
   num_rows_recv = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);

   /* must be true if indices contains proper offd indices */
   hypre_assert(indices_len == num_rows_recv);

   /* send_i/recv_i:
    * the arrays to send and recv: we first send and recv the row lengths */
   d_send_i   = hypre_TAlloc(HYPRE_Int, num_rows_send + 1, HYPRE_MEMORY_DEVICE);
   d_send_map = hypre_TAlloc(HYPRE_Int, num_rows_send,     HYPRE_MEMORY_DEVICE);
   send_i     = hypre_TAlloc(HYPRE_Int, num_rows_send,     HYPRE_MEMORY_HOST);
   recv_i     = hypre_TAlloc(HYPRE_Int, num_rows_recv + 1, HYPRE_MEMORY_HOST);
   d_recv_i   = hypre_TAlloc(HYPRE_Int, num_rows_recv + 1, HYPRE_MEMORY_DEVICE);

   /* fill the send array with row lengths */
   hypre_TMemcpy(d_send_map, hypre_ParCSRCommPkgSendMapElmts(comm_pkg), HYPRE_Int,
                 num_rows_send, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   hypre_Memset(d_send_i, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);
   hypreDevice_GetRowNnz(num_rows_send, d_send_map, A_diag_i, A_offd_i, d_send_i + 1);

   /* send array send_i out: deviceTohost first and MPI (async)
    * note the shift in recv_i by one */
   hypre_TMemcpy(send_i, d_send_i + 1, HYPRE_Int, num_rows_send, HYPRE_MEMORY_HOST,
                 HYPRE_MEMORY_DEVICE);

   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_i, recv_i + 1);

   hypreDevice_IntegerInclusiveScan(num_rows_send + 1, d_send_i);

   /* total number of nnz to send */
   hypre_TMemcpy(&num_nnz_send, d_send_i + num_rows_send, HYPRE_Int, 1, HYPRE_MEMORY_HOST,
                 HYPRE_MEMORY_DEVICE);

   /* prepare data to send out. overlap with the above commmunication */
   d_send_j = hypre_TAlloc(HYPRE_BigInt, num_nnz_send, HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      d_send_a = hypre_TAlloc(HYPRE_Complex, num_nnz_send, HYPRE_MEMORY_DEVICE);
   }

   if (d_col_map_offd_A == NULL)
   {
      d_col_map_offd_A = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(d_col_map_offd_A, col_map_offd_A, HYPRE_BigInt, num_cols_A_offd,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixDeviceColMapOffd(A) = d_col_map_offd_A;
   }

   /* job == 2, d_send_i is input that contains row ptrs (length num_rows_send) */
   hypreDevice_CopyParCSRRows(num_rows_send, d_send_map, 2, num_procs > 1,
                              first_col, d_col_map_offd_A,
                              A_diag_i, A_diag_j, A_diag_a,
                              A_offd_i, A_offd_j, A_offd_a,
                              d_send_i, d_send_j, d_send_a);

   /* pointers to each proc in send_j */
   send_jstarts = hypre_TAlloc(HYPRE_Int, num_sends + 1, HYPRE_MEMORY_HOST);
   send_jstarts[0] = 0;
   for (i = 1; i <= num_sends; i++)
   {
      send_jstarts[i] = send_jstarts[i - 1];
      for ( j = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i - 1);
            j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            j++ )
      {
         send_jstarts[i] += send_i[j];
      }
   }
   hypre_assert(send_jstarts[num_sends] == num_nnz_send);

   /* finish the above communication: send_i/recv_i */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* adjust recv_i to ptrs */
   recv_i[0] = 0;
   for (i = 1; i <= num_rows_recv; i++)
   {
      recv_i[i] += recv_i[i - 1];
   }
   num_nnz_recv = recv_i[num_rows_recv];

   /* allocate device memory for j and a */
   d_recv_j = hypre_TAlloc(HYPRE_BigInt, num_nnz_recv, HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      d_recv_a = hypre_TAlloc(HYPRE_Complex, num_nnz_recv, HYPRE_MEMORY_DEVICE);
   }

   recv_jstarts = hypre_TAlloc(HYPRE_Int, num_recvs + 1, HYPRE_MEMORY_HOST);
   recv_jstarts[0] = 0;
   for (i = 1; i <= num_recvs; i++)
   {
      j = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
      recv_jstarts[i] = recv_i[j];
   }

   /* ready to send and recv: create a communication package for data */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs,
                                    hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                    recv_jstarts,
                                    num_sends,
                                    hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                    send_jstarts,
                                    NULL,
                                    &comm_pkg_j);

   /* RL: make sure d_send_j/d_send_a is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }

   /* init communication */
   /* ja */
   comm_handle_j = hypre_ParCSRCommHandleCreate_v2(21, comm_pkg_j,
                                                   HYPRE_MEMORY_DEVICE, d_send_j,
                                                   HYPRE_MEMORY_DEVICE, d_recv_j);
   if (want_data)
   {
      /* a */
      comm_handle_a = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg_j,
                                                      HYPRE_MEMORY_DEVICE, d_send_a,
                                                      HYPRE_MEMORY_DEVICE, d_recv_a);
   }
   else
   {
      comm_handle_a = NULL;
   }

   hypre_TMemcpy(d_recv_i, recv_i, HYPRE_Int, num_rows_recv + 1, HYPRE_MEMORY_DEVICE,
                 HYPRE_MEMORY_HOST);

   /* create A_ext: on device */
   A_ext = hypre_CSRMatrixCreate(num_rows_recv, hypre_ParCSRMatrixGlobalNumCols(A), num_nnz_recv);
   hypre_CSRMatrixI   (A_ext) = d_recv_i;
   hypre_CSRMatrixBigJ(A_ext) = d_recv_j;
   hypre_CSRMatrixData(A_ext) = d_recv_a;
   hypre_CSRMatrixMemoryLocation(A_ext) = HYPRE_MEMORY_DEVICE;

   /* output */
   vrequest = hypre_TAlloc(void *, 3, HYPRE_MEMORY_HOST);
   vrequest[0] = (void *) comm_handle_j;
   vrequest[1] = (void *) comm_handle_a;
   vrequest[2] = (void *) A_ext;

   *request_ptr = (void *) vrequest;

   /* free */
   hypre_TFree(send_i,     HYPRE_MEMORY_HOST);
   hypre_TFree(recv_i,     HYPRE_MEMORY_HOST);
   hypre_TFree(d_send_i,   HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_send_map, HYPRE_MEMORY_DEVICE);

   hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j), HYPRE_MEMORY_HOST);
   hypre_TFree(comm_pkg_j, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

hypre_CSRMatrix*
hypre_ParcsrGetExternalRowsDeviceWait(void *vrequest)
{
   void **request = (void **) vrequest;

   hypre_ParCSRCommHandle *comm_handle_j = (hypre_ParCSRCommHandle *) request[0];
   hypre_ParCSRCommHandle *comm_handle_a = (hypre_ParCSRCommHandle *) request[1];
   hypre_CSRMatrix        *A_ext         = (hypre_CSRMatrix *)        request[2];
   HYPRE_BigInt           *send_j        = comm_handle_j ? (HYPRE_BigInt *)
                                           hypre_ParCSRCommHandleSendData(comm_handle_j) : NULL;
   HYPRE_Complex          *send_a        = comm_handle_a ? (HYPRE_Complex *)
                                           hypre_ParCSRCommHandleSendData(comm_handle_a) : NULL;

   hypre_ParCSRCommHandleDestroy(comm_handle_j);
   hypre_ParCSRCommHandleDestroy(comm_handle_a);

   hypre_TFree(send_j, HYPRE_MEMORY_DEVICE);
   hypre_TFree(send_a, HYPRE_MEMORY_DEVICE);

   hypre_TFree(request, HYPRE_MEMORY_HOST);

   return A_ext;
}

HYPRE_Int
hypre_ParCSRCommPkgCreateMatrixE( hypre_ParCSRCommPkg  *comm_pkg,
                                  HYPRE_Int             num_cols )
{
   /* Input variables */
   HYPRE_Int        num_sends      = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int        num_elements   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   HYPRE_Int        num_components = hypre_ParCSRCommPkgNumComponents(comm_pkg);
   HYPRE_Int       *send_map       = hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);
   HYPRE_Int       *send_map_def;

   /* Local variables */
   hypre_CSRMatrix *matrix_E;
   HYPRE_Int       *e_i;
   HYPRE_Int       *e_ii;
   HYPRE_Int       *e_j;
   HYPRE_Int       *new_end;
   HYPRE_Int        nid;

   /* Update number of elements exchanged when communicating multivectors */
   num_elements /= num_components;

   /* Create matrix_E */
   matrix_E = hypre_CSRMatrixCreate(num_cols, num_elements, num_elements);
   hypre_CSRMatrixMemoryLocation(matrix_E) = HYPRE_MEMORY_DEVICE;

   /* Build default (original) send_map_elements array */
   if (num_components > 1)
   {
      send_map_def = hypre_TAlloc(HYPRE_Int, num_elements, HYPRE_MEMORY_DEVICE);
      hypreDevice_IntStridedCopy(num_elements, num_components, send_map, send_map_def);
   }
   else
   {
      send_map_def = send_map;
   }

   /* Allocate arrays */
   e_ii = hypre_TAlloc(HYPRE_Int, num_elements, HYPRE_MEMORY_DEVICE);
   e_j  = hypre_TAlloc(HYPRE_Int, num_elements, HYPRE_MEMORY_DEVICE);

   /* Build e_ii and e_j */
   hypre_TMemcpy(e_ii, send_map_def, HYPRE_Int, num_elements,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   hypreSycl_sequence(e_j, e_j + num_elements, 0);
   hypreSycl_stable_sort_by_key(e_ii, e_ii + num_elements, e_j);
#else
   HYPRE_THRUST_CALL(sequence, e_j, e_j + num_elements);
   HYPRE_THRUST_CALL(stable_sort_by_key, e_ii, e_ii + num_elements, e_j);
#endif

   /* Construct row pointers from row indices */
   e_i = hypreDevice_CsrRowIndicesToPtrs(num_cols, num_elements, e_ii);

   /* Find row indices with nonzero coefficients */
#if defined(HYPRE_USING_SYCL)
   new_end = HYPRE_ONEDPL_CALL(std::unique, e_ii, e_ii + num_elements);
#else
   new_end = HYPRE_THRUST_CALL(unique, e_ii, e_ii + num_elements);
#endif
   nid = new_end - e_ii;
   e_ii = hypre_TReAlloc_v2(e_ii, HYPRE_Int, num_elements,
                            HYPRE_Int, nid, HYPRE_MEMORY_DEVICE);

   /* Set matrix_E pointers */
   hypre_CSRMatrixI(matrix_E) = e_i;
   hypre_CSRMatrixJ(matrix_E) = e_j;
   hypre_CSRMatrixNumRownnz(matrix_E) = nid;
   hypre_CSRMatrixRownnz(matrix_E) = e_ii;

   /* Set matrix_E */
   hypre_ParCSRCommPkgMatrixE(comm_pkg) = matrix_E;

   /* Free memory */
   if (num_components > 1)
   {
      hypre_TFree(send_map_def, HYPRE_MEMORY_DEVICE);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixCompressOffdMapDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixCompressOffdMapDevice(hypre_ParCSRMatrix *A)
{
   hypre_GpuProfilingPushRange("CompressOffdMap");
   hypre_ParCSRMatrixCopyColMapOffdToDevice(A);

   hypre_CSRMatrix *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int        num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt    *col_map_offd_A  = hypre_ParCSRMatrixDeviceColMapOffd(A);
   HYPRE_BigInt    *col_map_offd_A_new;
   HYPRE_Int        num_cols_A_offd_new;

   hypre_CSRMatrixCompressColumnsDevice(A_offd, col_map_offd_A, NULL, &col_map_offd_A_new);

   num_cols_A_offd_new = hypre_CSRMatrixNumCols(A_offd);

   if (num_cols_A_offd_new < num_cols_A_offd)
   {
      hypre_TFree(col_map_offd_A, HYPRE_MEMORY_DEVICE);
      hypre_ParCSRMatrixDeviceColMapOffd(A) = col_map_offd_A_new;

      hypre_ParCSRMatrixColMapOffd(A) = hypre_TReAlloc(hypre_ParCSRMatrixColMapOffd(A),
                                                       HYPRE_BigInt, num_cols_A_offd_new,
                                                       HYPRE_MEMORY_HOST);

      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(A),
                    hypre_ParCSRMatrixDeviceColMapOffd(A),
                    HYPRE_BigInt, num_cols_A_offd_new,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/* Get element-wise tolerances based on row norms for ParCSRMatrix
 * NOTE: Keep the diagonal, i.e. elmt_tol = 0.0 for diagonals
 * Output vectors have size nnz:
 *    elmt_tols_diag[j] = tol * (norm of row i) for j in [ A_diag_i[i] , A_diag_i[i+1] )
 *    elmt_tols_offd[j] = tol * (norm of row i) for j in [ A_offd_i[i] , A_offd_i[i+1] )
 * type == -1, infinity norm,
 *         1, 1-norm
 *         2, 2-norm
 */
template<HYPRE_Int type>
__global__ void
hypre_ParCSRMatrixDropSmallEntriesDevice_getElmtTols( hypre_DeviceItem &item,
                                                      HYPRE_Int      nrows,
                                                      HYPRE_Real     tol,
                                                      HYPRE_Int     *A_diag_i,
                                                      HYPRE_Int     *A_diag_j,
                                                      HYPRE_Complex *A_diag_a,
                                                      HYPRE_Int     *A_offd_i,
                                                      HYPRE_Complex *A_offd_a,
                                                      HYPRE_Real     *elmt_tols_diag,
                                                      HYPRE_Real     *elmt_tols_offd)
{
   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p_diag = 0, p_offd = 0, q_diag, q_offd;

   /* sum row norm over diag part */
   if (lane < 2)
   {
      p_diag = read_only_load(A_diag_i + row_i + lane);
   }
   q_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag, 0);

   HYPRE_Real row_norm_i = 0.0;

   for (HYPRE_Int j = p_diag + lane; j < q_diag; j += HYPRE_WARP_SIZE)
   {
      HYPRE_Complex val = A_diag_a[j];

      if (type == -1)
      {
         row_norm_i = hypre_max(row_norm_i, hypre_cabs(val));
      }
      else if (type == 1)
      {
         row_norm_i += hypre_cabs(val);
      }
      else if (type == 2)
      {
         row_norm_i += val * val;
      }
   }

   /* sum row norm over offd part */
   if (lane < 2)
   {
      p_offd = read_only_load(A_offd_i + row_i + lane);
   }
   q_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd, 0);

   for (HYPRE_Int j = p_offd + lane; j < q_offd; j += HYPRE_WARP_SIZE)
   {
      HYPRE_Complex val = A_offd_a[j];

      if (type == -1)
      {
         row_norm_i = hypre_max(row_norm_i, hypre_cabs(val));
      }
      else if (type == 1)
      {
         row_norm_i += hypre_cabs(val);
      }
      else if (type == 2)
      {
         row_norm_i += val * val;
      }
   }

   /* allreduce to get the row norm on all threads */
   if (type == -1)
   {
      row_norm_i = warp_allreduce_max(item, row_norm_i);
   }
   else
   {
      row_norm_i = warp_allreduce_sum(item, row_norm_i);
   }
   if (type == 2)
   {
      row_norm_i = hypre_sqrt(row_norm_i);
   }

   /* set elmt_tols_diag */
   for (HYPRE_Int j = p_diag + lane; j < q_diag; j += HYPRE_WARP_SIZE)
   {
      HYPRE_Int col = A_diag_j[j];

      /* elmt_tol = 0.0 ensures diagonal will be kept */
      if (col == row_i)
      {
         elmt_tols_diag[j] = 0.0;
      }
      else
      {
         elmt_tols_diag[j] = tol * row_norm_i;
      }
   }

   /* set elmt_tols_offd */
   for (HYPRE_Int j = p_offd + lane; j < q_offd; j += HYPRE_WARP_SIZE)
   {
      elmt_tols_offd[j] = tol * row_norm_i;
   }

}

/* drop the entries that are not on the diagonal and smaller than:
 *    type 0: tol
 *    type 1: tol*(1-norm of row)
 *    type 2: tol*(2-norm of row)
 *    type -1: tol*(infinity norm of row) */
HYPRE_Int
hypre_ParCSRMatrixDropSmallEntriesDevice( hypre_ParCSRMatrix *A,
                                          HYPRE_Complex       tol,
                                          HYPRE_Int           type)
{
   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int        num_cols_A_offd  = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt    *h_col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_BigInt    *col_map_offd_A = hypre_ParCSRMatrixDeviceColMapOffd(A);

   HYPRE_Real      *elmt_tols_diag = NULL;
   HYPRE_Real      *elmt_tols_offd = NULL;

   /* Exit if tolerance is zero */
   if (tol < HYPRE_REAL_MIN)
   {
      return hypre_error_flag;
   }

   hypre_GpuProfilingPushRange("ParCSRMatrixDropSmallEntries");

   if (col_map_offd_A == NULL)
   {
      col_map_offd_A = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(col_map_offd_A, h_col_map_offd_A, HYPRE_BigInt, num_cols_A_offd,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixDeviceColMapOffd(A) = col_map_offd_A;
   }

   /* get elmement-wise tolerances if needed */
   if (type != 0)
   {
      elmt_tols_diag = hypre_TAlloc(HYPRE_Real, hypre_CSRMatrixNumNonzeros(A_diag), HYPRE_MEMORY_DEVICE);
      elmt_tols_offd = hypre_TAlloc(HYPRE_Real, hypre_CSRMatrixNumNonzeros(A_offd), HYPRE_MEMORY_DEVICE);
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(hypre_CSRMatrixNumRows(A_diag), "warp", bDim);

   HYPRE_Int A_diag_nrows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Complex *A_offd_data = hypre_CSRMatrixData(A_offd);
   if (type == -1)
   {
      HYPRE_GPU_LAUNCH( hypre_ParCSRMatrixDropSmallEntriesDevice_getElmtTols < -1 >, gDim, bDim,
                        A_diag_nrows, tol, A_diag_i,
                        A_diag_j, A_diag_data, A_offd_i,
                        A_offd_data, elmt_tols_diag, elmt_tols_offd);
   }
   if (type == 1)
   {
      HYPRE_GPU_LAUNCH( hypre_ParCSRMatrixDropSmallEntriesDevice_getElmtTols<1>, gDim, bDim,
                        A_diag_nrows, tol, A_diag_i,
                        A_diag_j, A_diag_data, A_offd_i,
                        A_offd_data, elmt_tols_diag, elmt_tols_offd);
   }
   if (type == 2)
   {
      HYPRE_GPU_LAUNCH( hypre_ParCSRMatrixDropSmallEntriesDevice_getElmtTols<2>, gDim, bDim,
                        A_diag_nrows, tol, A_diag_i,
                        A_diag_j, A_diag_data, A_offd_i,
                        A_offd_data, elmt_tols_diag, elmt_tols_offd);
   }

   /* drop entries from diag and offd CSR matrices */
   hypre_CSRMatrixDropSmallEntriesDevice(A_diag, tol, elmt_tols_diag);
   hypre_CSRMatrixDropSmallEntriesDevice(A_offd, tol, elmt_tols_offd);

   hypre_ParCSRMatrixSetNumNonzeros(A);
   hypre_ParCSRMatrixDNumNonzeros(A) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(A);

   /* squeeze out zero columns of A_offd */
   hypre_ParCSRMatrixCompressOffdMapDevice(A);

   if (type != 0)
   {
      hypre_TFree(elmt_tols_diag, HYPRE_MEMORY_DEVICE);
      hypre_TFree(elmt_tols_offd, HYPRE_MEMORY_DEVICE);
   }

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

hypre_CSRMatrix*
hypre_MergeDiagAndOffdDevice(hypre_ParCSRMatrix *A)
{
   MPI_Comm         comm     = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex   *A_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex   *A_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int        local_num_rows   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     glbal_num_cols   = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt     first_col        = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_Int        num_cols_A_offd  = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt    *col_map_offd_A   = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_BigInt    *d_col_map_offd_A = hypre_ParCSRMatrixDeviceColMapOffd(A);

   hypre_CSRMatrix *B;
   HYPRE_Int        B_nrows = local_num_rows;
   HYPRE_BigInt     B_ncols = glbal_num_cols;
   HYPRE_Int       *B_i = hypre_TAlloc(HYPRE_Int, B_nrows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_BigInt    *B_j;
   HYPRE_Complex   *B_a;
   HYPRE_Int        B_nnz;

   HYPRE_Int        num_procs;

   hypre_MPI_Comm_size(comm, &num_procs);

   hypre_Memset(B_i, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);

   hypreDevice_GetRowNnz(B_nrows, NULL, A_diag_i, A_offd_i, B_i + 1);

   hypreDevice_IntegerInclusiveScan(B_nrows + 1, B_i);

   /* total number of nnz */
   hypre_TMemcpy(&B_nnz, B_i + B_nrows, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   B_j = hypre_TAlloc(HYPRE_BigInt,  B_nnz, HYPRE_MEMORY_DEVICE);
   B_a = hypre_TAlloc(HYPRE_Complex, B_nnz, HYPRE_MEMORY_DEVICE);

   if (d_col_map_offd_A == NULL)
   {
      d_col_map_offd_A = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(d_col_map_offd_A, col_map_offd_A, HYPRE_BigInt, num_cols_A_offd,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixDeviceColMapOffd(A) = d_col_map_offd_A;
   }

   hypreDevice_CopyParCSRRows(B_nrows, NULL, 2, num_procs > 1, first_col, d_col_map_offd_A,
                              A_diag_i, A_diag_j, A_diag_a, A_offd_i, A_offd_j, A_offd_a,
                              B_i, B_j, B_a);

   /* output */
   B = hypre_CSRMatrixCreate(B_nrows, B_ncols, B_nnz);
   hypre_CSRMatrixI   (B) = B_i;
   hypre_CSRMatrixBigJ(B) = B_j;
   hypre_CSRMatrixData(B) = B_a;
   hypre_CSRMatrixMemoryLocation(B) = HYPRE_MEMORY_DEVICE;

   hypre_SyncComputeStream(hypre_handle());

   return B;
}

HYPRE_Int
hypre_ParCSRMatrixGetRowDevice( hypre_ParCSRMatrix  *mat,
                                HYPRE_BigInt         row,
                                HYPRE_Int           *size,
                                HYPRE_BigInt       **col_ind,
                                HYPRE_Complex      **values )
{
   HYPRE_Int nrows, local_row;
   HYPRE_BigInt row_start, row_end;
   hypre_CSRMatrix *Aa;
   hypre_CSRMatrix *Ba;

   if (!mat)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   Aa = (hypre_CSRMatrix *) hypre_ParCSRMatrixDiag(mat);
   Ba = (hypre_CSRMatrix *) hypre_ParCSRMatrixOffd(mat);

   if (hypre_ParCSRMatrixGetrowactive(mat))
   {
      return (-1);
   }

   hypre_ParCSRMatrixGetrowactive(mat) = 1;

   row_start = hypre_ParCSRMatrixFirstRowIndex(mat);
   row_end = hypre_ParCSRMatrixLastRowIndex(mat) + 1;
   nrows = row_end - row_start;

   if (row < row_start || row >= row_end)
   {
      return (-1);
   }

   local_row = row - row_start;

   /* if buffer is not allocated and some information is requested, allocate buffer with the max row_nnz */
   if ( !hypre_ParCSRMatrixRowvalues(mat) && (col_ind || values) )
   {
      HYPRE_Int max_row_nnz;
      HYPRE_Int *row_nnz = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_DEVICE);

      hypreDevice_GetRowNnz(nrows, NULL, hypre_CSRMatrixI(Aa), hypre_CSRMatrixI(Ba), row_nnz);

      hypre_TMemcpy(size, row_nnz + local_row, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      max_row_nnz = HYPRE_ONEDPL_CALL(std::reduce, row_nnz, row_nnz + nrows, 0,
                                      oneapi::dpl::maximum<HYPRE_Int>());
#else
      max_row_nnz = HYPRE_THRUST_CALL(reduce, row_nnz, row_nnz + nrows, 0, thrust::maximum<HYPRE_Int>());
#endif

      /*
            HYPRE_Int *max_row_nnz_d = HYPRE_THRUST_CALL(max_element, row_nnz, row_nnz + nrows);
            hypre_TMemcpy( &max_row_nnz, max_row_nnz_d,
                           HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE );
      */

      hypre_TFree(row_nnz, HYPRE_MEMORY_DEVICE);

      hypre_ParCSRMatrixRowvalues(mat)  =
         (HYPRE_Complex *) hypre_TAlloc(HYPRE_Complex, max_row_nnz, hypre_ParCSRMatrixMemoryLocation(mat));
      hypre_ParCSRMatrixRowindices(mat) =
         (HYPRE_BigInt *)  hypre_TAlloc(HYPRE_BigInt,  max_row_nnz, hypre_ParCSRMatrixMemoryLocation(mat));
   }
   else
   {
      HYPRE_Int *size_d = hypre_TAlloc(HYPRE_Int, 1, HYPRE_MEMORY_DEVICE);
      hypreDevice_GetRowNnz(1, NULL, hypre_CSRMatrixI(Aa) + local_row, hypre_CSRMatrixI(Ba) + local_row,
                            size_d);
      hypre_TMemcpy(size, size_d, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_TFree(size_d, HYPRE_MEMORY_DEVICE);
   }

   if (col_ind || values)
   {
      if (hypre_ParCSRMatrixDeviceColMapOffd(mat) == NULL)
      {
         hypre_ParCSRMatrixDeviceColMapOffd(mat) =
            hypre_TAlloc(HYPRE_BigInt, hypre_CSRMatrixNumCols(Ba), HYPRE_MEMORY_DEVICE);

         hypre_TMemcpy( hypre_ParCSRMatrixDeviceColMapOffd(mat),
                        hypre_ParCSRMatrixColMapOffd(mat),
                        HYPRE_BigInt,
                        hypre_CSRMatrixNumCols(Ba),
                        HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST );
      }

      hypreDevice_CopyParCSRRows( 1, NULL, -1, Ba != NULL,
                                  hypre_ParCSRMatrixFirstColDiag(mat),
                                  hypre_ParCSRMatrixDeviceColMapOffd(mat),
                                  hypre_CSRMatrixI(Aa) + local_row,
                                  hypre_CSRMatrixJ(Aa),
                                  hypre_CSRMatrixData(Aa),
                                  hypre_CSRMatrixI(Ba) + local_row,
                                  hypre_CSRMatrixJ(Ba),
                                  hypre_CSRMatrixData(Ba),
                                  NULL,
                                  hypre_ParCSRMatrixRowindices(mat),
                                  hypre_ParCSRMatrixRowvalues(mat) );
   }

   if (col_ind)
   {
      *col_ind = hypre_ParCSRMatrixRowindices(mat);
   }

   if (values)
   {
      *values = hypre_ParCSRMatrixRowvalues(mat);
   }

   hypre_SyncComputeStream(hypre_handle());

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixTransposeDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixTransposeDevice( hypre_ParCSRMatrix  *A,
                                   hypre_ParCSRMatrix **AT_ptr,
                                   HYPRE_Int            data )
{
   hypre_CSRMatrix    *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix    *A_offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix    *A_diagT;
   hypre_CSRMatrix    *AT_offd;
   HYPRE_Int           num_procs;
   HYPRE_Int           num_cols_offd_AT = 0;
   HYPRE_BigInt       *col_map_offd_AT = NULL;
   hypre_ParCSRMatrix *AT;

   hypre_MPI_Comm_size(hypre_ParCSRMatrixComm(A), &num_procs);

   if (num_procs > 1)
   {
      void *request;
      hypre_CSRMatrix *A_offdT, *Aext;
      HYPRE_Int *Aext_ii, *Aext_j, Aext_nnz;
      HYPRE_Complex *Aext_data;
      HYPRE_BigInt *tmp_bigj;

      hypre_CSRMatrixTranspose(A_offd, &A_offdT, data);
      hypre_CSRMatrixBigJ(A_offdT) = hypre_TAlloc(HYPRE_BigInt, hypre_CSRMatrixNumNonzeros(A_offdT),
                                                  HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::transform,
                         hypre_CSRMatrixJ(A_offdT),
                         hypre_CSRMatrixJ(A_offdT) + hypre_CSRMatrixNumNonzeros(A_offdT),
                         hypre_CSRMatrixBigJ(A_offdT),
      [y = hypre_ParCSRMatrixFirstRowIndex(A)] (const auto & x) {return x + y;} );
#else
      HYPRE_THRUST_CALL( transform,
                         hypre_CSRMatrixJ(A_offdT),
                         hypre_CSRMatrixJ(A_offdT) + hypre_CSRMatrixNumNonzeros(A_offdT),
                         thrust::make_constant_iterator(hypre_ParCSRMatrixFirstRowIndex(A)),
                         hypre_CSRMatrixBigJ(A_offdT),
                         thrust::plus<HYPRE_BigInt>() );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure A_offdT is ready before issuing GPU-GPU MPI */
      if (hypre_GetGpuAwareMPI())
      {
         hypre_ForceSyncComputeStream(hypre_handle());
      }
#endif

      if (!hypre_ParCSRMatrixCommPkg(A))
      {
         hypre_MatvecCommPkgCreate(A);
      }

      hypre_ExchangeExternalRowsDeviceInit(A_offdT, hypre_ParCSRMatrixCommPkg(A), data, &request);

      hypre_CSRMatrixTranspose(A_diag, &A_diagT, data);

      Aext = hypre_ExchangeExternalRowsDeviceWait(request);

      hypre_CSRMatrixDestroy(A_offdT);

      // Aext contains offd of AT
      Aext_nnz = hypre_CSRMatrixNumNonzeros(Aext);
      Aext_ii = hypreDevice_CsrRowPtrsToIndices(hypre_CSRMatrixNumRows(Aext), Aext_nnz,
                                                hypre_CSRMatrixI(Aext));

      hypre_ParCSRCommPkgCopySendMapElmtsToDevice(hypre_ParCSRMatrixCommPkg(A));

#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather( Aext_ii,
                        Aext_ii + Aext_nnz,
                        hypre_ParCSRCommPkgDeviceSendMapElmts(hypre_ParCSRMatrixCommPkg(A)),
                        Aext_ii );
#else
      HYPRE_THRUST_CALL( gather,
                         Aext_ii,
                         Aext_ii + Aext_nnz,
                         hypre_ParCSRCommPkgDeviceSendMapElmts(hypre_ParCSRMatrixCommPkg(A)),
                         Aext_ii );
#endif

      tmp_bigj = hypre_TAlloc(HYPRE_BigInt, Aext_nnz, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_bigj, hypre_CSRMatrixBigJ(Aext), HYPRE_BigInt, Aext_nnz, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::sort,
                         tmp_bigj,
                         tmp_bigj + Aext_nnz );

      HYPRE_BigInt *new_end = HYPRE_ONEDPL_CALL( std::unique,
                                                 tmp_bigj,
                                                 tmp_bigj + Aext_nnz );
#else
      HYPRE_THRUST_CALL( sort,
                         tmp_bigj,
                         tmp_bigj + Aext_nnz );

      HYPRE_BigInt *new_end = HYPRE_THRUST_CALL( unique,
                                                 tmp_bigj,
                                                 tmp_bigj + Aext_nnz );
#endif

      num_cols_offd_AT = new_end - tmp_bigj;
      col_map_offd_AT = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_AT, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(col_map_offd_AT, tmp_bigj, HYPRE_BigInt, num_cols_offd_AT, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_DEVICE);

      hypre_TFree(tmp_bigj, HYPRE_MEMORY_DEVICE);

      Aext_j = hypre_TAlloc(HYPRE_Int, Aext_nnz, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                         col_map_offd_AT,
                         col_map_offd_AT + num_cols_offd_AT,
                         hypre_CSRMatrixBigJ(Aext),
                         hypre_CSRMatrixBigJ(Aext) + Aext_nnz,
                         Aext_j );
#else
      HYPRE_THRUST_CALL( lower_bound,
                         col_map_offd_AT,
                         col_map_offd_AT + num_cols_offd_AT,
                         hypre_CSRMatrixBigJ(Aext),
                         hypre_CSRMatrixBigJ(Aext) + Aext_nnz,
                         Aext_j );
#endif

      Aext_data = hypre_CSRMatrixData(Aext);
      hypre_CSRMatrixData(Aext) = NULL;
      hypre_CSRMatrixDestroy(Aext);

      if (data)
      {
         hypreDevice_StableSortByTupleKey(Aext_nnz, Aext_ii, Aext_j, Aext_data, 0);
      }
      else
      {
#if defined(HYPRE_USING_SYCL)
         HYPRE_ONEDPL_CALL( std::stable_sort,
                            oneapi::dpl::make_zip_iterator(Aext_ii, Aext_j),
                            oneapi::dpl::make_zip_iterator(Aext_ii, Aext_j) + Aext_nnz,
         [] (const auto & x, const auto & y) {return std::get<0>(x) < std::get<0>(y);} );
#else
         HYPRE_THRUST_CALL( stable_sort,
                            thrust::make_zip_iterator(thrust::make_tuple(Aext_ii, Aext_j)),
                            thrust::make_zip_iterator(thrust::make_tuple(Aext_ii, Aext_j)) + Aext_nnz );
#endif
      }

      AT_offd = hypre_CSRMatrixCreate(hypre_ParCSRMatrixNumCols(A), num_cols_offd_AT, Aext_nnz);
      hypre_CSRMatrixJ(AT_offd) = Aext_j;
      hypre_CSRMatrixData(AT_offd) = Aext_data;
      hypre_CSRMatrixInitialize_v2(AT_offd, 0, HYPRE_MEMORY_DEVICE);
      hypreDevice_CsrRowIndicesToPtrs_v2(hypre_CSRMatrixNumRows(AT_offd), Aext_nnz, Aext_ii,
                                         hypre_CSRMatrixI(AT_offd));
      hypre_TFree(Aext_ii, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      hypre_CSRMatrixTransposeDevice(A_diag, &A_diagT, data);
      AT_offd = hypre_CSRMatrixCreate(hypre_ParCSRMatrixNumCols(A), 0, 0);
      hypre_CSRMatrixInitialize_v2(AT_offd, 0, HYPRE_MEMORY_DEVICE);
   }

   AT = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumCols(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixColStarts(A),
                                 hypre_ParCSRMatrixRowStarts(A),
                                 num_cols_offd_AT,
                                 hypre_CSRMatrixNumNonzeros(A_diagT),
                                 hypre_CSRMatrixNumNonzeros(AT_offd));

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(AT));
   hypre_ParCSRMatrixDiag(AT) = A_diagT;

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(AT));
   hypre_ParCSRMatrixOffd(AT) = AT_offd;

   if (num_cols_offd_AT)
   {
      hypre_ParCSRMatrixDeviceColMapOffd(AT) = col_map_offd_AT;

      hypre_ParCSRMatrixColMapOffd(AT) = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_AT, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(AT), col_map_offd_AT, HYPRE_BigInt, num_cols_offd_AT,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }

   *AT_ptr = AT;

   return hypre_error_flag;
}

HYPRE_Int
hypre_ParCSRMatrixAddDevice( HYPRE_Complex        alpha,
                             hypre_ParCSRMatrix  *A,
                             HYPRE_Complex        beta,
                             hypre_ParCSRMatrix  *B,
                             hypre_ParCSRMatrix **C_ptr )
{
   hypre_CSRMatrix *A_diag           = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd           = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix *B_diag           = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix *B_offd           = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int        num_cols_offd_A  = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int        num_cols_offd_B  = hypre_CSRMatrixNumCols(B_offd);
   HYPRE_Int        num_cols_offd_C  = 0;
   HYPRE_BigInt    *d_col_map_offd_C = NULL;
   HYPRE_Int        num_procs;

   hypre_MPI_Comm_size(hypre_ParCSRMatrixComm(A), &num_procs);
   hypre_GpuProfilingPushRange("hypre_ParCSRMatrixAdd");

   hypre_CSRMatrix *C_diag = hypre_CSRMatrixAddDevice(alpha, A_diag, beta, B_diag);
   hypre_CSRMatrix *C_offd;

   //if (num_cols_offd_A || num_cols_offd_B)
   if (num_procs > 1)
   {
      hypre_ParCSRMatrixCopyColMapOffdToDevice(A);
      hypre_ParCSRMatrixCopyColMapOffdToDevice(B);

      HYPRE_BigInt *tmp = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_A + num_cols_offd_B,
                                       HYPRE_MEMORY_DEVICE);

      hypre_TMemcpy(tmp,                   hypre_ParCSRMatrixDeviceColMapOffd(A), HYPRE_BigInt,
                    num_cols_offd_A, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp + num_cols_offd_A, hypre_ParCSRMatrixDeviceColMapOffd(B), HYPRE_BigInt,
                    num_cols_offd_B, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::sort, tmp, tmp + num_cols_offd_A + num_cols_offd_B );
      HYPRE_BigInt *new_end = HYPRE_ONEDPL_CALL( std::unique, tmp,
                                                 tmp + num_cols_offd_A + num_cols_offd_B );
#else
      HYPRE_THRUST_CALL( sort, tmp, tmp + num_cols_offd_A + num_cols_offd_B );
      HYPRE_BigInt *new_end = HYPRE_THRUST_CALL( unique, tmp,
                                                 tmp + num_cols_offd_A + num_cols_offd_B );
#endif
      num_cols_offd_C = new_end - tmp;
      d_col_map_offd_C = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(d_col_map_offd_C, tmp, HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_DEVICE);

      /* reuse memory of tmp */
      HYPRE_Int *offd_A2C = (HYPRE_Int *) tmp;
      HYPRE_Int *offd_B2C = offd_A2C + num_cols_offd_A;
#if defined(HYPRE_USING_SYCL)
      /* WM: todo - getting an error when num_cols_offd_A is zero */
      if (num_cols_offd_A > 0)
      {
         HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                            d_col_map_offd_C,
                            d_col_map_offd_C + num_cols_offd_C,
                            hypre_ParCSRMatrixDeviceColMapOffd(A),
                            hypre_ParCSRMatrixDeviceColMapOffd(A) + num_cols_offd_A,
                            offd_A2C );
      }
      /* WM: todo - getting an error when num_cols_offd_B is zero */
      if (num_cols_offd_B > 0)
      {
         HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                            d_col_map_offd_C,
                            d_col_map_offd_C + num_cols_offd_C,
                            hypre_ParCSRMatrixDeviceColMapOffd(B),
                            hypre_ParCSRMatrixDeviceColMapOffd(B) + num_cols_offd_B,
                            offd_B2C );
      }
#else
      HYPRE_THRUST_CALL( lower_bound,
                         d_col_map_offd_C,
                         d_col_map_offd_C + num_cols_offd_C,
                         hypre_ParCSRMatrixDeviceColMapOffd(A),
                         hypre_ParCSRMatrixDeviceColMapOffd(A) + num_cols_offd_A,
                         offd_A2C );
      HYPRE_THRUST_CALL( lower_bound,
                         d_col_map_offd_C,
                         d_col_map_offd_C + num_cols_offd_C,
                         hypre_ParCSRMatrixDeviceColMapOffd(B),
                         hypre_ParCSRMatrixDeviceColMapOffd(B) + num_cols_offd_B,
                         offd_B2C );
#endif

      HYPRE_Int *C_offd_i, *C_offd_j, nnzC_offd;
      HYPRE_Complex *C_offd_a;

      hypreDevice_CSRSpAdd( hypre_CSRMatrixNumRows(A_offd),
                            hypre_CSRMatrixNumRows(B_offd),
                            num_cols_offd_C,
                            hypre_CSRMatrixNumNonzeros(A_offd),
                            hypre_CSRMatrixNumNonzeros(B_offd),
                            hypre_CSRMatrixI(A_offd),
                            hypre_CSRMatrixJ(A_offd),
                            alpha,
                            hypre_CSRMatrixData(A_offd),
                            offd_A2C,
                            hypre_CSRMatrixI(B_offd),
                            hypre_CSRMatrixJ(B_offd),
                            beta,
                            hypre_CSRMatrixData(B_offd),
                            offd_B2C,
                            NULL,
                            &nnzC_offd,
                            &C_offd_i,
                            &C_offd_j,
                            &C_offd_a );

      hypre_TFree(tmp, HYPRE_MEMORY_DEVICE);

      C_offd = hypre_CSRMatrixCreate(hypre_CSRMatrixNumRows(A_offd), num_cols_offd_C, nnzC_offd);
      hypre_CSRMatrixI(C_offd) = C_offd_i;
      hypre_CSRMatrixJ(C_offd) = C_offd_j;
      hypre_CSRMatrixData(C_offd) = C_offd_a;
      hypre_CSRMatrixMemoryLocation(C_offd) = HYPRE_MEMORY_DEVICE;
   }
   else
   {
      C_offd = hypre_CSRMatrixCreate(hypre_CSRMatrixNumRows(A_offd), 0, 0);
      hypre_CSRMatrixInitialize_v2(C_offd, 0, HYPRE_MEMORY_DEVICE);
   }

   /* Create ParCSRMatrix C */
   hypre_ParCSRMatrix *C = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                                    hypre_ParCSRMatrixGlobalNumRows(A),
                                                    hypre_ParCSRMatrixGlobalNumCols(A),
                                                    hypre_ParCSRMatrixRowStarts(A),
                                                    hypre_ParCSRMatrixColStarts(A),
                                                    num_cols_offd_C,
                                                    hypre_CSRMatrixNumNonzeros(C_diag),
                                                    hypre_CSRMatrixNumNonzeros(C_offd));

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;
   hypre_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      hypre_ParCSRMatrixDeviceColMapOffd(C) = d_col_map_offd_C;

      hypre_ParCSRMatrixColMapOffd(C) = hypre_TAlloc(HYPRE_BigInt,
                                                     num_cols_offd_C,
                                                     HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(C), d_col_map_offd_C,
                    HYPRE_BigInt, num_cols_offd_C,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }

   hypre_ParCSRMatrixSetNumNonzeros(C);
   hypre_ParCSRMatrixDNumNonzeros(C) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(C);

   /* create CommPkg of C */
   hypre_MatvecCommPkgCreate(C);

   *C_ptr = C;

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_GPU)

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixDiagScaleDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixDiagScaleDevice( hypre_ParCSRMatrix *par_A,
                                   hypre_ParVector    *par_ld,
                                   hypre_ParVector    *par_rd )
{
   /* Input variables */
   hypre_ParCSRCommPkg    *comm_pkg  = hypre_ParCSRMatrixCommPkg(par_A);
   hypre_ParCSRCommHandle *comm_handle;
   HYPRE_Int               num_sends;
   HYPRE_Int              *d_send_map_elmts;
   HYPRE_Int               send_map_num_elmts;

   hypre_CSRMatrix        *A_diag        = hypre_ParCSRMatrixDiag(par_A);
   hypre_CSRMatrix        *A_offd        = hypre_ParCSRMatrixOffd(par_A);
   HYPRE_Int               num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   hypre_Vector           *ld            = (par_ld) ? hypre_ParVectorLocalVector(par_ld) : NULL;
   hypre_Vector           *rd            = hypre_ParVectorLocalVector(par_rd);
   HYPRE_Complex          *rd_data       = hypre_VectorData(rd);

   /* Local variables */
   hypre_Vector           *rdbuf;
   HYPRE_Complex          *recv_rdbuf_data;
   HYPRE_Complex          *send_rdbuf_data;
   HYPRE_Int               sync_stream;

   /*---------------------------------------------------------------------
    * Setup communication info
    *--------------------------------------------------------------------*/

   hypre_GetSyncCudaCompute(&sync_stream);
   hypre_SetSyncCudaCompute(0);

   /* Create buffer vectors */
   rdbuf = hypre_SeqVectorCreate(num_cols_offd);

   /* If there exists no CommPkg for A, create it. */
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(par_A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(par_A);
   }

   /* Communicate a single vector component */
   hypre_ParCSRCommPkgUpdateVecStarts(comm_pkg,
                                      hypre_VectorNumVectors(rd),
                                      hypre_VectorVectorStride(rd),
                                      hypre_VectorIndexStride(rd));

   /* send_map_elmts on device */
   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   /* Set variables */
   num_sends          = hypre_ParCSRCommPkgNumSends(comm_pkg);
   d_send_map_elmts   = hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);
   send_map_num_elmts = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /*---------------------------------------------------------------------
    * Allocate/reuse receive data buffer
    *--------------------------------------------------------------------*/

   if (!hypre_ParCSRCommPkgTmpData(comm_pkg))
   {
      hypre_ParCSRCommPkgTmpData(comm_pkg) = hypre_TAlloc(HYPRE_Complex,
                                                          num_cols_offd,
                                                          HYPRE_MEMORY_DEVICE);
   }
   hypre_VectorData(rdbuf) = recv_rdbuf_data = hypre_ParCSRCommPkgTmpData(comm_pkg);
   hypre_SeqVectorSetDataOwner(rdbuf, 0);
   hypre_SeqVectorInitialize_v2(rdbuf, HYPRE_MEMORY_DEVICE);

   /*---------------------------------------------------------------------
    * Allocate/reuse send data buffer
    *--------------------------------------------------------------------*/

   if (!hypre_ParCSRCommPkgBufData(comm_pkg))
   {
      hypre_ParCSRCommPkgBufData(comm_pkg) = hypre_TAlloc(HYPRE_Complex,
                                                          send_map_num_elmts,
                                                          HYPRE_MEMORY_DEVICE);
   }
   send_rdbuf_data = hypre_ParCSRCommPkgBufData(comm_pkg);

   /*---------------------------------------------------------------------
    * Pack send data
    *--------------------------------------------------------------------*/

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int  i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(send_rdbuf_data, rd_data, d_send_map_elmts)
   for (i = 0; i < send_map_num_elmts; i++)
   {
      send_rdbuf_data[i] = rd_data[d_send_map_elmts[i]];
   }
#else
#if defined(HYPRE_USING_SYCL)
   auto permuted_source = oneapi::dpl::make_permutation_iterator(rd_data,
                                                                 d_send_map_elmts);
   HYPRE_ONEDPL_CALL( std::copy,
                      permuted_source,
                      permuted_source + send_map_num_elmts,
                      send_rdbuf_data );
#else
   HYPRE_THRUST_CALL( gather,
                      d_send_map_elmts,
                      d_send_map_elmts + send_map_num_elmts,
                      rd_data,
                      send_rdbuf_data );
#endif
#endif


#if defined(HYPRE_USING_THRUST_NOSYNC)
   /* make sure send_rdbuf_data is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }
#endif

   /* A_diag = diag(ld) * A_diag * diag(rd) */
   hypre_CSRMatrixDiagScale(A_diag, ld, rd);

   /* Communication phase */
   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg,
                                                 HYPRE_MEMORY_DEVICE, send_rdbuf_data,
                                                 HYPRE_MEMORY_DEVICE, recv_rdbuf_data);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* A_offd = diag(ld) * A_offd * diag(rd) */
   hypre_CSRMatrixDiagScale(A_offd, ld, rdbuf);

#if defined(HYPRE_USING_GPU)
   /*---------------------------------------------------------------------
    * Synchronize calls
    *--------------------------------------------------------------------*/
   hypre_SetSyncCudaCompute(sync_stream);
   hypre_SyncComputeStream(hypre_handle());
#endif

   /* Free memory */
   hypre_SeqVectorDestroy(rdbuf);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScaleVectorDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRDiagScaleVectorDevice( hypre_ParCSRMatrix *par_A,
                                   hypre_ParVector    *par_y,
                                   hypre_ParVector    *par_x )
{
   /* Local Matrix and Vectors */
   hypre_CSRMatrix    *A_diag        = hypre_ParCSRMatrixDiag(par_A);
   hypre_Vector       *x             = hypre_ParVectorLocalVector(par_x);
   hypre_Vector       *y             = hypre_ParVectorLocalVector(par_y);

   /* Local vector x info */
   HYPRE_Complex      *x_data        = hypre_VectorData(x);
   HYPRE_Int           x_size        = hypre_VectorSize(x);
   HYPRE_Int           x_num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int           x_vecstride   = hypre_VectorVectorStride(x);

   /* Local vector y info */
   HYPRE_Complex      *y_data        = hypre_VectorData(y);
   HYPRE_Int           y_size        = hypre_VectorSize(y);
   HYPRE_Int           y_num_vectors = hypre_VectorNumVectors(y);
   HYPRE_Int           y_vecstride   = hypre_VectorVectorStride(y);

   /* Local matrix A info */
   HYPRE_Int           num_rows      = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int          *A_i           = hypre_CSRMatrixI(A_diag);
   HYPRE_Complex      *A_data        = hypre_CSRMatrixData(A_diag);

   /* Sanity checks */
   hypre_assert(x_vecstride == x_size);
   hypre_assert(y_vecstride == y_size);
   hypre_assert(x_num_vectors == y_num_vectors);

   hypre_GpuProfilingPushRange("ParCSRDiagScaleVector");

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(x_data,y_data,A_data,A_i)
   for (i = 0; i < num_rows; i++)
   {
      x_data[i] = y_data[i] / A_data[A_i[i]];
   }
#else
   hypreDevice_DiagScaleVector(x_num_vectors, num_rows, A_i, A_data, y_data, 0.0, x_data);
#endif // #if defined(HYPRE_USING_DEVICE_OPENMP)

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)

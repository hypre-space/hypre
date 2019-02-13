/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Matrix operation functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"
#include "csr_matrix.h"
#include "csr_sparse_device.h"

#if defined(HYPRE_USING_CUDA)

struct in_range
{
   HYPRE_Int low, up;

   in_range(HYPRE_Int low_, HYPRE_Int up_) { low = low_; up = up_; }

   __host__ __device__ bool operator()(const HYPRE_Int &x)
   {
      return (x >= low && x <= up);
   }
};

struct out_of_range
{
   HYPRE_Int low, up;

   out_of_range(HYPRE_Int low_, HYPRE_Int up_) { low = low_; up = up_; }

   __host__ __device__ bool operator()(const HYPRE_Int &x)
   {
      return (x < low || x > up);
   }
};

hypre_CSRMatrix*
hypre_CSRMatrixAddDevice ( hypre_CSRMatrix *A,
                           hypre_CSRMatrix *B )
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         nnz_A    = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Complex    *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
   HYPRE_Int         nnz_B    = hypre_CSRMatrixNumNonzeros(B);
   HYPRE_Complex    *C_data;
   HYPRE_Int        *C_i;
   HYPRE_Int        *C_j;
   HYPRE_Int         nnzC;
   hypre_CSRMatrix  *C;

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! incompatible matrix dimensions!\n");

      return NULL;
   }

   hypreDevice_CSRSpAdd(nrows_A, nrows_B, ncols_A, nnz_A, nnz_B, A_i, A_j, A_data, B_i, B_j, B_data, NULL,
                        &nnzC, &C_i, &C_j, &C_data);

   C = hypre_CSRMatrixCreate(nrows_A, ncols_B, nnzC);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixJ(C) = C_j;
   hypre_CSRMatrixData(C) = C_data;
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

   return C;
}

hypre_CSRMatrix*
hypre_CSRMatrixMultiplyDevice( hypre_CSRMatrix *A,
                               hypre_CSRMatrix *B)
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Complex    *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
   HYPRE_Complex    *C_data;
   HYPRE_Int        *C_i;
   HYPRE_Int        *C_j;
   HYPRE_Int         nnzC;
   hypre_CSRMatrix  *C;

   /* HYPRE_Int         allsquare = 0; */

   if (ncols_A != nrows_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! incompatible matrix dimensions!\n");

      return NULL;
   }

   /*
   if (nrows_A == ncols_B)
   {
      allsquare = 1;
   }
   */

   hypreDevice_CSRSpGemm(nrows_A, ncols_A, ncols_B, A_i, A_j, A_data, B_i, B_j, B_data,
                         &C_i, &C_j, &C_data, &nnzC);

   C = hypre_CSRMatrixCreate(nrows_A, ncols_B, nnzC);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixJ(C) = C_j;
   hypre_CSRMatrixData(C) = C_data;
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

   return C;
}

/* split CSR matrix B_ext (extended rows of parcsr B) into diag part and offd part
 * corresponding to B.
 * Input  col_map_offd_B:
 * Output col_map_offd_C: union of col_map_offd_B and offd-indices of Bext_offd
 *        map_B_to_C: mapping from col_map_offd_B to col_map_offd_C
 */

HYPRE_Int
hypre_CSRMatrixSplitDevice(hypre_CSRMatrix  *B_ext,
                           HYPRE_Int         first_col_diag_B,
                           HYPRE_Int         last_col_diag_B,
                           HYPRE_Int         num_cols_offd_B,
                           HYPRE_Int        *col_map_offd_B,
                           HYPRE_Int       **map_B_to_C_ptr,
                           HYPRE_Int        *num_cols_offd_C_ptr,
                           HYPRE_Int       **col_map_offd_C_ptr,
                           hypre_CSRMatrix **B_ext_diag_ptr,
                           hypre_CSRMatrix **B_ext_offd_ptr)
{
   HYPRE_Int        num_rows   = hypre_CSRMatrixNumRows(B_ext);
   HYPRE_Int        B_ext_nnz  = hypre_CSRMatrixNumNonzeros(B_ext);
   HYPRE_Int       *B_ext_i    = hypre_CSRMatrixI(B_ext);
   HYPRE_Int       *B_ext_j    = hypre_CSRMatrixJ(B_ext);
   HYPRE_Complex   *B_ext_a = hypre_CSRMatrixData(B_ext);
   hypre_CSRMatrix *B_ext_diag = NULL;
   hypre_CSRMatrix *B_ext_offd = NULL;
   HYPRE_Int        B_ext_diag_nnz = 0;
   HYPRE_Int        B_ext_offd_nnz = 0;
   HYPRE_Int       *B_ext_row_indices, *B_ext_diag_row_indices, *B_ext_offd_row_indices;
   HYPRE_Int       *tmpi;
   HYPRE_Complex   *tmpc;

   HYPRE_Int     *B_ext_diag_i    = NULL;
   HYPRE_Int     *B_ext_diag_j    = NULL;
   HYPRE_Complex *B_ext_diag_a = NULL;
   HYPRE_Int     *B_ext_offd_i    = NULL;
   HYPRE_Int     *B_ext_offd_j    = NULL;
   HYPRE_Complex *B_ext_offd_a = NULL;

   HYPRE_Int     *col_map_offd_C, *map_B_to_C;
   HYPRE_Int      num_cols_offd_C;

   in_range     pred1(first_col_diag_B, last_col_diag_B);
   out_of_range pred2(first_col_diag_B, last_col_diag_B);

   /* get diag and offd nnz */
   B_ext_diag_nnz = thrust::count_if(thrust::device, B_ext_j, B_ext_j + B_ext_nnz, pred1);
   B_ext_offd_nnz = B_ext_nnz - B_ext_diag_nnz;

   /* allocate memory */
   B_ext_diag_j = hypre_TAlloc(HYPRE_Int,     B_ext_diag_nnz, HYPRE_MEMORY_DEVICE);
   B_ext_diag_a = hypre_TAlloc(HYPRE_Complex, B_ext_diag_nnz, HYPRE_MEMORY_DEVICE);

   B_ext_offd_j = hypre_TAlloc(HYPRE_Int,     B_ext_offd_nnz, HYPRE_MEMORY_DEVICE);
   B_ext_offd_a = hypre_TAlloc(HYPRE_Complex, B_ext_offd_nnz, HYPRE_MEMORY_DEVICE);

   B_ext_diag_row_indices = hypre_TAlloc(HYPRE_Int, B_ext_diag_nnz, HYPRE_MEMORY_DEVICE);
   B_ext_offd_row_indices = hypre_TAlloc(HYPRE_Int, B_ext_offd_nnz, HYPRE_MEMORY_DEVICE);

   B_ext_row_indices = hypreDevice_CsrRowPtrsToIndices(num_rows, B_ext_nnz, B_ext_i);

   /* copy to diag */
   tmpi = thrust::copy_if(thrust::device, B_ext_j, B_ext_j + B_ext_nnz, B_ext_j, B_ext_diag_j, pred1);
   hypre_assert(tmpi - B_ext_diag_j == B_ext_diag_nnz);

   tmpc = thrust::copy_if(thrust::device, B_ext_a, B_ext_a + B_ext_nnz, B_ext_j, B_ext_diag_a, pred1);
   hypre_assert(tmpc - B_ext_diag_a == B_ext_diag_nnz);

   tmpi = thrust::copy_if(thrust::device, B_ext_row_indices, B_ext_row_indices + B_ext_nnz, B_ext_j,
                          B_ext_diag_row_indices, pred1);
   hypre_assert(tmpi - B_ext_diag_row_indices == B_ext_diag_nnz);

   /* copy to offd */
   tmpi = thrust::copy_if(thrust::device, B_ext_j, B_ext_j + B_ext_nnz, B_ext_j, B_ext_offd_j, pred2);
   hypre_assert(tmpi - B_ext_offd_j == B_ext_offd_nnz);

   tmpc = thrust::copy_if(thrust::device, B_ext_a, B_ext_a + B_ext_nnz, B_ext_j, B_ext_offd_a, pred2);
   hypre_assert(tmpc - B_ext_offd_a == B_ext_offd_nnz);

   tmpi = thrust::copy_if(thrust::device, B_ext_row_indices, B_ext_row_indices + B_ext_nnz, B_ext_j,
                          B_ext_offd_row_indices, pred2);
   hypre_assert(tmpi - B_ext_offd_row_indices == B_ext_offd_nnz);

   /* offd map of B_ext_offd Union col_map_offd_B */
   col_map_offd_C = hypre_TAlloc(HYPRE_Int, B_ext_offd_nnz + num_cols_offd_B, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(col_map_offd_C, B_ext_offd_j, HYPRE_Int, B_ext_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(col_map_offd_C + B_ext_offd_nnz, col_map_offd_B, HYPRE_Int, num_cols_offd_B, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   thrust::sort(thrust::device, col_map_offd_C, col_map_offd_C + B_ext_offd_nnz + num_cols_offd_B);
   tmpi = thrust::unique(thrust::device, col_map_offd_C, col_map_offd_C + B_ext_offd_nnz + num_cols_offd_B);
   num_cols_offd_C = tmpi - col_map_offd_C;
   col_map_offd_C = hypre_TReAlloc(col_map_offd_C, HYPRE_Int, num_cols_offd_C, HYPRE_MEMORY_DEVICE);

   /* create map from col_map_offd_B */
   map_B_to_C = hypre_TAlloc(HYPRE_Int, num_cols_offd_B, HYPRE_MEMORY_DEVICE);
   thrust::lower_bound(thrust::device, col_map_offd_C, col_map_offd_C + num_cols_offd_C, col_map_offd_B,
                       col_map_offd_B + num_cols_offd_B, map_B_to_C);

   /* adjust diag and offd col indices */
   thrust::transform(thrust::device, B_ext_diag_j, B_ext_diag_j + B_ext_diag_nnz,
                     thrust::make_constant_iterator(first_col_diag_B), B_ext_diag_j,
                     thrust::minus<HYPRE_Int>());
   thrust::lower_bound(thrust::device, col_map_offd_C, col_map_offd_C + num_cols_offd_C,
                       B_ext_offd_j, B_ext_offd_j + B_ext_offd_nnz, B_ext_offd_j);

   /* convert to row ptrs */
   B_ext_diag_i = hypreDevice_CsrRowIndicesToPtrs(num_rows, B_ext_diag_nnz, B_ext_diag_row_indices);
   B_ext_offd_i = hypreDevice_CsrRowIndicesToPtrs(num_rows, B_ext_offd_nnz, B_ext_offd_row_indices);

   /* create diag and offd CSR */
   B_ext_diag = hypre_CSRMatrixCreate(num_rows, last_col_diag_B - first_col_diag_B + 1, B_ext_diag_nnz);
   B_ext_offd = hypre_CSRMatrixCreate(num_rows, num_cols_offd_C, B_ext_offd_nnz);

   hypre_CSRMatrixI(B_ext_diag) = B_ext_diag_i;
   hypre_CSRMatrixJ(B_ext_diag) = B_ext_diag_j;
   hypre_CSRMatrixData(B_ext_diag) = B_ext_diag_a;
   hypre_CSRMatrixNumNonzeros(B_ext_diag) = B_ext_diag_nnz;
   hypre_CSRMatrixMemoryLocation(B_ext_diag) = HYPRE_MEMORY_DEVICE;

   hypre_CSRMatrixI(B_ext_offd) = B_ext_offd_i;
   hypre_CSRMatrixJ(B_ext_offd) = B_ext_offd_j;
   hypre_CSRMatrixData(B_ext_offd) = B_ext_offd_a;
   hypre_CSRMatrixNumNonzeros(B_ext_offd) = B_ext_offd_nnz;
   hypre_CSRMatrixMemoryLocation(B_ext_offd) = HYPRE_MEMORY_DEVICE;

   /* free */
   hypre_TFree(B_ext_diag_row_indices, HYPRE_MEMORY_DEVICE);
   hypre_TFree(B_ext_offd_row_indices, HYPRE_MEMORY_DEVICE);
   hypre_TFree(B_ext_row_indices,      HYPRE_MEMORY_DEVICE);

   *map_B_to_C_ptr      = map_B_to_C;
   *num_cols_offd_C_ptr = num_cols_offd_C;
   *col_map_offd_C_ptr  = col_map_offd_C;
   *B_ext_diag_ptr      = B_ext_diag;
   *B_ext_offd_ptr      = B_ext_offd;

   return hypre_error_flag;
}


HYPRE_Int
hypre_CSRMatrixTransposeDevice(hypre_CSRMatrix  *A,
                               hypre_CSRMatrix **AT_ptr,
                               HYPRE_Int         data)
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         nnz_A    = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Complex    *C_data;
   HYPRE_Int        *C_i;
   HYPRE_Int        *C_j;
   hypre_CSRMatrix  *C;

   hypreDevice_CSRSpTrans(nrows_A, ncols_A, nnz_A, A_i, A_j, A_data, &C_i, &C_j, &C_data, data);

   C = hypre_CSRMatrixCreate(ncols_A, nrows_A, nnz_A);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixJ(C) = C_j;
   hypre_CSRMatrixData(C) = C_data;
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

   *AT_ptr = C;

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixAddPartial:
 * adds matrix rows in the CSR matrix B to the CSR Matrix A, where row_nums[i]
 * defines to which row of A the i-th row of B is added, and returns a CSR Matrix C;
 * Repeated row indices are allowed in row_nums
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 *       in A and B. To remove those, use hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixAddPartialDevice( hypre_CSRMatrix *A,
                                 hypre_CSRMatrix *B,
                                 HYPRE_Int       *row_nums)
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         nnz_A    = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Complex    *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
   HYPRE_Int         nnz_B    = hypre_CSRMatrixNumNonzeros(B);
   HYPRE_Complex    *C_data;
   HYPRE_Int        *C_i;
   HYPRE_Int        *C_j;
   HYPRE_Int         nnzC;
   hypre_CSRMatrix  *C;

   if (ncols_A != ncols_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! incompatible matrix dimensions!\n");

      return NULL;
   }

   hypreDevice_CSRSpAdd(nrows_A, nrows_B, ncols_A, nnz_A, nnz_B, A_i, A_j, A_data, B_i, B_j, B_data, row_nums,
                        &nnzC, &C_i, &C_j, &C_data);

   C = hypre_CSRMatrixCreate(nrows_A, ncols_B, nnzC);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixJ(C) = C_j;
   hypre_CSRMatrixData(C) = C_data;
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;


   return C;
}

#else

hypre_CSRMatrix*
hypre_CSRMatrixAddDevice ( hypre_CSRMatrix *A,
                           hypre_CSRMatrix *B )
{
   return NULL;
}

hypre_CSRMatrix*
hypre_CSRMatrixMultiplyDevice( hypre_CSRMatrix *A,
                               hypre_CSRMatrix *B)
{
   return NULL;
}

HYPRE_Int
hypre_CSRMatrixSplitDevice(hypre_CSRMatrix  *B_ext,
                           HYPRE_Int         first_col_diag_B,
                           HYPRE_Int         last_col_diag_B,
                           HYPRE_Int         num_cols_offd_B,
                           HYPRE_Int        *col_map_offd_B,
                           HYPRE_Int       **map_B_to_C_ptr,
                           HYPRE_Int        *num_cols_offd_C_ptr,
                           HYPRE_Int       **col_map_offd_C_ptr,
                           hypre_CSRMatrix **B_ext_diag_ptr,
                           hypre_CSRMatrix **B_ext_offd_ptr)
{
   return -1;
}

HYPRE_Int
hypre_CSRMatrixTransposeDevice(hypre_CSRMatrix  *A,
                               hypre_CSRMatrix **AT_ptr,
                               HYPRE_Int         data)
{
   return -1;
}

hypre_CSRMatrix*
hypre_CSRMatrixAddPartialDevice( hypre_CSRMatrix *A,
                                 hypre_CSRMatrix *B,
                                 HYPRE_Int       *row_nums)
{
   return NULL;
}

#endif /* HYPRE_USING_CUDA */


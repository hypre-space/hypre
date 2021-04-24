/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "_hypre_utilities.hpp"
#include "csr_matrix_sycl_utils.hpp"

#if defined(HYPRE_USING_SYCL)

/*
 * @brief Uses oneMKL to calculate a sparse-matrix x sparse-matrix product in CSRS format.
 *
 * @param[in] m Number of rows of A,C
 * @param[in] k Number of columns of B,C
 * @param[in] n Number of columns of A, number of rows of B
 * @param[in] nnzA Number of nonzeros in A
 * @param[in] *d_ia Array containing the row pointers of A
 * @param[in] *d_ja Array containing the column indices of A
 * @param[in] *d_a Array containing values of A
 * @param[in] nnzB Number of nonzeros in B
 * @param[in] *d_ib Array containing the row pointers of B
 * @param[in] *d_jb Array containing the column indices of B
 * @param[in] *d_b Array containing values of B
 * @param[out] *nnzC_out Pointer to address with number of nonzeros in C
 * @param[out] *d_ic_out Array containing the row pointers of C
 * @param[out] *d_jc_out Array containing the column indices of C
 * @param[out] *d_c_out Array containing values of C
 */

HYPRE_Int
hypreDevice_CSRSpGemmOneMKL(HYPRE_Int       m,
			    HYPRE_Int       k,
			    HYPRE_Int       n,
			    HYPRE_Int       nnzA,
			    HYPRE_Int      *d_ia,
			    HYPRE_Int      *d_ja,
			    HYPRE_Complex  *d_a,
			    HYPRE_Int       nnzB,
			    HYPRE_Int      *d_ib,
			    HYPRE_Int      *d_jb,
			    HYPRE_Complex  *d_b,
			    HYPRE_Int      *nnzC_out,
			    HYPRE_Int     **d_ic_out,
			    HYPRE_Int     **d_jc_out,
			    HYPRE_Complex **d_c_out)
{
   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_SYCL)

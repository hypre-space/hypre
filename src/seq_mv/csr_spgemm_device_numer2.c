/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#include "seq_mv.h"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#include <csr_spgemm_device_numer.h>

template HYPRE_Int
hypre_spgemm_numerical_with_rownnz < 2, HYPRE_SPGEMM_NUMER_HASH_SIZE / 8,
                                     HYPRE_SPGEMM_BASE_GROUP_SIZE / 8, true >
( HYPRE_Int m, HYPRE_Int *row_ind, HYPRE_Int k, HYPRE_Int n, bool need_ghash,
  HYPRE_Int exact_rownnz,
  HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int *d_ib, HYPRE_Int *d_jb,
  HYPRE_Complex *d_b,
  HYPRE_Int *d_rc, HYPRE_Int *d_ic, HYPRE_Int *d_jc, HYPRE_Complex *d_c );

template HYPRE_Int hypre_spgemm_numerical_max_num_blocks < HYPRE_SPGEMM_NUMER_HASH_SIZE / 8,
                                                           HYPRE_SPGEMM_BASE_GROUP_SIZE / 8 >
( HYPRE_Int multiProcessorCount, HYPRE_Int *num_blocks_ptr, HYPRE_Int *block_size_ptr );

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */


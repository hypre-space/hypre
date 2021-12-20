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
hypre_spgemm_numerical_with_rownnz<HYPRE_SPGEMM_NUMER_HASH_SIZE * 2, HYPRE_WARP_SIZE * 2, true, false>
( HYPRE_Int m, HYPRE_Int *row_ind, HYPRE_Int k, HYPRE_Int n, HYPRE_Int EXACT_ROWNNZ, char hash_type,
  HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int *d_ib, HYPRE_Int *d_jb,
  HYPRE_Complex *d_b, HYPRE_Int *d_rc, HYPRE_Int *d_ic, HYPRE_Int *d_jc, HYPRE_Complex *d_c );

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */


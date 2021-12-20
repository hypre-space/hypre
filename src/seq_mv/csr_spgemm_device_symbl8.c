/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#include "seq_mv.h"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#include <csr_spgemm_device_symbl.h>

template HYPRE_Int
hypre_spgemm_symbolic_rownnz<8 * HYPRE_SPGEMM_SYMBL_HASH_SIZE, 8 * HYPRE_WARP_SIZE, true>
( HYPRE_Int m, HYPRE_Int *row_ind, HYPRE_Int k, HYPRE_Int n, bool need_ghash, HYPRE_Int *d_ia,
  HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int *d_rc, bool can_fail, char *d_rf );

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */


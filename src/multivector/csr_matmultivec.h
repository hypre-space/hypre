/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef CSR_MULTIMATVEC_H
#define CSR_MULTIMATVEC_H

#include "seq_mv.h"
#include "seq_multivector.h"

#ifdef __cplusplus
extern "C" {
#endif
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatMultivec
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_CSRMatrixMatMultivec(HYPRE_Complex alpha, hypre_CSRMatrix *A,
                           hypre_Multivector *x, HYPRE_Complex beta,
                           hypre_Multivector *y);
                            

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMultiMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixMatMultivecT(HYPRE_Complex alpha, hypre_CSRMatrix *A,
                            hypre_Multivector *x, HYPRE_Complex beta,
                            hypre_Multivector *y);
                             
#ifdef __cplusplus
}
#endif

#endif /* CSR_MATMULTIVEC_H */

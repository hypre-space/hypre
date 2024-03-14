/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"

/* Mixed precision function protos */
/* hypre_seq_mv_mp.h */

#ifdef HYPRE_MIXED_PRECISION
HYPRE_Int
hypre_SeqVectorCopy_mp( hypre_Vector_mp *x,
                     hypre_Vector_mp *y );

HYPRE_Int
hypre_SeqVectorAxpy_mp( hypre_double alpha,
                     hypre_Vector_mp *x,
                     hypre_Vector_mp *y     );

HYPRE_Int
hypre_CSRMatrixConvert_mp (hypre_CSRMatrix_mp *A,
                          HYPRE_Precision new_precision);

HYPRE_Int
hypre_SeqVectorConvert_mp (hypre_Vector_mp *v,
                          HYPRE_Precision new_precision);
                     
#endif

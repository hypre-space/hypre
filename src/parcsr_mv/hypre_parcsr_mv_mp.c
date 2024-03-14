/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * hypre seq_mv mixed-precision interface
 *
 *****************************************************************************/

#include "hypre_parcsr_mv_mp.h"

#ifdef HYPRE_MIXED_PRECISION

/******************************************************************************
 *
 * Member functions for hypre_ParVector class.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Mixed-precision hypre_ParVectorCopy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorCopy_mp( hypre_ParVector *x,
                     hypre_ParVector *y )
{
   hypre_Vector_mp *x_local = (hypre_Vector_mp *)hypre_ParVectorLocalVector(x);
   hypre_Vector_mp *y_local = (hypre_Vector_mp *)hypre_ParVectorLocalVector(y);
   return hypre_SeqVectorCopy_mp(x_local, y_local);
}

/*--------------------------------------------------------------------------
 * Mixed-Precision hypre_ParVectorAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorAxpy_mp( hypre_double    alpha,
                     hypre_ParVector *x,
                     hypre_ParVector *y )
{
   hypre_Vector_mp *x_local = (hypre_Vector_mp *)hypre_ParVectorLocalVector(x);
   hypre_Vector_mp *y_local = (hypre_Vector_mp *)hypre_ParVectorLocalVector(y);
           
   return hypre_SeqVectorAxpy_mp( alpha, x_local, y_local);
}

/*--------------------------------------------------------------------------
 * Mixed-Precision Vector conversion
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorConvert_mp( hypre_ParVector *v,
                           HYPRE_Precision new_precision)
{
   hypre_Vector_mp *v_local = (hypre_Vector_mp *) hypre_ParVectorLocalVector(v);
   hypre_SeqVectorConvert_mp (v_local, new_precision);

   return (hypre_error_flag);
}
/*--------------------------------------------------------------------------
 * Mixed-Precision hypre_ParVectorAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixConvert_mp( hypre_ParCSRMatrix *A,
                              HYPRE_Precision new_precision)
{
   hypre_CSRMatrix_mp *A_diag = (hypre_CSRMatrix_mp *) hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix_mp *A_offd = (hypre_CSRMatrix_mp *) hypre_ParCSRMatrixOffd(A);

   hypre_CSRMatrixConvert_mp (A_diag, new_precision);
   hypre_CSRMatrixConvert_mp (A_offd, new_precision);

   return (hypre_error_flag);
}

#endif

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

#include "_hypre_parcsr_mv.h"

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
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   return hypre_SeqVectorCopy_mp(x_local, y_local);
}

/*--------------------------------------------------------------------------
 * Mixed-Precision hypre_ParVectorAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorAxpy_mp( hypre_long_double    alpha,
                        hypre_ParVector *x,
                        hypre_ParVector *y )
{
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);

   return hypre_SeqVectorAxpy_mp( alpha, x_local, y_local);
}

/*--------------------------------------------------------------------------
 * Mixed-Precision Vector conversion
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorConvert_mp( hypre_ParVector *v,
                           HYPRE_Precision new_precision)
{
   hypre_Vector *v_local = hypre_ParVectorLocalVector(v);
   hypre_SeqVectorConvert_mp (v_local, new_precision);
   hypre_VectorPrecision(v) = new_precision;
   return (hypre_error_flag);
}
/*--------------------------------------------------------------------------
 * Mixed-precision matrix conversion
 * Note: This converts only the diag and offd matrices
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixConvert_mp( hypre_ParCSRMatrix *A,
                              HYPRE_Precision new_precision)
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);

   hypre_CSRMatrixConvert_mp (A_diag, new_precision);
   hypre_CSRMatrixConvert_mp (A_offd, new_precision);

   hypre_ParCSRMatrixPrecision(A) = new_precision;

   return (hypre_error_flag);
}

/*--------------------------------------------------------------------------
 * Mixed-precision ParCSR matrix copy: Copies A to B.
 * The routine does not check whether the dimensions of A and B are compatible
 * TODO: update d_num_nonzeros not fixed as hypre_double
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixCopy_mp( hypre_ParCSRMatrix *A,
                           hypre_ParCSRMatrix *B )
{
   hypre_CSRMatrix *A_diag;
   hypre_CSRMatrix *A_offd;
   HYPRE_BigInt *col_map_offd_A;
   hypre_CSRMatrix *B_diag;
   hypre_CSRMatrix *B_offd;
   HYPRE_BigInt *col_map_offd_B;
   HYPRE_Int num_cols_offd_A;
   HYPRE_Int num_cols_offd_B;

   if (!A)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!B)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (hypre_ParCSRMatrixPrecision(A) == hypre_ParCSRMatrixPrecision(B))
   {
      return hypre_ParCSRMatrixCopy_pre( hypre_ParCSRMatrixPrecision(A), A, B, 1 );
   }

   A_diag = hypre_ParCSRMatrixDiag(A);
   A_offd = hypre_ParCSRMatrixOffd(A);
   B_diag = hypre_ParCSRMatrixDiag(B);
   B_offd = hypre_ParCSRMatrixOffd(B);

   num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);
   num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);

   hypre_assert(num_cols_offd_A == num_cols_offd_B);

   col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
   col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);

   hypre_CSRMatrixCopy_mp(A_diag, B_diag);
   hypre_CSRMatrixCopy_mp(A_offd, B_offd);

   /* should not happen if B has been initialized */
   if (num_cols_offd_B && col_map_offd_B == NULL)
   {
      col_map_offd_B = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_B, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixColMapOffd(B) = col_map_offd_B;
   }

   hypre_TMemcpy(col_map_offd_B, col_map_offd_A, HYPRE_BigInt, num_cols_offd_B,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Mixed-precision clone of ParCSR matrix.
 * New matrix resides in the same memory location
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRMatrixClone_mp(hypre_ParCSRMatrix   *A, HYPRE_Precision new_precision)
{
   hypre_ParCSRMatrix *S;

   hypre_GpuProfilingPushRange("hypre_ParCSRMatrixClone_mp");

   if (hypre_ParCSRMatrixPrecision(A) == new_precision)
   {
      return hypre_ParCSRMatrixClone_pre( hypre_ParCSRMatrixPrecision(A), A, 1 );
   }

   S = hypre_ParCSRMatrixCreate_pre( new_precision, hypre_ParCSRMatrixComm(A),
                                     hypre_ParCSRMatrixGlobalNumRows(A),
                                     hypre_ParCSRMatrixGlobalNumCols(A),
                                     hypre_ParCSRMatrixRowStarts(A),
                                     hypre_ParCSRMatrixColStarts(A),
                                     hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)),
                                     hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(A)),
                                     hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(A)) );

   hypre_ParCSRMatrixNumNonzeros(S)  = hypre_ParCSRMatrixNumNonzeros(A);
   hypre_ParCSRMatrixDNumNonzeros(S) = hypre_ParCSRMatrixNumNonzeros(A);

   hypre_ParCSRMatrixInitialize_v2_pre(new_precision, S, hypre_ParCSRMatrixMemoryLocation(A));

   hypre_ParCSRMatrixCopy_mp(A, S);

   hypre_GpuProfilingPopRange();

   return S;
}

#endif

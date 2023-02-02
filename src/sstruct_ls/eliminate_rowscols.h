/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_PARCSR_ELIMINATE_ROWSCOLS
#define hypre_PARCSR_ELIMINATE_ROWSCOLS

#ifdef __cplusplus
extern "C" {
#endif

/*
  Function:  hypre_ParCSRMatrixEliminateRowsCols

  This function eliminates the global rows and columns of a matrix
  A corresponding to given lists of sorted (!) local row numbers.

  The elimination is done as follows:

                / A_ii | A_ib \          / A_ii |  0   \
    (input) A = | -----+----- |   --->   | -----+----- | (output)
                \ A_bi | A_bb /          \   0  |  I   /
*/
HYPRE_Int hypre_ParCSRMatrixEliminateRowsCols (hypre_ParCSRMatrix *A,
                                               HYPRE_Int nrows_to_eliminate,
                                               HYPRE_Int *rows_to_eliminate);


/*
  Function:  hypre_CSRMatrixEliminateRowsColsDiag

  Eliminate the rows and columns of Adiag corresponding to the
  given sorted (!) list of rows. Put I on the eliminated diagonal.
*/
HYPRE_Int hypre_CSRMatrixEliminateRowsColsDiag (hypre_ParCSRMatrix *A,
                                                HYPRE_Int nrows_to_eliminate,
                                                HYPRE_Int *rows_to_eliminate);

/*
  Function:  hypre_CSRMatrixEliminateRowsOffd

  Eliminate the given list of rows of Aoffd.
*/
HYPRE_Int hypre_CSRMatrixEliminateRowsOffd (hypre_ParCSRMatrix *A,
                                            HYPRE_Int nrows_to_eliminate,
                                            HYPRE_Int *rows_to_eliminate);

/*
  Function:  hypre_CSRMatrixEliminateColsOffd

  Eliminate the given sorted (!) list of columns of Aoffd.
*/
HYPRE_Int hypre_CSRMatrixEliminateColsOffd (hypre_CSRMatrix *Aoffd,
                                            HYPRE_Int ncols_to_eliminate,
                                            HYPRE_Int *cols_to_eliminate);

#ifdef __cplusplus
}
#endif

#endif

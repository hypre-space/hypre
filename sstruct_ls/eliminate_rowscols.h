/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef hypre_PARCSR_ELIMINATE_ROWSCOLS
#define hypre_PARCSR_ELIMINATE_ROWSCOLS

/*
  Function:  hypre_ParCSRMatrixEliminateRowsCols

  This function eliminates the global rows and columns of a matrix
  A corresponding to given lists of sorted (!) local row numbers.

  The elimination is done as follows:

                / A_ii | A_ib \          / A_ii |  0   \
    (input) A = | -----+----- |   --->   | -----+----- | (output)
                \ A_bi | A_bb /          \   0  |  I   /
*/
int hypre_ParCSRMatrixEliminateRowsCols (hypre_ParCSRMatrix *A,
                                         int nrows_to_eliminate,
                                         int *rows_to_eliminate);


/*
  Function:  hypre_CSRMatrixEliminateRowsColsDiag

  Eliminate the rows and columns of Adiag corresponding to the
  given sorted (!) list of rows. Put I on the eliminated diagonal.
*/
int hypre_CSRMatrixEliminateRowsColsDiag (hypre_ParCSRMatrix *A,
                                          int nrows_to_eliminate,
                                          int *rows_to_eliminate);

/*
  Function:  hypre_CSRMatrixEliminateRowsOffd

  Eliminate the given list of rows of Aoffd.
*/
int hypre_CSRMatrixEliminateRowsOffd (hypre_ParCSRMatrix *A,
                                      int nrows_to_eliminate,
                                      int *rows_to_eliminate);

/*
  Function:  hypre_CSRMatrixEliminateColsOffd

  Eliminate the given sorted (!) list of columns of Aoffd.
*/
int hypre_CSRMatrixEliminateColsOffd (hypre_CSRMatrix *Aoffd,
                                      int ncols_to_eliminate,
                                      int *cols_to_eliminate);

#endif

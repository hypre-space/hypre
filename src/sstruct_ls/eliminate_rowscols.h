/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/





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

#endif

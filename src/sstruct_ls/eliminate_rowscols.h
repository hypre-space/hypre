/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.3 $
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

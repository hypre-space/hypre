/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Auxiliary Parallel CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_AUX_PARCSR_MATRIX_HEADER
#define hypre_AUX_PARCSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   int     local_num_rows;   /* defines number of rows on this processors */
   int     local_num_cols;   /* defines number of cols of diag */

   int	   diag_size;	    /* size of aux_diag_j or aux_diag_data */
   int	   indx_diag;	   /* first empty element of aux_diag_j(data) */
   int	   nnz_diag;   /* number of nonzeros entered into the structure */
   int    *row_start_diag; /* row_start_diag[i] points to first element 
				of i-th row */
   int    *row_end_diag; /* row_end_diag[i] points to last element 
				of i-th row */
   int    *aux_diag_j;	/* contains collected column indices */
   double *aux_diag_data; /* contains collected data */

   int	   offd_size; /* size of aux_offd_j or aux_offd_data */
   int	   indx_offd; /* first empty element of aux_offd_j(data) */
   int	   nnz_offd;  /* number of nonzeros entered into the structure */
   int    *row_start_offd;  /* row_start_offd[i] points to first element 
                                of i-th row */
   int    *row_end_offd;  /* row_end_offd[i] points to last element 
                                of i-th row */
   int    *aux_offd_j;  /* contains collected column indices */
   double *aux_offd_data;  /* contains collected data */

} hypre_AuxParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParCSRMatrixLocalNumRows(matrix)  ((matrix) -> local_num_rows)
#define hypre_AuxParCSRMatrixLocalNumCols(matrix)  ((matrix) -> local_num_cols)

#define hypre_AuxParCSRMatrixDiagSize(matrix)      ((matrix) -> diag_size)
#define hypre_AuxParCSRMatrixIndxDiag(matrix)      ((matrix) -> indx_diag)
#define hypre_AuxParCSRMatrixNnzDiag(matrix)       ((matrix) -> nnz_diag)
#define hypre_AuxParCSRMatrixRowStartDiag(matrix)  ((matrix) -> row_start_diag)
#define hypre_AuxParCSRMatrixRowEndDiag(matrix)    ((matrix) -> row_end_diag)
#define hypre_AuxParCSRMatrixAuxDiagJ(matrix)  	   ((matrix) -> aux_diag_j)
#define hypre_AuxParCSRMatrixAuxDiagData(matrix)   ((matrix) -> aux_diag_data)

#define hypre_AuxParCSRMatrixOffdSize(matrix)      ((matrix) -> offd_size)
#define hypre_AuxParCSRMatrixIndxOffd(matrix)      ((matrix) -> indx_offd)
#define hypre_AuxParCSRMatrixNnzOffd(matrix)       ((matrix) -> nnz_offd)
#define hypre_AuxParCSRMatrixRowStartOffd(matrix)  ((matrix) -> row_start_offd)
#define hypre_AuxParCSRMatrixRowEndOffd(matrix)    ((matrix) -> row_end_offd)
#define hypre_AuxParCSRMatrixAuxOffdJ(matrix)  	   ((matrix) -> aux_offd_j)
#define hypre_AuxParCSRMatrixAuxOffdData(matrix)   ((matrix) -> aux_offd_data)

#endif

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
 * Header info for Parallel CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Parallel CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm		comm;

   int     		global_num_rows;
   int     		global_num_cols;
   int     		first_row_index;
   int     		first_col_diag;
   hypre_CSRMatrix	*diag;
   hypre_CSRMatrix	*offd;
   int			*col_map_offd;

   hypre_CommPkg	*comm_pkg;
   
   /* Does the CSRMatrix create/destroy `data', `i', `j'? */
   int      owns_data;

} hypre_ParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRMatrixComm(matrix)		((matrix) -> comm)
#define hypre_ParCSRMatrixGlobalNumRows(matrix) ((matrix) -> global_num_rows)
#define hypre_ParCSRMatrixGlobalNumCols(matrix) ((matrix) -> global_num_cols)
#define hypre_ParCSRMatrixFirstRowIndex(matrix) ((matrix) -> first_row_index)
#define hypre_ParCSRMatrixFirstColDiag(matrix)  ((matrix) -> first_col_diag)
#define hypre_ParCSRMatrixDiag(matrix)  	((matrix) -> diag)
#define hypre_ParCSRMatrixOffd(matrix)  	((matrix) -> offd)
#define hypre_ParCSRMatrixColMapOffd(matrix)  	((matrix) -> col_map_offd)
#define hypre_ParCSRMatrixCommPkg(matrix)	((matrix) -> comm_pkg)
#define hypre_ParCSRMatrixOwnsData(matrix)      ((matrix) -> owns_data)


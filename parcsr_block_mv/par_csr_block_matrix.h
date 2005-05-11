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

#include <HYPRE_config.h>
                                                                                                               
#ifndef hypre_PAR_CSR_BLOCK_MATRIX_HEADER
#define hypre_PAR_CSR_BLOCK_MATRIX_HEADER
                                                                                                               
#include "utilities.h"
#include "csr_block_matrix.h"
#include "../parcsr_mv/parcsr_mv.h"
                                                                                                               
#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Parallel CSR Block Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm		comm;

   int     		global_num_rows;
   int     		global_num_cols;
   int			first_row_index;
   int			first_col_diag;
   hypre_CSRBlockMatrix	*diag;
   hypre_CSRBlockMatrix	*offd;
   int			*col_map_offd; 
	/* maps columns of offd to global columns */
   int 			*row_starts; 
	/* array of length num_procs+1, row_starts[i] contains the 
	   global number of the first row on proc i,  
	   first_row_index = row_starts[my_id],
	   row_starts[num_procs] = global_num_rows */
   int 			*col_starts;
	/* array of length num_procs+1, col_starts[i] contains the 
	   global number of the first column of diag on proc i,  
	   first_col_diag = col_starts[my_id],
	   col_starts[num_procs] = global_num_cols */

   hypre_ParCSRCommPkg	*comm_pkg;
   hypre_ParCSRCommPkg	*comm_pkgT;
   
   /* Does the ParCSRBlockMatrix create/destroy `diag', `offd', `col_map_offd'? */
   int      owns_data;
   /* Does the ParCSRBlockMatrix create/destroy `row_starts', `col_starts'? */
   int      owns_row_starts;
   int      owns_col_starts;

   int      num_nonzeros;

   /* Buffers used by GetRow to hold row currently being accessed. AJC, 4/99 */
   int     *rowindices;
   double  *rowvalues;
   int      getrowactive;

} hypre_ParCSRBlockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Block Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRBlockMatrixComm(matrix)	       ((matrix)->comm)
#define hypre_ParCSRBlockMatrixGlobalNumRows(matrix)   ((matrix)->global_num_rows)
#define hypre_ParCSRBlockMatrixGlobalNumCols(matrix)   ((matrix)->global_num_cols)
#define hypre_ParCSRBlockMatrixFirstRowIndex(matrix)   ((matrix)->first_row_index)
#define hypre_ParCSRBlockMatrixFirstColDiag(matrix)    ((matrix)->first_col_diag)
#define hypre_ParCSRBlockMatrixBlockSize(matrix)       ((matrix)->diag->block_size)
#define hypre_ParCSRBlockMatrixDiag(matrix)  	       ((matrix) -> diag)
#define hypre_ParCSRBlockMatrixOffd(matrix)  	       ((matrix) -> offd)
#define hypre_ParCSRBlockMatrixColMapOffd(matrix)      ((matrix) -> col_map_offd)
#define hypre_ParCSRBlockMatrixRowStarts(matrix)       ((matrix) -> row_starts)
#define hypre_ParCSRBlockMatrixColStarts(matrix)       ((matrix) -> col_starts)
#define hypre_ParCSRBlockMatrixCommPkg(matrix)	       ((matrix) -> comm_pkg)
#define hypre_ParCSRBlockMatrixCommPkgT(matrix)	       ((matrix) -> comm_pkgT)
#define hypre_ParCSRBlockMatrixOwnsData(matrix)        ((matrix) -> owns_data)
#define hypre_ParCSRBlockMatrixOwnsRowStarts(matrix)   ((matrix) -> owns_row_starts)
#define hypre_ParCSRBlockMatrixOwnsColStarts(matrix)   ((matrix) -> owns_col_starts)
#define hypre_ParCSRBlockMatrixNumRows(matrix) \
hypre_CSRBlockMatrixNumRows(hypre_ParCSRBlockMatrixDiag(matrix))
#define hypre_ParCSRBlockMatrixNumCols(matrix) \
hypre_CSRBlockMatrixNumCols(hypre_ParCSRBlockMatrixDiag(matrix))
#define hypre_ParCSRBlockMatrixNumNonzeros(matrix)     ((matrix) -> num_nonzeros)
#define hypre_ParCSRBlockMatrixRowindices(matrix)      ((matrix) -> rowindices)
#define hypre_ParCSRBlockMatrixRowvalues(matrix)       ((matrix) -> rowvalues)
#define hypre_ParCSRBlockMatrixGetrowactive(matrix)    ((matrix) -> getrowactive)

hypre_CSRBlockMatrix *
hypre_ParCSRBlockMatrixExtractBExt(hypre_ParCSRBlockMatrix *B,
                                   hypre_ParCSRBlockMatrix *A, int data);

hypre_ParCSRBlockMatrix *
hypre_ParCSRBlockMatrixCreate(MPI_Comm comm, int block_size, int global_num_rows,
                              int global_num_cols, int *row_starts, int *col_starts,
                              int num_cols_offd, int num_nonzeros_diag,
                              int num_nonzeros_offd);

#ifdef __cplusplus
}
#endif
#endif

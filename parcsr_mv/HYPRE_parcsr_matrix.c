/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_ParCSRMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRMatrixCreate( MPI_Comm  comm,
                          int       global_num_rows,
                          int       global_num_cols,
                          int      *row_starts,
                          int      *col_starts,
                          int       num_cols_offd,
                          int       num_nonzeros_diag,
                          int       num_nonzeros_offd,
			  HYPRE_ParCSRMatrix *matrix )
{
   *matrix = (HYPRE_ParCSRMatrix)
	hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                     row_starts, col_starts, num_cols_offd,
                                     num_nonzeros_diag, num_nonzeros_offd);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRMatrixDestroy( HYPRE_ParCSRMatrix matrix )
{
   return( hypre_ParCSRMatrixDestroy( (hypre_ParCSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRMatrixInitialize( HYPRE_ParCSRMatrix matrix )
{
   return ( hypre_ParCSRMatrixInitialize( (hypre_ParCSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixRead
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRMatrixRead( MPI_Comm comm,
                        char    *file_name, 
			HYPRE_ParCSRMatrix *matrix)
{
   *matrix = (HYPRE_ParCSRMatrix) hypre_ParCSRMatrixRead( comm, file_name );
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixPrint
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRMatrixPrint( HYPRE_ParCSRMatrix  matrix,
                         char               *file_name )
{
   hypre_ParCSRMatrixPrint( (hypre_ParCSRMatrix *) matrix,
                            file_name );
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetComm
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRMatrixGetComm( HYPRE_ParCSRMatrix  matrix,
                         MPI_Comm *comm )
{  
   int ierr = 0;

   *comm = hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) matrix);

   return( ierr );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetDims
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRMatrixGetDims( HYPRE_ParCSRMatrix  matrix,
                         int *M, int *N )
{  
   int ierr = 0;

   *M = hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix *) matrix);
   *N = hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix *) matrix);

   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRMatrixGetRowPartitioning( HYPRE_ParCSRMatrix  matrix,
                                int **row_partitioning_ptr)
{  
   int ierr = 0;
   int *row_partitioning, *row_starts;
   int num_procs, i;

   MPI_Comm_size(hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) matrix), 
			&num_procs);
   row_starts = hypre_ParCSRMatrixRowStarts((hypre_ParCSRMatrix *) matrix);
   if (!row_starts) return -1;
   row_partitioning = hypre_CTAlloc(int, num_procs+1);
   for (i=0; i < num_procs + 1; i++)
	row_partitioning[i] = row_starts[i];

   *row_partitioning_ptr = row_partitioning;
   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRMatrixGetColPartitioning( HYPRE_ParCSRMatrix  matrix,
                                int **col_partitioning_ptr)
{  
   int ierr = 0;
   int *col_partitioning, *col_starts;
   int num_procs, i;

   MPI_Comm_size(hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) matrix), 
			&num_procs);
   col_starts = hypre_ParCSRMatrixColStarts((hypre_ParCSRMatrix *) matrix);
   if (!col_starts) return -1;
   col_partitioning = hypre_CTAlloc(int, num_procs+1);
   for (i=0; i < num_procs + 1; i++)
	col_partitioning[i] = col_starts[i];

   *col_partitioning_ptr = col_partitioning;
   return( ierr );
}



/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetLocalRange
 *--------------------------------------------------------------------------*/
/**
Returns range of rows and columns owned by this processor.
Not collective.

@return integer error code
@param HYPRE_ParCSRMatrix matrix [IN]
 the matrix to be operated on. 
@param int *row_start [OUT]
 the global number of the first row stored on this processor
@param int *row_end [OUT]
 the global number of the first row stored on this processor
@param int *col_start [OUT]
 the global number of the first column stored on this processor
@param int *col_end [OUT]
 the global number of the first column stored on this processor
*/

int
HYPRE_ParCSRMatrixGetLocalRange( HYPRE_ParCSRMatrix  matrix,
                         int               *row_start,
                         int               *row_end,
                         int               *col_start,
                         int               *col_end )
{  
   int ierr = 0;

   ierr = hypre_ParCSRMatrixGetLocalRange( (hypre_ParCSRMatrix *) matrix,
                            row_start, row_end, col_start, col_end );
   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetRow
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRMatrixGetRow( HYPRE_ParCSRMatrix  matrix,
                         int                row,
                         int               *size,
                         int               **col_ind,
                         double            **values )
{  
   int ierr=0;
   
   ierr = hypre_ParCSRMatrixGetRow( (hypre_ParCSRMatrix *) matrix,
                            row, size, col_ind, values );
   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixRestoreRow
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRMatrixRestoreRow( HYPRE_ParCSRMatrix  matrix,
                         int                row,
                         int               *size,
                         int               **col_ind,
                         double            **values )
{  
   int ierr = 0;

   ierr = hypre_ParCSRMatrixRestoreRow( (hypre_ParCSRMatrix *) matrix,
                            row, size, col_ind, values );
   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixToParCSRMatrix
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRMatrixToParCSRMatrix( MPI_Comm comm,
			       HYPRE_CSRMatrix A_CSR,
			       int *row_partitioning,
                               int *col_partitioning,
			       HYPRE_ParCSRMatrix *matrix)
{
   *matrix = (HYPRE_ParCSRMatrix) hypre_CSRMatrixToParCSRMatrix( comm, 	
		(hypre_CSRMatrix *) A_CSR, row_partitioning, 
		col_partitioning) ;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRMatrixMatvec( double alpha,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector x,
                          double beta,
                          HYPRE_ParVector y     )
{
   return ( hypre_ParCSRMatrixMatvec( alpha, (hypre_ParCSRMatrix *) A,
		(hypre_ParVector *) x, beta, (hypre_ParVector *) y) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixMatvecT
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRMatrixMatvecT( double alpha,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector x,
                           double beta,
                           HYPRE_ParVector y     )
{
   return ( hypre_ParCSRMatrixMatvecT( alpha, (hypre_ParCSRMatrix *) A,
		(hypre_ParVector *) x, beta, (hypre_ParVector *) y) );
}

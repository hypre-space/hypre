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
 * HYPRE_CreateParCSRMatrix
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix 
HYPRE_CreateParCSRMatrix( MPI_Comm  comm,
                          int       global_num_rows,
                          int       global_num_cols,
                          int      *row_starts,
                          int      *col_starts,
                          int       num_cols_offd,
                          int       num_nonzeros_diag,
                          int       num_nonzeros_offd )
{
   hypre_ParCSRMatrix *matrix;

   matrix = hypre_CreateParCSRMatrix(comm, global_num_rows, global_num_cols,
                                     row_starts, col_starts, num_cols_offd,
                                     num_nonzeros_diag, num_nonzeros_offd);

   return ( (HYPRE_ParCSRMatrix) matrix );
}

/*--------------------------------------------------------------------------
 * HYPRE_DestroyParCSRMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_DestroyParCSRMatrix( HYPRE_ParCSRMatrix matrix )
{
   return( hypre_DestroyParCSRMatrix( (hypre_ParCSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeParCSRMatrix
 *--------------------------------------------------------------------------*/

int
HYPRE_InitializeParCSRMatrix( HYPRE_ParCSRMatrix matrix )
{
   return ( hypre_InitializeParCSRMatrix( (hypre_ParCSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PrintParCSRMatrix
 *--------------------------------------------------------------------------*/

void 
HYPRE_PrintParCSRMatrix( HYPRE_ParCSRMatrix  matrix,
                         char               *file_name )
{
   hypre_PrintParCSRMatrix( (hypre_ParCSRMatrix *) matrix,
                            file_name );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetCommParCSR
 *--------------------------------------------------------------------------*/

int
HYPRE_GetCommParCSR( HYPRE_ParCSRMatrix  matrix,
                         MPI_Comm *comm )
{  
   int ierr = 0;

   *comm = hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) matrix);

   return( ierr );
}
/*--------------------------------------------------------------------------
 * HYPRE_GetDimsParCSR
 *--------------------------------------------------------------------------*/

int
HYPRE_GetDimsParCSR( HYPRE_ParCSRMatrix  matrix,
                         int *M, int *N )
{  
   int ierr = 0;

   *M = hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix *) matrix);
   *N = hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix *) matrix);

   return( ierr );
}



/*--------------------------------------------------------------------------
 * HYPRE_GetLocalRangeParcsr
 *--------------------------------------------------------------------------*/

int
HYPRE_GetLocalRangeParcsr( HYPRE_ParCSRMatrix  matrix,
                         int               *start,
                         int               *end )
{  
   int ierr = 0;

   ierr = hypre_GetLocalRangeParCSRMatrix( (hypre_ParCSRMatrix *) matrix,
                            start, end );
   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetRowParCSRMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_GetRowParCSRMatrix( HYPRE_ParCSRMatrix  matrix,
                         int                row,
                         int               *size,
                         int               **col_ind,
                         double            **values )
{  
   int ierr=0;
   
   ierr = hypre_GetRowParCSRMatrix( (hypre_ParCSRMatrix *) matrix,
                            row, size, col_ind, values );
   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_RestoreRowParCSRMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_RestoreRowParCSRMatrix( HYPRE_ParCSRMatrix  matrix,
                         int                row,
                         int               *size,
                         int               **col_ind,
                         double            **values )
{  
   int ierr = 0;

   ierr = hypre_RestoreRowParCSRMatrix( (hypre_ParCSRMatrix *) matrix,
                            row, size, col_ind, values );
   return( ierr );
}

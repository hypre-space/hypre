/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_ParCSRMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixCreate( MPI_Comm  comm,
                          HYPRE_Int       global_num_rows,
                          HYPRE_Int       global_num_cols,
                          HYPRE_Int      *row_starts,
                          HYPRE_Int      *col_starts,
                          HYPRE_Int       num_cols_offd,
                          HYPRE_Int       num_nonzeros_diag,
                          HYPRE_Int       num_nonzeros_offd,
			  HYPRE_ParCSRMatrix *matrix )
{
   *matrix = (HYPRE_ParCSRMatrix)
	hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                     row_starts, col_starts, num_cols_offd,
                                     num_nonzeros_diag, num_nonzeros_offd);

   if (!matrix) hypre_error_in_arg(9);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRMatrixDestroy( HYPRE_ParCSRMatrix matrix )
{
   return( hypre_ParCSRMatrixDestroy( (hypre_ParCSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixInitialize( HYPRE_ParCSRMatrix matrix )
{
   return ( hypre_ParCSRMatrixInitialize( (hypre_ParCSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixRead
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixRead( MPI_Comm            comm,
                        const char         *file_name, 
			HYPRE_ParCSRMatrix *matrix)
{
   *matrix = (HYPRE_ParCSRMatrix) hypre_ParCSRMatrixRead( comm, file_name );
   if (!matrix) hypre_error_in_arg(3);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixPrint( HYPRE_ParCSRMatrix  matrix,
                         const char         *file_name )
{
   hypre_ParCSRMatrixPrint( (hypre_ParCSRMatrix *) matrix,
                            file_name );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetComm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixGetComm( HYPRE_ParCSRMatrix  matrix,
                         MPI_Comm *comm )
{  
   if (!matrix) hypre_error_in_arg(1);
   *comm = hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) matrix);

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetDims
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixGetDims( HYPRE_ParCSRMatrix  matrix,
                         HYPRE_Int *M, HYPRE_Int *N )
{  
   if (!matrix) hypre_error_in_arg(1);

   *M = hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix *) matrix);
   *N = hypre_ParCSRMatrixGlobalNumCols((hypre_ParCSRMatrix *) matrix);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixGetRowPartitioning( HYPRE_ParCSRMatrix  matrix,
                                HYPRE_Int **row_partitioning_ptr)
{  
   HYPRE_Int *row_partitioning, *row_starts;
   HYPRE_Int num_procs, i;

   if (!matrix) 
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_MPI_Comm_size(hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) matrix), 
			&num_procs);
   row_starts = hypre_ParCSRMatrixRowStarts((hypre_ParCSRMatrix *) matrix);
   if (!row_starts) return -1;
   row_partitioning = hypre_CTAlloc(HYPRE_Int, num_procs+1);
   for (i=0; i < num_procs + 1; i++)
	row_partitioning[i] = row_starts[i];

   *row_partitioning_ptr = row_partitioning;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixGetColPartitioning( HYPRE_ParCSRMatrix  matrix,
                                HYPRE_Int **col_partitioning_ptr)
{  
   HYPRE_Int *col_partitioning, *col_starts;
   HYPRE_Int num_procs, i;

   if (!matrix) 
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_MPI_Comm_size(hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) matrix), 
			&num_procs);
   col_starts = hypre_ParCSRMatrixColStarts((hypre_ParCSRMatrix *) matrix);
   if (!col_starts) return -1;
   col_partitioning = hypre_CTAlloc(HYPRE_Int, num_procs+1);
   for (i=0; i < num_procs + 1; i++)
	col_partitioning[i] = col_starts[i];

   *col_partitioning_ptr = col_partitioning;
   return hypre_error_flag;
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
@param HYPRE_Int *row_start [OUT]
 the global number of the first row stored on this processor
@param HYPRE_Int *row_end [OUT]
 the global number of the first row stored on this processor
@param HYPRE_Int *col_start [OUT]
 the global number of the first column stored on this processor
@param HYPRE_Int *col_end [OUT]
 the global number of the first column stored on this processor
*/

HYPRE_Int
HYPRE_ParCSRMatrixGetLocalRange( HYPRE_ParCSRMatrix  matrix,
                         HYPRE_Int               *row_start,
                         HYPRE_Int               *row_end,
                         HYPRE_Int               *col_start,
                         HYPRE_Int               *col_end )
{  
   if (!matrix) 
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }


   hypre_ParCSRMatrixGetLocalRange( (hypre_ParCSRMatrix *) matrix,
                            row_start, row_end, col_start, col_end );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetRow
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRMatrixGetRow( HYPRE_ParCSRMatrix  matrix,
                         HYPRE_Int                row,
                         HYPRE_Int               *size,
                         HYPRE_Int               **col_ind,
                         double            **values )
{  
   if (!matrix) 
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   
   hypre_ParCSRMatrixGetRow( (hypre_ParCSRMatrix *) matrix,
                            row, size, col_ind, values );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixRestoreRow
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRMatrixRestoreRow( HYPRE_ParCSRMatrix  matrix,
                         HYPRE_Int                row,
                         HYPRE_Int               *size,
                         HYPRE_Int               **col_ind,
                         double            **values )
{  
   if (!matrix) 
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }


   hypre_ParCSRMatrixRestoreRow( (hypre_ParCSRMatrix *) matrix,
                            row, size, col_ind, values );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixToParCSRMatrix
 * Output argument (fifth argument): a new ParCSRmatrix.
 * Input arguments: MPI communicator, CSR matrix, and optional partitionings.
 * If you don't have partitionings, just pass a null pointer for the third
 * and fourth arguments and they will be computed.
 * Note that it is not possible to provide a null pointer if this is called
 * from Fortran code; so you must provide the paritionings from Fortran.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRMatrixToParCSRMatrix( MPI_Comm comm,
			       HYPRE_CSRMatrix A_CSR,
			       HYPRE_Int *row_partitioning,
                               HYPRE_Int *col_partitioning,
			       HYPRE_ParCSRMatrix *matrix)
{
   *matrix = (HYPRE_ParCSRMatrix) hypre_CSRMatrixToParCSRMatrix( comm, 	
		(hypre_CSRMatrix *) A_CSR, row_partitioning, 
		col_partitioning) ;
   if (!matrix) 
      hypre_error_in_arg(5);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning
 * Output argument (third argument): a new ParCSRmatrix.
 * Input arguments: MPI communicator, CSR matrix.
 * Row and column partitionings are computed for the output matrix.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning(
   MPI_Comm comm, HYPRE_CSRMatrix A_CSR, HYPRE_ParCSRMatrix *matrix)
{
   *matrix = (HYPRE_ParCSRMatrix) hypre_CSRMatrixToParCSRMatrix(
      comm, (hypre_CSRMatrix *) A_CSR, NULL, NULL ) ;
   if (!matrix) 
      hypre_error_in_arg(3);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
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

HYPRE_Int
HYPRE_ParCSRMatrixMatvecT( double alpha,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector x,
                           double beta,
                           HYPRE_ParVector y     )
{
   return ( hypre_ParCSRMatrixMatvecT( alpha, (hypre_ParCSRMatrix *) A,
		(hypre_ParVector *) x, beta, (hypre_ParVector *) y) );
}

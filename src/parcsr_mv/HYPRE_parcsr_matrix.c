/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSRMatrix interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixCreate( MPI_Comm            comm,
                          HYPRE_BigInt        global_num_rows,
                          HYPRE_BigInt        global_num_cols,
                          HYPRE_BigInt       *row_starts,
                          HYPRE_BigInt       *col_starts,
                          HYPRE_Int           num_cols_offd,
                          HYPRE_Int           num_nonzeros_diag,
                          HYPRE_Int           num_nonzeros_offd,
                          HYPRE_ParCSRMatrix *matrix )
{
   if (!matrix)
   {
      hypre_error_in_arg(9);
      return hypre_error_flag;
   }

   *matrix = (HYPRE_ParCSRMatrix)
             hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                      row_starts, col_starts, num_cols_offd,
                                      num_nonzeros_diag, num_nonzeros_offd);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixDestroy( HYPRE_ParCSRMatrix matrix )
{
   return ( hypre_ParCSRMatrixDestroy( (hypre_ParCSRMatrix *) matrix ) );
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
   if (!matrix)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   *matrix = (HYPRE_ParCSRMatrix) hypre_ParCSRMatrixRead( comm, file_name );
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
                           MPI_Comm           *comm )
{
   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *comm = hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) matrix);

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetDims
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixGetDims( HYPRE_ParCSRMatrix  matrix,
                           HYPRE_BigInt       *M,
                           HYPRE_BigInt       *N )
{
   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *M = hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix *) matrix);
   *N = hypre_ParCSRMatrixGlobalNumCols((hypre_ParCSRMatrix *) matrix);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixGetRowPartitioning( HYPRE_ParCSRMatrix   matrix,
                                      HYPRE_BigInt       **row_partitioning_ptr )
{
   HYPRE_BigInt *row_partitioning, *row_starts;
   HYPRE_Int num_procs, i;

   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_MPI_Comm_size(hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) matrix),
                       &num_procs);
   row_starts = hypre_ParCSRMatrixRowStarts((hypre_ParCSRMatrix *) matrix);
   if (!row_starts) { return -1; }
   row_partitioning = hypre_CTAlloc(HYPRE_BigInt,  num_procs + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_procs + 1; i++)
   {
      row_partitioning[i] = row_starts[i];
   }

   *row_partitioning_ptr = row_partitioning;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetGlobalRowPartitioning
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ParCSRMatrixGetGlobalRowPartitioning( HYPRE_ParCSRMatrix   matrix,
                                            HYPRE_Int            all_procs,
                                            HYPRE_BigInt       **row_partitioning_ptr )
{
   MPI_Comm        comm;
   HYPRE_Int       my_id;
   HYPRE_BigInt   *row_partitioning = NULL;

   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   comm = hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) matrix);
   hypre_MPI_Comm_rank(comm, &my_id);

   HYPRE_Int       num_procs;
   HYPRE_BigInt    row_start;

   hypre_MPI_Comm_size(comm, &num_procs);
   if (my_id == 0 || all_procs)
   {
      row_partitioning = hypre_CTAlloc(HYPRE_BigInt, num_procs + 1, HYPRE_MEMORY_HOST);
   }

   row_start = hypre_ParCSRMatrixFirstRowIndex((hypre_ParCSRMatrix *) matrix);
   if (all_procs)
   {
      hypre_MPI_Allgather(&row_start, 1, HYPRE_MPI_BIG_INT, row_partitioning,
                          1, HYPRE_MPI_BIG_INT, comm);
   }
   else
   {
      hypre_MPI_Gather(&row_start, 1, HYPRE_MPI_BIG_INT, row_partitioning,
                       1, HYPRE_MPI_BIG_INT, 0, comm);
   }

   if (my_id == 0 || all_procs)
   {
      row_partitioning[num_procs] = hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix *) matrix);
   }

   *row_partitioning_ptr = row_partitioning;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixGetColPartitioning( HYPRE_ParCSRMatrix   matrix,
                                      HYPRE_BigInt       **col_partitioning_ptr )
{
   HYPRE_BigInt *col_partitioning, *col_starts;
   HYPRE_Int num_procs, i;

   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_MPI_Comm_size(hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) matrix),
                       &num_procs);
   col_starts = hypre_ParCSRMatrixColStarts((hypre_ParCSRMatrix *) matrix);
   if (!col_starts) { return -1; }
   col_partitioning = hypre_CTAlloc(HYPRE_BigInt,  num_procs + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_procs + 1; i++)
   {
      col_partitioning[i] = col_starts[i];
   }

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
                                 HYPRE_BigInt       *row_start,
                                 HYPRE_BigInt       *row_end,
                                 HYPRE_BigInt       *col_start,
                                 HYPRE_BigInt       *col_end )
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
                          HYPRE_BigInt        row,
                          HYPRE_Int          *size,
                          HYPRE_BigInt      **col_ind,
                          HYPRE_Complex     **values )
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
                              HYPRE_BigInt        row,
                              HYPRE_Int          *size,
                              HYPRE_BigInt      **col_ind,
                              HYPRE_Complex     **values )
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
HYPRE_CSRMatrixToParCSRMatrix( MPI_Comm            comm,
                               HYPRE_CSRMatrix     A_CSR,
                               HYPRE_BigInt       *row_partitioning,
                               HYPRE_BigInt       *col_partitioning,
                               HYPRE_ParCSRMatrix *matrix)
{
   if (!matrix)
   {
      hypre_error_in_arg(5);
      return hypre_error_flag;
   }
   *matrix = (HYPRE_ParCSRMatrix)
             hypre_CSRMatrixToParCSRMatrix( comm, (hypre_CSRMatrix *) A_CSR,
                                            row_partitioning, col_partitioning) ;
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
   MPI_Comm            comm,
   HYPRE_CSRMatrix     A_CSR,
   HYPRE_ParCSRMatrix *matrix )
{
   if (!matrix)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   *matrix = (HYPRE_ParCSRMatrix)
             hypre_CSRMatrixToParCSRMatrix( comm, (hypre_CSRMatrix *) A_CSR, NULL, NULL ) ;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixMatvec( HYPRE_Complex      alpha,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector    x,
                          HYPRE_Complex      beta,
                          HYPRE_ParVector    y )
{
   return ( hypre_ParCSRMatrixMatvec(
               alpha, (hypre_ParCSRMatrix *) A,
               (hypre_ParVector *) x, beta, (hypre_ParVector *) y) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixMatvecT
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRMatrixMatvecT( HYPRE_Complex      alpha,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    x,
                           HYPRE_Complex      beta,
                           HYPRE_ParVector    y )
{
   return ( hypre_ParCSRMatrixMatvecT(
               alpha, (hypre_ParCSRMatrix *) A,
               (hypre_ParVector *) x, beta, (hypre_ParVector *) y) );
}

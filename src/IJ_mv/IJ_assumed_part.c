/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*----------------------------------------------------
 * Functions for the IJ assumed partition fir IJ_Matrix
 *-----------------------------------------------------*/

#include "_hypre_IJ_mv.h"

/*------------------------------------------------------------------
 * hypre_IJMatrixCreateAssumedPartition -
 * Each proc gets it own range. Then
 * each needs to reconcile its actual range with its assumed
 * range - the result is essentila a partition of its assumed range -
 * this is the assumed partition.
 *--------------------------------------------------------------------*/


HYPRE_Int
hypre_IJMatrixCreateAssumedPartition( hypre_IJMatrix *matrix)
{
   HYPRE_BigInt global_num_rows;
   HYPRE_BigInt global_first_row;
   HYPRE_Int myid;
   HYPRE_BigInt row_start = 0, row_end = 0;
   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);

   MPI_Comm   comm;

   hypre_IJAssumedPart *apart;

   global_num_rows = hypre_IJMatrixGlobalNumRows(matrix);
   global_first_row = hypre_IJMatrixGlobalFirstRow(matrix);
   comm = hypre_IJMatrixComm(matrix);

   /* find out my actual range of rows and rowumns */
   row_start = row_partitioning[0];
   row_end = row_partitioning[1] - 1;
   hypre_MPI_Comm_rank(comm, &myid );

   /* allocate space */
   apart = hypre_CTAlloc(hypre_IJAssumedPart,  1, HYPRE_MEMORY_HOST);

   /* get my assumed partitioning  - we want row partitioning of the matrix
      for off processor values - so we use the row start and end
      Note that this is different from the assumed partitioning for the parcsr matrix
      which needs it for matvec multiplications and therefore needs to do it for
      the col partitioning */
   hypre_GetAssumedPartitionRowRange( comm, myid, global_first_row,
                                      global_num_rows, &(apart->row_start), &(apart->row_end));

   /*allocate some space for the partition of the assumed partition */
   apart->length = 0;
   /*room for 10 owners of the assumed partition*/
   apart->storage_length = 10; /*need to be >=1 */
   apart->proc_list = hypre_TAlloc(HYPRE_Int,  apart->storage_length, HYPRE_MEMORY_HOST);
   apart->row_start_list =   hypre_TAlloc(HYPRE_BigInt,  apart->storage_length, HYPRE_MEMORY_HOST);
   apart->row_end_list =   hypre_TAlloc(HYPRE_BigInt,  apart->storage_length, HYPRE_MEMORY_HOST);

   /* now we want to reconcile our actual partition with the assumed partition */
   hypre_LocateAssumedPartition(comm, row_start, row_end, global_first_row,
                                global_num_rows, apart, myid);

   /* this partition will be saved in the matrix data structure until the matrix is destroyed */
   hypre_IJMatrixAssumedPart(matrix) = apart;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_IJVectorCreateAssumedPartition -

 * Essentially the same as for a matrix!

 * Each proc gets it own range. Then
 * each needs to reconcile its actual range with its assumed
 * range - the result is essentila a partition of its assumed range -
 * this is the assumed partition.
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_IJVectorCreateAssumedPartition( hypre_IJVector *vector)
{
   HYPRE_BigInt  global_num, global_first_row;
   HYPRE_Int     myid;
   HYPRE_BigInt  start, end;
   HYPRE_BigInt *partitioning = hypre_IJVectorPartitioning(vector);
   MPI_Comm      comm;

   hypre_IJAssumedPart *apart;

   global_num = hypre_IJVectorGlobalNumRows(vector);
   global_first_row = hypre_IJVectorGlobalFirstRow(vector);
   comm = hypre_ParVectorComm(vector);

   /* find out my actualy range of rows */
   start = partitioning[0];
   end   = partitioning[1] - 1;

   hypre_MPI_Comm_rank(comm, &myid);

   /* allocate space */
   apart = hypre_CTAlloc(hypre_IJAssumedPart, 1, HYPRE_MEMORY_HOST);

   /* get my assumed partitioning  - we want partitioning of the vector that the
      matrix multiplies - so we use the col start and end */
   hypre_GetAssumedPartitionRowRange(comm, myid, global_first_row,
                                     global_num, &(apart->row_start), &(apart->row_end));

   /*allocate some space for the partition of the assumed partition */
   apart->length = 0;
   /*room for 10 owners of the assumed partition*/
   apart->storage_length = 10; /*need to be >=1 */
   apart->proc_list      = hypre_TAlloc(HYPRE_Int, apart->storage_length, HYPRE_MEMORY_HOST);
   apart->row_start_list = hypre_TAlloc(HYPRE_BigInt, apart->storage_length, HYPRE_MEMORY_HOST);
   apart->row_end_list   = hypre_TAlloc(HYPRE_BigInt, apart->storage_length, HYPRE_MEMORY_HOST);

   /* now we want to reconcile our actual partition with the assumed partition */
   hypre_LocateAssumedPartition(comm, start, end, global_first_row,
                                global_num, apart, myid);

   /* this partition will be saved in the vector data structure until the vector is destroyed */
   hypre_IJVectorAssumedPart(vector) = apart;

   return hypre_error_flag;
}

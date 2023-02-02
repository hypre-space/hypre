/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_block_mv.h"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockCommHandleCreate
 *--------------------------------------------------------------------------*/

hypre_ParCSRCommHandle *
hypre_ParCSRBlockCommHandleCreate(HYPRE_Int job,
                                  HYPRE_Int bnnz,
                                  hypre_ParCSRCommPkg *comm_pkg,
                                  void *send_data,
                                  void *recv_data )
{
   HYPRE_Int      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   MPI_Comm comm      = hypre_ParCSRCommPkgComm(comm_pkg);
   hypre_ParCSRCommHandle *comm_handle;
   HYPRE_Int         num_requests;
   hypre_MPI_Request *requests;
   HYPRE_Int    i, j, my_id, num_procs, ip, vec_start, vec_len;
   HYPRE_Complex *d_send_data = (HYPRE_Complex *) send_data;
   HYPRE_Complex *d_recv_data = (HYPRE_Complex *) recv_data;

   /*---------------------------------------------------------------------------
    * job = 1 : is used to initialize communication exchange for the parts
    *           of vector needed to perform a Matvec,  it requires send_data
    *           and recv_data to be doubles, recv_vec_starts and
    *           send_map_starts need to be set in comm_pkg.
    * job = 2 : is used to initialize communication exchange for the parts
    *           of vector needed to perform a MatvecT,  it requires send_data
    *           and recv_data to be doubles, recv_vec_starts and
    *           send_map_starts need to be set in comm_pkg.
    *------------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests = hypre_CTAlloc(hypre_MPI_Request,  num_requests, HYPRE_MEMORY_HOST);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   j = 0;

   switch (job)
   {
      case  1:
      {
         for (i = 0; i < num_recvs; i++)
         {
            ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len =
               (hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start) * bnnz;
            hypre_MPI_Irecv(&d_recv_data[vec_start * bnnz], vec_len,
                            HYPRE_MPI_COMPLEX, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_sends; i++)
         {
            vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len =
               (hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start) * bnnz;
            ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            hypre_MPI_Isend(&d_send_data[vec_start * bnnz], vec_len,
                            HYPRE_MPI_COMPLEX, ip, 0, comm, &requests[j++]);
         }
         break;
      }
      case  2:
      {

         for (i = 0; i < num_sends; i++)
         {
            vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len =
               (hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start) * bnnz;
            ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            hypre_MPI_Irecv(&d_recv_data[vec_start * bnnz], vec_len,
                            HYPRE_MPI_COMPLEX, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_recvs; i++)
         {
            ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len =
               (hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start) * bnnz;
            hypre_MPI_Isend(&d_send_data[vec_start * bnnz], vec_len,
                            HYPRE_MPI_COMPLEX, ip, 0, comm, &requests[j++]);
         }
         break;
      }
   }

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle,  1, HYPRE_MEMORY_HOST);

   hypre_ParCSRCommHandleCommPkg(comm_handle)     = comm_pkg;
   hypre_ParCSRCommHandleSendData(comm_handle)    = send_data;
   hypre_ParCSRCommHandleRecvData(comm_handle)    = recv_data;
   hypre_ParCSRCommHandleNumRequests(comm_handle) = num_requests;
   hypre_ParCSRCommHandleRequests(comm_handle)    = requests;
   return ( comm_handle );
}

/*--------------------------------------------------------------------
  hypre_ParCSRBlockCommHandleDestroy
  *--------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRBlockCommHandleDestroy(hypre_ParCSRCommHandle *comm_handle)
{
   hypre_MPI_Status          *status0;

   if ( comm_handle == NULL ) { return hypre_error_flag; }

   if (hypre_ParCSRCommHandleNumRequests(comm_handle))
   {
      status0 = hypre_CTAlloc(hypre_MPI_Status,
                              hypre_ParCSRCommHandleNumRequests(comm_handle), HYPRE_MEMORY_HOST);
      hypre_MPI_Waitall(hypre_ParCSRCommHandleNumRequests(comm_handle),
                        hypre_ParCSRCommHandleRequests(comm_handle), status0);
      hypre_TFree(status0, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(hypre_ParCSRCommHandleRequests(comm_handle), HYPRE_MEMORY_HOST);
   hypre_TFree(comm_handle, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixCreateAssumedPartition -
 * Each proc gets it own range. Then
 * each needs to reconcile its actual range with its assumed
 * range - the result is essentila a partition of its assumed range -
 * this is the assumed partition.
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRBlockMatrixCreateAssumedPartition( hypre_ParCSRBlockMatrix *matrix)
{
   HYPRE_BigInt global_num_cols;
   HYPRE_Int myid;
   HYPRE_BigInt  col_start = 0, col_end = 0;

   MPI_Comm   comm;

   hypre_IJAssumedPart *apart;

   global_num_cols = hypre_ParCSRBlockMatrixGlobalNumCols(matrix);
   comm = hypre_ParCSRBlockMatrixComm(matrix);

   /* find out my actualy range of rows and columns */
   col_start =  hypre_ParCSRBlockMatrixFirstColDiag(matrix);
   col_end =  hypre_ParCSRBlockMatrixLastColDiag(matrix);

   hypre_MPI_Comm_rank(comm, &myid );

   /* allocate space */
   apart = hypre_CTAlloc(hypre_IJAssumedPart,  1, HYPRE_MEMORY_HOST);

   /* get my assumed partitioning  - we want partitioning of the vector that the
      matrix multiplies - so we use the col start and end */
   hypre_GetAssumedPartitionRowRange(comm, myid, 0, global_num_cols,
                                     &(apart->row_start), &(apart->row_end));

   /*allocate some space for the partition of the assumed partition */
   apart->length = 0;
   /*room for 10 owners of the assumed partition*/
   apart->storage_length = 10; /*need to be >=1 */
   apart->proc_list = hypre_TAlloc(HYPRE_Int,  apart->storage_length, HYPRE_MEMORY_HOST);
   apart->row_start_list =   hypre_TAlloc(HYPRE_BigInt,  apart->storage_length, HYPRE_MEMORY_HOST);
   apart->row_end_list =   hypre_TAlloc(HYPRE_BigInt,  apart->storage_length, HYPRE_MEMORY_HOST);

   /* now we want to reconcile our actual partition with the assumed partition */
   hypre_LocateAssumedPartition(comm, col_start, col_end,
                                0, global_num_cols, apart, myid);

   /* this partition will be saved in the matrix data structure until the matrix
    * is destroyed */
   hypre_ParCSRBlockMatrixAssumedPartition(matrix) = apart;

   return hypre_error_flag;

}

/*--------------------------------------------------------------------
 * hypre_ParCSRMatrixDestroyAssumedPartition
 *--------------------------------------------------------------------*/
HYPRE_Int
hypre_ParCSRBlockMatrixDestroyAssumedPartition( hypre_ParCSRBlockMatrix *matrix )
{

   hypre_IJAssumedPart *apart;

   apart = hypre_ParCSRMatrixAssumedPartition(matrix);

   if (apart->storage_length > 0)
   {
      hypre_TFree(apart->proc_list, HYPRE_MEMORY_HOST);
      hypre_TFree(apart->row_start_list, HYPRE_MEMORY_HOST);
      hypre_TFree(apart->row_end_list, HYPRE_MEMORY_HOST);
      hypre_TFree(apart->sort_index, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(apart, HYPRE_MEMORY_HOST);

   return (0);
}

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvecOutOfPlaceDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixMatvecOutOfPlaceDevice( HYPRE_Complex       alpha,
                                          hypre_ParCSRMatrix *A,
                                          hypre_ParVector    *x,
                                          HYPRE_Complex       beta,
                                          hypre_ParVector    *b,
                                          hypre_ParVector    *y )
{
   hypre_GpuProfilingPushRange("Matvec");

   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int               *d_send_map_elmts;
   HYPRE_Int                send_map_num_elmts;

   hypre_CSRMatrix         *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix         *offd = hypre_ParCSRMatrixOffd(A);

   hypre_Vector            *x_local  = hypre_ParVectorLocalVector(x);
   hypre_Vector            *b_local  = hypre_ParVectorLocalVector(b);
   hypre_Vector            *y_local  = hypre_ParVectorLocalVector(y);
   hypre_Vector            *x_tmp;

   HYPRE_BigInt             num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt             num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt             x_size   = hypre_ParVectorGlobalSize(x);
   HYPRE_BigInt             b_size   = hypre_ParVectorGlobalSize(b);
   HYPRE_BigInt             y_size   = hypre_ParVectorGlobalSize(y);

   HYPRE_Int                num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_Int                num_recvs, num_sends;
   HYPRE_Int                ierr = 0;

   HYPRE_Int                idxstride    = hypre_VectorIndexStride(x_local);
   HYPRE_Int                num_vectors  = hypre_VectorNumVectors(x_local);
   HYPRE_Complex           *x_local_data = hypre_VectorData(x_local);
   HYPRE_Complex           *x_tmp_data;
   HYPRE_Complex           *x_buf_data;

   HYPRE_Int                sync_stream;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_GetSyncCudaCompute(&sync_stream);
   hypre_SetSyncCudaCompute(0);

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  ParMatvec returns ierr = 11 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 12 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 13 if both are true.
    *
    *  Because temporary vectors are often used in ParMatvec, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
   hypre_assert( idxstride > 0 );

   if (num_cols != x_size)
   {
      ierr = 11;
   }

   if (num_rows != y_size || num_rows != b_size)
   {
      ierr = 12;
   }

   if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
   {
      ierr = 13;
   }

   hypre_assert( hypre_VectorNumVectors(b_local) == num_vectors );
   hypre_assert( hypre_VectorNumVectors(y_local) == num_vectors );

   if (num_vectors == 1)
   {
      x_tmp = hypre_SeqVectorCreate(num_cols_offd);
   }
   else
   {
      hypre_assert(num_vectors > 1);
      x_tmp = hypre_SeqMultiVectorCreate(num_cols_offd, num_vectors);
      hypre_VectorMultiVecStorageMethod(x_tmp) = 1;
   }

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* Update send_map_starts, send_map_elmts, and recv_vec_starts when doing
      sparse matrix/multivector product  */
   hypre_ParCSRCommPkgUpdateVecStarts(comm_pkg, num_vectors,
                                      hypre_VectorVectorStride(hypre_ParVectorLocalVector(x)),
                                      hypre_VectorIndexStride(hypre_ParVectorLocalVector(x)));

   /* Copy send_map_elmts to the device if not already there */
   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   /* Get information from the communication package*/
   num_recvs          = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   num_sends          = hypre_ParCSRCommPkgNumSends(comm_pkg);
   d_send_map_elmts   = hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);
   send_map_num_elmts = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /* Sanity checks */
   hypre_assert( num_cols_offd * num_vectors ==
                 hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) );
   hypre_assert( hypre_ParCSRCommPkgRecvVecStart(comm_pkg, 0) == 0 );
   hypre_assert( hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0) == 0 );

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

   /*---------------------------------------------------------------------
    * Allocate or reuse receive data buffer for x_tmp
    *--------------------------------------------------------------------*/

   if (!hypre_ParCSRCommPkgTmpData(comm_pkg))
   {
      hypre_ParCSRCommPkgTmpData(comm_pkg) = hypre_TAlloc(HYPRE_Complex,
                                                          num_cols_offd * num_vectors,
                                                          HYPRE_MEMORY_DEVICE);
   }
   hypre_VectorData(x_tmp) = x_tmp_data = hypre_ParCSRCommPkgTmpData(comm_pkg);
   hypre_SeqVectorSetDataOwner(x_tmp, 0);
   hypre_SeqVectorInitialize_v2(x_tmp, HYPRE_MEMORY_DEVICE);

   /*---------------------------------------------------------------------
    * Allocate or reuse send data buffer
    *--------------------------------------------------------------------*/

   if (!hypre_ParCSRCommPkgBufData(comm_pkg))
   {
      hypre_ParCSRCommPkgBufData(comm_pkg) = hypre_TAlloc(HYPRE_Complex,
                                                          send_map_num_elmts,
                                                          HYPRE_MEMORY_DEVICE);
   }
   x_buf_data = hypre_ParCSRCommPkgBufData(comm_pkg);

   /* The assert is because this code has been tested for column-wise vector storage only. */
   hypre_assert(idxstride == 1);

   //hypre_SeqVectorPrefetch(x_local, HYPRE_MEMORY_DEVICE);

   /*---------------------------------------------------------------------
    * Pack send data
    *--------------------------------------------------------------------*/

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int  i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(x_buf_data, x_local_data, d_send_map_elmts)
   for (i = 0; i < send_map_num_elmts; i++)
   {
      x_buf_data[i] = x_local_data[d_send_map_elmts[i]];
   }
#else
#if defined(HYPRE_USING_SYCL)
   auto permuted_source = oneapi::dpl::make_permutation_iterator(x_local_data,
                                                                 d_send_map_elmts);
   HYPRE_ONEDPL_CALL( std::copy,
                      permuted_source,
                      permuted_source + send_map_num_elmts,
                      x_buf_data );
#else
   HYPRE_THRUST_CALL( gather,
                      d_send_map_elmts,
                      d_send_map_elmts + send_map_num_elmts,
                      x_local_data,
                      x_buf_data );
#endif
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure x_buf_data is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }
#endif

   /* when using GPUs, start local matvec first in order to overlap with communication */
   hypre_CSRMatrixMatvecOutOfPlace(alpha, diag, x_local, beta, b_local, y_local, 0);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

   /* Non-blocking communication starts */
   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg,
                                                 HYPRE_MEMORY_DEVICE, x_buf_data,
                                                 HYPRE_MEMORY_DEVICE, x_tmp_data);

   /* Non-blocking communication ends */
   hypre_ParCSRCommHandleDestroy(comm_handle);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif

   /* computation offd part */
   if (num_cols_offd)
   {
      hypre_CSRMatrixMatvec(alpha, offd, x_tmp, 1.0, y_local);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

   /*---------------------------------------------------------------------
    * Free memory
    *--------------------------------------------------------------------*/
   hypre_SeqVectorDestroy(x_tmp);

   /*---------------------------------------------------------------------
    * Synchronize calls
    *--------------------------------------------------------------------*/
   hypre_SetSyncCudaCompute(sync_stream);
   hypre_SyncComputeStream(hypre_handle());

   /*---------------------------------------------------------------------
    * Performance profiling
    *--------------------------------------------------------------------*/
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
#endif

   HYPRE_ANNOTATE_FUNC_END;

   hypre_GpuProfilingPopRange();

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvecTDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixMatvecTDevice( HYPRE_Complex       alpha,
                                 hypre_ParCSRMatrix *A,
                                 hypre_ParVector    *x,
                                 HYPRE_Complex       beta,
                                 hypre_ParVector    *y )
{
   hypre_GpuProfilingPushRange("MatvecT");

   hypre_ParCSRCommPkg     *comm_pkg      = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int                send_map_num_elmts;

   hypre_CSRMatrix         *diag          = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix         *offd          = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix         *diagT         = hypre_ParCSRMatrixDiagT(A);
   hypre_CSRMatrix         *offdT         = hypre_ParCSRMatrixOffdT(A);

   hypre_Vector            *x_local       = hypre_ParVectorLocalVector(x);
   hypre_Vector            *y_local       = hypre_ParVectorLocalVector(y);
   hypre_Vector            *y_tmp;

   HYPRE_Int                num_cols_diag = hypre_CSRMatrixNumCols(diag);
   HYPRE_Int                num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_BigInt             num_rows      = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt             num_cols      = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt             x_size        = hypre_ParVectorGlobalSize(x);
   HYPRE_BigInt             y_size        = hypre_ParVectorGlobalSize(y);

   HYPRE_Complex           *y_tmp_data;
   HYPRE_Complex           *y_buf_data;
   HYPRE_Complex           *y_local_data  = hypre_VectorData(y_local);
   HYPRE_Int                idxstride     = hypre_VectorIndexStride(y_local);
   HYPRE_Int                num_vectors   = hypre_VectorNumVectors(y_local);
   HYPRE_Int                num_sends;
   HYPRE_Int                num_recvs;
   HYPRE_Int                ierr = 0;
   HYPRE_Int                sync_stream;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_GetSyncCudaCompute(&sync_stream);
   hypre_SetSyncCudaCompute(0);

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
   if (num_rows != x_size)
   {
      ierr = 1;
   }

   if (num_cols != y_size)
   {
      ierr = 2;
   }

   if (num_rows != x_size && num_cols != y_size)
   {
      ierr = 3;
   }

   hypre_assert( hypre_VectorNumVectors(x_local) == num_vectors );
   hypre_assert( hypre_VectorNumVectors(y_local) == num_vectors );

   if (num_vectors == 1)
   {
      y_tmp = hypre_SeqVectorCreate(num_cols_offd);
   }
   else
   {
      hypre_assert(num_vectors > 1);
      y_tmp = hypre_SeqMultiVectorCreate(num_cols_offd, num_vectors);
      hypre_VectorMultiVecStorageMethod(y_tmp) = 1;
   }

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* Update send_map_starts, send_map_elmts, and recv_vec_starts for SpMV with multivecs */
   hypre_ParCSRCommPkgUpdateVecStarts(comm_pkg, num_vectors,
                                      hypre_VectorVectorStride(hypre_ParVectorLocalVector(y)),
                                      hypre_VectorIndexStride(hypre_ParVectorLocalVector(y)));

   /* Update send_map_elmts on device */
   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   /* Get information from the communication package*/
   num_recvs          = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   num_sends          = hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_map_num_elmts = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /* Sanity checks */
   hypre_assert( num_cols_offd * num_vectors ==
                 hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) );
   hypre_assert( hypre_ParCSRCommPkgRecvVecStart(comm_pkg, 0) == 0 );
   hypre_assert( hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0) == 0 );

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

   /*---------------------------------------------------------------------
    * Allocate or reuse send data buffer for y_tmp
    *--------------------------------------------------------------------*/

   if (!hypre_ParCSRCommPkgTmpData(comm_pkg))
   {
      hypre_ParCSRCommPkgTmpData(comm_pkg) = hypre_TAlloc(HYPRE_Complex,
                                                          num_cols_offd * num_vectors,
                                                          HYPRE_MEMORY_DEVICE);
   }
   hypre_VectorData(y_tmp) = y_tmp_data = hypre_ParCSRCommPkgTmpData(comm_pkg);
   hypre_SeqVectorSetDataOwner(y_tmp, 0);
   hypre_SeqVectorInitialize_v2(y_tmp, HYPRE_MEMORY_DEVICE);

   /*---------------------------------------------------------------------
    * Allocate receive data buffer
    *--------------------------------------------------------------------*/

   if (!hypre_ParCSRCommPkgBufData(comm_pkg))
   {
      hypre_ParCSRCommPkgBufData(comm_pkg) = hypre_TAlloc(HYPRE_Complex,
                                                          send_map_num_elmts,
                                                          HYPRE_MEMORY_DEVICE);
   }
   y_buf_data = hypre_ParCSRCommPkgBufData(comm_pkg);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
#endif

   /* Compute y_tmp = offd^T * x_local */
   if (num_cols_offd)
   {
      if (offdT)
      {
         // offdT is optional. Used only if it's present
         hypre_CSRMatrixMatvec(alpha, offdT, x_local, 0.0, y_tmp);
      }
      else
      {
         hypre_CSRMatrixMatvecT(alpha, offd, x_local, 0.0, y_tmp);
      }
   }

   /* RL: make sure y_tmp is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }

   /* when using GPUs, start local matvec first in order to overlap with communication */
   if (diagT)
   {
      // diagT is optional. Used only if it's present.
      hypre_CSRMatrixMatvec(alpha, diagT, x_local, beta, y_local);
   }
   else
   {
      hypre_CSRMatrixMatvecT(alpha, diag, x_local, beta, y_local);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

   /* Non-blocking communication starts */
   comm_handle = hypre_ParCSRCommHandleCreate_v2(2, comm_pkg,
                                                 HYPRE_MEMORY_DEVICE, y_tmp_data,
                                                 HYPRE_MEMORY_DEVICE, y_buf_data );

   /* Non-blocking communication ends */
   hypre_ParCSRCommHandleDestroy(comm_handle);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK]   -= hypre_MPI_Wtime();
#endif

   /* The assert is here because this code has been tested for column-wise vector storage only. */
   hypre_assert( idxstride == 1 );

   /*---------------------------------------------------------------------
    * Unpack receive data
    *--------------------------------------------------------------------*/

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int  *d_send_map_elmts = hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);
   HYPRE_Int   i, j;

   for (i = 0; i < num_sends; i++)
   {
      HYPRE_Int  start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      HYPRE_Int  end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);

      #pragma omp target teams distribute parallel for private(j) is_device_ptr(y_buf_data, y_local_data, d_send_map_elmts)
      for (j = start; j < end; j++)
      {
         y_local_data[d_send_map_elmts[j]] += y_buf_data[j];
      }
   }
#else
   /* Use SpMV to unpack data */
   hypre_ParCSRMatrixMatvecT_unpack(comm_pkg, num_cols_diag, y_buf_data, y_local_data);
#endif

   /*---------------------------------------------------------------------
    * Free memory
    *--------------------------------------------------------------------*/

   hypre_SeqVectorDestroy(y_tmp);

   /*---------------------------------------------------------------------
    * Synchronize when using GPUs
    *--------------------------------------------------------------------*/

#if defined(HYPRE_USING_GPU)
   hypre_SetSyncCudaCompute(sync_stream);
   hypre_SyncComputeStream(hypre_handle());
#endif

   /*---------------------------------------------------------------------
    * Performance profiling
    *--------------------------------------------------------------------*/

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
#endif

   HYPRE_ANNOTATE_FUNC_END;

   hypre_GpuProfilingPopRange();

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvecT_unpack
 *
 * Computes on the device:
 *
 *   local_data[send_map_elmts] += recv_data
 *
 * with hypre's internal SpMV.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixMatvecT_unpack( hypre_ParCSRCommPkg *comm_pkg,
                                  HYPRE_Int            num_cols,
                                  HYPRE_Complex       *recv_data,
                                  HYPRE_Complex       *local_data )
{
   /* Input variables */
   hypre_CSRMatrix  *matrix_E       = hypre_ParCSRCommPkgMatrixE(comm_pkg);
   HYPRE_Int         num_sends      = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int         num_elements   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   HYPRE_Int         num_components = hypre_ParCSRCommPkgNumComponents(comm_pkg);

   /* Local variables */
   hypre_Vector      vec_x;
   hypre_Vector      vec_y;
   HYPRE_Int         trans = 0;
   HYPRE_Int         fill  = 0;
   HYPRE_Complex     alpha = 1.0;
   HYPRE_Complex     beta  = 1.0;

   if (num_elements == 0)
   {
      return hypre_error_flag;
   }

   /* Create matrix E if it not exists */
   if (!matrix_E)
   {
      hypre_ParCSRCommPkgCreateMatrixE(comm_pkg, num_cols);
      matrix_E = hypre_ParCSRCommPkgMatrixE(comm_pkg);
   }

   /* Set vector x */
   hypre_VectorData(&vec_x)                  = recv_data;
   hypre_VectorOwnsData(&vec_x)              = 0;
   hypre_VectorSize(&vec_x)                  = num_elements / num_components;
   hypre_VectorVectorStride(&vec_x)          = 1;
   hypre_VectorIndexStride(&vec_x)           = num_components;
   hypre_VectorNumVectors(&vec_x)            = num_components;
   hypre_VectorMultiVecStorageMethod(&vec_x) = 1;

   /* Set vector y */
   hypre_VectorData(&vec_y)                  = local_data;
   hypre_VectorOwnsData(&vec_y)              = 0;
   hypre_VectorSize(&vec_y)                  = num_cols;
   hypre_VectorVectorStride(&vec_y)          = num_cols;
   hypre_VectorIndexStride(&vec_y)           = 1;
   hypre_VectorNumVectors(&vec_y)            = num_components;
   hypre_VectorMultiVecStorageMethod(&vec_y) = 0;

   /* WM: todo - port hypre_CSRMatrixSpMVDevice() to sycl */
#if defined(HYPRE_USING_SYCL)
   HYPRE_Complex *data = hypre_TAlloc(HYPRE_Complex,
                                      hypre_CSRMatrixNumNonzeros(matrix_E),
                                      HYPRE_MEMORY_DEVICE);
   hypreDevice_ComplexFilln(data, hypre_CSRMatrixNumNonzeros(matrix_E), 1.0);
   hypre_CSRMatrixData(matrix_E) = data;

   hypre_CSRMatrixMatvecDevice(trans, alpha, matrix_E, &vec_x, beta, &vec_y, &vec_y, 0);
#else
   /* Compute y += E*x */
   hypre_CSRMatrixSpMVDevice(trans, alpha, matrix_E, &vec_x, beta, &vec_y, fill);
#endif

   return hypre_error_flag;
}

#endif /* #if defined(HYPRE_USING_GPU) */

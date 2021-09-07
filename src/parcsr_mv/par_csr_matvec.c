/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.hpp" //RL: TODO par_csr_matvec_device.c, include cuda there

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/
// y = alpha*A*x + beta*b
HYPRE_Int
hypre_ParCSRMatrixMatvecOutOfPlace( HYPRE_Complex       alpha,
                                    hypre_ParCSRMatrix *A,
                                    hypre_ParVector    *x,
                                    HYPRE_Complex       beta,
                                    hypre_ParVector    *b,
                                    hypre_ParVector    *y )
{
   hypre_ParCSRCommHandle **comm_handle;
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);

   hypre_Vector *x_local  = hypre_ParVectorLocalVector(x);
   hypre_Vector *b_local  = hypre_ParVectorLocalVector(b);
   hypre_Vector *y_local  = hypre_ParVectorLocalVector(y);
   hypre_Vector *x_tmp;

   HYPRE_BigInt num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt x_size   = hypre_ParVectorGlobalSize(x);
   HYPRE_BigInt b_size   = hypre_ParVectorGlobalSize(b);
   HYPRE_BigInt y_size   = hypre_ParVectorGlobalSize(y);

   HYPRE_Int num_vectors   = hypre_VectorNumVectors(x_local);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_Int ierr = 0;
   HYPRE_Int num_sends, jv;

   HYPRE_Int vecstride = hypre_VectorVectorStride( x_local );
   HYPRE_Int idxstride = hypre_VectorIndexStride( x_local );

   HYPRE_Complex *x_tmp_data, **x_buf_data;
   HYPRE_Complex *x_local_data = hypre_VectorData(x_local);

#if defined(HYPRE_USING_GPU)
   HYPRE_Int sync_stream;
   hypre_GetSyncCudaCompute(&sync_stream);
   hypre_SetSyncCudaCompute(0);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;

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
   hypre_assert( idxstride>0 );

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

   if ( num_vectors == 1 )
   {
      x_tmp = hypre_SeqVectorCreate( num_cols_offd );
   }
   else
   {
      hypre_assert( num_vectors > 1 );
      x_tmp = hypre_SeqMultiVectorCreate( num_cols_offd, num_vectors );
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

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   hypre_assert( num_cols_offd == hypre_ParCSRCommPkgRecvVecStart(comm_pkg, hypre_ParCSRCommPkgNumRecvs(comm_pkg)) );
   hypre_assert( hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0) == 0 );

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

   HYPRE_Int use_persistent_comm = 0;

#ifdef HYPRE_USING_PERSISTENT_COMM
   use_persistent_comm = num_vectors == 1;
   // JSP TODO: we can use persistent communication for multi-vectors,
   // but then we need different communication handles for different
   // num_vectors.
   hypre_ParCSRPersistentCommHandle *persistent_comm_handle;
#endif

   if (use_persistent_comm)
   {
#ifdef HYPRE_USING_PERSISTENT_COMM
      persistent_comm_handle = hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);
#endif
   }
   else
   {
      comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle*, num_vectors, HYPRE_MEMORY_HOST);
   }

   /* x_tmp */
#if defined(HYPRE_USING_GPU)
   /* for GPU and single vector, alloc persistent memory for x_tmp (in comm_pkg) and reuse */
   if (num_vectors == 1)
   {
      if (!hypre_ParCSRCommPkgTmpData(comm_pkg))
      {
#if 1
         hypre_ParCSRCommPkgTmpData(comm_pkg) = hypre_TAlloc(HYPRE_Complex, num_cols_offd, HYPRE_MEMORY_DEVICE);
#else
         hypre_ParCSRCommPkgTmpData(comm_pkg) = _hypre_TAlloc(HYPRE_Complex, num_cols_offd, hypre_MEMORY_DEVICE);
#endif
      }
      hypre_VectorData(x_tmp) = hypre_ParCSRCommPkgTmpData(comm_pkg);
      hypre_SeqVectorSetDataOwner(x_tmp, 0);
   }
#else
   if (use_persistent_comm)
   {
#ifdef HYPRE_USING_PERSISTENT_COMM
      hypre_VectorData(x_tmp) = (HYPRE_Complex *) hypre_ParCSRCommHandleRecvDataBuffer(persistent_comm_handle);
      hypre_SeqVectorSetDataOwner(x_tmp, 0);
#endif
   }
#endif

   hypre_SeqVectorInitialize_v2(x_tmp, HYPRE_MEMORY_DEVICE);
   x_tmp_data = hypre_VectorData(x_tmp);

   /* x_buff_data */
   x_buf_data = hypre_CTAlloc(HYPRE_Complex*, num_vectors, HYPRE_MEMORY_HOST);

   for (jv = 0; jv < num_vectors; ++jv)
   {
#if defined(HYPRE_USING_GPU)
      if (jv == 0)
      {
         if (!hypre_ParCSRCommPkgBufData(comm_pkg))
         {
#if 1
            hypre_ParCSRCommPkgBufData(comm_pkg) = hypre_TAlloc(HYPRE_Complex,
                                                                hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                                                HYPRE_MEMORY_DEVICE);
#else
            hypre_ParCSRCommPkgBufData(comm_pkg) = _hypre_TAlloc(HYPRE_Complex,
                                                                 hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                                                 hypre_MEMORY_DEVICE);
#endif
         }
         x_buf_data[0] = hypre_ParCSRCommPkgBufData(comm_pkg);
         continue;
      }
#endif
      if (use_persistent_comm)
      {
#ifdef HYPRE_USING_PERSISTENT_COMM
         x_buf_data[0] = (HYPRE_Complex *) hypre_ParCSRCommHandleSendDataBuffer(persistent_comm_handle);
         continue;
#endif
      }

      x_buf_data[jv] = hypre_TAlloc(HYPRE_Complex,
                                    hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                    HYPRE_MEMORY_DEVICE);
   }

   /* The assert is because the following loop only works for 'column'
      storage of a multivector. This needs to be fixed to work more generally,
      at least for 'row' storage. This in turn, means either change CommPkg so
      num_sends is no.zones*no.vectors (not no.zones) or, less dangerously, put
      a stride in the logic of CommHandleCreate (stride either from a new arg or
      a new variable inside CommPkg).  Or put the num_vector iteration inside
      CommHandleCreate (perhaps a new multivector variant of it).
   */

   hypre_assert( idxstride == 1 );

   //hypre_SeqVectorPrefetch(x_local, HYPRE_MEMORY_DEVICE);

   /* send_map_elmts on device */
   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   for (jv = 0; jv < num_vectors; ++jv)
   {
      HYPRE_Complex *send_data = (HYPRE_Complex *) x_buf_data[jv];
      HYPRE_Complex *locl_data = x_local_data + jv * vecstride;

      /* if on device, no need to Sync: send_data is on device memory */
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      /* pack send data on device */
      HYPRE_THRUST_CALL( gather,
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) +
                         hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                         locl_data,
                         send_data );
#elif defined(HYPRE_USING_DEVICE_OPENMP)
      /* pack send data on device */
      HYPRE_Int i;
      HYPRE_Int *device_send_map_elmts = hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);
      HYPRE_Int start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
      HYPRE_Int end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
#pragma omp target teams distribute parallel for private(i) is_device_ptr(send_data, locl_data, device_send_map_elmts)
      for (i = start; i < end; i++)
      {
         send_data[i] = locl_data[device_send_map_elmts[i]];
      }
#else
      HYPRE_Int i;
      /* pack send data on host */
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
      for (i = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
           i < hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
           i ++)
      {
         send_data[i] = locl_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)];
      }
#endif
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK]   += hypre_MPI_Wtime();
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

   /* nonblocking communication starts */
   if (use_persistent_comm)
   {
#ifdef HYPRE_USING_PERSISTENT_COMM
      hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle, HYPRE_MEMORY_DEVICE, x_buf_data[0]);
#endif
   }
   else
   {
      for ( jv = 0; jv < num_vectors; ++jv )
      {
         comm_handle[jv] = hypre_ParCSRCommHandleCreate_v2( 1, comm_pkg, HYPRE_MEMORY_DEVICE, x_buf_data[jv],
                                                            HYPRE_MEMORY_DEVICE, &x_tmp_data[jv*num_cols_offd] );
      }
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif

   /* overlapped local computation */
   hypre_CSRMatrixMatvecOutOfPlace( alpha, diag, x_local, beta, b_local, y_local, 0 );

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

   /* nonblocking communication ends */
   if (use_persistent_comm)
   {
#ifdef HYPRE_USING_PERSISTENT_COMM
      hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle, HYPRE_MEMORY_DEVICE, x_tmp_data);
#endif
   }
   else
   {
      for ( jv = 0; jv < num_vectors; ++jv )
      {
         hypre_ParCSRCommHandleDestroy(comm_handle[jv]);
         comm_handle[jv] = NULL;
      }
      hypre_TFree(comm_handle, HYPRE_MEMORY_HOST);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif

   /* computation offd part */
   if (num_cols_offd)
   {
      hypre_CSRMatrixMatvec( alpha, offd, x_tmp, 1.0, y_local );
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

   hypre_SeqVectorDestroy(x_tmp);  x_tmp = NULL;

   if (!use_persistent_comm)
   {
      for ( jv = 0; jv < num_vectors; ++jv )
      {
#if defined(HYPRE_USING_GPU)
         if (jv == 0)
         {
            continue;
         }
#endif
         hypre_TFree(x_buf_data[jv], HYPRE_MEMORY_DEVICE);
      }
      hypre_TFree(x_buf_data, HYPRE_MEMORY_HOST);
   }

#if defined(HYPRE_USING_GPU)
   hypre_SetSyncCudaCompute(sync_stream);
   hypre_SyncCudaComputeStream(hypre_handle());
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
#endif

   HYPRE_ANNOTATE_FUNC_END;

   return ierr;
}

HYPRE_Int
hypre_ParCSRMatrixMatvec( HYPRE_Complex       alpha,
                          hypre_ParCSRMatrix *A,
                          hypre_ParVector    *x,
                          HYPRE_Complex       beta,
                          hypre_ParVector    *y )
{
   return hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, x, beta, y, y);
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixMatvecT( HYPRE_Complex       alpha,
                           hypre_ParCSRMatrix *A,
                           hypre_ParVector    *x,
                           HYPRE_Complex       beta,
                           hypre_ParVector    *y )
{
   hypre_ParCSRCommHandle **comm_handle;
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);

   hypre_CSRMatrix *diag  = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd  = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix *diagT = hypre_ParCSRMatrixDiagT(A);
   hypre_CSRMatrix *offdT = hypre_ParCSRMatrixOffdT(A);

   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   hypre_Vector *y_tmp;

   HYPRE_BigInt num_rows  = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt num_cols  = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt x_size    = hypre_ParVectorGlobalSize(x);
   HYPRE_BigInt y_size    = hypre_ParVectorGlobalSize(y);

   HYPRE_Int num_vectors   = hypre_VectorNumVectors(y_local);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_Int ierr = 0;
   HYPRE_Int num_sends, jv;

   HYPRE_Int vecstride     = hypre_VectorVectorStride(y_local);
   HYPRE_Int idxstride     = hypre_VectorIndexStride(y_local);

   HYPRE_Complex *y_tmp_data, **y_buf_data;
   HYPRE_Complex *y_local_data = hypre_VectorData(y_local);

#if defined(HYPRE_USING_GPU)
   HYPRE_Int sync_stream;
   hypre_GetSyncCudaCompute(&sync_stream);
   hypre_SetSyncCudaCompute(0);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;

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

   if ( num_vectors == 1 )
   {
      y_tmp = hypre_SeqVectorCreate(num_cols_offd);
   }
   else
   {
      hypre_assert( num_vectors > 1 );
      y_tmp = hypre_SeqMultiVectorCreate(num_cols_offd, num_vectors);
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

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   hypre_assert( num_cols_offd == hypre_ParCSRCommPkgRecvVecStart(comm_pkg, hypre_ParCSRCommPkgNumRecvs(comm_pkg)) );
   hypre_assert( hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0) == 0 );

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

   HYPRE_Int use_persistent_comm = 0;

#ifdef HYPRE_USING_PERSISTENT_COMM
   use_persistent_comm = num_vectors == 1;
   // JSP TODO: we can use persistent communication for multi-vectors,
   // but then we need different communication handles for different
   // num_vectors.
   hypre_ParCSRPersistentCommHandle *persistent_comm_handle;
#endif

   if (use_persistent_comm)
   {
#ifdef HYPRE_USING_PERSISTENT_COMM
      persistent_comm_handle = hypre_ParCSRCommPkgGetPersistentCommHandle(2, comm_pkg);
#endif
   }
   else
   {
      comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle*, num_vectors, HYPRE_MEMORY_HOST);
   }

   /* y_tmp */
#if defined(HYPRE_USING_GPU)
   /* for GPU and single vector, alloc persistent memory for y_tmp (in comm_pkg) and reuse */
   if (num_vectors == 1)
   {
      if (!hypre_ParCSRCommPkgTmpData(comm_pkg))
      {
#if 1
         hypre_ParCSRCommPkgTmpData(comm_pkg) = hypre_TAlloc(HYPRE_Complex, num_cols_offd, HYPRE_MEMORY_DEVICE);
#else
         hypre_ParCSRCommPkgTmpData(comm_pkg) = _hypre_TAlloc(HYPRE_Complex, num_cols_offd, hypre_MEMORY_DEVICE);
#endif
      }
      hypre_VectorData(y_tmp) = hypre_ParCSRCommPkgTmpData(comm_pkg);
      hypre_SeqVectorSetDataOwner(y_tmp, 0);
   }
#else
   if (use_persistent_comm)
   {
#ifdef HYPRE_USING_PERSISTENT_COMM
      hypre_VectorData(y_tmp) = (HYPRE_Complex *) hypre_ParCSRCommHandleSendDataBuffer(persistent_comm_handle);
      hypre_SeqVectorSetDataOwner(y_tmp, 0);
#endif
   }
#endif

   hypre_SeqVectorInitialize_v2(y_tmp, HYPRE_MEMORY_DEVICE);
   y_tmp_data = hypre_VectorData(y_tmp);

   /* y_buf_data */
   y_buf_data = hypre_CTAlloc(HYPRE_Complex*, num_vectors, HYPRE_MEMORY_HOST);

   for (jv = 0; jv < num_vectors; ++jv)
   {
#if defined(HYPRE_USING_GPU)
      if (jv == 0)
      {
         if (!hypre_ParCSRCommPkgBufData(comm_pkg))
         {
#if 1
            hypre_ParCSRCommPkgBufData(comm_pkg) = hypre_TAlloc(HYPRE_Complex,
                                                                hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                                                HYPRE_MEMORY_DEVICE);
#else
            hypre_ParCSRCommPkgBufData(comm_pkg) = _hypre_TAlloc(HYPRE_Complex,
                                                                 hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                                                 hypre_MEMORY_DEVICE);
#endif
         }
         y_buf_data[0] = hypre_ParCSRCommPkgBufData(comm_pkg);
         continue;
      }
#endif
      if (use_persistent_comm)
      {
#ifdef HYPRE_USING_PERSISTENT_COMM
         y_buf_data[0] = (HYPRE_Complex *) hypre_ParCSRCommHandleRecvDataBuffer(persistent_comm_handle);
         continue;
#endif
      }

      y_buf_data[jv] = hypre_TAlloc(HYPRE_Complex,
                                    hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                    HYPRE_MEMORY_DEVICE);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
#endif

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

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

   if (use_persistent_comm)
   {
#ifdef HYPRE_USING_PERSISTENT_COMM
      hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle, HYPRE_MEMORY_DEVICE, y_tmp_data);
#endif
   }
   else
   {
      for ( jv = 0; jv < num_vectors; ++jv )
      {
         /* this is where we assume multivectors are 'column' storage */
         comm_handle[jv] = hypre_ParCSRCommHandleCreate_v2( 2, comm_pkg, HYPRE_MEMORY_DEVICE, &y_tmp_data[jv*num_cols_offd],
                                                            HYPRE_MEMORY_DEVICE, y_buf_data[jv] );
      }
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif

   /* overlapped local computation */
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

   /* nonblocking communication ends */
   if (use_persistent_comm)
   {
#ifdef HYPRE_USING_PERSISTENT_COMM
      hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle, HYPRE_MEMORY_DEVICE, y_buf_data[0]);
#endif
   }
   else
   {
      for ( jv = 0; jv < num_vectors; ++jv )
      {
         hypre_ParCSRCommHandleDestroy(comm_handle[jv]);
         comm_handle[jv] = NULL;
      }
      hypre_TFree(comm_handle, HYPRE_MEMORY_HOST);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

   /* The assert is because the following loop only works for 'column'
      storage of a multivector. This needs to be fixed to work more generally,
      at least for 'row' storage. This in turn, means either change CommPkg so
      num_sends is no.zones*no.vectors (not no.zones) or, less dangerously, put
      a stride in the logic of CommHandleCreate (stride either from a new arg or
      a new variable inside CommPkg).  Or put the num_vector iteration inside
      CommHandleCreate (perhaps a new multivector variant of it).
   */

   hypre_assert( idxstride == 1 );

   /* send_map_elmts on device */
   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   for (jv = 0; jv < num_vectors; ++jv)
   {
      HYPRE_Complex *recv_data = (HYPRE_Complex *) y_buf_data[jv];
      HYPRE_Complex *locl_data = y_local_data + jv * vecstride;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      /* unpack recv data on device */
      if (!hypre_ParCSRCommPkgWorkSpace(comm_pkg))
      {
         hypre_ParCSRCommPkgWorkSpace(comm_pkg) =
            hypre_TAlloc( char,
                          (2*sizeof(HYPRE_Int)+sizeof(HYPRE_Real)) * hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                          HYPRE_MEMORY_DEVICE );
      }
      hypreDevice_GenScatterAdd(locl_data,
                                hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                                recv_data,
                                hypre_ParCSRCommPkgWorkSpace(comm_pkg));
#elif defined(HYPRE_USING_DEVICE_OPENMP)
      HYPRE_Int i, j;
      /* unpack recv data on device */
      for (i = 0; i < num_sends; i++)
      {
         HYPRE_Int *device_send_map_elmts = hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);
         HYPRE_Int start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         HYPRE_Int end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
#pragma omp target teams distribute parallel for private(j) is_device_ptr(recv_data, locl_data, device_send_map_elmts)
         for (j = start; j < end; j++)
         {
            locl_data[device_send_map_elmts[j]] += recv_data[j];
         }
      }
#else
      HYPRE_Int i;
      /* unpack recv data on host, TODO OMP? */
      for (i = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
           i < hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
           i ++)
      {
         locl_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)] += recv_data[i];
      }
#endif
   }

   hypre_SeqVectorDestroy(y_tmp);  y_tmp = NULL;

   if (!use_persistent_comm)
   {
      for ( jv = 0; jv < num_vectors; ++jv )
      {
#if defined(HYPRE_USING_GPU)
         if (jv == 0)
         {
            continue;
         }
#endif
         hypre_TFree(y_buf_data[jv], HYPRE_MEMORY_DEVICE);
      }
      hypre_TFree(y_buf_data, HYPRE_MEMORY_HOST);
   }

#if defined(HYPRE_USING_GPU)
   hypre_SetSyncCudaCompute(sync_stream);
   hypre_SyncCudaComputeStream(hypre_handle());
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
#endif

   HYPRE_ANNOTATE_FUNC_END;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvec_FF
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_ParCSRMatrixMatvec_FF( HYPRE_Complex       alpha,
                             hypre_ParCSRMatrix *A,
                             hypre_ParVector    *x,
                             HYPRE_Complex       beta,
                             hypre_ParVector    *y,
                             HYPRE_Int          *CF_marker,
                             HYPRE_Int           fpt )
{
   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix        *diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix        *offd   = hypre_ParCSRMatrixOffd(A);
   hypre_Vector           *x_local  = hypre_ParVectorLocalVector(x);
   hypre_Vector           *y_local  = hypre_ParVectorLocalVector(y);
   HYPRE_BigInt            num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt            num_cols = hypre_ParCSRMatrixGlobalNumCols(A);

   hypre_Vector      *x_tmp;
   HYPRE_BigInt       x_size = hypre_ParVectorGlobalSize(x);
   HYPRE_BigInt       y_size = hypre_ParVectorGlobalSize(y);
   HYPRE_Int          num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_Int          ierr = 0;
   HYPRE_Int          num_sends, i, j, index, start, num_procs;
   HYPRE_Int         *int_buf_data = NULL;
   HYPRE_Int         *CF_marker_offd = NULL;

   HYPRE_Complex     *x_tmp_data = NULL;
   HYPRE_Complex     *x_buf_data = NULL;
   HYPRE_Complex     *x_local_data = hypre_VectorData(x_local);
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

   hypre_MPI_Comm_size(comm,&num_procs);

   if (num_cols != x_size)
      ierr = 11;

   if (num_rows != y_size)
      ierr = 12;

   if (num_cols != x_size && num_rows != y_size)
      ierr = 13;

   if (num_procs > 1)
   {
      if (num_cols_offd)
      {
         x_tmp = hypre_SeqVectorCreate( num_cols_offd );
         hypre_SeqVectorInitialize(x_tmp);
         x_tmp_data = hypre_VectorData(x_tmp);
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

      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      if (num_sends)
         x_buf_data = hypre_CTAlloc(HYPRE_Complex,  hypre_ParCSRCommPkgSendMapStart
                                    (comm_pkg,  num_sends), HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            x_buf_data[index++]
               = x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
      comm_handle =
         hypre_ParCSRCommHandleCreate ( 1, comm_pkg, x_buf_data, x_tmp_data );
   }
   hypre_CSRMatrixMatvec_FF( alpha, diag, x_local, beta, y_local, CF_marker,
                             CF_marker, fpt);

   if (num_procs > 1)
   {
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      if (num_sends)
         int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart
                                      (comm_pkg,  num_sends), HYPRE_MEMORY_HOST);
      if (num_cols_offd) CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            int_buf_data[index++]
               = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
      comm_handle =
         hypre_ParCSRCommHandleCreate(11,comm_pkg,int_buf_data,CF_marker_offd );

      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      if (num_cols_offd) hypre_CSRMatrixMatvec_FF( alpha, offd, x_tmp, 1.0, y_local,
                                                   CF_marker, CF_marker_offd, fpt);

      hypre_SeqVectorDestroy(x_tmp);
      x_tmp = NULL;
      hypre_TFree(x_buf_data, HYPRE_MEMORY_HOST);
      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
      hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   }

   return ierr;
}

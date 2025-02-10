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

#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvecOutOfPlaceHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixMatvecOutOfPlaceHost( HYPRE_Complex       alpha,
                                        hypre_ParCSRMatrix *A,
                                        hypre_ParVector    *x,
                                        HYPRE_Complex       beta,
                                        hypre_ParVector    *b,
                                        hypre_ParVector    *y )
{
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);

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

   HYPRE_Int                i;
   HYPRE_Int                idxstride    = hypre_VectorIndexStride(x_local);
   HYPRE_Int                num_vectors  = hypre_VectorNumVectors(x_local);
   HYPRE_Complex           *x_local_data = hypre_VectorData(x_local);
   HYPRE_Complex           *x_tmp_data;
   HYPRE_Complex           *x_buf_data;

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

   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   hypre_assert( num_cols_offd * num_vectors ==
                 hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) );
   hypre_assert( hypre_ParCSRCommPkgRecvVecStart(comm_pkg, 0) == 0 );
   hypre_assert( hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0) == 0 );

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_ParCSRPersistentCommHandle *persistent_comm_handle =
      hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);
#else
   hypre_ParCSRCommHandle *comm_handle;
#endif

   /*---------------------------------------------------------------------
    * Allocate (during hypre_SeqVectorInitialize_v2) or retrieve
    * persistent receive data buffer for x_tmp (if persistent is enabled).
    *--------------------------------------------------------------------*/

#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_VectorData(x_tmp) = (HYPRE_Complex *)
                             hypre_ParCSRCommHandleRecvDataBuffer(persistent_comm_handle);
   hypre_SeqVectorSetDataOwner(x_tmp, 0);
#endif

   hypre_SeqVectorInitialize_v2(x_tmp, HYPRE_MEMORY_HOST);
   x_tmp_data = hypre_VectorData(x_tmp);

   /*---------------------------------------------------------------------
    * Allocate data send buffer
    *--------------------------------------------------------------------*/

#if defined(HYPRE_USING_PERSISTENT_COMM)
   x_buf_data = (HYPRE_Complex *) hypre_ParCSRCommHandleSendDataBuffer(persistent_comm_handle);

#else
   x_buf_data = hypre_TAlloc(HYPRE_Complex,
                             hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                             HYPRE_MEMORY_HOST);
#endif

   /* The assert is because this code has been tested for column-wise vector storage only. */
   hypre_assert(idxstride == 1);

   /*---------------------------------------------------------------------
    * Pack send data
    *--------------------------------------------------------------------*/

#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
   for (i = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
        i < hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
        i++)
   {
      x_buf_data[i] = x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK]   += hypre_MPI_Wtime();
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

   /* Non-blocking communication starts */
#ifdef HYPRE_USING_PERSISTENT_COMM
   hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle,
                                         HYPRE_MEMORY_HOST, x_buf_data);
#else
   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg,
                                                 HYPRE_MEMORY_HOST, x_buf_data,
                                                 HYPRE_MEMORY_HOST, x_tmp_data);
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif

   /* overlapped local computation */
   hypre_CSRMatrixMatvecOutOfPlace(alpha, diag, x_local, beta, b_local, y_local, 0);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

   /* Non-blocking communication ends */
#ifdef HYPRE_USING_PERSISTENT_COMM
   hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle, HYPRE_MEMORY_HOST, x_tmp_data);
#else
   hypre_ParCSRCommHandleDestroy(comm_handle);
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif

   /* computation offd part */
   if (num_cols_offd)
   {
      hypre_CSRMatrixMatvec(alpha, offd, x_tmp, 1.0, y_local);
   }

   /*---------------------------------------------------------------------
    * Free memory
    *--------------------------------------------------------------------*/
   hypre_SeqVectorDestroy(x_tmp);

#if !defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_TFree(x_buf_data, HYPRE_MEMORY_HOST);
#endif

   HYPRE_ANNOTATE_FUNC_END;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvecOutOfPlace
 *
 * Performs y <- alpha * A * x + beta * b
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixMatvecOutOfPlace( HYPRE_Complex       alpha,
                                    hypre_ParCSRMatrix *A,
                                    hypre_ParVector    *x,
                                    HYPRE_Complex       beta,
                                    hypre_ParVector    *b,
                                    hypre_ParVector    *y )
{
   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_ParCSRMatrixMemoryLocation(A),
                                                      hypre_ParVectorMemoryLocation(x) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_ParCSRMatrixMatvecOutOfPlaceDevice(alpha, A, x, beta, b, y);
   }
   else
#endif
   {
      ierr = hypre_ParCSRMatrixMatvecOutOfPlaceHost(alpha, A, x, beta, b, y);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvec
 *
 * Performs y <- alpha * A * x + beta * y
 *--------------------------------------------------------------------------*/

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
 * hypre_ParCSRMatrixMatvecTHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixMatvecTHost( HYPRE_Complex       alpha,
                               hypre_ParCSRMatrix *A,
                               hypre_ParVector    *x,
                               HYPRE_Complex       beta,
                               hypre_ParVector    *y )
{
   hypre_ParCSRCommPkg     *comm_pkg      = hypre_ParCSRMatrixCommPkg(A);

   hypre_CSRMatrix         *diag          = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix         *offd          = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix         *diagT         = hypre_ParCSRMatrixDiagT(A);
   hypre_CSRMatrix         *offdT         = hypre_ParCSRMatrixOffdT(A);

   hypre_Vector            *x_local       = hypre_ParVectorLocalVector(x);
   hypre_Vector            *y_local       = hypre_ParVectorLocalVector(y);
   hypre_Vector            *y_tmp;

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
   HYPRE_Int                i;
   HYPRE_Int                ierr = 0;

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

   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   hypre_assert( num_cols_offd * num_vectors ==
                 hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) );
   hypre_assert( hypre_ParCSRCommPkgRecvVecStart(comm_pkg, 0) == 0 );
   hypre_assert( hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0) == 0 );

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_ParCSRPersistentCommHandle *persistent_comm_handle =
      hypre_ParCSRCommPkgGetPersistentCommHandle(2, comm_pkg);
#else
   hypre_ParCSRCommHandle *comm_handle;
#endif

   /*---------------------------------------------------------------------
    * Allocate (during hypre_SeqVectorInitialize_v2) or retrieve
    * persistent send data buffer for y_tmp (if persistent is enabled).
    *--------------------------------------------------------------------*/

#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_VectorData(y_tmp) = (HYPRE_Complex *)
                             hypre_ParCSRCommHandleSendDataBuffer(persistent_comm_handle);
   hypre_SeqVectorSetDataOwner(y_tmp, 0);
#endif

   hypre_SeqVectorInitialize_v2(y_tmp, HYPRE_MEMORY_HOST);
   y_tmp_data = hypre_VectorData(y_tmp);

   /*---------------------------------------------------------------------
    * Allocate receive data buffer
    *--------------------------------------------------------------------*/

#if defined(HYPRE_USING_PERSISTENT_COMM)
   y_buf_data = (HYPRE_Complex *) hypre_ParCSRCommHandleRecvDataBuffer(persistent_comm_handle);

#else
   y_buf_data = hypre_TAlloc(HYPRE_Complex,
                             hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                             HYPRE_MEMORY_HOST);
#endif

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

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

   /* Non-blocking communication starts */
#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle, HYPRE_MEMORY_HOST, y_tmp_data);

#else
   comm_handle = hypre_ParCSRCommHandleCreate_v2(2, comm_pkg,
                                                 HYPRE_MEMORY_HOST, y_tmp_data,
                                                 HYPRE_MEMORY_HOST, y_buf_data );
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif

   /* Overlapped local computation.
      diagT is optional. Used only if it's present. */
   if (diagT)
   {
      hypre_CSRMatrixMatvec(alpha, diagT, x_local, beta, y_local);
   }
   else
   {
      hypre_CSRMatrixMatvecT(alpha, diag, x_local, beta, y_local);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

   /* Non-blocking communication ends */
#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle,
                                        HYPRE_MEMORY_HOST, y_buf_data);
#else
   hypre_ParCSRCommHandleDestroy(comm_handle);
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK]   -= hypre_MPI_Wtime();
#endif

   /* The assert is here because this code has been tested for column-wise vector storage only. */
   hypre_assert(idxstride == 1);

   /* unpack recv data on host, TODO OMP? */
   for (i = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
        i < hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
        i ++)
   {
      y_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)] += y_buf_data[i];
   }

   /*---------------------------------------------------------------------
    * Free memory
    *--------------------------------------------------------------------*/
   hypre_SeqVectorDestroy(y_tmp);

#if !defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_TFree(y_buf_data, HYPRE_MEMORY_HOST);
#endif

   HYPRE_ANNOTATE_FUNC_END;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
#endif

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvecT
 *
 * Performs y <- alpha * A^T * x + beta * y
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixMatvecT( HYPRE_Complex       alpha,
                           hypre_ParCSRMatrix *A,
                           hypre_ParVector    *x,
                           HYPRE_Complex       beta,
                           hypre_ParVector    *y )
{
   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_ParCSRMatrixMemoryLocation(A),
                                                      hypre_ParVectorMemoryLocation(x) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_ParCSRMatrixMatvecTDevice(alpha, A, x, beta, y);
   }
   else
#endif
   {
      ierr = hypre_ParCSRMatrixMatvecTHost(alpha, A, x, beta, y);
   }

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
   hypre_ParCSRCommHandle *comm_handle = NULL;
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix        *diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix        *offd   = hypre_ParCSRMatrixOffd(A);
   hypre_Vector           *x_local  = hypre_ParVectorLocalVector(x);
   hypre_Vector           *y_local  = hypre_ParVectorLocalVector(y);
   HYPRE_BigInt            num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt            num_cols = hypre_ParCSRMatrixGlobalNumCols(A);

   hypre_Vector      *x_tmp = NULL;
   HYPRE_BigInt       x_size = hypre_ParVectorGlobalSize(x);
   HYPRE_BigInt       y_size = hypre_ParVectorGlobalSize(y);
   HYPRE_Int          num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_Int          ierr = 0;
   HYPRE_Int          num_sends = 0, i, j, index, start, num_procs;
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

   hypre_MPI_Comm_size(comm, &num_procs);

   if (num_cols != x_size)
   {
      ierr = 11;
   }

   if (num_rows != y_size)
   {
      ierr = 12;
   }

   if (num_cols != x_size && num_rows != y_size)
   {
      ierr = 13;
   }

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
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            x_buf_data[index++]
               = x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
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
      if (num_cols_offd) { CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_HOST); }
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            int_buf_data[index++]
               = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
      comm_handle =
         hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd );

      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      if (num_cols_offd) hypre_CSRMatrixMatvec_FF(alpha, offd, x_tmp, 1.0, y_local,
                                                     CF_marker, CF_marker_offd, fpt);

      hypre_SeqVectorDestroy(x_tmp);
      x_tmp = NULL;
      hypre_TFree(x_buf_data, HYPRE_MEMORY_HOST);
      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
      hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   }

   return ierr;
}

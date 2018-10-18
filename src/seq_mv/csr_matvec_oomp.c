/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

//#if defined(HYPRE_USING_UNIFIED_MEMORY)

#include "seq_mv.h"
#include <assert.h>

#define NUM_TEAMS 2048
#define NUM_THREADS 1024

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec
 *--------------------------------------------------------------------------*/

/* y[offset:end] = alpha*A[offset:end,:]*x + beta*b[offset:end] */
HYPRE_Int
hypre_CSRMatrixMatvecOutOfPlaceOOMP2( HYPRE_Complex    alpha,
                                 hypre_CSRMatrix *A,
                                 hypre_Vector    *x,
                                 HYPRE_Complex    beta,
                                 hypre_Vector    *b,
                                 hypre_Vector    *y,
                                 HYPRE_Int        offset     )
{
   /* printf("CALLING OOOMP MATVE\n"); */
#ifdef HYPRE_PROFILE
   HYPRE_Real time_begin = hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
   PUSH_RANGE_PAYLOAD("MATVEC_OOMP",0, hypre_CSRMatrixNumRows(A));
   HYPRE_Int ierr = hypre_CSRMatrixMatvecDevice( alpha,A,x,beta,b,y,offset );
   POP_RANGE;
#else
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A) + offset;
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A) - offset;
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);
   /*HYPRE_Int         num_nnz  = hypre_CSRMatrixNumNonzeros(A);*/

   HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *b_data = hypre_VectorData(b) + offset;
   HYPRE_Complex    *y_data = hypre_VectorData(y) + offset;
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         b_size = hypre_VectorSize(b) - offset;
   HYPRE_Int         y_size = hypre_VectorSize(y) - offset;
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   /*HYPRE_Int         idxstride_b = hypre_VectorIndexStride(b);
   HYPRE_Int         vecstride_b = hypre_VectorVectorStride(b);*/
   HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

   HYPRE_Complex     temp, tempx;

   HYPRE_Int         i, j, jj;

   HYPRE_Int         m;

   HYPRE_Real        xpar=0.7;

   HYPRE_Int         ierr = 0;
   hypre_Vector     *x_tmp = NULL;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/

   hypre_assert( num_vectors == hypre_VectorNumVectors(y) );
   hypre_assert( num_vectors == hypre_VectorNumVectors(b) );

   if (num_cols != x_size)
      ierr = 1;

   if (num_rows != y_size || num_rows != b_size)
      ierr = 2;

   if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
      ierr = 3;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
     if (y_data!=b_data){
#ifdef HYPRE_USING_OPENMP_OFFLOAD
       //printf("Sub loop 0\n");
#pragma omp target teams  distribute  parallel for num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data,b_data)
#endif
       for (i = 0; i < num_rows*num_vectors; i++)
         y_data[i] = beta*b_data[i];
     } else {
#ifdef HYPRE_USING_OPENMP_OFFLOAD
       //printf("Sub loop 1\n");
#pragma omp target teams  distribute  parallel for num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data)
#endif
       for (i = 0; i < num_rows*num_vectors; i++)
         y_data[i] *= beta;
     }
#ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_MATVEC] += hypre_MPI_Wtime() - time_begin;
#endif

      return ierr;
   }

   if (x == y)
   {
     x_tmp = hypre_SeqVectorCloneDeep(x);
     x_data = hypre_VectorData(x_tmp);
    }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/

   temp = beta / alpha;

/* use rownnz pointer to do the A*x multiplication  when num_rownnz is smaller than num_rows */

   if (num_rownnz < xpar*(num_rows) || num_vectors > 1)
   {
      /*-----------------------------------------------------------------------
       * y = (beta/alpha)*y
       *-----------------------------------------------------------------------*/

      if (temp != 1.0)
      {
         if (temp == 0.0)
         {
#ifdef HYPRE_USING_OPENMP_OFFLOAD
            //printf("Sub loop 2\n");
#pragma omp target teams  distribute  parallel for num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data)
#endif
            for (i = 0; i < num_rows*num_vectors; i++)
               y_data[i] = 0.0;
         }
         else
         {
            if (y_data!=b_data){
#ifdef HYPRE_USING_OPENMP_OFFLOAD
               //printf("Sub loop 3\n");
#pragma omp target teams  distribute  parallel for num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data,b_data)
#endif
            for (i = 0; i < num_rows*num_vectors; i++)
               y_data[i] = b_data[i]*temp;

            } else {
#ifdef HYPRE_USING_OPENMP_OFFLOAD
               //printf("Sub loop 4\n");
#pragma omp target teams  distribute  parallel for num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data)
#endif
               for (i = 0; i < num_rows*num_vectors; i++)
                  y_data[i] = y_data[i]*temp;
         }

         }
      }
      else
      {
         if (y_data!=b_data){
#ifdef HYPRE_USING_OPENMP_OFFLOAD
            //printf("Sub loop 5\n");
#pragma omp target teams  distribute  parallel for num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data)
#endif
            for (i = 0; i < num_rows*num_vectors; i++)
               y_data[i] = b_data[i];
         }
      }


      /*-----------------------------------------------------------------
       * y += A*x
       *-----------------------------------------------------------------*/

      if (num_rownnz < xpar*(num_rows))
      {
#ifdef HYPRE_USING_OPENMP_OFFLOAD
         //printf("Sub loop 6\n");
#pragma omp target teams  distribute  parallel for private(i,j,jj,m,tempx) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data,A_data,x_data,A_i,A_j)
#endif
         for (i = 0; i < num_rownnz; i++)
         {
            m = A_rownnz[i];

            /*
             * for (jj = A_i[m]; jj < A_i[m+1]; jj++)
             * {
             *         j = A_j[jj];
             *  y_data[m] += A_data[jj] * x_data[j];
             * } */
            if ( num_vectors==1 )
            {
               tempx = 0;
               for (jj = A_i[m]; jj < A_i[m+1]; jj++)
                  tempx +=  A_data[jj] * x_data[A_j[jj]];
               y_data[m] += tempx;
            }
            else
               for ( j=0; j<num_vectors; ++j )
               {
                  tempx = 0;
                  for (jj = A_i[m]; jj < A_i[m+1]; jj++)
                     tempx +=  A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
                  y_data[ j*vecstride_y + m*idxstride_y] += tempx;
               }
         }
      }
      else // num_vectors > 1
      {
#ifdef HYPRE_USING_OPENMP_OFFLOAD
         //printf("Sub loop 7\n");
#pragma omp target teams  distribute  parallel for private(i,j,jj,m,tempx) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data,A_data,x_data,A_i,A_j)
#endif
         for (i = 0; i < num_rows; i++)
         {
            for (j = 0; j < num_vectors; ++j)
            {
               tempx = 0;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
               }
               y_data[ j*vecstride_y + i*idxstride_y ] += tempx;
            }
         }
      }

      /*-----------------------------------------------------------------
       * y = alpha*y
       *-----------------------------------------------------------------*/

      if (alpha != 1.0)
      {
#ifdef HYPRE_USING_OPENMP_OFFLOAD
         //printf("Alph!=1 loop 0\n");
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data)
#endif
         for (i = 0; i < num_rows*num_vectors; i++)
            y_data[i] *= alpha; // WHAT is going on here ?
      }
   }
   else
   { // JSP: this is currently the only path optimized

     if (y_data!=b_data){

#ifdef HYPRE_USING_OPENMP_OFFLOAD
       //printf("Main work loop 1\n");
#pragma omp target teams  distribute  parallel for private(i,j,jj,m,tempx) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data,A_data,x_data,A_i,A_j)
#endif
       for(i=0;i<num_rows;i++)
       {
          y_data[i]=beta*b_data[i];
          HYPRE_Complex temp = 0.0;
          for (jj = A_i[i]; jj < A_i[i+1]; jj++)
             temp += A_data[jj] * x_data[A_j[jj]];
          y_data[i] += alpha*temp;
          //y_data[i] *= alpha;
       }
     } else {
        /*printf("Main work loop 2 %d offset = %d alpha =%lf beta = %lf \n",num_rows,offset,alpha,beta);*/
#ifdef HYPRE_USING_OPENMP_OFFLOAD22
#pragma omp target teams  distribute  parallel for private(i,j,jj,tempx) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data,A_data,x_data,A_i,A_j)
#endif
       for(i=0;i<num_rows;i++)
       {
          //y_data[i]=beta*y_data[i];
          HYPRE_Complex tempx = 0.0;
          for (jj = A_i[i]; jj < A_i[i+1]; jj++)
             tempx += A_data[jj] * x_data[A_j[jj]];
          y_data[i] = alpha*tempx+beta*y_data[i];
       //y_data[i] *= alpha;
       }
     }

   }
   if (x == y) hypre_SeqVectorDestroy(x_tmp);

#endif /* defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY) */

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MATVEC] += hypre_MPI_Wtime() - time_begin;
#endif

   return ierr;
}

HYPRE_Int
hypre_CSRMatrixMatvecOutOfPlaceOOMP( HYPRE_Complex    alpha,
                                 hypre_CSRMatrix *A,
                                 hypre_Vector    *x,
                                 HYPRE_Complex    beta,
                                 hypre_Vector    *b,
                                 hypre_Vector    *y,
                                 HYPRE_Int        offset     )
{
#ifdef HYPRE_PROFILE
   HYPRE_Real time_begin = hypre_MPI_Wtime();
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MATVEC] += hypre_MPI_Wtime() - time_begin;
#endif

   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A) + offset;
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A) - offset;
#ifdef HYPRE_USING_CUSPARSE
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         num_nnz  = hypre_CSRMatrixNumNonzeros(A);

   HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         b_size = hypre_VectorSize(b) - offset;
   HYPRE_Int         y_size = hypre_VectorSize(y) - offset;
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   /*HYPRE_Int         idxstride_b = hypre_VectorIndexStride(b);
   HYPRE_Int         vecstride_b = hypre_VectorVectorStride(b);*/
   HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

   HYPRE_Real        xpar=0.7;
#endif
   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *b_data = hypre_VectorData(b) + offset;
   HYPRE_Complex    *y_data = hypre_VectorData(y) + offset;

   HYPRE_Int         ierr = 0;
   hypre_Vector     *x_tmp = NULL;

   if (offset!=0) {
     hypre_error_w_msg(HYPRE_ERROR_GENERIC,"WARNING :: NON ZERO OFFSET\n OPENMP version with no-zero offset not tested\n");
     return hypre_error_flag;
   }


#ifdef HYPRE_USING_CUSPARSE
   static cusparseHandle_t handle;
   static cusparseMatDescr_t descr;
   static HYPRE_Int FirstCall=1;
   cusparseStatus_t status;
   static cudaStream_t s[10];
   static HYPRE_Int myid;
   if (FirstCall){
    PUSH_RANGE("FIRST_CALL",4);

    handle=getCusparseHandle();

    status= cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR:: Matrix descriptor initialization failed\n");
      return hypre_error_flag;
    }

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    FirstCall=0;
    hypre_int jj;
    for(jj=0;jj<5;jj++)
      s[jj]=HYPRE_STREAM(jj);
    nvtxNameCudaStreamA(s[4], "HYPRE_COMPUTE_STREAM");
    hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
    myid++;
    POP_RANGE;
  }
#endif


#ifdef HYPRE_USING_UNIFIED_MEMORY
   hypre_CSRMatrixPrefetchToDevice(A);
   hypre_SeqVectorPrefetchToDevice(x);
   hypre_SeqVectorPrefetchToDevice(y);
   if (b!=y) hypre_SeqVectorPrefetchToDevice(b);
#endif

#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
   if (A->mapped==-1) {
     //hypre_CSRMatrixSortHost(A);
     hypre_CSRMatrixMapToDevice(A);
     //printf("MAPPING %p \n",A);
     hypre_CSRMatrixUpdateToDevice(A);
     //printf("DONE MAPPING %p \n",A);

   }
   //printf("Mapping X::");
   if (!x->mapped) hypre_SeqVectorMapToDevice(x);
   else SyncVectorToDevice(x);
   //printf("Mapping Y::");
   if (!y->mapped) hypre_SeqVectorMapToDevice(y);
   else SyncVectorToDevice(y);

   if (b!=y){
     if(!b->mapped)  {
       //printf("Mapping B::");
       hypre_SeqVectorMapToDevice(b);
     } else
       SyncVectorToDevice(b);
   }
#endif

   if (x == y)
   {
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
     hypre_error_w_msg(HYPRE_ERROR_GENERIC,"WARNING:: x_tmp is not mapped in Mapped OMP Offload version\n");
#endif
     x_tmp = hypre_SeqVectorCloneDeep(x);
     x_data = hypre_VectorData(x_tmp);
   }
   HYPRE_Int i;


#ifdef HYPRE_USING_CUSPARSE

#if defined(TRACK_MEMORY_ALLOCATIONS)
   ASSERT_MANAGED(A_data);
   ASSERT_MANAGED(A_i);
   ASSERT_MANAGED(A_j);
   ASSERT_MANAGED(x_data);
   ASSERT_MANAGED(y_data);
   ASSERT_MANAGED(b_data);
#endif

   if (b!=y){
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
#pragma omp target teams  distribute  parallel for private(i)
#else
#pragma omp target teams  distribute  parallel for private(i) is_device_ptr(y_data,b_data)
#endif
     for(i=0;i<y_size;i++) y_data[i] = b_data[i];
   }

   if (A->num_rows>0){
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
#pragma omp target data use_device_ptr(A_data,x_data,y_data,A_i,A_j)
#endif
   cusparseErrchk(cusparseDcsrmv(handle ,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 num_rows, num_cols, num_nnz,
                                 &alpha, descr,
                                 A_data ,A_i,A_j,
                                 x_data, &beta, y_data));
   }

   hypre_CheckErrorDevice(cudaStreamSynchronize(s[4]));
#else
#ifdef HYPRE_USING_OPENMP_OFFLOAD
   HYPRE_Int num_threads=64; // >64  for 100% Theoritical occupancy
   HYPRE_Int num_teams = (num_rows+num_rows%num_threads)/num_threads;
#pragma omp target teams  distribute  parallel for private(i) num_teams(num_teams) thread_limit(num_threads) is_device_ptr(A_data,A_i,A_j,y_data,b_data,x_data) schedule(static,1)
#endif
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
   HYPRE_Int num_threads=64; // >64  for 100% Theoritical occupancy
   HYPRE_Int num_teams = (num_rows+num_rows%num_threads)/num_threads;
   //   printf("Matvec with %d teams & %d tehreads \n",num_teams,num_threads);
   //   printf("Mapping map %d %d %d %d\n",omp_target_is_present(A,0),omp_target_is_present(A_data,0),omp_target_is_present(A_i,0),omp_target_is_present(A_j,0));
#pragma omp target teams  distribute  parallel for private(i) num_teams(num_teams) thread_limit(num_threads) schedule(static,1)
#endif
   for(i=0;i<num_rows;i++)
     {
       HYPRE_Complex tempx = 0.0;
       HYPRE_Int jj;
       for (jj = A_i[i]; jj < A_i[i+1]; jj++){
          tempx += A_data[jj] * x_data[A_j[jj]];
       }
       y_data[i] = alpha*tempx+beta*b_data[i];
     }
#endif
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
UpdateDRC(y);
#endif
   if (x == y) hypre_SeqVectorDestroy(x_tmp);
   //printRC(y,"Inside MatvecOOMP");
   //hypre_SeqVectorUpdateHost(y);
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
   //hypre_SeqVectorUnMapFromDevice(y);
   //hypre_SeqVectorUnMapFromDevice(x);
   //if ((b!=y)&&(b->mapped)) hypre_SeqVectorUnMapFromDevice(b);
#endif
   //printf("DONE WITH OOMP\n");
   return ierr;
}
HYPRE_Int
hypre_CSRMatrixMatvecOutOfPlaceOOMP3( HYPRE_Complex    alpha,
                                 hypre_CSRMatrix *A,
                                 hypre_Vector    *x,
                                 HYPRE_Complex    beta,
                                 hypre_Vector    *b,
                                 hypre_Vector    *y,
                                 HYPRE_Int        offset     )
{
  return 0;
  /*
  hypre_CSRMatrixMatvecOutOfPlaceOOMP(alpha,A,x,beta,b,y,offset);
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
  hypre_SeqVectorUpdateHost(y);
#endif
  return 0;
  */
}
/* HYPRE_Int hypre_CSRMatrixSortHost(hypre_CSRMatrix *A){ */
/*   HYPRE_Int      ierr=0; */
/*   HYPRE_Int      num_rows = hypre_CSRMatrixNumRows(A); */
/*   HYPRE_Int     *A_i = hypre_CSRMatrixI(A); */
/*   HYPRE_Int     *A_j = hypre_CSRMatrixJ(A); */
/*   HYPRE_Complex *A_data=hypre_CSRMatrixData(A); */

/*   HYPRE_Int i, j; */
/*   //printf("hypre_CSRMatrixSortHost\n"); */
/*   for (i=0; i < num_rows; i++){ */
/*     //printf("Row %d size %d \n",i,(A_i[i+1]-A_i[i])); */
/*     mysort(&A_data[A_i[i]],&A_j[A_i[i]],(A_i[i+1]-A_i[i])); */
/*   } */
/* } */
//#endif

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
  //printf("CALLING OOOMP MATVE\n");
#ifdef HYPRE_PROFILE
   HYPRE_Real time_begin = hypre_MPI_Wtime();
#endif
#ifdef HYPRE_USE_GPU
   PUSH_RANGE_PAYLOAD("MATVEC_OOMP",0, hypre_CSRMatrixNumRows(A));
   HYPRE_Int ret=hypre_CSRMatrixMatvecDevice( alpha,A,x,beta,b,y,offset);
   POP_RANGE;
  return ret;
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MATVEC] += hypre_MPI_Wtime() - time_begin;
#endif
#endif
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
   hypre_Vector	    *x_tmp = NULL;
   hypre_Vector	    *b_tmp = NULL;

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
       printf("Sub loop 0\n");
#pragma omp target teams  distribute  parallel for num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data,b_data)
#endif
       for (i = 0; i < num_rows*num_vectors; i++)
         y_data[i] = beta*b_data[i];
     } else {
#ifdef HYPRE_USING_OPENMP_OFFLOAD
       printf("Sub loop 1\n");
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
	   printf("Sub loop 2\n");
#pragma omp target teams  distribute  parallel for num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data)
#endif
            for (i = 0; i < num_rows*num_vectors; i++)
               y_data[i] = 0.0;
         }
         else
         {
	   if (y_data!=b_data){
#ifdef HYPRE_USING_OPENMP_OFFLOAD
	     printf("Sub loop 3\n");
#pragma omp target teams  distribute  parallel for num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data,b_data)
#endif
            for (i = 0; i < num_rows*num_vectors; i++)
               y_data[i] = b_data[i]*temp;
	    
	 } else {
#ifdef HYPRE_USING_OPENMP_OFFLOAD
	     printf("Sub loop 4\n");
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
	    printf("Sub loop 5\n");
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
	printf("Sub loop 6\n");
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
	printf("Sub loop 7\n");
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
	printf("Alph!=1 loop 0\n");
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
       printf("Main work loop 1\n");
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
       printf("Main work loop 2 %d offset = %d alpha =%lf beta = %lf \n",num_rows,offset,alpha,beta);
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
  //printf("CALLING OOOMP MATVE\n");
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

 
   HYPRE_Real        xpar=0.7;

   HYPRE_Int         ierr = 0;
   hypre_Vector	    *x_tmp = NULL;

   if (offset!=0) {
     fprintf(stderr,"WARNING :: NON ZERO OFFSET\n OPENMP version with no-zero offset not tested\n");
     exit(2);
   }

   if (x == y)
   {
     x_tmp = hypre_SeqVectorCloneDeep(x);
     x_data = hypre_VectorData(x_tmp);
   }
   int i;
#ifdef HYPRE_USING_OPENMP_OFFLOAD
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(A_data,A_i,A_j,y_data,b_data,x_data)
#endif
   for(i=0;i<num_rows;i++)
     {
       HYPRE_Complex tempx = 0.0;
       int jj;
       for (jj = A_i[i]; jj < A_i[i+1]; jj++){
	 tempx += A_data[jj] * x_data[A_j[jj]];
       }
       y_data[i] = alpha*tempx+beta*b_data[i];
     }
    
   if (x == y) hypre_SeqVectorDestroy(x_tmp);
   
   return ierr;
}

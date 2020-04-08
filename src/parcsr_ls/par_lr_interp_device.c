/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "aux_interp.h"

#define MAX_C_CONNECTIONS 100
#define HAVE_COMMON_C 1


__global__ void create_str_array( HYPRE_Int nr_of_rows, HYPRE_Int* A_i, HYPRE_Int* A_j, 
                                   HYPRE_Int* S_i, HYPRE_Int* S_j, HYPRE_Int* S_flag );

__global__ void compute_weak_rowsums( HYPRE_Int nr_of_rows, HYPRE_Int* A_i,
                                      HYPRE_Int* A_j, HYPRE_Real* A_data, 
                                      HYPRE_Int* A_str, HYPRE_Real* rs );

__global__ void compute_rowsums( HYPRE_Int nr_of_rows, 
                                 HYPRE_Int* A_diag_i, HYPRE_Real* A_diag_data,
                                 HYPRE_Int* A_offd_i, HYPRE_Real* A_offd_data,
                                 HYPRE_Int* S_diag_flag, HYPRE_Int* S_offd_flag, HYPRE_Real* rs );

__global__ void compute_rowsum( HYPRE_Int nr_of_rows, 
                                HYPRE_Int* A_diag_i, HYPRE_Real* A_diag_data,
                                HYPRE_Int* A_offd_i, HYPRE_Real* A_offd_data,
                                HYPRE_Real* rs );

__global__ void compute_twiaff_t( HYPRE_Int nr_of_rows, 
                                  HYPRE_Int* AFF_diag_i, HYPRE_Real* AFF_diag_data,
                                  HYPRE_Int* SFF_diag_f, 
                                  HYPRE_Int* AFF_offd_i, HYPRE_Real* AFF_offd_data, 
                                  HYPRE_Int* SFF_offd_f, 
                                  HYPRE_Real* rsFF, HYPRE_Real* rsFC );

__global__ void compute_twiafc_t( HYPRE_Int nr_of_rows,
                                  HYPRE_Int* AFC_diag_i, HYPRE_Real* AFC_diag_data,
                                  HYPRE_Int* SFC_diag_f, 
                                  HYPRE_Int* AFC_offd_i, HYPRE_Real* AFC_offd_data,
                                  HYPRE_Int* SFC_offd_f, 
                                  HYPRE_Real* rsFC);

__global__ void compute_twiafc_w( HYPRE_Int nr_of_rows,
                                  HYPRE_Int* AFC_diag_i, HYPRE_Real* AFC_diag_data,
                                  HYPRE_Int* SFC_diag_f, 
                                  HYPRE_Int* AFC_offd_i, HYPRE_Real* AFC_offd_data,
                                  HYPRE_Int* SFC_offd_f, 
                                  HYPRE_Real* rsFC);

__global__ void extendWtoP( HYPRE_Int nr_of_rowsP, HYPRE_Int* CF_marker, HYPRE_Int* PWoffset,
                            HYPRE_Int* W_diag_i, HYPRE_Int* W_diag_j, HYPRE_Real* W_diag_data,
                            HYPRE_Int* P_diag_i, HYPRE_Int* P_diag_j, HYPRE_Real* P_diag_data,
                            HYPRE_Int* W_offd_i, HYPRE_Int* P_offd_i );

struct is_pos : public thrust::unary_function<int,int>
{
   __host__ __device__ bool operator()(int x) { return x>0;}
};

struct is_neg  : public thrust::unary_function<int,int>
{
   __host__ __device__ bool operator()(int x) { return x<0;}
};
      
void moveParCSRMatrix( hypre_ParCSRMatrix* A, HYPRE_MemoryLocation to_memory );

HYPRE_Int
hypre_BoomerAMGBuildExtInterpDevice(hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                              hypre_ParCSRMatrix   *S, HYPRE_BigInt *num_cpts_global,
                              HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag,
                              HYPRE_Real trunc_factor, HYPRE_Int max_elmts,
                              HYPRE_Int *col_offd_S_to_A,
                              hypre_ParCSRMatrix  **P_ptr)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);


   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);


   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);
   HYPRE_Int       *A_diag_str, *A_offd_str; 
   HYPRE_Int *CF_marker_dev;

   hypre_ParCSRMatrix* AFF, *AFC;
   HYPRE_Int  *AFF_diag_i, *AFF_diag_j, *AFF_offd_i, *AFF_offd_j;
   HYPRE_Real *AFF_diag_data, *AFF_offd_data;
   HYPRE_Int  *AFC_diag_i, *AFC_diag_j, *AFC_offd_i, *AFC_offd_j;
   HYPRE_Real *AFC_diag_data, *AFC_offd_data;

   HYPRE_Int *AFF_diag_str, *AFF_offd_str;
   HYPRE_Int *AFC_diag_str, *AFC_offd_str;
   HYPRE_Int nnzFFdiag, nnzFFoffd, nnzFCdiag, nnzFCoffd, ncoarse;

   hypre_ParCSRMatrix* W;
   HYPRE_Int nr_of_rows, P_nr_of_rows, P_diag_nnz;

   HYPRE_Int* PWoffset;
   HYPRE_Real* rsFC, *rsWA, *rsW;

   HYPRE_Int*  W_diag_i, *W_diag_j, *W_offd_i;
   HYPRE_Real* W_diag_data;
   HYPRE_Int*  P_diag_i, *P_diag_j, *P_offd_i;
   HYPRE_Real* P_diag_data;

   dim3 bDim, gDim, gwDim;
   int myrank;

   //   HYPRE_Int memory_location=HYPRE_MEMORY_SHARED;
   //#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   //   if (hypre_handle->no_cuda_um == 1)
   //   {
   //      memory_location = HYPRE_MEMORY_DEVICE;
   //   }
   //#endif
   bool trace=false, debug=false;
   int dbgproc = 0;
   
   hypre_MPI_Comm_rank(comm, &myrank);

   HYPRE_Int A_nr_of_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   CF_marker_dev = hypre_CTAlloc( HYPRE_Int, A_nr_of_rows, HYPRE_MEMORY_DEVICE );
   hypre_TMemcpy( CF_marker_dev, CF_marker, HYPRE_Int, A_nr_of_rows, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST );

   if( trace )
      printf("In extintp\n");

/* 0.Find row sums of weak elements */
   bDim  = hypre_GetDefaultCUDABlockDimension();
   gDim  = hypre_GetDefaultCUDAGridDimension(A_nr_of_rows, "thread", bDim);
   gwDim = hypre_GetDefaultCUDAGridDimension(A_nr_of_rows, "warp",   bDim);

   A_diag_str = hypre_CTAlloc( HYPRE_Int, A_diag_i[A_nr_of_rows], HYPRE_MEMORY_DEVICE);
   HYPRE_CUDA_LAUNCH( create_str_array, gDim, bDim,
                      A_nr_of_rows, A_diag_i, A_diag_j, S_diag_i, S_diag_j, A_diag_str );   
   rsWA = hypre_CTAlloc( HYPRE_Real, A_nr_of_rows, HYPRE_MEMORY_DEVICE);
   HYPRE_CUDA_LAUNCH( compute_weak_rowsums, gwDim, bDim,
                      A_nr_of_rows, A_diag_i, A_diag_j, A_diag_data, A_diag_str, rsWA );
   if( A_offd_i[A_nr_of_rows] > 0 )
   {
      A_offd_str = hypre_CTAlloc( HYPRE_Int, A_offd_i[A_nr_of_rows], HYPRE_MEMORY_DEVICE);
      HYPRE_CUDA_LAUNCH( create_str_array, gDim, bDim, 
                      A_nr_of_rows, A_offd_i, A_offd_j, S_offd_i, S_offd_j, A_offd_str );
      HYPRE_CUDA_CALL(cudaDeviceSynchronize());
      HYPRE_CUDA_LAUNCH( compute_weak_rowsums, gwDim, bDim,
                      A_nr_of_rows, A_offd_i, A_offd_j, A_offd_data, A_offd_str, rsWA );
   }
   else
      A_offd_str=NULL;

   if( debug && myrank == dbgproc )
   {
      printf("A: \n");
      for( int i=0 ; i < A_nr_of_rows ; i++ )
      {
         for( int ind=A_diag_i[i] ; ind < A_diag_i[i+1] ; ind++ )
            printf("A(%d,%d) %d \n",i+1,A_diag_j[ind]+1,A_diag_str[ind]);
         for( int ind=A_offd_i[i] ; ind < A_offd_i[i+1] ; ind++ )
            printf("A(%d,%d) %d\n",i+1,A_offd_j[ind]+1,A_offd_str[ind]);
      }
      printf("S: \n");
      for( int i=0 ; i < A_nr_of_rows ; i++ )
      {
         for( int ind=S_diag_i[i] ; ind < S_diag_i[i+1] ; ind++ )
            printf("S(%d,%d)=1 \n",i+1,S_diag_j[ind]+1);
         for( int ind=S_offd_i[i] ; ind < S_offd_i[i+1] ; ind++ )
            printf("S(%d,%d)=1\n",i+1,S_offd_j[ind]+1);
      }
   }

   /* 1.Submatrices of A */
   moveParCSRMatrix(A,HYPRE_MEMORY_HOST);

   if( trace )
      printf("Before extract submatrix\n");

   hypre_ParCSRMatrixExtractSubmatrixFC( A, CF_marker, num_cpts_global, "FF", &AFF, 0.25 );
   hypre_ParCSRMatrixExtractSubmatrixFC( A, CF_marker, num_cpts_global, "FC", &AFC, 0.25 ); 

   if( trace )
      printf("After extract submatrix\n");
   moveParCSRMatrix(A,HYPRE_MEMORY_DEVICE);

   nr_of_rows    = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(AFF));
   AFF_diag_i    = hypre_CSRMatrixI(   hypre_ParCSRMatrixDiag(AFF));
   AFF_offd_i    = hypre_CSRMatrixI(   hypre_ParCSRMatrixOffd(AFF));
   AFC_diag_i    = hypre_CSRMatrixI(   hypre_ParCSRMatrixDiag(AFC));
   AFC_offd_i    = hypre_CSRMatrixI(   hypre_ParCSRMatrixOffd(AFC));

   nnzFFdiag = AFF_diag_i[nr_of_rows];
   nnzFFoffd = AFF_offd_i[nr_of_rows];   
   nnzFCdiag = AFC_diag_i[nr_of_rows];
   nnzFCoffd = AFC_offd_i[nr_of_rows];

   moveParCSRMatrix(AFF,HYPRE_MEMORY_DEVICE);
   moveParCSRMatrix(AFC,HYPRE_MEMORY_DEVICE);

   if( trace )
      printf("After move to device\n");

   AFF_diag_i    = hypre_CSRMatrixI(   hypre_ParCSRMatrixDiag(AFF));
   AFF_diag_j    = hypre_CSRMatrixJ(   hypre_ParCSRMatrixDiag(AFF));
   AFF_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(AFF));

   AFF_offd_i    = hypre_CSRMatrixI(   hypre_ParCSRMatrixOffd(AFF));
   AFF_offd_j    = hypre_CSRMatrixJ(   hypre_ParCSRMatrixOffd(AFF));
   AFF_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(AFF));

   AFC_diag_i    = hypre_CSRMatrixI(   hypre_ParCSRMatrixDiag(AFC));
   AFC_diag_j    = hypre_CSRMatrixJ(   hypre_ParCSRMatrixDiag(AFC));
   AFC_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(AFC));

   AFC_offd_i    = hypre_CSRMatrixI(   hypre_ParCSRMatrixOffd(AFC));
   AFC_offd_j    = hypre_CSRMatrixJ(   hypre_ParCSRMatrixOffd(AFC));
   AFC_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(AFC));

   hypre_TFree( A_diag_str, HYPRE_MEMORY_DEVICE);
   hypre_TFree( A_offd_str, HYPRE_MEMORY_DEVICE);

   bDim  = hypre_GetDefaultCUDABlockDimension();
   gDim  = hypre_GetDefaultCUDAGridDimension(nr_of_rows, "thread", bDim);
   gwDim = hypre_GetDefaultCUDAGridDimension(nr_of_rows, "warp",   bDim);

   HYPRE_Int* tmp=hypre_CTAlloc( HYPRE_Int, nnzFFdiag+nnzFFoffd+nnzFCdiag+nnzFCoffd, HYPRE_MEMORY_DEVICE);
   AFF_diag_str = tmp;
   AFF_offd_str = &tmp[nnzFFdiag];
   AFC_diag_str = &tmp[nnzFFdiag+nnzFFoffd];
   AFC_offd_str = &tmp[nnzFFdiag+nnzFFoffd+nnzFCdiag];

   /* Assume we have only strong connections for now, need to improve extractSubmatrixFC */
   /*   discovered later, extractSubmatrixFC keeps some weak connections, the following calls */
   /*   set the weakly connected elements to zero */
   
   //   zero_out_weak<<<gDim,bDim>>>( nr_of_rows, AFF_diag_i, AFF_diag_j, AFF_diag_data,
   //                                             AFC_diag_i, AFC_diag_j, AFC_diag_data );

   /* account for weak connections in submatrices AFF, AFC */
   thrust::device_ptr<HYPRE_Int> dev(tmp);
   HYPRE_THRUST_CALL( fill, dev, dev + nnzFFdiag+nnzFFoffd+nnzFCdiag+nnzFCoffd, 1 );

   rsW = hypre_CTAlloc( HYPRE_Real, nr_of_rows, HYPRE_MEMORY_DEVICE);
  /* restrict weak connection to F-rows:*/
   thrust::device_ptr<HYPRE_Real> rsWAd(rsWA), rsWd(rsW);
   thrust::device_ptr<HYPRE_Int>  CF_marker_devd(CF_marker_dev);
   HYPRE_THRUST_CALL( copy_if, rsWAd, rsWAd+A_nr_of_rows, CF_marker_devd, rsWd, is_neg() );
   hypre_Free( rsWA, HYPRE_MEMORY_DEVICE );

   if( debug && myrank==0 )
   {
      printf("AFF: \n");
      for( int i=0 ; i < nr_of_rows ; i++ )
      {
         for( int ind=AFF_diag_i[i] ; ind < AFF_diag_i[i+1] ; ind++ )
            printf("AFF(%d,%d)=%g\n",i+1,AFF_diag_j[ind]+1,AFF_diag_data[ind]);
         for( int ind=AFF_offd_i[i] ; ind < AFF_offd_i[i+1] ; ind++ )
            printf("AFF(%d,%d)=%g\n",i+1,AFF_offd_j[ind]+1,AFF_offd_data[ind]);
      }
      printf("done AFF\n");
      printf("AFC: \n");
      for( int i=0 ; i < nr_of_rows ; i++ )
      {
         for( int ind=AFC_diag_i[i] ; ind < AFC_diag_i[i+1] ; ind++ )
            printf("AFC(%d,%d)=%g\n",i+1,AFC_diag_j[ind]+1,AFC_diag_data[ind]);
         for( int ind=AFC_offd_i[i] ; ind < AFC_offd_i[i+1] ; ind++ )
            printf("AFC(%d,%d)=%g\n",i+1,AFC_offd_j[ind]+1,AFC_offd_data[ind]);
      }
      printf("done AFC\n");
   }

   rsFC = hypre_CTAlloc( HYPRE_Real, nr_of_rows, HYPRE_MEMORY_DEVICE );
   HYPRE_CUDA_LAUNCH( compute_rowsum, gwDim, bDim,
                      nr_of_rows, AFC_diag_i, AFC_diag_data, AFC_offd_i, AFC_offd_data, rsFC );


   /* 5. Form matrix ~{A_FF}, (return twAFF in AFF data structure ) */
   HYPRE_CUDA_CALL(cudaDeviceSynchronize());
   
   HYPRE_CUDA_LAUNCH( compute_twiaff_t, gDim, bDim,
                      nr_of_rows, AFF_diag_i, AFF_diag_data, AFF_diag_str,
                      AFF_offd_i, AFF_offd_data, AFF_offd_str, rsW, rsFC );

   /* 6. Form matrix ~{A_FC}, (return twAFC in AFC data structure) */
 
   HYPRE_CUDA_LAUNCH( compute_twiafc_w, gwDim, bDim,
                    nr_of_rows, AFC_diag_i, AFC_diag_data, AFC_diag_str,
                    AFC_offd_i, AFC_offd_data, AFC_offd_str, rsFC );
 //   compute_twiafc_w<<<gwDim,bDim>>>( nr_of_rows, AFC_diag_i, AFC_diag_data, AFC_diag_str,
 //                                     AFC_offd_i, AFC_offd_data, AFC_offd_str, rsFC );

   //   HYPRE_CUDA_CALL(cudaDeviceSynchronize());

   /* 7. Perform matrix-matrix multiplication */
   W = hypre_ParCSRMatMat(AFF, AFC );

   /* 8. Construct P from matrix product W.*/
   ncoarse   = hypre_CSRMatrixNumCols( hypre_ParCSRMatrixDiag(AFC) );
   P_nr_of_rows = nr_of_rows+ncoarse;

   W_diag_i    = hypre_CSRMatrixI(    hypre_ParCSRMatrixDiag(W));
   W_diag_j    = hypre_CSRMatrixJ(    hypre_ParCSRMatrixDiag(W));
   W_diag_data = hypre_CSRMatrixData( hypre_ParCSRMatrixDiag(W));
   W_offd_i    = hypre_CSRMatrixI(    hypre_ParCSRMatrixOffd(W));

   PWoffset = hypre_CTAlloc( HYPRE_Int, P_nr_of_rows+1, HYPRE_MEMORY_DEVICE );
   P_diag_i = hypre_CTAlloc( HYPRE_Int, P_nr_of_rows+1, HYPRE_MEMORY_DEVICE );
   P_offd_i = hypre_CTAlloc( HYPRE_Int, P_nr_of_rows+1, HYPRE_MEMORY_DEVICE );

   P_diag_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W))+ncoarse;

   P_diag_j    = hypre_CTAlloc( HYPRE_Int,  P_diag_nnz, HYPRE_MEMORY_DEVICE );
   P_diag_data = hypre_CTAlloc( HYPRE_Real, P_diag_nnz, HYPRE_MEMORY_DEVICE );

   //   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(W)) = HYPRE_MEMORY_DEVICE;
   //   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(W)) = HYPRE_MEMORY_DEVICE;


   gDim = hypre_GetDefaultCUDAGridDimension( P_nr_of_rows, "thread", bDim);
   if( trace )
      printf("before thrust \n");
   thrust::device_ptr<HYPRE_Int> PWoffsetd(PWoffset);
   HYPRE_THRUST_CALL( transform,  CF_marker_devd, &CF_marker_devd[P_nr_of_rows], 
                      PWoffsetd, is_pos());
   HYPRE_THRUST_CALL( exclusive_scan, PWoffsetd, &PWoffsetd[P_nr_of_rows+1], PWoffsetd );
   if( trace )
      printf("after thrust \n");
   if( debug && myrank == dbgproc )
   {
      for( int i=0 ; i < P_nr_of_rows ; i++ )
      printf("pwo(%d) = %d \n",i,PWoffset[i]);
      printf("--------------- \n");
   }
   HYPRE_CUDA_LAUNCH( extendWtoP, gDim, bDim,
                      P_nr_of_rows, CF_marker_dev, PWoffset, W_diag_i, W_diag_j, W_diag_data,
                      P_diag_i, P_diag_j, P_diag_data, W_offd_i, P_offd_i );
//   extendWtoP<<<gDim,bDim>>>( P_nr_of_rows, CF_marker_dev, PWoffset, W_diag_i, W_diag_j, W_diag_data,
//                              P_diag_i, P_diag_j, P_diag_data, W_offd_i, P_offd_i );

   HYPRE_CUDA_CALL(cudaDeviceSynchronize());
   P_diag_i[P_nr_of_rows]=P_diag_nnz;
   P_offd_i[P_nr_of_rows]=hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(W));

   if( debug && myrank ==dbgproc )
   {
      printf("W: (%d rows)\n",nr_of_rows);
      for( int i=0 ; i < nr_of_rows ; i++ )
      {
         for( int ind=W_diag_i[i] ; ind < W_diag_i[i+1] ; ind++ )
              printf("W(%d,%d)=%g\n",i+1,W_diag_j[ind]+1,W_diag_data[ind]);
        }
        printf("done W\n");

      printf("P: (%d rows)\n",P_nr_of_rows);
      for( int i=0 ; i < P_nr_of_rows ; i++ )
      {
         for( int ind=P_diag_i[i] ; ind < P_diag_i[i+1] ; ind++ )
              printf("P(%d,%d)=%g\n",i+1,P_diag_j[ind]+1,P_diag_data[ind]);
        }
        printf("done P\n");
   }

   hypre_CSRMatrixI(    hypre_ParCSRMatrixDiag(W) ) = P_diag_i;
   hypre_CSRMatrixJ(    hypre_ParCSRMatrixDiag(W) ) = P_diag_j;
   hypre_CSRMatrixData( hypre_ParCSRMatrixDiag(W) ) = P_diag_data;

   hypre_CSRMatrixI(    hypre_ParCSRMatrixOffd(W) ) = P_offd_i;

   hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixDiag(W) ) = P_diag_nnz;
   hypre_CSRMatrixNumRows(     hypre_ParCSRMatrixDiag(W) ) = P_nr_of_rows;
   hypre_CSRMatrixNumRows(     hypre_ParCSRMatrixOffd(W) ) = P_nr_of_rows;
   //   hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixOffd(W) ) = P_offd_nnz;

   //   hypre_ParCSRMatrixSetNumNonzeros(B);
   //   hypre_ParCSRMatrixDNumNonzeros(B) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(B);

   hypre_MatvecCommPkgCreate(W);

   hypre_ParCSRMatrixGlobalNumRows(W) = hypre_ParCSRMatrixGlobalNumRows(A);

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncationDevice(W, trunc_factor, max_elmts );
   }
   for (int i=0; i < nr_of_rows; i++)
      if (CF_marker[i] == -3) CF_marker[i] = -1;

   *P_ptr = W;
   if( trace )
      printf("Exit extintp \n");

   /* 9. Free memory   */
   hypre_TFree( W_diag_i, HYPRE_MEMORY_DEVICE );
   hypre_TFree( W_diag_j, HYPRE_MEMORY_DEVICE );
   hypre_TFree( W_diag_data, HYPRE_MEMORY_DEVICE );
   hypre_TFree( W_offd_i, HYPRE_MEMORY_DEVICE );

   hypre_TFree( PWoffset, HYPRE_MEMORY_DEVICE );
   hypre_TFree( rsW, HYPRE_MEMORY_DEVICE );
   hypre_TFree( rsFC, HYPRE_MEMORY_DEVICE );
   hypre_TFree( tmp, HYPRE_MEMORY_DEVICE );

   return hypre_error_flag;
}

//-----------------------------------------------------------------------
__global__ void compute_weak_rowsums( HYPRE_Int nr_of_rows, HYPRE_Int* A_i,
                                      HYPRE_Int* A_j, HYPRE_Real* A_data, 
                                      HYPRE_Int* A_str, HYPRE_Real* rs )
{
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>(), lane, i, ib, ie;
   //   HYPRE_Int i= hypre_cuda_get_grid_thread_id<1,1>(); //threadIdx.x + blockIdx.x * blockDim.x,
   HYPRE_Real rl=0;
   if( row >= nr_of_rows )
      return;
   lane = hypre_cuda_get_lane_id<1>();

   if( lane < 2 )
      ib = read_only_load(A_i+row+lane);
   ie = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 1);
   ib = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 0);

   for( i = ib + lane ; __any_sync(HYPRE_WARP_FULL_MASK, i < ie); i += HYPRE_WARP_SIZE )
   {
      if( i < ie )
         rl += A_data[i]*(1-A_str[i]);
   }
   rl = warp_reduce_sum(rl);
   if( lane == 0 )
      rs[row] += rl;
}

//-----------------------------------------------------------------------
__global__ void create_str_array( HYPRE_Int nr_of_rows, HYPRE_Int* A_i, HYPRE_Int* A_j, 
                                   HYPRE_Int* S_i, HYPRE_Int* S_j, HYPRE_Int* A_str )
{
   HYPRE_Int i= hypre_cuda_get_grid_thread_id<1,1>(); //threadIdx.x + blockIdx.x * blockDim.x,
   
   //   const HYPRE_Int nthreads = gridDim.x * blockDim.x;
   HYPRE_Int ind, sind, sf;
   if( i < nr_of_rows )
   {
      sind=S_i[i];
      for( ind=A_i[i] ; ind < A_i[i+1] ; ind++ )
      {
         if( sind < S_i[i+1] )
            sf = A_j[ind]==S_j[sind];
         else
            sf = 0;
         A_str[ind]=sf;
         sind += sf;
         //      cind=A_j[ind];
         //      if( cind == sind )
         //      {
         //         A_str[ind] = 1;
         //         sind++;
         //      }
         //      else
         //         S_str[ind] = 0;
      }
   }
}

//-----------------------------------------------------------------------
__global__ void compute_rowsum( HYPRE_Int nr_of_rows, 
                      HYPRE_Int* A_diag_i, HYPRE_Real* A_diag_data,
                      HYPRE_Int* A_offd_i, HYPRE_Real* A_offd_data,
                      HYPRE_Real* rs )
{
   HYPRE_Real rl=0;
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>(), lane, i, ib, ie, ibo, ieo;

   if( row >= nr_of_rows )
      return;
   lane = hypre_cuda_get_lane_id<1>();
   if( lane < 2 )
      ib = read_only_load(A_diag_i+row+lane);
   ie = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 1);
   ib = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 0);

   for( i = ib + lane ; __any_sync(HYPRE_WARP_FULL_MASK, i < ie); i += HYPRE_WARP_SIZE )
   {
      if( i < ie )
      {
         rl += read_only_load(A_diag_data + i);
      }
   }

 /* Off-diagonal part of matrix */
   if( lane < 2 )
      ibo = read_only_load(A_offd_i+row+lane);
   ieo = __shfl_sync(HYPRE_WARP_FULL_MASK, ibo, 1);
   ibo = __shfl_sync(HYPRE_WARP_FULL_MASK, ibo, 0);

   for( i = ibo + lane ; __any_sync(HYPRE_WARP_FULL_MASK, i < ieo); i += HYPRE_WARP_SIZE )
   {
      if( i < ieo )
      {
         rl += read_only_load(A_offd_data + i);
      }
   }
   rl=warp_reduce_sum(rl);
   if( lane == 0 )
      rs[row]  = rl;
}

//-----------------------------------------------------------------------
__global__ void compute_rowsums( HYPRE_Int nr_of_rows, 
                      HYPRE_Int* A_diag_i, HYPRE_Real* A_diag_data,
                      HYPRE_Int* A_offd_i, HYPRE_Real* A_offd_data,
                      HYPRE_Int* A_diag_str, HYPRE_Int* A_offd_str, HYPRE_Real* rs )
{
 // one thread/row
   //   for( int ind=A_diag_i[i] ; ind < A_diag_i[i+1] ; ind++ )
   //      rs[2*i+A_diag_str[ind]] += A_diag_data[ind];
   //   for( int ind=A_offd_i[i] ; ind < A_offd_i[i+1] ; ind++ )
   //      rs[2*i+A_offd_str[ind]] += A_offd_data[ind];

 // one warp/row
   //   HYPRE_Real rl[2]={0,0}, term, rl0, rl1;
   HYPRE_Real term, rl0=0, rl1=0;   
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>(), lane, i, ib, ie, fl, ibo, ieo;

   if( row >= nr_of_rows )
      return;
   lane = hypre_cuda_get_lane_id<1>();

 /* Diagonal part of matrix */
   if( lane < 2 )
      ib = read_only_load(A_diag_i+row+lane);
   ie = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 1);
   ib = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 0);

   for( i = ib + lane ; __any_sync(HYPRE_WARP_FULL_MASK, i < ie); i += HYPRE_WARP_SIZE )
   {
      if( i < ie )
      {
         term = read_only_load(A_diag_data + i);
         fl   = read_only_load(A_diag_str + i);
         //         rl[fl] += term;
         rl0 += term*(1-fl);
         rl1 += term*fl;
      }
   }

 /* Off-diagonal part of matrix */
   if( lane < 2 )
      ibo = read_only_load(A_offd_i+row+lane);
   ieo = __shfl_sync(HYPRE_WARP_FULL_MASK, ibo, 1);
   ibo = __shfl_sync(HYPRE_WARP_FULL_MASK, ibo, 0);

   for( i = ibo + lane ; __any_sync(HYPRE_WARP_FULL_MASK, i < ieo); i += HYPRE_WARP_SIZE )
   {
      if( i < ieo )
      {
         term = read_only_load(A_offd_data + i);
         fl   = read_only_load(A_offd_str + i);
         //         rl[fl] += term;
         rl0 += term*(1-fl);
         rl1 += term*fl;
      }
   }
   rl0=warp_reduce_sum(rl0);
   rl1=warp_reduce_sum(rl1);
   if( lane == 0 )
   {
      rs[2*row]  = rl0;
      rs[2*row+1]= rl1;
   }
}


//-----------------------------------------------------------------------
__global__ void compute_twiaff_t( HYPRE_Int nr_of_rows, 
                     HYPRE_Int* AFF_diag_i, HYPRE_Real* AFF_diag_data,
                     HYPRE_Int* AFF_diag_str, 
                     HYPRE_Int* AFF_offd_i, HYPRE_Real* AFF_offd_data, 
                     HYPRE_Int* AFF_offd_str, 
                     HYPRE_Real* rsW, HYPRE_Real* rsFC )
{
   // One thread/row
   HYPRE_Real iscale; //, diag_el;
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>(), ind;


   if( i < nr_of_rows )
   {
      /* Diagonal element */
      ind=AFF_diag_i[i];
      //      diag_el=AFF_diag_data[ind];
      //      iscale =-1.0/(rsW[i]+diag_el);
      iscale =-1.0/(rsW[i]);
      AFF_diag_data[ind] = rsFC[i]*iscale;
      for( ind=AFF_diag_i[i]+1 ; ind < AFF_diag_i[i+1] ; ind++ )
         AFF_diag_data[ind] *= iscale;
      for( ind=AFF_offd_i[i] ; ind < AFF_offd_i[i+1] ; ind++ )
         AFF_offd_data[ind] *= iscale;
      
      //      ind=AFF_diag_i[i];
      //      iscale = -1.0/(rsFF[2*i]+rsFC[2*i]);
      //      AFF_diag_data[ind] = rsFC[2*i+1]*iscale;
      //      /* AFF^(s)*/
      //      for( ind=AFF_diag_i[i]+1 ; ind < AFF_diag_i[i+1] ; ind++ )
      //         AFF_diag_data[ind] *= iscale*AFF_diag_str[ind];
      //      for( ind=AFF_offd_i[i] ; ind < AFF_offd_i[i+1] ; ind++ )
      //         AFF_offd_data[ind] *= iscale*AFF_offd_str[ind]; 

   }
}

//-----------------------------------------------------------------------
__global__ void compute_twiafc_t( HYPRE_Int nr_of_rows,
                       HYPRE_Int* AFC_diag_i, HYPRE_Real* AFC_diag_data,
                       HYPRE_Int* AFC_diag_str,
                       HYPRE_Int* AFC_offd_i, HYPRE_Real* AFC_offd_data,
                       HYPRE_Int* AFC_offd_str,
                       HYPRE_Real* rsFC)
{
   // One thread/row
   HYPRE_Real scale,iscale;
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>(), ind;

   if( i < nr_of_rows )
   {
      scale = read_only_load(rsFC+2*i+1);
      if( scale != 0 )
      {
         //         iscale = 1.0/read_only_load(rsFC+2*i+1);
         iscale =1.0/scale;
         for( ind=AFC_diag_i[i] ; ind < AFC_diag_i[i+1] ; ind++ )
            AFC_diag_data[ind] *= iscale*AFC_diag_str[ind];
         for( ind=AFC_offd_i[i] ; ind < AFC_offd_i[i+1] ; ind++ )
            AFC_offd_data[ind] *= iscale*AFC_offd_str[ind];
      }
   }
}

//-----------------------------------------------------------------------
__global__ void compute_twiafc_w( HYPRE_Int nr_of_rows,
                       HYPRE_Int* AFC_diag_i, HYPRE_Real* AFC_diag_data,
                       HYPRE_Int* AFC_diag_str,
                       HYPRE_Int* AFC_offd_i, HYPRE_Real* AFC_offd_data,
                       HYPRE_Int* AFC_offd_str,
                       HYPRE_Real* rsFC)
{
   // One warp/row
   HYPRE_Real iscale;
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>(), lane, i, ib, ie, ibo, ieo;
   if( row >= nr_of_rows )
      return;
   lane = hypre_cuda_get_lane_id<1>();
   if( lane < 2 )
      ib = read_only_load(AFC_diag_i+row+lane);
   ie = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 1);
   ib = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 0);
   //   iscale = read_only_load(rsFC+2*row+1);
   iscale = read_only_load(rsFC+row);
   if( iscale != 0 )
      iscale = 1.0/iscale;

   for( i = ib + lane ; __any_sync(HYPRE_WARP_FULL_MASK, i < ie); i += HYPRE_WARP_SIZE )
   {
      if( i < ie )
         AFC_diag_data[i] *= iscale;
      //         AFC_diag_data[i] *= iscale*AFC_diag_str[i];
   }
   if( lane < 2 )
      ibo = read_only_load(AFC_offd_i+row+lane);
   ieo = __shfl_sync(HYPRE_WARP_FULL_MASK, ibo, 1);
   ibo = __shfl_sync(HYPRE_WARP_FULL_MASK, ibo, 0);

   for( i = ibo + lane ; __any_sync(HYPRE_WARP_FULL_MASK, i < ieo); i += HYPRE_WARP_SIZE )
   {
      if( i < ieo )
         //         AFC_offd_data[i] *= iscale*AFC_offd_str[i];
         AFC_offd_data[i] *= iscale;
   }
}

//-----------------------------------------------------------------------
__global__ void extendWtoP( HYPRE_Int nr_of_rowsP, HYPRE_Int* CF_marker, HYPRE_Int* PWoffset,
                            HYPRE_Int* W_diag_i, HYPRE_Int* W_diag_j, HYPRE_Real* W_diag_data,
                            HYPRE_Int* P_diag_i, HYPRE_Int* P_diag_j, HYPRE_Real* P_diag_data,
                            HYPRE_Int* W_offd_i, HYPRE_Int* P_offd_i )
{
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>(), ind, s, pd;
   if( i < nr_of_rowsP )
   {
      s  = PWoffset[i];
      pd = W_diag_i[i-s]+s;
      P_diag_i[i] = pd;
      P_offd_i[i] = W_offd_i[i-s];
      if( CF_marker[i] < 0 )
      {
          // Fine pt, copy P from W with shift
         for( ind = W_diag_i[i-s]; ind < W_diag_i[i-s+1] ; ind++ )
         {
            P_diag_j[ind+s]    = W_diag_j[ind];
            P_diag_data[ind+s] = W_diag_data[ind];
         }
      }
      else
      {
          // Coarse pt, add unit row to P_diag
         P_diag_j[pd]    = s;
         P_diag_data[pd] = 1.0;
      }
   }
}

/*-----------------------------------------------------------------------*/
__global__ void zero_out_weak( HYPRE_Int nr_of_rows, HYPRE_Int* AFF_diag_i,
                               HYPRE_Int* AFF_diag_j, HYPRE_Real* AFF_diag_data,
                               HYPRE_Int* AFC_diag_i, HYPRE_Int* AFC_diag_j,
                               HYPRE_Real* AFC_diag_data )
{
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>();
   if( i < nr_of_rows )
   {
      HYPRE_Real delm = AFF_diag_data[AFF_diag_i[i]];
      for( int ind=AFF_diag_i[i]+1 ; ind < AFF_diag_i[i+1] ; ind++ )
         AFF_diag_data[ind] = AFF_diag_data[ind]*delm < 0 ? AFF_diag_data[ind]:0;
      for( int ind=AFC_diag_i[i] ; ind < AFC_diag_i[i+1] ; ind++ )
         AFC_diag_data[ind] = AFC_diag_data[ind]*delm < 0 ? AFC_diag_data[ind]:0;
   }
}

/*-----------------------------------------------------------------------*/
void moveParCSRMatrix( hypre_ParCSRMatrix* A, HYPRE_MemoryLocation to_memory )
{
   if( hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(A)) != to_memory ||
       hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(A)) != to_memory )
   {
      HYPRE_Int nrowsd = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
      HYPRE_Int nrowso = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixOffd(A));
      HYPRE_Int nnzd   = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(A));
      HYPRE_Int nnzo   = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(A));

      HYPRE_MemoryLocation from_memoryd = hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(A));
      HYPRE_MemoryLocation from_memoryo = hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(A));

      HYPRE_Int* Ad_i= hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A));
      HYPRE_Int* tmp = hypre_CTAlloc(HYPRE_Int, nrowsd+1, to_memory );
      hypre_Memcpy(tmp,Ad_i,sizeof(HYPRE_Int)*(nrowsd+1), to_memory, from_memoryd );
      hypre_TFree( Ad_i, from_memoryd );
      hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A)) = tmp;

      HYPRE_Int* Ao_i=hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A));
      tmp = hypre_CTAlloc(HYPRE_Int, nrowso+1, to_memory );
      hypre_Memcpy(tmp,Ao_i,sizeof(HYPRE_Int)*(nrowso+1), to_memory, from_memoryo);
      hypre_TFree( Ao_i, from_memoryo );
      hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A)) = tmp;

      HYPRE_Int* Ad_j=hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A));
      tmp = hypre_CTAlloc(HYPRE_Int, nnzd, to_memory );
      hypre_Memcpy(tmp,Ad_j,sizeof(HYPRE_Int)*nnzd, to_memory, from_memoryd);
      hypre_TFree( Ad_j, from_memoryd );
      hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A)) = tmp;

      if( nnzo >0 )
      {
         HYPRE_Int* Ao_j=hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(A));
         tmp = hypre_CTAlloc(HYPRE_Int, nnzo, to_memory );
         hypre_Memcpy(tmp,Ao_j,sizeof(HYPRE_Int)*nnzo, to_memory, from_memoryo);
         hypre_TFree( Ao_j, from_memoryo );
         hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(A)) = tmp;
      }
      HYPRE_Real* Ad_r =hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A));
      HYPRE_Real* tmpr;

      if( Ad_r != NULL )      
      {
         tmpr = hypre_CTAlloc(HYPRE_Real, nnzd, to_memory );
         hypre_Memcpy(tmpr,Ad_r,sizeof(HYPRE_Real)*nnzd, to_memory, from_memoryd);
         hypre_TFree( Ad_r, from_memoryd );
         hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A)) = tmpr;
      }

      HYPRE_Real* Ao_r=hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A));
      if( Ao_r != NULL && nnzo >0 )
      {
         tmpr = hypre_CTAlloc(HYPRE_Real, nnzo, to_memory );
         hypre_Memcpy(tmpr,Ao_r,sizeof(HYPRE_Real)*nnzo, to_memory, from_memoryo);
         hypre_TFree( Ao_r, from_memoryo );
         hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A)) = tmpr;
      }

      hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(A)) = to_memory;
      hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(A)) = to_memory;
   }
}

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

#if defined(HYPRE_USING_CUDA)


__global__ void create_flag_array( HYPRE_Int nr_of_rows, HYPRE_Int* A_i, HYPRE_Int* A_j,
                                   HYPRE_Int* S_i, HYPRE_Int* S_j, HYPRE_Int* S_flag );

__global__ void compute_rowsums( HYPRE_Int nr_of_rows,
                                 HYPRE_Int* A_diag_i, HYPRE_Real* A_diag_data,
                                 HYPRE_Int* A_offd_i, HYPRE_Real* A_offd_data,
                                 HYPRE_Int* S_diag_flag, HYPRE_Int* S_offd_flag, HYPRE_Real* rs );

__global__ void compute_twiaff_t( HYPRE_Int nr_of_rows,
                                  HYPRE_Int* AFF_diag_i, HYPRE_Real* AFF_diag_data,
                                  HYPRE_Int* AFF_offd_i, HYPRE_Real* AFF_offd_data,
                                  HYPRE_Real* rsFF, HYPRE_Real* rsFC );

__global__ void compute_twiafc_t( HYPRE_Int nr_of_rows,
                                  HYPRE_Int* AFC_diag_i, HYPRE_Real* AFC_diag_data,
                                  HYPRE_Int* AFC_offd_i, HYPRE_Real* AFC_offd_data,
                                  HYPRE_Real* rsFC);

__global__ void compute_twiafc_w( HYPRE_Int nr_of_rows,
                                  HYPRE_Int* AFC_diag_i, HYPRE_Real* AFC_diag_data,
                                  HYPRE_Int* AFC_offd_i, HYPRE_Real* AFC_offd_data,
                                  HYPRE_Real* rsFC);

__global__ void extendWtoP( HYPRE_Int nr_of_rowsP, HYPRE_Int* CF_marker, HYPRE_Int* PWoffset,
                            HYPRE_Int* W_diag_i, HYPRE_Int* W_diag_j, HYPRE_Real* W_diag_data,
                            HYPRE_Int* P_diag_i, HYPRE_Int* P_diag_j, HYPRE_Real* P_diag_data,
                            HYPRE_Int* W_offd_i, HYPRE_Int* P_offd_i );




HYPRE_Int
hypre_BoomerAMGBuildExtInterpDevice(hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                              hypre_ParCSRMatrix   *S, HYPRE_BigInt *num_cpts_global,
                              HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag,
                              HYPRE_Real trunc_factor, HYPRE_Int max_elmts,
                              HYPRE_Int *col_offd_S_to_A,
                              hypre_ParCSRMatrix  **P_ptr)
{
   //   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   //   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   //   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   //   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);


   //   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   //   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   //   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   //   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   //   hypre_CSRMatrix *P_diag, *P_offd;

   //   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   // Extract AFFd, AFFo, AFCd, AFCo

   hypre_ParCSRMatrix* AFF, *AFC, *SFF, *SFC, *W;
   //   hypre_CSRMatrix *AFF_diag, *AFF_offd, *AFC_diag, *AFC_Offd;
   HYPRE_Int  *AFF_diag_i, *AFF_diag_j, *AFF_offd_i, *AFF_offd_j;
   HYPRE_Real *AFF_diag_data, *AFF_offd_data;
   HYPRE_Int  *AFC_diag_i, *AFC_diag_j, *AFC_offd_i, *AFC_offd_j;
   HYPRE_Real *AFC_diag_data, *AFC_offd_data;

   // and corresponding strength matrix arrays
   HYPRE_Int *SFF_diag_i, *SFF_diag_j, *SFF_offd_i, *SFF_offd_j;
   HYPRE_Int *SFC_diag_i, *SFC_diag_j, *SFC_offd_i, *SFC_offd_j;
   HYPRE_Int nr_of_rows, P_nr_of_rows, P_diag_nnz;

   HYPRE_Int nnzFFdiag, nnzFFoffd, nnzFCdiag, nnzFCoffd, ncoarse;

   HYPRE_Int* PWoffset;
   HYPRE_Int* SFF_diag_f, *SFF_offd_f, *SFC_diag_f, *SFC_offd_f;
   HYPRE_Real* rsFF, *rsFC;

   HYPRE_Int* W_diag_i, *W_diag_j, *W_offd_i;
   HYPRE_Real* W_diag_data;
   HYPRE_Int* P_diag_i, *P_diag_j, *P_offd_i;
   HYPRE_Real* P_diag_data;

   dim3 bDim, gDim, gwDim;

   hypre_ParCSRMatrixExtractSubmatrixFC( A, CF_marker, num_cpts_global, "FF", &AFF, 0 );
   hypre_ParCSRMatrixExtractSubmatrixFC( A, CF_marker, num_cpts_global, "FC", &AFC, 0 );

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

   nr_of_rows    = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(AFF));

   nnzFFdiag = AFF_diag_i[nr_of_rows];
   nnzFFoffd = AFF_offd_i[nr_of_rows];
   nnzFCdiag = AFC_diag_i[nr_of_rows];
   nnzFCoffd = AFC_offd_i[nr_of_rows];

   hypre_ParCSRMatrixExtractSubmatrixFC( S, CF_marker, num_cpts_global, "FF", &SFF, 0 );
   hypre_ParCSRMatrixExtractSubmatrixFC( S, CF_marker, num_cpts_global, "FC", &SFC, 0 );
   SFF_diag_i    = hypre_CSRMatrixI(   hypre_ParCSRMatrixDiag(SFF));
   SFF_diag_j    = hypre_CSRMatrixJ(   hypre_ParCSRMatrixDiag(SFF));

   SFF_offd_i    = hypre_CSRMatrixI(   hypre_ParCSRMatrixOffd(SFF));
   SFF_offd_j    = hypre_CSRMatrixJ(   hypre_ParCSRMatrixOffd(SFF));

   SFC_diag_i    = hypre_CSRMatrixI(   hypre_ParCSRMatrixDiag(SFC));
   SFC_diag_j    = hypre_CSRMatrixJ(   hypre_ParCSRMatrixDiag(SFC));

   SFC_offd_i    = hypre_CSRMatrixI(   hypre_ParCSRMatrixOffd(SFC));
   SFC_offd_j    = hypre_CSRMatrixJ(   hypre_ParCSRMatrixOffd(SFC));


 // 1. Compute flag array for strong/weak connections
       // SFF_diag_f = 1 for strong connection, 0 for weak connection, etc..


   /* Alternative;*/
   HYPRE_Int* tmp=hypre_CTAlloc( HYPRE_Int, nnzFFdiag+nnzFFoffd+nnzFCdiag+nnzFCoffd, HYPRE_MEMORY_DEVICE);
   SFF_diag_f = tmp;
   SFF_offd_f = &tmp[nnzFFdiag];
   SFC_diag_f = &tmp[nnzFFdiag+nnzFFoffd];
   SFC_offd_f = &tmp[nnzFFdiag+nnzFFoffd+nnzFCdiag];

   bDim = hypre_GetDefaultCUDABlockDimension();
   gDim = hypre_GetDefaultCUDAGridDimension(nr_of_rows, "thread", bDim);

   /*   SFF_diag_f = hypre_CTAlloc( HYPRE_Int, nnzFFdiag, HYPRE_MEMORY_DEVICE );*/
   create_flag_array<<<gDim,bDim>>>( nr_of_rows, AFF_diag_i, AFF_diag_j, SFF_diag_i, SFF_diag_j, SFF_diag_f );

   /*   SFF_offd_f = hypre_CTAlloc( HYPRE_Int, nnzFFoffd, HYPRE_MEMORY_DEVICE );*/
   create_flag_array<<<gDim,bDim >>>( nr_of_rows, AFF_offd_i, AFF_offd_j, SFF_offd_i, SFF_offd_j, SFF_offd_f );

   /*   SFC_diag_f = hypre_CTAlloc( HYPRE_Int, nnzFCdiag, HYPRE_MEMORY_DEVICE );*/
   create_flag_array<<<gDim,bDim>>>( nr_of_rows, AFC_diag_i, AFC_diag_j, SFC_diag_i, SFC_diag_j, SFC_diag_f );

   /*   SFC_offd_f = hypre_CTAlloc( HYPRE_Int, nnzFCoffd, HYPRE_MEMORY_DEVICE );*/
   create_flag_array<<<gDim,bDim>>>( nr_of_rows, AFC_offd_i, AFC_offd_j, SFC_offd_i, SFC_offd_j, SFC_offd_f );

// 2. Compute row sums
   gwDim = hypre_GetDefaultCUDAGridDimension(nr_of_rows, "warp", bDim);
//   rs[2*i] weak row sums, rs[2*i+1] strong row sums
   rsFF = hypre_CTAlloc( HYPRE_Real, 2*nr_of_rows, HYPRE_MEMORY_DEVICE );
   compute_rowsums<<<gwDim,bDim>>>( nr_of_rows, AFF_diag_i, AFF_diag_data, AFF_offd_i, AFF_offd_data,
                                    SFF_diag_f, SFF_offd_f, rsFF );

   rsFC = hypre_CTAlloc( HYPRE_Real, 2*nr_of_rows, HYPRE_MEMORY_DEVICE );
   compute_rowsums<<<gwDim,bDim>>>( nr_of_rows, AFC_diag_i, AFC_diag_data, AFC_offd_i, AFC_diag_data,
                                    SFC_diag_f, SFC_offd_f, rsFC );

// 3. Form matrix ~{A_FF}, (return twAFF in AFF data structure )
   compute_twiaff_t<<<gDim,bDim>>>( nr_of_rows, AFF_diag_i, AFF_diag_data, AFF_offd_i, AFF_offd_data, rsFF, rsFC);

// 4. Form matrix ~{A_FC}, (return twAFC in AFC data structure)
   compute_twiafc_w<<<gwDim,bDim>>>( nr_of_rows, AFC_diag_i, AFC_diag_data, AFC_offd_i, AFC_offd_data, rsFC );

// 5. Perform matrix-matrix multiplication
   W = hypre_ParCSRMatMat(AFF, AFC );

// 6. Construct P from matrix product W.
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

   gDim = hypre_GetDefaultCUDAGridDimension( P_nr_of_rows, "thread", bDim);

   thrust::transform( thrust::device, CF_marker, &CF_marker[P_nr_of_rows], PWoffset, is_positive<HYPRE_Int>());
   thrust::exclusive_scan(thrust::device, PWoffset, &PWoffset[P_nr_of_rows+1], PWoffset );

   extendWtoP<<<gDim,bDim>>>( P_nr_of_rows, CF_marker, PWoffset, W_diag_i, W_diag_j, W_diag_data,
                              P_diag_i, P_diag_j, P_diag_data, W_offd_i, P_offd_i );

   hypre_CSRMatrixI(    hypre_ParCSRMatrixDiag(W) ) = P_diag_i;
   hypre_CSRMatrixJ(    hypre_ParCSRMatrixDiag(W) ) = P_diag_j;
   hypre_CSRMatrixData( hypre_ParCSRMatrixDiag(W) ) = P_diag_data;

   hypre_CSRMatrixI(    hypre_ParCSRMatrixOffd(W) ) = P_offd_i;

   hypre_CSRMatrixNumNonzeros( hypre_ParCSRMatrixDiag(W) ) = P_diag_nnz;
   hypre_CSRMatrixNumRows(     hypre_ParCSRMatrixDiag(W) ) = P_nr_of_rows;
   hypre_CSRMatrixNumRows(     hypre_ParCSRMatrixOffd(W) ) = P_nr_of_rows;

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncationDevice(W, trunc_factor, max_elmts );
   }

   *P_ptr = W;

// 7. Free memory
   hypre_TFree( W_diag_i, HYPRE_MEMORY_DEVICE );
   hypre_TFree( W_diag_j, HYPRE_MEMORY_DEVICE );
   hypre_TFree( W_diag_data, HYPRE_MEMORY_DEVICE );
   hypre_TFree( W_offd_i, HYPRE_MEMORY_DEVICE );

   hypre_TFree( PWoffset, HYPRE_MEMORY_DEVICE );
   hypre_TFree( rsFF, HYPRE_MEMORY_DEVICE );
   hypre_TFree( rsFC, HYPRE_MEMORY_DEVICE );
   hypre_TFree( tmp, HYPRE_MEMORY_DEVICE );
   /*   hypre_TFree( SFF_diag_f, HYPRE_MEMORY_DEVICE );
   hypre_TFree( SFF_offd_f, HYPRE_MEMORY_DEVICE );
   hypre_TFree( SFC_diag_f, HYPRE_MEMORY_DEVICE );
   hypre_TFree( SFC_offd_f, HYPRE_MEMORY_DEVICE );*/

   return hypre_error_flag;
}

//-----------------------------------------------------------------------
__global__ void create_flag_array( HYPRE_Int nr_of_rows, HYPRE_Int* A_i, HYPRE_Int* A_j,
                        HYPRE_Int* S_i, HYPRE_Int* S_j, HYPRE_Int* S_flag )
{
   HYPRE_Int i= hypre_cuda_get_grid_thread_id<1,1>(), ind; //threadIdx.x + blockIdx.x * blockDim.x,
   //   const HYPRE_Int nthreads = gridDim.x * blockDim.x;
   HYPRE_Int sind, sf;
   if( i < nr_of_rows )
   {
      sind=S_i[i];
      for( ind=A_i[i] ; ind < A_i[i+1] ; ind++ )
      {
         sf = A_j[ind]==S_j[sind];
         S_flag[ind]=sf;
         sind += sf;
         //      cind=A_j[ind];
         //      if( cind == sind )
         //      {
         //         S_flag[ind] = 1;
         //         sind++;
         //      }
         //      else
         //         S_flag[ind] = 0;
      }
   }
}

//-----------------------------------------------------------------------
__global__ void compute_rowsums( HYPRE_Int nr_of_rows,
                      HYPRE_Int* A_diag_i, HYPRE_Real* A_diag_data,
                      HYPRE_Int* A_offd_i, HYPRE_Real* A_offd_data,
                      HYPRE_Int* S_diag_flag, HYPRE_Int* S_offd_flag, HYPRE_Real* rs )
{
 // one thread/row
   //   for( HYPRE_Int ind=A_diag_i[i] ; ind < A_diag_i[i+1] ; ind++ )
   //      rs[2*i+S_diag_flag[ind]] += A_diag_data[ind];
   //   for( HYPRE_Int ind=A_offd_i[i] ; ind < A_offd_i[i+1] ; ind++ )
   //      rs[2*i+S_offd_flag[ind]] += A_offd_data[ind];

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
         fl   = read_only_load(S_diag_flag + i);
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
         fl   = read_only_load(S_offd_flag + i);
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
                     HYPRE_Int* AFF_offd_i, HYPRE_Real* AFF_offd_data,
                     HYPRE_Real* rsFF, HYPRE_Real* rsFC )
{
   // One thread/row
   HYPRE_Real iscale;
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>(), ind;

   // diagonal element
   if( i < nr_of_rows )
   {
      ind=AFF_diag_i[i];
      iscale = -1.0/(rsFF[2*i]+rsFC[2*i]);
      AFF_diag_data[ind] = rsFC[2*i+1]*iscale;
      for( ind=AFF_diag_i[i]+1 ; ind < AFF_diag_i[i+1] ; ind++ )
         AFF_diag_data[ind] *= iscale;
      for( ind=AFF_offd_i[i] ; ind < AFF_offd_i[i+1] ; ind++ )
         AFF_offd_data[ind] *= iscale;
   }
}

//-----------------------------------------------------------------------
__global__ void compute_twiafc_t( HYPRE_Int nr_of_rows,
                       HYPRE_Int* AFC_diag_i, HYPRE_Real* AFC_diag_data,
                       HYPRE_Int* AFC_offd_i, HYPRE_Real* AFC_offd_data,
                       HYPRE_Real* rsFC)
{
   // One thread/row
   HYPRE_Real iscale;
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>(), ind;

   if( i < nr_of_rows )
   {
      iscale = 1.0/read_only_load(rsFC+2*i+1);
      for( ind=AFC_diag_i[i] ; ind < AFC_diag_i[i+1] ; ind++ )
         AFC_diag_data[ind] *= iscale;
      for( ind=AFC_offd_i[i] ; ind < AFC_offd_i[i+1] ; ind++ )
         AFC_offd_data[ind] *= iscale;
   }
}

//-----------------------------------------------------------------------
__global__ void compute_twiafc_w( HYPRE_Int nr_of_rows,
                       HYPRE_Int* AFC_diag_i, HYPRE_Real* AFC_diag_data,
                       HYPRE_Int* AFC_offd_i, HYPRE_Real* AFC_offd_data,
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
   iscale = read_only_load(rsFC+2*row+1);
   iscale = 1.0/iscale;

   for( i = ib + lane ; __any_sync(HYPRE_WARP_FULL_MASK, i < ie); i += HYPRE_WARP_SIZE )
   {
      if( i < ie )
         AFC_diag_data[i] *= iscale;
   }
   if( lane < 2 )
      ibo = read_only_load(AFC_offd_i+row+lane);
   ieo = __shfl_sync(HYPRE_WARP_FULL_MASK, ibo, 1);
   ibo = __shfl_sync(HYPRE_WARP_FULL_MASK, ibo, 0);

   for( i = ibo + lane ; __any_sync(HYPRE_WARP_FULL_MASK, i < ieo); i += HYPRE_WARP_SIZE )
   {
      if( i < ieo )
         AFC_offd_data[i] *= iscale;
   }
}

//-----------------------------------------------------------------------
__global__ void create_CFflag( HYPRE_Int P_nr_of_rows, HYPRE_Int* CF_marker, HYPRE_Int* CFflag )
{
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>();
   CFflag[i] = CF_marker[i]>0;
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
         P_diag_j[pd]    = i;
         P_diag_data[pd] = 1.0;
      }
   }
}

#endif

/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

#if defined(HYPRE_USING_CUDA)

#if 1

__global__ void
hypreCUDAKernel_InterpTruncation( HYPRE_Int   nrows,
                                  HYPRE_Real  trunc_factor,
                                  HYPRE_Int   max_elmts,
                                  HYPRE_Int  *P_i,
                                  HYPRE_Int  *P_j,
                                  HYPRE_Real *P_a)
{
   HYPRE_Real row_max = 0.0, row_sum = 0.0, row_scal = 0.0;
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>();

   if (row >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>(), p, q;

   /* 1. compute row max, rowsum */
   if (lane < 2)
   {
      p = read_only_load(P_i + row + lane);
   }
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int i = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < q); i += HYPRE_WARP_SIZE)
   {
      if (i < q)
      {
         HYPRE_Real v = read_only_load(&P_a[i]);
         row_max = hypre_max(row_max, fabs(v));
         row_sum += v;
      }
   }

   row_max = warp_allreduce_max(row_max) * trunc_factor;
   row_sum = warp_allreduce_sum(row_sum);

   /* 2. mark dropped entries by -1 in P_j, and compute row_scal */
   HYPRE_Int last_pos = -1;
   for (HYPRE_Int i = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < q); i += HYPRE_WARP_SIZE)
   {
      HYPRE_Int cond = 0, cond_prev;

      cond_prev = i == p + lane || warp_allreduce_min(cond);

      if (i < q)
      {
         HYPRE_Real v;
         cond = cond_prev && i < p + max_elmts;
         if (cond)
         {
            v = read_only_load(&P_a[i]);
         }
         cond = cond && fabs(v) >= row_max;

         if (cond)
         {
            last_pos = i;
            row_scal += v;
         }
         else
         {
            P_j[i] = -1;
         }
      }
   }

   row_scal = row_sum / warp_allreduce_sum(row_scal);

   /* 3. scale the row */
   for (HYPRE_Int i = p + lane; i <= last_pos; i += HYPRE_WARP_SIZE)
   {
      P_a[i] *= row_scal;
   }
}

/*-----------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGInterpTruncationDevice( hypre_ParCSRMatrix *P, HYPRE_Real trunc_factor, HYPRE_Int max_elmts )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_INTERP_TRUNC] -= hypre_MPI_Wtime();
#endif

   hypre_CSRMatrix *P_diag      = hypre_ParCSRMatrixDiag(P);
   HYPRE_Int       *P_diag_i    = hypre_CSRMatrixI(P_diag);
   HYPRE_Int       *P_diag_j    = hypre_CSRMatrixJ(P_diag);
   HYPRE_Real      *P_diag_a    = hypre_CSRMatrixData(P_diag);

   hypre_CSRMatrix *P_offd      = hypre_ParCSRMatrixOffd(P);
   HYPRE_Int       *P_offd_i    = hypre_CSRMatrixI(P_offd);
   HYPRE_Int       *P_offd_j    = hypre_CSRMatrixJ(P_offd);
   HYPRE_Real      *P_offd_a    = hypre_CSRMatrixData(P_offd);

   //HYPRE_Int        ncols       = hypre_CSRMatrixNumCols(P_diag);
   HYPRE_Int        nrows       = hypre_CSRMatrixNumRows(P_diag);
   HYPRE_Int        nnz_diag    = hypre_CSRMatrixNumNonzeros(P_diag);
   HYPRE_Int        nnz_offd    = hypre_CSRMatrixNumNonzeros(P_offd);
   HYPRE_Int        nnz_P       = nnz_diag + nnz_offd;
   HYPRE_Int       *P_i         = hypre_TAlloc(HYPRE_Int,  nnz_P,   HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *P_j         = hypre_TAlloc(HYPRE_Int,  nnz_P,   HYPRE_MEMORY_DEVICE);
   HYPRE_Real      *P_a         = hypre_TAlloc(HYPRE_Real, nnz_P,   HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *P_rowptr    = hypre_TAlloc(HYPRE_Int,  nrows+1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *tmp_rowid   = hypre_TAlloc(HYPRE_Int,  nnz_P,   HYPRE_MEMORY_DEVICE);

   HYPRE_Int        memory_loc   = hypre_CSRMatrixMemoryLocation(P_diag);
   HYPRE_Int        new_nnz_diag = 0, new_nnz_offd = 0;
   HYPRE_Int        ierr         = 0;

   /*
   HYPRE_Int        num_procs, my_id;
   MPI_Comm         comm = hypre_ParCSRMatrixComm(P);
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   */

   hypreDevice_CsrRowPtrsToIndices_v2(nrows, P_diag_i, P_i);
   hypreDevice_CsrRowPtrsToIndices_v2(nrows, P_offd_i, P_i + nnz_diag);

   hypre_TMemcpy(P_j, P_diag_j, HYPRE_Int, nnz_diag, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   /* offd col id := -2 - offd col id */
   HYPRE_THRUST_CALL(transform, P_offd_j, P_offd_j + nnz_offd, P_j + nnz_diag, -_1 - 2);

   hypre_TMemcpy(P_a,            P_diag_a, HYPRE_Real, nnz_diag, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(P_a + nnz_diag, P_offd_a, HYPRE_Real, nnz_offd, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* sort rows based on (rowind, abs(P_a)) */
   hypreDevice_StableSortByTupleKey(nnz_P, P_i, P_a, P_j, 1);

   hypreDevice_CsrRowIndicesToPtrs_v2(nrows, nnz_P, P_i, P_rowptr);

   /* truncate P, unwanted entries are marked -1 in P_j */
   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(nrows, "warp", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_InterpTruncation, gDim, bDim,
                      nrows, trunc_factor, max_elmts, P_rowptr, P_j, P_a );

   /* build new P_diag and P_offd */
   if (nnz_diag)
   {
      auto new_end = HYPRE_THRUST_CALL(
            copy_if,
            thrust::make_zip_iterator(thrust::make_tuple(P_i,       P_j,       P_a)),
            thrust::make_zip_iterator(thrust::make_tuple(P_i+nnz_P, P_j+nnz_P, P_a+nnz_P)),
            P_j,
            thrust::make_zip_iterator(thrust::make_tuple(tmp_rowid, P_diag_j,  P_diag_a)),
            is_nonnegative() );

      new_nnz_diag = thrust::get<0>(new_end.get_iterator_tuple()) - tmp_rowid;

      hypre_assert(new_nnz_diag <= nnz_diag);

      hypreDevice_CsrRowIndicesToPtrs_v2(nrows, new_nnz_diag, tmp_rowid, P_diag_i);
   }

   if (nnz_offd)
   {
      less_than pred(-1);
      auto new_end = HYPRE_THRUST_CALL(
            copy_if,
            thrust::make_zip_iterator(thrust::make_tuple(P_i,       P_j,       P_a)),
            thrust::make_zip_iterator(thrust::make_tuple(P_i+nnz_P, P_j+nnz_P, P_a+nnz_P)),
            P_j,
            thrust::make_zip_iterator(thrust::make_tuple(tmp_rowid, P_offd_j,  P_offd_a)),
            pred );

      new_nnz_offd = thrust::get<0>(new_end.get_iterator_tuple()) - tmp_rowid;

      hypre_assert(new_nnz_offd <= nnz_offd);

      HYPRE_THRUST_CALL(transform, P_offd_j, P_offd_j + new_nnz_offd, P_offd_j, -_1 - 2);

      hypreDevice_CsrRowIndicesToPtrs_v2(nrows, new_nnz_offd, tmp_rowid, P_offd_i);
   }

   /*
   printf("nnz_diag %d, new nnz_diag %d\n", nnz_diag, new_nnz_diag);
   printf("nnz_offd %d, new nnz_offd %d\n", nnz_offd, new_nnz_offd);
   */

   hypre_CSRMatrixJ   (P_diag) = hypre_TReAlloc(P_diag_j, HYPRE_Int,  new_nnz_diag, memory_loc);
   hypre_CSRMatrixData(P_diag) = hypre_TReAlloc(P_diag_a, HYPRE_Real, new_nnz_diag, memory_loc);
   hypre_CSRMatrixJ   (P_offd) = hypre_TReAlloc(P_offd_j, HYPRE_Int,  new_nnz_offd, memory_loc);
   hypre_CSRMatrixData(P_offd) = hypre_TReAlloc(P_offd_a, HYPRE_Real, new_nnz_offd, memory_loc);
   hypre_CSRMatrixNumNonzeros(P_diag) = new_nnz_diag;
   hypre_CSRMatrixNumNonzeros(P_offd) = new_nnz_offd;

   hypre_TFree(P_i,       HYPRE_MEMORY_DEVICE);
   hypre_TFree(P_j,       HYPRE_MEMORY_DEVICE);
   hypre_TFree(P_a,       HYPRE_MEMORY_DEVICE);
   hypre_TFree(P_rowptr,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(tmp_rowid, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_INTERP_TRUNC] += hypre_MPI_Wtime();
#endif

   return ierr;
}

#else

__global__ void hypre_BoomerAMGInterpTruncationDevice_dev1( HYPRE_Int num_rows_P,
      HYPRE_Int* P_diag_i,
      HYPRE_Int* P_diag_j,
      HYPRE_Real* P_diag_data,
      HYPRE_Int* P_offd_i,
      HYPRE_Int* P_offd_j,
      HYPRE_Real* P_offd_data,
      HYPRE_Int* P_aux_diag_i,
      HYPRE_Int* P_aux_offd_i,
      HYPRE_Real trunc_factor );

__global__ void hypre_BoomerAMGInterpTruncationDevice_dev2( HYPRE_Int  num_rows_P,
      HYPRE_Int* P_diag_i,
      HYPRE_Int* P_offd_i,
      HYPRE_Int* P_aux_diag_i,
      HYPRE_Int* P_aux_offd_i );

__global__ void hypre_BoomerAMGInterpTruncationDevice_dev3( HYPRE_Int   num_rows_P,
      HYPRE_Int*  P_diag_i,
      HYPRE_Int*  P_diag_j,
      HYPRE_Real* P_diag_data,
      HYPRE_Int*  P_offd_i,
      HYPRE_Int*  P_offd_j,
      HYPRE_Real* P_offd_data,
      HYPRE_Int*  P_aux_diag_i,
      HYPRE_Int*  P_aux_offd_i,
      HYPRE_Int   max_elements );

__global__ void hypre_BoomerAMGInterpTruncationDevice_dev4( HYPRE_Int   num_rows_P,
      HYPRE_Int*  P_diag_i,
      HYPRE_Int*  P_diag_j,
      HYPRE_Real* P_diag_data,
      HYPRE_Int*  P_diag_i_new,
      HYPRE_Int*  P_diag_j_new,
      HYPRE_Real* P_diag_data_new,
      HYPRE_Int*  P_offd_i,
      HYPRE_Int*  P_offd_j,
      HYPRE_Real* P_offd_data,
      HYPRE_Int*  P_offd_i_new,
      HYPRE_Int*  P_offd_j_new,
      HYPRE_Real* P_offd_data_new );

__device__ void hypre_qsort2abs_dev( HYPRE_Int *v,
      HYPRE_Real *w,
      HYPRE_Int  left,
      HYPRE_Int  right );

__device__ void hypre_isort2abs_dev( HYPRE_Int  *v,
                                     HYPRE_Real *w,
                                     HYPRE_Int  n );

/*-----------------------------------------------------------------------*/
/*
   typedef thrust::zip_iterator<thrust::tuple<thrust::device_vector<HYPRE_Int>::iterator,thrust::device_vector<HYPRE_Int>::iterator > > ZIvec2Iterator;
   struct compare_tuple
   {
      template<typename Tuple>
      __host__ __device__
      bool operator()(Tuple lhs, Tuple rhs)
      {
         return thrust::get<0>(lhs)+thrust::get<1>(lhs) < thrust::get<0>(rhs)+thrust::get<1>(rhs);
      }
   };
*/

/*-----------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGInterpTruncationDevice( hypre_ParCSRMatrix *P, HYPRE_Real trunc_factor, HYPRE_Int max_elmts)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_INTERP_TRUNC] -= hypre_MPI_Wtime();
#endif

   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);
   HYPRE_Int *P_diag_i = hypre_CSRMatrixI(P_diag);
   HYPRE_Int *P_diag_j = hypre_CSRMatrixJ(P_diag);
   HYPRE_Real *P_diag_data = hypre_CSRMatrixData(P_diag);
   HYPRE_Int *P_diag_j_new;
   HYPRE_Real *P_diag_data_new;

   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(P);
   HYPRE_Int *P_offd_i = hypre_CSRMatrixI(P_offd);
   HYPRE_Int *P_offd_j = hypre_CSRMatrixJ(P_offd);
   HYPRE_Real *P_offd_data = hypre_CSRMatrixData(P_offd);
   HYPRE_Int *P_offd_j_new;
   HYPRE_Real *P_offd_data_new;
   HYPRE_Int* P_aux_diag_i=NULL;
   HYPRE_Int* P_aux_offd_i=NULL;
   HYPRE_Int* nel_per_row;

   HYPRE_Int n_fine = hypre_CSRMatrixNumRows(P_diag);
   HYPRE_Int P_diag_size;
   HYPRE_Int P_offd_size;
   HYPRE_Int mx_row;
   HYPRE_Int limit=1048576;/* arbitrarily choosen limit on CUDA grid size, most devices seem to have 2^31 as grid size limit */
   bool truncated = false;

   dim3 grid, block(32,1,1);
   grid.x = n_fine/block.x;
   if( n_fine % block.x != 0 )
      grid.x++;
   if( grid.x > limit )
      grid.x = limit;
   grid.y = 1;
   grid.z = 1;

   if( 0.0 < trunc_factor && trunc_factor < 1.0 )
   {
      /* truncate with trunc_factor, return number of remaining elements/row in P_aux_diag_i and P_aux_offd_i */
      P_aux_diag_i = hypre_CTAlloc(HYPRE_Int, n_fine+1, HYPRE_MEMORY_SHARED);
      P_aux_offd_i = hypre_CTAlloc(HYPRE_Int, n_fine+1, HYPRE_MEMORY_SHARED);
      HYPRE_CUDA_LAUNCH( hypre_BoomerAMGInterpTruncationDevice_dev1, grid, block,
                         n_fine, P_diag_i, P_diag_j, P_diag_data,
                         P_offd_i, P_offd_j, P_offd_data,
                         P_aux_diag_i, P_aux_offd_i, trunc_factor );
      truncated = true;
   }

   if( max_elmts > 0 )
   {
      if( !truncated )
      {
         /* If not previously truncated, set up P_aux_diag_i and P_aux_offd_i with full number of elements/row */
         P_aux_diag_i = hypre_CTAlloc(HYPRE_Int, n_fine+1, HYPRE_MEMORY_SHARED);
         P_aux_offd_i = hypre_CTAlloc(HYPRE_Int, n_fine+1, HYPRE_MEMORY_SHARED);
         HYPRE_CUDA_LAUNCH( hypre_BoomerAMGInterpTruncationDevice_dev2, grid,block,
                            n_fine, P_diag_i, P_offd_i, P_aux_diag_i, P_aux_offd_i );
      }
      nel_per_row = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL(transform,&P_aux_diag_i[0],&P_aux_diag_i[n_fine],&P_aux_offd_i[0],&nel_per_row[0],thrust::plus<HYPRE_Int>() );
      mx_row = HYPRE_THRUST_CALL(reduce,&nel_per_row[0],&nel_per_row[n_fine],0,thrust::maximum<HYPRE_Int>());
      hypre_TFree(nel_per_row,HYPRE_MEMORY_DEVICE);

      /* Use zip_iterator to avoid creating help array nel_per_row */
      /*
      ZIvec2Iterator i = thrust::max_element(thrust::device,thrust::make_zip_iterator(&P_aux_diag_i[0],&P_aux_offd_i[0]),
      thrust::make_zip_iterator(&P_aux_diag_i[n_fine],&P_aux_offd_i[n_fine],compare_tuple() ));
      mx_row = thrust::get<0>(*i)+thrust::get<1>(*i);
       */
      if( mx_row > max_elmts )
      {
       /* Truncate with respect to maximum number of elements per row */
         HYPRE_CUDA_LAUNCH( hypre_BoomerAMGInterpTruncationDevice_dev3, grid, block,
                            n_fine, P_diag_i, P_diag_j, P_diag_data,
                            P_offd_i, P_offd_j, P_offd_data, P_aux_diag_i,
                            P_aux_offd_i, max_elmts );
         truncated = true;
      }
   }

   if( truncated )
   {
      cudaDeviceSynchronize();
      /* Matrix has been truncated, reshuffle it into shorter arrays */
      HYPRE_THRUST_CALL(exclusive_scan, &P_aux_diag_i[0],&P_aux_diag_i[n_fine+1],&P_aux_diag_i[0]);
      P_diag_size = P_aux_diag_i[n_fine];
      HYPRE_THRUST_CALL(exclusive_scan, &P_aux_offd_i[0],&P_aux_offd_i[n_fine+1],&P_aux_offd_i[0]);
      P_offd_size = P_aux_offd_i[n_fine];

      P_diag_j_new    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_SHARED);
      P_diag_data_new = hypre_CTAlloc(HYPRE_Real, P_diag_size, HYPRE_MEMORY_SHARED);
      P_offd_j_new    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_SHARED);
      P_offd_data_new = hypre_CTAlloc(HYPRE_Real, P_offd_size, HYPRE_MEMORY_SHARED);

      HYPRE_CUDA_LAUNCH( hypre_BoomerAMGInterpTruncationDevice_dev4, grid,block,
                         n_fine, P_diag_i, P_diag_j, P_diag_data, P_aux_diag_i,
                         P_diag_j_new, P_diag_data_new,
                         P_offd_i, P_offd_j, P_offd_data, P_aux_offd_i,
                         P_offd_j_new, P_offd_data_new );
      cudaDeviceSynchronize();

      //      P_diag_i[n_fine] = P_diag_size ;
      hypre_TFree(P_diag_i, HYPRE_MEMORY_SHARED);
      hypre_TFree(P_diag_j, HYPRE_MEMORY_SHARED);
      hypre_TFree(P_diag_data, HYPRE_MEMORY_SHARED);
      hypre_CSRMatrixI(P_diag) = P_aux_diag_i;
      hypre_CSRMatrixJ(P_diag) = P_diag_j_new;
      hypre_CSRMatrixData(P_diag) = P_diag_data_new;
      hypre_CSRMatrixNumNonzeros(P_diag) = P_diag_size;

      //      P_offd_i[n_fine] = P_offd_size ;
      hypre_TFree(P_offd_i, HYPRE_MEMORY_SHARED);
      hypre_TFree(P_offd_j, HYPRE_MEMORY_SHARED);
      hypre_TFree(P_offd_data, HYPRE_MEMORY_SHARED);
      hypre_CSRMatrixI(P_offd) = P_aux_offd_i;
      hypre_CSRMatrixJ(P_offd) = P_offd_j_new;
      hypre_CSRMatrixData(P_offd) = P_offd_data_new;
      hypre_CSRMatrixNumNonzeros(P_offd) = P_offd_size;
   }
   else if( P_aux_diag_i != NULL )
   {
      hypre_TFree(P_aux_diag_i, HYPRE_MEMORY_SHARED);
      hypre_TFree(P_aux_offd_i, HYPRE_MEMORY_SHARED);
   }
   return 0;
}


/* -----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGInterpTruncationDevice_dev1( HYPRE_Int num_rows_P,
      HYPRE_Int* P_diag_i,
      HYPRE_Int* P_diag_j,
      HYPRE_Real* P_diag_data,
      HYPRE_Int* P_offd_i,
      HYPRE_Int* P_offd_j,
      HYPRE_Real* P_offd_data,
      HYPRE_Int* P_aux_diag_i,
      HYPRE_Int* P_aux_offd_i,
      HYPRE_Real trunc_factor )
   /*
    Perform truncation by eleminating all elements from row i whose absolute value is
    smaller than trunc_factor*max_k|P_{i,k}|.
    The matrix is rescaled after truncation to conserve its row sums.

    Input: num_rows_P - Number of rows of matrix in this MPI-task.
           P_diag_i, P_diag_j, P_diag_data - CSR representation of block diagonal part of matrix.
           P_offd_i, P_offd_j, P_offd_data - CSR representation of off-block diagonal part of matrix.
           trunc_factor - Factor in truncation threshold.

    Output:  P_aux_diag_i - P_aux_diag_i[i] holds the number of non-truncated elements on row i of P_diag.
    P_aux_offd_i - P_aux_offd_i[i] holds the number of non-truncated elements on row i of P_offd.
    P_diag_j, P_diag_data, P_offd_j, P_offd_data - For rows where truncation occurs, elements are
                 reordered to have the non-truncated elements first on each row, and the data arrays are rescaled.

   */
{
   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i, ind, indp,nel_diag, nel_offd;
   HYPRE_Real max_coef, row_sum, row_sum_trunc;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;

   for( i = myid ; i < num_rows_P ; i += nthreads )
   {
  /* 1. Compute maximum absolute value element in row */
      max_coef = 0;
      for (ind = P_diag_i[i]; ind < P_diag_i[i+1]; ind++)
         max_coef = (max_coef < fabs(P_diag_data[ind])) ?
            fabs(P_diag_data[ind]) : max_coef;
      for (ind = P_offd_i[i]; ind < P_offd_i[i+1]; ind++)
         max_coef = (max_coef < fabs(P_offd_data[ind])) ?
            fabs(P_offd_data[ind]) : max_coef;
      max_coef *= trunc_factor;

  /* 2. Eliminate small elements and compress row */
      nel_diag = 0;
      row_sum = 0;
      row_sum_trunc = 0;
      indp = P_diag_i[i];
      for (ind = P_diag_i[i]; ind < P_diag_i[i+1]; ind++)
      {
         row_sum += P_diag_data[ind];
         if( fabs(P_diag_data[ind]) >= max_coef)
         {
            row_sum_trunc += P_diag_data[ind];
            P_diag_data[indp+nel_diag] = P_diag_data[ind];
            P_diag_j[indp+nel_diag++]  = P_diag_j[ind];
         }
      }
      nel_offd = 0;
      indp=P_offd_i[i];
      for (ind = P_offd_i[i]; ind < P_offd_i[i+1]; ind++)
      {
         row_sum += P_offd_data[ind];
         if( fabs(P_offd_data[ind]) >= max_coef)
         {
            row_sum_trunc += P_offd_data[ind];
            P_offd_data[indp+nel_offd] = P_offd_data[ind];
            P_offd_j[indp+nel_offd++]  = P_offd_j[ind];
         }
      }

  /* 3. Rescale row to conserve row sum */
      if( row_sum_trunc != 0 )
      {
         if( row_sum_trunc != row_sum )
         {
            row_sum_trunc = row_sum/row_sum_trunc;
            for (ind = P_diag_i[i]; ind < P_diag_i[i]+nel_diag; ind++)
               P_diag_data[ind] *= row_sum_trunc;
            for (ind = P_offd_i[i]; ind < P_offd_i[i]+nel_offd; ind++)
               P_offd_data[ind] *= row_sum_trunc;
         }
      }
  /* 4. Remember number of elements of compressed matrix */
      P_aux_diag_i[i] = nel_diag;
      P_aux_offd_i[i] = nel_offd;
   }
}

/*-----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGInterpTruncationDevice_dev2( HYPRE_Int  num_rows_P,
      HYPRE_Int* P_diag_i,
      HYPRE_Int* P_offd_i,
      HYPRE_Int* P_aux_diag_i,
      HYPRE_Int* P_aux_offd_i )
/*
   Construct P_aux_diag_i and P_aux_offd_i from a non-truncated matrix.

   Input: num_rows_P - Number of rows of matrix in this MPI-task.
          P_diag_i - CSR vector I of P_diag.
          P_offd_i - CSR vector I of P_offd.

   Output: P_aux_diag_i - P_aux_diag_i[i] holds the number of elements on row i in P_diag.
   P_aux_offd_i - P_aux_offd_i[i] holds the number of elements on row i in P_offd.
 */
{
   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;
   for( i = myid ; i < num_rows_P ; i += nthreads )
      P_aux_diag_i[i] = P_diag_i[i+1]-P_diag_i[i];
   for( i = myid ; i < num_rows_P ; i += nthreads )
      P_aux_offd_i[i] = P_offd_i[i+1]-P_offd_i[i];
}

/*-----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGInterpTruncationDevice_dev3( HYPRE_Int   num_rows_P,
      HYPRE_Int*  P_diag_i,
      HYPRE_Int*  P_diag_j,
      HYPRE_Real* P_diag_data,
      HYPRE_Int*  P_offd_i,
      HYPRE_Int*  P_offd_j,
      HYPRE_Real* P_offd_data,
      HYPRE_Int*  P_aux_diag_i,
      HYPRE_Int*  P_aux_offd_i,
      HYPRE_Int   max_elements )
   /*
    Perform truncation by retaining the max_elements largest (absolute value) elements of each row.
    The matrix is rescaled after truncation to conserve its row sums.

    Input: num_rows_P - Number of rows of matrix in this MPI-task.
           P_diag_i, P_diag_j, P_diag_data - CSR representation of block diagonal part of matrix
           P_offd_i, P_offd_j, P_offd_data - CSR representation of off-block diagonal part of matrix
           P_aux_diag_i - P_aux_diag_i[i] holds the number of non-truncated elements on row i in P_diag.
           P_aux_offd_i - P_aux_offd_i[i] holds the number of non-truncated elements on row i in P_offd.

    Output: P_aux_diag_i, P_aux_offd_i - Updated with the new number of elements per row, after truncation.
            P_diag_j, P_diag_data, P_offd_j, P_offd_data - Reordered so that the first P_aux_diag_i[i] and
                                  the first P_aux_offd_i[i] elements on each row form the truncated matrix.
*/
{
   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i, nel, ind, indo;
   HYPRE_Real row_sum, row_sum_trunc;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;

   for( i = myid ; i < num_rows_P ; i += nthreads )
   {
   /* count number of elements in row */
      nel = P_aux_diag_i[i]+P_aux_offd_i[i];

  /* 0. Do we need to do anything ? */
      if( nel > max_elements )
      {
  /* 1. Save row sum before truncation, for rescaling below */
         row_sum = 0;
         for (ind = P_diag_i[i]; ind < P_diag_i[i]+P_aux_diag_i[i]; ind++)
            row_sum += P_diag_data[ind];
         for (ind = P_offd_i[i]; ind < P_offd_i[i]+P_aux_offd_i[i]; ind++)
            row_sum += P_offd_data[ind];

         /* Sort in place, avoid allocation of extra array */
         hypre_isort2abs_dev(&P_diag_j[P_diag_i[i]], &P_diag_data[P_diag_i[i]], P_aux_diag_i[i] );
         if( P_aux_offd_i[i] > 0 )
         hypre_isort2abs_dev(&P_offd_j[P_offd_i[i]], &P_offd_data[P_offd_i[i]], P_aux_offd_i[i] );
         /* The routine hypre_qsort2abs(v,w,i0,i1) sorts (v,w) in decreasing order w.r.t w */
         /* hypre_qsort2abs_dev(&P_diag_j[i], &P_diag_data[i], 0, P_aux_diag_i[i]-1 );
                 hypre_qsort2abs_dev(&P_offd_j[i], &P_offd_data[i], 0, P_aux_offd_i[i]-1 );*/

  /* 2. Retain the max_elements largest elements, only index of last element
        needs to be computed, since data is now sorted                        */
         nel = 0;
         ind =P_diag_i[i];
         indo=P_offd_i[i];

  /* 2a. Also, keep track of row sum of truncated matrix, for rescaling below */
         row_sum_trunc = 0;
         while( nel < max_elements )
         {
            if( ind  < P_diag_i[i]+P_aux_diag_i[i] && indo < P_offd_i[i]+P_aux_offd_i[i] )
            {
               if( fabs(P_diag_data[ind])>fabs(P_offd_data[indo]) )
               {
                  row_sum_trunc += P_diag_data[ind];
                  ind++;
               }
               else
               {
                  row_sum_trunc += P_offd_data[indo];
                  indo++;
               }
            }
            else if( ind < P_diag_i[i]+P_aux_diag_i[i] )
            {
               row_sum_trunc += P_diag_data[ind];
               ind++;
            }
            else
            {
               row_sum_trunc += P_offd_data[indo];
               indo++;
            }
            nel++;
         }
  /* 3. Remember new row sizes */
         P_aux_diag_i[i] = ind-P_diag_i[i];
         P_aux_offd_i[i] = indo-P_offd_i[i];

  /* 4. Rescale row to conserve row sum */
         if( row_sum_trunc != 0 )
         {
            if( row_sum_trunc != row_sum )
            {
               row_sum_trunc = row_sum/row_sum_trunc;
               for (ind = P_diag_i[i]; ind < P_diag_i[i]+P_aux_diag_i[i]; ind++)
                  P_diag_data[ind] *= row_sum_trunc;
               for (ind = P_offd_i[i]; ind < P_offd_i[i]+P_aux_offd_i[i]; ind++)
                  P_offd_data[ind] *= row_sum_trunc;
            }
         }
      }
   }
}


/*-----------------------------------------------------------------------*/
__global__ void hypre_BoomerAMGInterpTruncationDevice_dev4( HYPRE_Int   num_rows_P,
      HYPRE_Int*  P_diag_i,
      HYPRE_Int*  P_diag_j,
      HYPRE_Real* P_diag_data,
      HYPRE_Int*  P_diag_i_new,
      HYPRE_Int*  P_diag_j_new,
      HYPRE_Real* P_diag_data_new,
      HYPRE_Int*  P_offd_i,
      HYPRE_Int*  P_offd_j,
      HYPRE_Real* P_offd_data,
      HYPRE_Int*  P_offd_i_new,
      HYPRE_Int*  P_offd_j_new,
      HYPRE_Real* P_offd_data_new )
/*
   Copy truncated matrix to smaller storage. In the previous kernels, the number of elements per row
   has been reduced, but the matrix is still stored in the old CSR arrays.

   Input:  num_rows_P - Number of rows of matrix in this MPI-task.
           P_diag_i, P_diag_j, P_diag_data - CSR representation of block diagonal part of matrix
           P_offd_i, P_offd_j, P_offd_data - CSR representation of off-block diagonal part of matrix
           P_diag_i_new - P_diag has been truncated, this is the new CSR I-vector, pointing to beginnings of rows.
           P_offd_i_new - P_offd has been truncated, this is the new CSR I-vector, pointing to beginnings of rows.

   Output: P_diag_j_new, P_diag_data_new, P_offd_j_new, P_offd_data_new - These are the resized CSR arrays of the truncated matrix.
 */
{
   HYPRE_Int myid= threadIdx.x + blockIdx.x * blockDim.x, i, ind, indp, indo;
   const HYPRE_Int nthreads = gridDim.x * blockDim.x;

   for( i = myid ; i < num_rows_P ; i += nthreads )
   {
      //      indp = P_diag_i[i];
      indp = P_diag_i[i]-P_diag_i_new[i];
      for( ind = P_diag_i_new[i] ; ind < P_diag_i_new[i+1]; ind++ )
      {
         // P_diag_j_new[ind]    = P_diag_j[indp];
         // P_diag_data_new[ind] = P_diag_data[indp++];
         P_diag_j_new[ind]    = P_diag_j[indp+ind];
         P_diag_data_new[ind] = P_diag_data[indp+ind];
      }
      //      indo = P_offd_i[i];
      indo = P_offd_i[i]-P_offd_i_new[i];
      for( ind = P_offd_i_new[i] ; ind < P_offd_i_new[i+1]; ind++ )
      {
         // P_offd_j_new[ind]    = P_offd_j[indo];
         // P_offd_data_new[ind] = P_offd_data[indo++];
         P_offd_j_new[ind]    = P_offd_j[indo+ind];
         P_offd_data_new[ind] = P_offd_data[indo+ind];
      }
   }
}

/*-----------------------------------------------------------------------*/
__device__ void hypre_swap2_dev(HYPRE_Int  *v,
      HYPRE_Real *w,
      HYPRE_Int  i,
      HYPRE_Int  j )
{
   HYPRE_Int temp;
   HYPRE_Real temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/*-----------------------------------------------------------------------*/
/* sort both v and w, in place, but based only on entries in w */
/* Sorts in decreasing order */
__device__ void hypre_qsort2abs_dev( HYPRE_Int *v,
      HYPRE_Real *w,
      HYPRE_Int  left,
      HYPRE_Int  right )
{
   HYPRE_Int i, last;
   if (left >= right)
      return;
   hypre_swap2_dev( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (fabs(w[i]) > fabs(w[left]))
      {
         hypre_swap2_dev(v, w, ++last, i);
      }
   hypre_swap2_dev(v, w, left, last);
   hypre_qsort2abs_dev(v, w, left, last-1);
   hypre_qsort2abs_dev(v, w, last+1, right);
}

/*-----------------------------------------------------------------------*/
/* sort both v and w, in place, but based only on entries in w
   Sorts in decreasing order, insertion sort, slower than quicksort
   but avoids compiler warning message on stack size limit unknown. */
__device__ void hypre_isort2abs_dev( HYPRE_Int  *v,
                                     HYPRE_Real *w,
                                     HYPRE_Int  n )
{
   HYPRE_Int i, j, y;
   HYPRE_Real x;
   for( i=1 ; i < n ; i++ )
   {
      x = w[i];
      y = v[i];
      j = i-1;
      while( j >= 0 && fabs(w[j]) < fabs(x) )
      {
         w[j+1] = w[j];
         v[j+1] = v[j];
         j--;
      }
      w[j+1] = x;
      v[j+1] = y;
   }
}

#endif

#endif /* #if defined(HYPRE_USING_CUDA) */

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

__global__ void hypre_BoomerAMGBuildRestrNeumannAIR_assembleRdiag( hypre_DeviceItem &item,
                                                                   HYPRE_Int nr_of_rows,
                                                                   HYPRE_Int *Fmap, HYPRE_Int *Cmap, HYPRE_Int *Z_diag_i, HYPRE_Int *Z_diag_j, HYPRE_Complex *Z_diag_a,
                                                                   HYPRE_Int *R_diag_i, HYPRE_Int *R_diag_j, HYPRE_Complex *R_diag_a);

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildRestrNeumannAIR
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildRestrNeumannAIRDevice( hypre_ParCSRMatrix   *A,
                                           HYPRE_Int            *CF_marker,
                                           HYPRE_BigInt         *num_cpts_global,
                                           HYPRE_Int             num_functions,
                                           HYPRE_Int            *dof_func,
                                           HYPRE_Int             NeumannDeg,
                                           HYPRE_Real            strong_thresholdR,
                                           HYPRE_Real            filter_thresholdR,
                                           HYPRE_Int             debug_flag,
                                           hypre_ParCSRMatrix  **R_ptr)
{
   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommHandle  *comm_handle;

   /* diag part of A */
   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);

   /* Restriction matrix R and CSR's */
   hypre_ParCSRMatrix *R;
   hypre_CSRMatrix *R_diag;

   /* arrays */
   HYPRE_Complex      *R_diag_a;
   HYPRE_Int          *R_diag_i;
   HYPRE_Int          *R_diag_j;
   HYPRE_BigInt       *col_map_offd_R;
   HYPRE_Int           num_cols_offd_R;
   HYPRE_Int           my_id, num_procs;
   HYPRE_BigInt        total_global_cpts;
   HYPRE_Int           nnz_diag, nnz_offd;
   HYPRE_BigInt       *send_buf_i;
   HYPRE_Int           i;

   /* local size */
   HYPRE_Int n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt col_start = hypre_ParCSRMatrixFirstRowIndex(A);

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* global number of C points and my start position */
   if (my_id == (num_procs - 1))
   {
      total_global_cpts = num_cpts_global[1];
   }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /* get AFF and ACF */
   hypre_ParCSRMatrix *AFF, *ACF, *Dinv, *N, *X, *X2, *Z, *Z2;
   if (strong_thresholdR > 0)
   {
      hypre_ParCSRMatrix *S;
      hypre_BoomerAMGCreateSabs(A,
                                strong_thresholdR,
                                0.9,
                                num_functions,
                                dof_func,
                                &S);
      hypre_ParCSRMatrixGenerateFFCFDevice(A, CF_marker, num_cpts_global, S, &ACF, &AFF);
      hypre_ParCSRMatrixDestroy(S);
   }
   else
   {
      hypre_ParCSRMatrixGenerateFFCFDevice(A, CF_marker, num_cpts_global, NULL, &ACF, &AFF);
   }

   HYPRE_Int        n_fpts = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(AFF));
   HYPRE_Int        n_cpts = n_fine - n_fpts;
   hypre_assert(n_cpts == hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(ACF)));

   /* maps from F-pts and C-pts to all points */
   HYPRE_Int       *Fmap = hypre_TAlloc(HYPRE_Int, n_fpts, HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *Cmap = hypre_TAlloc(HYPRE_Int, n_cpts, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   oneapi::dpl::counting_iterator<HYPRE_Int> count(0);
   hypreSycl_copy_if( count,
                      count + n_fine,
                      CF_marker,
                      Fmap,
                      is_negative<HYPRE_Int>());
   hypreSycl_copy_if( count,
                      count + n_fine,
                      CF_marker,
                      Cmap,
                      is_positive<HYPRE_Int>());
#else
   HYPRE_THRUST_CALL( copy_if,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(n_fine),
                      CF_marker,
                      Fmap,
                      is_negative<HYPRE_Int>());
   HYPRE_THRUST_CALL( copy_if,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(n_fine),
                      CF_marker,
                      Cmap,
                      is_positive<HYPRE_Int>());
#endif

   /* setup Dinv = 1/(diagonal of AFF) */
   Dinv = hypre_ParCSRMatrixCreate(comm,
                                   hypre_ParCSRMatrixGlobalNumRows(AFF),
                                   hypre_ParCSRMatrixGlobalNumCols(AFF),
                                   hypre_ParCSRMatrixRowStarts(AFF),
                                   hypre_ParCSRMatrixColStarts(AFF),
                                   0,
                                   hypre_ParCSRMatrixNumRows(AFF),
                                   0);
   hypre_ParCSRMatrixAssumedPartition(Dinv) = hypre_ParCSRMatrixAssumedPartition(AFF);
   hypre_ParCSRMatrixOwnsAssumedPartition(Dinv) = 0;
   hypre_ParCSRMatrixInitialize(Dinv);
   hypre_CSRMatrix *Dinv_diag = hypre_ParCSRMatrixDiag(Dinv);
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::copy,
                      count,
                      count + hypre_CSRMatrixNumRows(Dinv_diag) + 1,
                      hypre_CSRMatrixI(Dinv_diag) );
   HYPRE_ONEDPL_CALL( std::copy,
                      count,
                      count + hypre_CSRMatrixNumRows(Dinv_diag),
                      hypre_CSRMatrixJ(Dinv_diag) );
#else
   HYPRE_THRUST_CALL( copy,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(hypre_CSRMatrixNumRows(Dinv_diag) + 1),
                      hypre_CSRMatrixI(Dinv_diag) );
   HYPRE_THRUST_CALL( copy,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(hypre_CSRMatrixNumRows(Dinv_diag)),
                      hypre_CSRMatrixJ(Dinv_diag) );
#endif
   hypre_CSRMatrixExtractDiagonalDevice(hypre_ParCSRMatrixDiag(AFF), hypre_CSRMatrixData(Dinv_diag),
                                        2);

   /* N = I - D^{-1}*A_FF */
   if (NeumannDeg >= 1)
   {
      N = hypre_ParCSRMatMat(Dinv, AFF);

      hypre_CSRMatrixRemoveDiagonalDevice(hypre_ParCSRMatrixDiag(N));

#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::transform,
                         hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(N)),
                         hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(N)) + hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(
                                                                                                        N)),
                         hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(N)),
                         std::negate<HYPRE_Complex>() );
      HYPRE_ONEDPL_CALL( std::transform,
                         hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(N)),
                         hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(N)) + hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(
                                                                                                        N)),
                         hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(N)),
                         std::negate<HYPRE_Complex>() );
#else
      HYPRE_THRUST_CALL( transform,
                         hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(N)),
                         hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(N)) + hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(
                                                                                                        N)),
                         hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(N)),
                         thrust::negate<HYPRE_Complex>() );
      HYPRE_THRUST_CALL( transform,
                         hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(N)),
                         hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(N)) + hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(
                                                                                                        N)),
                         hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(N)),
                         thrust::negate<HYPRE_Complex>() );
#endif
   }

   /* Z = Acf * (I + N + N^2 + ... + N^k) * D^{-1} */
   if (NeumannDeg < 1)
   {
      Z = ACF;
   }
   else if (NeumannDeg == 1)
   {
      X = hypre_ParCSRMatMat(ACF, N);
      hypre_ParCSRMatrixAdd(1.0, ACF, 1.0, X, &Z);
      hypre_ParCSRMatrixDestroy(X);
   }
   else
   {
      X = hypre_ParCSRMatMat(N, N);
      hypre_ParCSRMatrixAdd(1.0, N, 1.0, X, &Z);
      for (i = 2; i < NeumannDeg; i++)
      {
         X2 = hypre_ParCSRMatMat(X, N);
         hypre_ParCSRMatrixAdd(1.0, Z, 1.0, X2, &Z2);
         hypre_ParCSRMatrixDestroy(X);
         hypre_ParCSRMatrixDestroy(Z);
         Z = Z2;
         X = X2;
      }
      hypre_ParCSRMatrixDestroy(X);
      X = hypre_ParCSRMatMat(ACF, Z);
      hypre_ParCSRMatrixDestroy(Z);
      hypre_ParCSRMatrixAdd(1.0, ACF, 1.0, X, &Z);
      hypre_ParCSRMatrixDestroy(X);
   }

   X = Z;
   Z = hypre_ParCSRMatMat(X, Dinv);

   hypre_ParCSRMatrixDestroy(X);
   hypre_ParCSRMatrixDestroy(Dinv);
   hypre_ParCSRMatrixDestroy(AFF);
   if (NeumannDeg >= 1)
   {
      hypre_ParCSRMatrixDestroy(ACF);
      hypre_ParCSRMatrixDestroy(N);
   }

   hypre_CSRMatrix *Z_diag = hypre_ParCSRMatrixDiag(Z);
   hypre_CSRMatrix *Z_offd = hypre_ParCSRMatrixOffd(Z);
   HYPRE_Complex   *Z_diag_a = hypre_CSRMatrixData(Z_diag);
   HYPRE_Int       *Z_diag_i = hypre_CSRMatrixI(Z_diag);
   HYPRE_Int       *Z_diag_j = hypre_CSRMatrixJ(Z_diag);
   HYPRE_Int        num_cols_offd_Z = hypre_CSRMatrixNumCols(Z_offd);
   HYPRE_Int        nnz_diag_Z = hypre_CSRMatrixNumNonzeros(Z_diag);
   HYPRE_BigInt    *Fmap_offd_global = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_Z,
                                                    HYPRE_MEMORY_DEVICE);

   /* send and recv Fmap (wrt Z): global */
   if (num_procs > 1)
   {
      hypre_MatvecCommPkgCreate(Z);

      hypre_ParCSRCommPkg *comm_pkg_Z = hypre_ParCSRMatrixCommPkg(Z);
      HYPRE_Int num_sends_Z = hypre_ParCSRCommPkgNumSends(comm_pkg_Z);
      HYPRE_Int num_elems_send_Z = hypre_ParCSRCommPkgSendMapStart(comm_pkg_Z, num_sends_Z);
      send_buf_i = hypre_TAlloc(HYPRE_BigInt, num_elems_send_Z, HYPRE_MEMORY_DEVICE);

      hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg_Z);
#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg_Z),
                        hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg_Z) +
                        hypre_ParCSRCommPkgSendMapStart(comm_pkg_Z, num_sends_Z),
                        Fmap,
                        send_buf_i );
      HYPRE_ONEDPL_CALL( std::transform,
                         send_buf_i,
                         send_buf_i + num_elems_send_Z,
                         send_buf_i,
      [y = col_start](auto & x) {return x + y;} );
#else
      HYPRE_THRUST_CALL( gather,
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg_Z),
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg_Z) +
                         hypre_ParCSRCommPkgSendMapStart(comm_pkg_Z, num_sends_Z),
                         Fmap,
                         send_buf_i );
      HYPRE_THRUST_CALL( transform,
                         send_buf_i,
                         send_buf_i + num_elems_send_Z,
                         thrust::make_constant_iterator(col_start),
                         send_buf_i,
                         thrust::plus<HYPRE_BigInt>() );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure send_buf_i is ready before issuing GPU-GPU MPI */
      if (hypre_GetGpuAwareMPI())
      {
         hypre_ForceSyncComputeStream(hypre_handle());
      }
#endif

      comm_handle = hypre_ParCSRCommHandleCreate_v2(21, comm_pkg_Z, HYPRE_MEMORY_DEVICE, send_buf_i,
                                                    HYPRE_MEMORY_DEVICE, Fmap_offd_global);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      hypre_TFree(send_buf_i, HYPRE_MEMORY_DEVICE);
   }

   /* Assemble R = [-Z I] */
   nnz_diag = nnz_diag_Z + n_cpts;
   nnz_offd = hypre_CSRMatrixNumNonzeros(Z_offd);

   /* allocate arrays for R diag */
   R_diag_i = hypre_CTAlloc(HYPRE_Int,  n_cpts + 1, HYPRE_MEMORY_DEVICE);
   R_diag_j = hypre_CTAlloc(HYPRE_Int,  nnz_diag, HYPRE_MEMORY_DEVICE);
   R_diag_a = hypre_CTAlloc(HYPRE_Complex, nnz_diag, HYPRE_MEMORY_DEVICE);

   /* setup R row indices (just Z row indices plus one extra entry for each C-pt)*/
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::transform,
                      Z_diag_i,
                      Z_diag_i + n_cpts + 1,
                      count,
                      R_diag_i,
                      std::plus<HYPRE_Int>() );
#else
   HYPRE_THRUST_CALL( transform,
                      Z_diag_i,
                      Z_diag_i + n_cpts + 1,
                      thrust::make_counting_iterator(0),
                      R_diag_i,
                      thrust::plus<HYPRE_Int>() );
#endif

   /* assemble the diagonal part of R from Z */
   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(n_fine, "warp", bDim);
   HYPRE_GPU_LAUNCH( hypre_BoomerAMGBuildRestrNeumannAIR_assembleRdiag, gDim, bDim,
                     n_cpts, Fmap, Cmap, Z_diag_i, Z_diag_j, Z_diag_a, R_diag_i, R_diag_j, R_diag_a);

   num_cols_offd_R = num_cols_offd_Z;
   col_map_offd_R = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_Z, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(col_map_offd_R, Fmap_offd_global, HYPRE_BigInt, num_cols_offd_Z, HYPRE_MEMORY_HOST,
                 HYPRE_MEMORY_DEVICE);

   /* Now, we should have everything of Parcsr matrix R */
   R = hypre_ParCSRMatrixCreate(comm,
                                total_global_cpts, /* global num of rows */
                                hypre_ParCSRMatrixGlobalNumRows(A), /* global num of cols */
                                num_cpts_global, /* row_starts */
                                hypre_ParCSRMatrixRowStarts(A), /* col_starts */
                                num_cols_offd_R, /* num cols offd */
                                nnz_diag,
                                nnz_offd);

   R_diag = hypre_ParCSRMatrixDiag(R);
   hypre_CSRMatrixData(R_diag) = R_diag_a;
   hypre_CSRMatrixI(R_diag)    = R_diag_i;
   hypre_CSRMatrixJ(R_diag)    = R_diag_j;

   /* R_offd is simply a clone of -Z_offd */
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(R));
   hypre_ParCSRMatrixOffd(R) = hypre_CSRMatrixClone(Z_offd, 1);
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::transform,
                      hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(R)),
                      hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(R)) + hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(
                                                                                                     R)),
                      hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(R)),
                      std::negate<HYPRE_Complex>() );
#else
   HYPRE_THRUST_CALL( transform,
                      hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(R)),
                      hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(R)) + hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(
                                                                                                     R)),
                      hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(R)),
                      thrust::negate<HYPRE_Complex>() );
#endif

   hypre_ParCSRMatrixColMapOffd(R) = col_map_offd_R;

   /* create CommPkg of R */
   hypre_ParCSRMatrixAssumedPartition(R) = hypre_ParCSRMatrixAssumedPartition(A);
   hypre_ParCSRMatrixOwnsAssumedPartition(R) = 0;
   hypre_MatvecCommPkgCreate(R);

   /* Filter small entries from R */
   if (filter_thresholdR > 0)
   {
      hypre_ParCSRMatrixDropSmallEntries(R, filter_thresholdR, -1);
   }

   *R_ptr = R;

   hypre_ParCSRMatrixDestroy(Z);
   hypre_TFree(Fmap, HYPRE_MEMORY_DEVICE);
   hypre_TFree(Cmap, HYPRE_MEMORY_DEVICE);
   hypre_TFree(Fmap_offd_global, HYPRE_MEMORY_DEVICE);

   return 0;
}

/*-----------------------------------------------------------------------*/
__global__ void
hypre_BoomerAMGBuildRestrNeumannAIR_assembleRdiag( hypre_DeviceItem    &item,
                                                   HYPRE_Int      nr_of_rows,
                                                   HYPRE_Int     *Fmap,
                                                   HYPRE_Int     *Cmap,
                                                   HYPRE_Int     *Z_diag_i,
                                                   HYPRE_Int     *Z_diag_j,
                                                   HYPRE_Complex *Z_diag_a,
                                                   HYPRE_Int     *R_diag_i,
                                                   HYPRE_Int     *R_diag_j,
                                                   HYPRE_Complex *R_diag_a)
{
   /*-----------------------------------------------------------------------*/
   /* Assemble diag part of R = [-Z I]

      Input: nr_of_rows - Number of rows in matrix (local in processor)
             CSR represetnation of Z diag, assuming column indices of Z are
             already mapped appropriately

      Output: CSR representation of R diag
    */
   /*-----------------------------------------------------------------------*/

   HYPRE_Int i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nr_of_rows)
   {
      return;
   }

   HYPRE_Int p = 0, q, pZ = 0;
   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);

   /* diag part */
   if (lane < 2)
   {
      p = read_only_load(R_diag_i + i + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);
   if (lane < 1)
   {
      pZ = read_only_load(Z_diag_i + i + lane);
   }
   pZ = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pZ, 0);

   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      if (j == q - 1)
      {
         R_diag_j[j] = Cmap[i];
         R_diag_a[j] = 1.0;
      }
      else
      {
         HYPRE_Int jZ = pZ + (j - p);
         R_diag_j[j] = Fmap[ Z_diag_j[jZ] ];
         R_diag_a[j] = -Z_diag_a[jZ];
      }
   }
}

#if !defined(HYPRE_USING_SYCL)
struct setTo1minus1 : public thrust::unary_function<HYPRE_Int, HYPRE_Int>
{
   __host__ __device__ HYPRE_Int operator()(const HYPRE_Int &x) const
   {
      return x > 0 ? 1 : -1;
   }
};
#endif

HYPRE_Int
hypre_BoomerAMGCFMarkerTo1minus1Device( HYPRE_Int *CF_marker,
                                        HYPRE_Int size )
{
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::transform,
                      CF_marker,
                      CF_marker + size,
                      CF_marker,
   [] (const auto & x) {return x > 0 ? 1 : -1;} );
#else
   HYPRE_THRUST_CALL( transform,
                      CF_marker,
                      CF_marker + size,
                      CF_marker,
                      setTo1minus1() );
#endif

   return hypre_error_flag;
}

#endif // defined(HYPRE_USING_GPU)

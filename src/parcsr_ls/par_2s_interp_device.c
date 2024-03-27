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

#if defined(HYPRE_USING_SYCL)
SYCL_EXTERNAL
#endif
__global__ void hypreGPUKernel_compute_weak_rowsums( hypre_DeviceItem &item, HYPRE_Int nr_of_rows,
                                                     bool has_offd,
                                                     HYPRE_Int *CF_marker, HYPRE_Int *A_diag_i, HYPRE_Complex *A_diag_a, HYPRE_Int *S_diag_j,
                                                     HYPRE_Int *A_offd_i, HYPRE_Complex *A_offd_a, HYPRE_Int *S_offd_j, HYPRE_Real *rs, HYPRE_Int flag );

__global__ void hypreGPUKernel_MMInterpScaleAFF( hypre_DeviceItem &item, HYPRE_Int AFF_nrows,
                                                 HYPRE_Int *AFF_diag_i,
                                                 HYPRE_Int *AFF_diag_j, HYPRE_Complex *AFF_diag_a, HYPRE_Int *AFF_offd_i, HYPRE_Int *AFF_offd_j,
                                                 HYPRE_Complex *AFF_offd_a, HYPRE_Complex *beta_diag, HYPRE_Complex *beta_offd, HYPRE_Int *F2_to_F,
                                                 HYPRE_Real *rsW );

#if defined(HYPRE_USING_SYCL)
SYCL_EXTERNAL
#endif
__global__ void hypreGPUKernel_compute_dlam_dtmp( hypre_DeviceItem &item, HYPRE_Int nr_of_rows,
                                                  HYPRE_Int *AFF_diag_i,
                                                  HYPRE_Int *AFF_diag_j, HYPRE_Complex *AFF_diag_data, HYPRE_Int *AFF_offd_i,
                                                  HYPRE_Complex *AFF_offd_data, HYPRE_Complex *rsFC, HYPRE_Complex *dlam, HYPRE_Complex *dtmp );

__global__ void hypreGPUKernel_MMPEInterpScaleAFF( hypre_DeviceItem &item, HYPRE_Int AFF_nrows,
                                                   HYPRE_Int *AFF_diag_i,
                                                   HYPRE_Int *AFF_diag_j, HYPRE_Complex *AFF_diag_a, HYPRE_Int *AFF_offd_i, HYPRE_Int *AFF_offd_j,
                                                   HYPRE_Complex *AFF_offd_a, HYPRE_Complex *tmp_diag, HYPRE_Complex *tmp_offd,
                                                   HYPRE_Complex *lam_diag, HYPRE_Complex *lam_offd, HYPRE_Int *F2_to_F, HYPRE_Real *rsW );

/*--------------------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildModPartialExtInterpDevice( hypre_ParCSRMatrix  *A,
                                               HYPRE_Int           *CF_marker,
                                               hypre_ParCSRMatrix  *S,
                                               HYPRE_BigInt        *num_cpts_global,     /* C2 */
                                               HYPRE_BigInt        *num_old_cpts_global, /* C2 + F2 */
                                               HYPRE_Int            debug_flag,
                                               HYPRE_Real           trunc_factor,
                                               HYPRE_Int            max_elmts,
                                               hypre_ParCSRMatrix **P_ptr )
{
   HYPRE_Int           A_nr_local   = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix    *A_diag       = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i     = hypre_CSRMatrixI(A_diag);
   hypre_CSRMatrix    *A_offd       = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Int           A_offd_nnz   = hypre_CSRMatrixNumNonzeros(A_offd);
   HYPRE_Complex      *Dbeta, *Dbeta_offd, *rsWA, *rsW;
   hypre_ParCSRMatrix *As_F2F, *As_FC, *W, *P;

   hypre_BoomerAMGMakeSocFromSDevice(A, S);

   HYPRE_Int          *Soc_diag_j   = hypre_ParCSRMatrixSocDiagJ(S);
   HYPRE_Int          *Soc_offd_j   = hypre_ParCSRMatrixSocOffdJ(S);

   /* As_F2F = As_{F2, F}, As_FC = As_{F, C2} */
   hypre_ParCSRMatrixGenerateFFFC3Device(A, CF_marker, num_cpts_global, S, &As_FC, &As_F2F);

   HYPRE_Int AFC_nr_local = hypre_ParCSRMatrixNumRows(As_FC);
   HYPRE_Int AF2F_nr_local = hypre_ParCSRMatrixNumRows(As_F2F);

   /* row sum of AFC, i.e., D_beta */
   Dbeta = hypre_TAlloc(HYPRE_Complex, AFC_nr_local, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixDiag(As_FC), NULL, NULL, Dbeta, 0, 1.0, "set");
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixOffd(As_FC), NULL, NULL, Dbeta, 0, 1.0, "add");

   /* collect off-processor D_beta */
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(As_F2F);
   hypre_ParCSRCommHandle *comm_handle;
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(As_F2F);
      comm_pkg = hypre_ParCSRMatrixCommPkg(As_F2F);
   }
   Dbeta_offd = hypre_TAlloc(HYPRE_Complex, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(As_F2F)),
                             HYPRE_MEMORY_DEVICE);
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int num_elmts_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   HYPRE_Complex *send_buf = hypre_TAlloc(HYPRE_Complex, num_elmts_send, HYPRE_MEMORY_DEVICE);
   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
#if defined(HYPRE_USING_SYCL)
   hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                     Dbeta,
                     send_buf );
#else
   HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                      Dbeta,
                      send_buf );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }
#endif

   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg, HYPRE_MEMORY_DEVICE, send_buf,
                                                 HYPRE_MEMORY_DEVICE, Dbeta_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   hypre_TFree(send_buf, HYPRE_MEMORY_DEVICE);

   /* weak row sum and diagonal, i.e., DF2F2 + Dgamma */
   rsWA = hypre_TAlloc(HYPRE_Complex, A_nr_local, HYPRE_MEMORY_DEVICE);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(A_nr_local, "warp", bDim);

   /* only for rows corresponding to F2 (notice flag == -1) */
   HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_weak_rowsums,
                     gDim, bDim,
                     A_nr_local,
                     A_offd_nnz > 0,
                     CF_marker,
                     A_diag_i,
                     A_diag_data,
                     Soc_diag_j,
                     A_offd_i,
                     A_offd_data,
                     Soc_offd_j,
                     rsWA,
                     -1 );

   rsW = hypre_TAlloc(HYPRE_Complex, AF2F_nr_local, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   HYPRE_Complex *new_end = hypreSycl_copy_if( rsWA,
                                               rsWA + A_nr_local,
                                               CF_marker,
                                               rsW,
                                               equal<HYPRE_Int>(-2) );
#else
   HYPRE_Complex *new_end = HYPRE_THRUST_CALL( copy_if,
                                               rsWA,
                                               rsWA + A_nr_local,
                                               CF_marker,
                                               rsW,
                                               equal<HYPRE_Int>(-2) );
#endif

   hypre_assert(new_end - rsW == AF2F_nr_local);

   hypre_TFree(rsWA, HYPRE_MEMORY_DEVICE);

   /* map from F2 to F */
   HYPRE_Int *map_to_F = hypre_TAlloc(HYPRE_Int, A_nr_local, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,              is_negative<HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + A_nr_local, is_negative<HYPRE_Int>()),
                      map_to_F,
                      HYPRE_Int(0) );/* *MUST* pass init value since input and output types diff. */
#else
   HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,              is_negative<HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + A_nr_local, is_negative<HYPRE_Int>()),
                      map_to_F,
                      HYPRE_Int(0) );/* *MUST* pass init value since input and output types diff. */
#endif

   HYPRE_Int *map_F2_to_F = hypre_TAlloc(HYPRE_Int, AF2F_nr_local, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   HYPRE_Int *tmp_end = hypreSycl_copy_if( map_to_F,
                                           map_to_F + A_nr_local,
                                           CF_marker,
                                           map_F2_to_F,
                                           equal<HYPRE_Int>(-2) );
#else
   HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( copy_if,
                                           map_to_F,
                                           map_to_F + A_nr_local,
                                           CF_marker,
                                           map_F2_to_F,
                                           equal<HYPRE_Int>(-2) );
#endif

   hypre_assert(tmp_end - map_F2_to_F == AF2F_nr_local);

   hypre_TFree(map_to_F, HYPRE_MEMORY_DEVICE);

   /* add to rsW those in AF2F that correspond to Dbeta == 0
    * diagnoally scale As_F2F (from both sides) and replace the diagonal */
   gDim = hypre_GetDefaultDeviceGridDimension(AF2F_nr_local, "warp", bDim);

   HYPRE_Int *As_F2F_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(As_F2F));
   HYPRE_Int *As_F2F_diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(As_F2F));
   HYPRE_Complex *As_F2F_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(As_F2F));
   HYPRE_Int *As_F2F_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(As_F2F));
   HYPRE_Int *As_F2F_offd_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(As_F2F));
   HYPRE_Complex *As_F2F_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(As_F2F));
   HYPRE_GPU_LAUNCH( hypreGPUKernel_MMInterpScaleAFF,
                     gDim, bDim,
                     AF2F_nr_local,
                     As_F2F_diag_i,
                     As_F2F_diag_j,
                     As_F2F_diag_data,
                     As_F2F_offd_i,
                     As_F2F_offd_j,
                     As_F2F_offd_data,
                     Dbeta,
                     Dbeta_offd,
                     map_F2_to_F,
                     rsW );

   hypre_TFree(Dbeta, HYPRE_MEMORY_DEVICE);
   hypre_TFree(Dbeta_offd, HYPRE_MEMORY_DEVICE);
   hypre_TFree(map_F2_to_F, HYPRE_MEMORY_DEVICE);
   hypre_TFree(rsW, HYPRE_MEMORY_DEVICE);

   /* Perform matrix-matrix multiplication */
   W = hypre_ParCSRMatMatDevice(As_F2F, As_FC);

   hypre_ParCSRMatrixDestroy(As_F2F);
   hypre_ParCSRMatrixDestroy(As_FC);

   /* Construct P from matrix product W */
   HYPRE_Int     *P_diag_i, *P_diag_j, *P_offd_i;
   HYPRE_Complex *P_diag_data;
   HYPRE_Int      P_nr_local = A_nr_local - (AFC_nr_local - AF2F_nr_local);
   HYPRE_Int      P_diag_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)) +
                               hypre_ParCSRMatrixNumCols(W);

   hypre_assert(P_nr_local == hypre_ParCSRMatrixNumRows(W) + hypre_ParCSRMatrixNumCols(W));

   P_diag_i    = hypre_TAlloc(HYPRE_Int,     P_nr_local + 1, HYPRE_MEMORY_DEVICE);
   P_diag_j    = hypre_TAlloc(HYPRE_Int,     P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(HYPRE_Complex, P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_offd_i    = hypre_TAlloc(HYPRE_Int,     P_nr_local + 1, HYPRE_MEMORY_DEVICE);

   HYPRE_Int *C2F2_marker = hypre_TAlloc(HYPRE_Int, P_nr_local, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   tmp_end = hypreSycl_copy_if( CF_marker,
                                CF_marker + A_nr_local,
                                CF_marker,
                                C2F2_marker,
                                out_of_range<HYPRE_Int>(-1, 0) /* -2 or 1 */ );
#else
   tmp_end = HYPRE_THRUST_CALL( copy_if,
                                CF_marker,
                                CF_marker + A_nr_local,
                                CF_marker,
                                C2F2_marker,
                                out_of_range<HYPRE_Int>(-1, 0) /* -2 or 1 */ );
#endif

   hypre_assert(tmp_end - C2F2_marker == P_nr_local);

   hypreDevice_extendWtoP( P_nr_local,
                           AF2F_nr_local,
                           hypre_ParCSRMatrixNumCols(W),
                           C2F2_marker,
                           hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(W)),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(W)),
                           P_offd_i );

   hypre_TFree(C2F2_marker, HYPRE_MEMORY_DEVICE);

   // final P
   P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(W) + hypre_ParCSRMatrixGlobalNumCols(W),
                                hypre_ParCSRMatrixGlobalNumCols(W),
                                num_old_cpts_global,
                                num_cpts_global,
                                hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(W)),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(W)));

   hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W))    = NULL;
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W)) = NULL;

   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(P)) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(P)) = HYPRE_MEMORY_DEVICE;

   hypre_ParCSRMatrixDeviceColMapOffd(P) = hypre_ParCSRMatrixDeviceColMapOffd(W);
   hypre_ParCSRMatrixColMapOffd(P)       = hypre_ParCSRMatrixColMapOffd(W);
   hypre_ParCSRMatrixDeviceColMapOffd(W) = NULL;
   hypre_ParCSRMatrixColMapOffd(W)       = NULL;

   hypre_ParCSRMatrixNumNonzeros(P)  = hypre_ParCSRMatrixNumNonzeros(W) +
                                       hypre_ParCSRMatrixGlobalNumCols(W);
   hypre_ParCSRMatrixDNumNonzeros(P) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);

   hypre_ParCSRMatrixDestroy(W);

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts );
   }

   *P_ptr = P;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildModPartialExtPEInterpDevice( hypre_ParCSRMatrix  *A,
                                                 HYPRE_Int           *CF_marker,
                                                 hypre_ParCSRMatrix  *S,
                                                 HYPRE_BigInt        *num_cpts_global,     /* C2 */
                                                 HYPRE_BigInt        *num_old_cpts_global, /* C2 + F2 */
                                                 HYPRE_Int            debug_flag,
                                                 HYPRE_Real           trunc_factor,
                                                 HYPRE_Int            max_elmts,
                                                 hypre_ParCSRMatrix **P_ptr )
{
   HYPRE_Int           A_nr_local   = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix    *A_diag       = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i     = hypre_CSRMatrixI(A_diag);
   hypre_CSRMatrix    *A_offd       = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Int           A_offd_nnz   = hypre_CSRMatrixNumNonzeros(A_offd);
   HYPRE_Complex      *Dbeta, *rsWA, *rsW, *dlam, *dlam_offd, *dtmp, *dtmp_offd;
   hypre_ParCSRMatrix *As_F2F, *As_FF, *As_FC, *W, *P;

   hypre_BoomerAMGMakeSocFromSDevice(A, S);

   HYPRE_Int          *Soc_diag_j   = hypre_ParCSRMatrixSocDiagJ(S);
   HYPRE_Int          *Soc_offd_j   = hypre_ParCSRMatrixSocOffdJ(S);

   /* As_F2F = As_{F2, F}, As_FC = As_{F, C2} */
   hypre_ParCSRMatrixGenerateFFFC3Device(A, CF_marker, num_cpts_global, S, &As_FC, &As_F2F);

   HYPRE_Int AFC_nr_local = hypre_ParCSRMatrixNumRows(As_FC);
   HYPRE_Int AF2F_nr_local = hypre_ParCSRMatrixNumRows(As_F2F);

   /* row sum of AFC, i.e., D_beta */
   Dbeta = hypre_TAlloc(HYPRE_Complex, AFC_nr_local, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixDiag(As_FC), NULL, NULL, Dbeta, 0, 1.0, "set");
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixOffd(As_FC), NULL, NULL, Dbeta, 0, 1.0, "add");

   /* As_FF = As_{F,F} */
   hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, num_cpts_global, S, NULL, &As_FF);

   hypre_assert(AFC_nr_local == hypre_ParCSRMatrixNumRows(As_FF));

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(AFC_nr_local, "warp", bDim);

   /* Generate D_lambda in the paper: D_beta + (row sum of AFF without diagonal elements / row_nnz) */
   /* Generate D_tmp, i.e., D_mu / D_lambda */
   dlam = hypre_TAlloc(HYPRE_Complex, AFC_nr_local, HYPRE_MEMORY_DEVICE);
   dtmp = hypre_TAlloc(HYPRE_Complex, AFC_nr_local, HYPRE_MEMORY_DEVICE);

   HYPRE_Int *As_FF_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(As_FF));
   HYPRE_Int *As_FF_diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(As_FF));
   HYPRE_Complex *As_FF_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(As_FF));
   HYPRE_Int *As_FF_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(As_FF));
   HYPRE_Complex *As_FF_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(As_FF));
   HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_dlam_dtmp,
                     gDim, bDim,
                     AFC_nr_local,
                     As_FF_diag_i,
                     As_FF_diag_j,
                     As_FF_diag_data,
                     As_FF_offd_i,
                     As_FF_offd_data,
                     Dbeta,
                     dlam,
                     dtmp );

   hypre_ParCSRMatrixDestroy(As_FF);
   hypre_TFree(Dbeta, HYPRE_MEMORY_DEVICE);

   /* collect off-processor dtmp and dlam */
   dtmp_offd = hypre_TAlloc(HYPRE_Complex, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(As_F2F)),
                            HYPRE_MEMORY_DEVICE);
   dlam_offd = hypre_TAlloc(HYPRE_Complex, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(As_F2F)),
                            HYPRE_MEMORY_DEVICE);

   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(As_F2F);
   hypre_ParCSRCommHandle *comm_handle;
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(As_F2F);
      comm_pkg = hypre_ParCSRMatrixCommPkg(As_F2F);
   }
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int num_elmts_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   HYPRE_Complex *send_buf = hypre_TAlloc(HYPRE_Complex, num_elmts_send, HYPRE_MEMORY_DEVICE);
   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

#if defined(HYPRE_USING_SYCL)
   hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                     dtmp,
                     send_buf );
#else
   HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                      dtmp,
                      send_buf );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }
#endif

   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg, HYPRE_MEMORY_DEVICE, send_buf,
                                                 HYPRE_MEMORY_DEVICE, dtmp_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

#if defined(HYPRE_USING_SYCL)
   hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                     dlam,
                     send_buf );
#else
   HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                      dlam,
                      send_buf );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }
#endif

   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg, HYPRE_MEMORY_DEVICE, send_buf,
                                                 HYPRE_MEMORY_DEVICE, dlam_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   hypre_TFree(send_buf, HYPRE_MEMORY_DEVICE);

   /* weak row sum and diagonal, i.e., DFF + Dgamma */
   rsWA = hypre_TAlloc(HYPRE_Complex, A_nr_local, HYPRE_MEMORY_DEVICE);

   gDim = hypre_GetDefaultDeviceGridDimension(A_nr_local, "warp", bDim);

   /* only for rows corresponding to F2 (notice flag == -1) */
   HYPRE_GPU_LAUNCH( hypreGPUKernel_compute_weak_rowsums,
                     gDim, bDim,
                     A_nr_local,
                     A_offd_nnz > 0,
                     CF_marker,
                     A_diag_i,
                     A_diag_data,
                     Soc_diag_j,
                     A_offd_i,
                     A_offd_data,
                     Soc_offd_j,
                     rsWA,
                     -1 );

   rsW = hypre_TAlloc(HYPRE_Complex, AF2F_nr_local, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   HYPRE_Complex *new_end = hypreSycl_copy_if( rsWA,
                                               rsWA + A_nr_local,
                                               CF_marker,
                                               rsW,
                                               equal<HYPRE_Int>(-2) );
#else
   HYPRE_Complex *new_end = HYPRE_THRUST_CALL( copy_if,
                                               rsWA,
                                               rsWA + A_nr_local,
                                               CF_marker,
                                               rsW,
                                               equal<HYPRE_Int>(-2) );
#endif

   hypre_assert(new_end - rsW == AF2F_nr_local);

   hypre_TFree(rsWA, HYPRE_MEMORY_DEVICE);

   /* map from F2 to F */
   HYPRE_Int *map_to_F = hypre_TAlloc(HYPRE_Int, A_nr_local, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,              is_negative<HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + A_nr_local, is_negative<HYPRE_Int>()),
                      map_to_F,
                      HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */
#else
   HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,              is_negative<HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + A_nr_local, is_negative<HYPRE_Int>()),
                      map_to_F,
                      HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */
#endif
   HYPRE_Int *map_F2_to_F = hypre_TAlloc(HYPRE_Int, AF2F_nr_local, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   HYPRE_Int *tmp_end = hypreSycl_copy_if( map_to_F,
                                           map_to_F + A_nr_local,
                                           CF_marker,
                                           map_F2_to_F,
                                           equal<HYPRE_Int>(-2) );
#else
   HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( copy_if,
                                           map_to_F,
                                           map_to_F + A_nr_local,
                                           CF_marker,
                                           map_F2_to_F,
                                           equal<HYPRE_Int>(-2) );
#endif

   hypre_assert(tmp_end - map_F2_to_F == AF2F_nr_local);

   hypre_TFree(map_to_F, HYPRE_MEMORY_DEVICE);

   /* add to rsW those in AFF that correspond to lam == 0
    * diagnoally scale As_F2F (from both sides) and replace the diagonal */
   gDim = hypre_GetDefaultDeviceGridDimension(AF2F_nr_local, "warp", bDim);

   HYPRE_Int *As_F2F_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(As_F2F));
   HYPRE_Int *As_F2F_diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(As_F2F));
   HYPRE_Complex *As_F2F_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(As_F2F));
   HYPRE_Int *As_F2F_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(As_F2F));
   HYPRE_Int *As_F2F_offd_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(As_F2F));
   HYPRE_Complex *As_F2F_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(As_F2F));
   HYPRE_GPU_LAUNCH( hypreGPUKernel_MMPEInterpScaleAFF,
                     gDim, bDim,
                     AF2F_nr_local,
                     As_F2F_diag_i,
                     As_F2F_diag_j,
                     As_F2F_diag_data,
                     As_F2F_offd_i,
                     As_F2F_offd_j,
                     As_F2F_offd_data,
                     dtmp,
                     dtmp_offd,
                     dlam,
                     dlam_offd,
                     map_F2_to_F,
                     rsW );

   hypre_TFree(dlam,        HYPRE_MEMORY_DEVICE);
   hypre_TFree(dlam_offd,   HYPRE_MEMORY_DEVICE);
   hypre_TFree(dtmp,        HYPRE_MEMORY_DEVICE);
   hypre_TFree(dtmp_offd,   HYPRE_MEMORY_DEVICE);
   hypre_TFree(map_F2_to_F, HYPRE_MEMORY_DEVICE);
   hypre_TFree(rsW,         HYPRE_MEMORY_DEVICE);

   /* Perform matrix-matrix multiplication */
   W = hypre_ParCSRMatMatDevice(As_F2F, As_FC);

   hypre_ParCSRMatrixDestroy(As_F2F);
   hypre_ParCSRMatrixDestroy(As_FC);

   /* Construct P from matrix product W */
   HYPRE_Int     *P_diag_i, *P_diag_j, *P_offd_i;
   HYPRE_Complex *P_diag_data;
   HYPRE_Int      P_nr_local = A_nr_local - (AFC_nr_local - AF2F_nr_local);
   HYPRE_Int      P_diag_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)) +
                               hypre_ParCSRMatrixNumCols(W);

   hypre_assert(P_nr_local == hypre_ParCSRMatrixNumRows(W) + hypre_ParCSRMatrixNumCols(W));

   P_diag_i    = hypre_TAlloc(HYPRE_Int,     P_nr_local + 1, HYPRE_MEMORY_DEVICE);
   P_diag_j    = hypre_TAlloc(HYPRE_Int,     P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(HYPRE_Complex, P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_offd_i    = hypre_TAlloc(HYPRE_Int,     P_nr_local + 1, HYPRE_MEMORY_DEVICE);

   HYPRE_Int *C2F2_marker = hypre_TAlloc(HYPRE_Int, P_nr_local, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   tmp_end = hypreSycl_copy_if( CF_marker,
                                CF_marker + A_nr_local,
                                CF_marker,
                                C2F2_marker,
                                out_of_range<HYPRE_Int>(-1, 0) /* -2 or 1 */ );
#else
   tmp_end = HYPRE_THRUST_CALL( copy_if,
                                CF_marker,
                                CF_marker + A_nr_local,
                                CF_marker,
                                C2F2_marker,
                                out_of_range<HYPRE_Int>(-1, 0) /* -2 or 1 */ );
#endif

   hypre_assert(tmp_end - C2F2_marker == P_nr_local);

   hypreDevice_extendWtoP( P_nr_local,
                           AF2F_nr_local,
                           hypre_ParCSRMatrixNumCols(W),
                           C2F2_marker,
                           hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(W)),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(W)),
                           P_offd_i );

   hypre_TFree(C2F2_marker, HYPRE_MEMORY_DEVICE);

   // final P
   P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(W) + hypre_ParCSRMatrixGlobalNumCols(W),
                                hypre_ParCSRMatrixGlobalNumCols(W),
                                num_old_cpts_global,
                                num_cpts_global,
                                hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(W)),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(W)));

   hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W));
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(W))    = NULL;
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(W)) = NULL;

   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(P)) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(P)) = HYPRE_MEMORY_DEVICE;

   hypre_ParCSRMatrixDeviceColMapOffd(P) = hypre_ParCSRMatrixDeviceColMapOffd(W);
   hypre_ParCSRMatrixColMapOffd(P)       = hypre_ParCSRMatrixColMapOffd(W);
   hypre_ParCSRMatrixDeviceColMapOffd(W) = NULL;
   hypre_ParCSRMatrixColMapOffd(W)       = NULL;

   hypre_ParCSRMatrixNumNonzeros(P)  = hypre_ParCSRMatrixNumNonzeros(W) +
                                       hypre_ParCSRMatrixGlobalNumCols(W);
   hypre_ParCSRMatrixDNumNonzeros(P) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);

   hypre_ParCSRMatrixDestroy(W);

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts );
   }

   *P_ptr = P;

   return hypre_error_flag;
}

//-----------------------------------------------------------------------
__global__
void hypreGPUKernel_MMInterpScaleAFF( hypre_DeviceItem    &item,
                                      HYPRE_Int      AFF_nrows,
                                      HYPRE_Int     *AFF_diag_i,
                                      HYPRE_Int     *AFF_diag_j,
                                      HYPRE_Complex *AFF_diag_a,
                                      HYPRE_Int     *AFF_offd_i,
                                      HYPRE_Int     *AFF_offd_j,
                                      HYPRE_Complex *AFF_offd_a,
                                      HYPRE_Complex *beta_diag,
                                      HYPRE_Complex *beta_offd,
                                      HYPRE_Int     *F2_to_F,
                                      HYPRE_Real    *rsW )
{
   HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= AFF_nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int ib_diag = 0, ie_diag;
   HYPRE_Int rowF = 0;

   if (lane == 0)
   {
      rowF = read_only_load(&F2_to_F[row]);
   }
   rowF = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, rowF, 0);

   if (lane < 2)
   {
      ib_diag = read_only_load(AFF_diag_i + row + lane);
   }
   ie_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib_diag, 1);
   ib_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib_diag, 0);

   HYPRE_Complex rl = 0.0;

   for (HYPRE_Int i = ib_diag + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, i < ie_diag);
        i += HYPRE_WARP_SIZE)
   {
      if (i < ie_diag)
      {
         HYPRE_Int j = read_only_load(&AFF_diag_j[i]);

         if (j == rowF)
         {
            /* diagonal */
            AFF_diag_a[i] = 1.0;
         }
         else
         {
            /* off-diagonal */
            HYPRE_Complex beta = read_only_load(&beta_diag[j]);
            HYPRE_Complex val = AFF_diag_a[i];

            if (beta == 0.0)
            {
               rl += val;
               AFF_diag_a[i] = 0.0;
            }
            else
            {
               AFF_diag_a[i] = val / beta;
            }
         }
      }
   }

   HYPRE_Int ib_offd = 0, ie_offd;

   if (lane < 2)
   {
      ib_offd = read_only_load(AFF_offd_i + row + lane);
   }
   ie_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib_offd, 1);
   ib_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib_offd, 0);

   for (HYPRE_Int i = ib_offd + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, i < ie_offd);
        i += HYPRE_WARP_SIZE)
   {
      if (i < ie_offd)
      {
         HYPRE_Int j = read_only_load(&AFF_offd_j[i]);
         HYPRE_Complex beta = read_only_load(&beta_offd[j]);
         HYPRE_Complex val = AFF_offd_a[i];

         if (beta == 0.0)
         {
            rl += val;
            AFF_offd_a[i] = 0.0;
         }
         else
         {
            AFF_offd_a[i] = val / beta;
         }
      }
   }

   rl = warp_reduce_sum(item, rl);

   if (lane == 0)
   {
      rl += read_only_load(&rsW[row]);
      rl = rl == 0.0 ? 0.0 : -1.0 / rl;
   }

   rl = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, rl, 0);

   for (HYPRE_Int i = ib_diag + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, i < ie_diag);
        i += HYPRE_WARP_SIZE)
   {
      if (i < ie_diag)
      {
         AFF_diag_a[i] *= rl;
      }
   }

   for (HYPRE_Int i = ib_offd + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, i < ie_offd);
        i += HYPRE_WARP_SIZE)
   {
      if (i < ie_offd)
      {
         AFF_offd_a[i] *= rl;
      }
   }
}

//-----------------------------------------------------------------------
__global__
void hypreGPUKernel_MMPEInterpScaleAFF( hypre_DeviceItem    &item,
                                        HYPRE_Int      AFF_nrows,
                                        HYPRE_Int     *AFF_diag_i,
                                        HYPRE_Int     *AFF_diag_j,
                                        HYPRE_Complex *AFF_diag_a,
                                        HYPRE_Int     *AFF_offd_i,
                                        HYPRE_Int     *AFF_offd_j,
                                        HYPRE_Complex *AFF_offd_a,
                                        HYPRE_Complex *tmp_diag,
                                        HYPRE_Complex *tmp_offd,
                                        HYPRE_Complex *lam_diag,
                                        HYPRE_Complex *lam_offd,
                                        HYPRE_Int     *F2_to_F,
                                        HYPRE_Real    *rsW )
{
   HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= AFF_nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int ib_diag = 0, ie_diag;
   HYPRE_Int rowF = 0;

   if (lane == 0)
   {
      rowF = read_only_load(&F2_to_F[row]);
   }
   rowF = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, rowF, 0);

   if (lane < 2)
   {
      ib_diag = read_only_load(AFF_diag_i + row + lane);
   }
   ie_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib_diag, 1);
   ib_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib_diag, 0);

   HYPRE_Complex rl = 0.0;

   for (HYPRE_Int i = ib_diag + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, i < ie_diag);
        i += HYPRE_WARP_SIZE)
   {
      if (i < ie_diag)
      {
         HYPRE_Int j = read_only_load(&AFF_diag_j[i]);

         if (j == rowF)
         {
            /* diagonal */
            AFF_diag_a[i] = 1.0;
         }
         else
         {
            /* off-diagonal */
            HYPRE_Complex lam = read_only_load(&lam_diag[j]);
            HYPRE_Complex val = AFF_diag_a[i];

            if (lam == 0.0)
            {
               rl += val;
               AFF_diag_a[i] = 0.0;
            }
            else
            {
               rl += val * read_only_load(&tmp_diag[j]);
               AFF_diag_a[i] = val / lam;
            }
         }
      }
   }

   HYPRE_Int ib_offd = 0, ie_offd;

   if (lane < 2)
   {
      ib_offd = read_only_load(AFF_offd_i + row + lane);
   }
   ie_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib_offd, 1);
   ib_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, ib_offd, 0);

   for (HYPRE_Int i = ib_offd + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, i < ie_offd);
        i += HYPRE_WARP_SIZE)
   {
      if (i < ie_offd)
      {
         HYPRE_Int j = read_only_load(&AFF_offd_j[i]);
         HYPRE_Complex lam = read_only_load(&lam_offd[j]);
         HYPRE_Complex val = AFF_offd_a[i];

         if (lam == 0.0)
         {
            rl += val;
            AFF_offd_a[i] = 0.0;
         }
         else
         {
            rl += val * read_only_load(&tmp_offd[j]);
            AFF_offd_a[i] = val / lam;
         }
      }
   }

   rl = warp_reduce_sum(item, rl);

   if (lane == 0)
   {
      rl += read_only_load(&rsW[row]);
      rl = rl == 0.0 ? 0.0 : -1.0 / rl;
   }

   rl = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, rl, 0);

   for (HYPRE_Int i = ib_diag + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, i < ie_diag);
        i += HYPRE_WARP_SIZE)
   {
      if (i < ie_diag)
      {
         AFF_diag_a[i] *= rl;
      }
   }

   for (HYPRE_Int i = ib_offd + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, i < ie_offd);
        i += HYPRE_WARP_SIZE)
   {
      if (i < ie_offd)
      {
         AFF_offd_a[i] *= rl;
      }
   }
}

#endif /* #if defined(HYPRE_USING_GPU) */

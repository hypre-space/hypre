/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "aux_interp.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA)

#define MAX_C_CONNECTIONS 100
#define HAVE_COMMON_C 1

__global__ void compute_weak_rowsums( HYPRE_Int nr_of_rows, bool has_offd, HYPRE_Int *CF_marker, HYPRE_Int *A_diag_i, HYPRE_Complex *A_diag_a, HYPRE_Int *S_diag_j, HYPRE_Int *A_offd_i, HYPRE_Complex *A_offd_a, HYPRE_Int *S_offd_j, HYPRE_Real *rs );

__global__ void compute_aff_afc( HYPRE_Int nr_of_rows, HYPRE_Int *AFF_diag_i, HYPRE_Int *AFF_diag_j, HYPRE_Complex *AFF_diag_data, HYPRE_Int *AFF_offd_i, HYPRE_Complex *AFF_offd_data, HYPRE_Int *AFC_diag_i, HYPRE_Complex *AFC_diag_data, HYPRE_Int *AFC_offd_i, HYPRE_Complex *AFC_offd_data, HYPRE_Complex *rsW, HYPRE_Complex *rsFC );

void hypreDevice_extendWtoP( HYPRE_Int P_nr_of_rows, HYPRE_Int W_nr_of_rows, HYPRE_Int W_nr_of_cols, HYPRE_Int *CF_marker, HYPRE_Int W_diag_nnz, HYPRE_Int *W_diag_i, HYPRE_Int *W_diag_j, HYPRE_Complex *W_diag_data, HYPRE_Int *P_diag_i, HYPRE_Int *P_diag_j, HYPRE_Complex *P_diag_data, HYPRE_Int *W_offd_i, HYPRE_Int *P_offd_i );

__global__ void compute_twiaff_w( HYPRE_Int nr_of_rows, HYPRE_Int first_index, HYPRE_Int *AFF_diag_i, HYPRE_Int *AFF_diag_j, HYPRE_Complex *AFF_diag_data, HYPRE_Complex *AFF_diag_data_old, HYPRE_Int *AFF_offd_i, HYPRE_Int *AFF_offd_j, HYPRE_Complex *AFF_offd_data, HYPRE_Int *AFF_ext_i, HYPRE_BigInt *AFF_ext_j, HYPRE_Complex *AFF_ext_data, HYPRE_Complex *rsW, HYPRE_Complex *rsFC, HYPRE_Complex *rsFC_offd );

/*---------------------------------------------------------------------
 * Extended Interpolation in the form of Mat-Mat
 *---------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildExtInterpDevice(hypre_ParCSRMatrix  *A,
                                    HYPRE_Int           *CF_marker,
                                    hypre_ParCSRMatrix  *S,
                                    HYPRE_BigInt        *num_cpts_global,
                                    HYPRE_Int            num_functions,
                                    HYPRE_Int           *dof_func,
                                    HYPRE_Int            debug_flag,
                                    HYPRE_Real           trunc_factor,
                                    HYPRE_Int            max_elmts,
                                    HYPRE_Int           *col_offd_S_to_A,
                                    hypre_ParCSRMatrix **P_ptr)
{
   HYPRE_Int           A_nr_of_rows = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix    *A_diag       = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i     = hypre_CSRMatrixI(A_diag);
   hypre_CSRMatrix    *A_offd       = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Int           A_offd_nnz   = hypre_CSRMatrixNumNonzeros(A_offd);
   HYPRE_Int          *Soc_diag_j   = hypre_ParCSRMatrixSocDiagJ(S);
   HYPRE_Int          *Soc_offd_j   = hypre_ParCSRMatrixSocOffdJ(S);
   HYPRE_Int          *CF_marker_dev;
   hypre_ParCSRMatrix *AFF, *AFC;
   hypre_ParCSRMatrix *W, *P;
   HYPRE_Int           W_nr_of_rows, P_diag_nnz, i;
   HYPRE_Complex      *rsFC, *rsWA, *rsW;
   HYPRE_Int          *P_diag_i, *P_diag_j, *P_offd_i;
   HYPRE_Complex      *P_diag_data;

   CF_marker_dev = hypre_TAlloc(HYPRE_Int, A_nr_of_rows, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(CF_marker_dev, CF_marker, HYPRE_Int, A_nr_of_rows,
                  HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   /* 0. Find row sums of weak elements */
   /* row sum of A-weak + Diag(A), i.e., (D_gamma + D_alpha) in the notes, only for F-pts */
   rsWA = hypre_TAlloc(HYPRE_Complex, A_nr_of_rows, HYPRE_MEMORY_DEVICE);

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(A_nr_of_rows, "warp", bDim);

   HYPRE_CUDA_LAUNCH( compute_weak_rowsums,
                      gDim, bDim,
                      A_nr_of_rows,
                      A_offd_nnz > 0,
                      CF_marker_dev,
                      A_diag_i,
                      A_diag_data,
                      Soc_diag_j,
                      A_offd_i,
                      A_offd_data,
                      Soc_offd_j,
                      rsWA );

   // AFF AFC
   hypre_NvtxPushRangeColor("Extract Submatrix", 2);
   hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, num_cpts_global, S, &AFC, &AFF);
   hypre_NvtxPopRange();

   W_nr_of_rows = hypre_ParCSRMatrixNumRows(AFF);
   hypre_assert(A_nr_of_rows == W_nr_of_rows + hypre_ParCSRMatrixNumCols(AFC));

   rsW = hypre_TAlloc(HYPRE_Complex, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *new_end = HYPRE_THRUST_CALL( copy_if,
                                               rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker_dev,
                                               rsW,
                                               is_negative<HYPRE_Int>() );
   hypre_assert(new_end - rsW == W_nr_of_rows);
   hypre_TFree(rsWA, HYPRE_MEMORY_DEVICE);

   /* row sum of AFC, i.e., D_beta */
   rsFC = hypre_TAlloc(HYPRE_Complex, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixDiag(AFC), NULL, NULL, rsFC, 0, 1.0, "set");
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixOffd(AFC), NULL, NULL, rsFC, 0, 1.0, "add");

   /* 5. Form matrix ~{A_FF}, (return twAFF in AFF data structure ) */
   /* 6. Form matrix ~{A_FC}, (return twAFC in AFC data structure) */
   hypre_NvtxPushRangeColor("Compute interp matrix", 4);
   gDim = hypre_GetDefaultCUDAGridDimension(W_nr_of_rows, "warp", bDim);
   HYPRE_CUDA_LAUNCH( compute_aff_afc,
                      gDim, bDim,
                      W_nr_of_rows,
                      hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(AFF)),
                      hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(AFF)),
                      hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(AFF)),
                      hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(AFF)),
                      hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(AFF)),
                      hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(AFC)),
                      hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(AFC)),
                      hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(AFC)),
                      hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(AFC)),
                      rsW,
                      rsFC );
   hypre_TFree(rsW,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(rsFC, HYPRE_MEMORY_DEVICE);
   hypre_NvtxPopRange();

   /* 7. Perform matrix-matrix multiplication */
   hypre_NvtxPushRangeColor("Matrix-matrix mult", 3);
   W = hypre_ParCSRMatMatDevice(AFF, AFC);
   hypre_NvtxPopRange();

   /* 8. Construct P from matrix product W */
   P_diag_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)) +
                hypre_ParCSRMatrixNumCols(W);

   P_diag_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows+1, HYPRE_MEMORY_DEVICE);
   P_diag_j    = hypre_TAlloc(HYPRE_Int,     P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(HYPRE_Complex, P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_offd_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows+1, HYPRE_MEMORY_DEVICE);

   hypre_NvtxPushRangeColor("Extend matrix", 4);
   hypreDevice_extendWtoP( A_nr_of_rows,
                           W_nr_of_rows,
                           hypre_ParCSRMatrixNumCols(W),
                           CF_marker_dev,
                           hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(W)),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(W)),
                           P_offd_i );
   hypre_TFree(CF_marker_dev, HYPRE_MEMORY_DEVICE);
   hypre_NvtxPopRange();

   // final P
   P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixGlobalNumCols(W),
                                hypre_ParCSRMatrixColStarts(A),
                                hypre_ParCSRMatrixColStarts(W),
                                hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(W)),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(W)));

   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;
   hypre_ParCSRMatrixOwnsColStarts(P) = 0;

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

   hypre_NvtxPushRangeColor("Truncation", 4);
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts );
   }
   hypre_NvtxPopRange();

   hypre_MatvecCommPkgCreate(P);

   for (i = 0; i < A_nr_of_rows; i++)
   {
      if (CF_marker[i] == -3)
      {
         CF_marker[i] = -1;
      }
   }

   *P_ptr = P;

   /* 9. Free memory */
   hypre_ParCSRMatrixDestroy(W);

   return hypre_error_flag;
}

/*-----------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildExtPIInterpDevice( hypre_ParCSRMatrix  *A,
                                       HYPRE_Int           *CF_marker,
                                       hypre_ParCSRMatrix  *S,
                                       HYPRE_BigInt        *num_cpts_global,
                                       HYPRE_Int            num_functions,
                                       HYPRE_Int           *dof_func,
                                       HYPRE_Int            debug_flag,
                                       HYPRE_Real           trunc_factor,
                                       HYPRE_Int            max_elmts,
                                       hypre_ParCSRMatrix **P_ptr)
{
   HYPRE_Int           A_nr_of_rows = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix    *A_diag       = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i     = hypre_CSRMatrixI(A_diag);
   hypre_CSRMatrix    *A_offd       = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Int           A_offd_nnz   = hypre_CSRMatrixNumNonzeros(A_offd);
   HYPRE_Int          *Soc_diag_j   = hypre_ParCSRMatrixSocDiagJ(S);
   HYPRE_Int          *Soc_offd_j   = hypre_ParCSRMatrixSocOffdJ(S);
   HYPRE_Int          *CF_marker_dev;
   hypre_CSRMatrix    *AFF_ext = NULL;
   hypre_ParCSRMatrix *AFF, *AFC;
   hypre_ParCSRMatrix *W, *P;
   HYPRE_Int           W_nr_of_rows, P_diag_nnz, i;
   HYPRE_Complex      *rsFC, *rsFC_offd, *rsWA, *rsW;
   HYPRE_Int          *P_diag_i, *P_diag_j, *P_offd_i, num_procs;
   HYPRE_Complex      *P_diag_data;

   hypre_MPI_Comm_size(hypre_ParCSRMatrixComm(A), &num_procs);

   CF_marker_dev = hypre_TAlloc(HYPRE_Int, A_nr_of_rows, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(CF_marker_dev, CF_marker, HYPRE_Int, A_nr_of_rows,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   /* 0.Find row sums of weak elements */
   /* row sum of A-weak + Diag(A), i.e., (D_gamma + D_alpha) in the notes, only for F-pts */
   rsWA = hypre_TAlloc(HYPRE_Complex, A_nr_of_rows, HYPRE_MEMORY_DEVICE);

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(A_nr_of_rows, "warp",   bDim);

   HYPRE_CUDA_LAUNCH( compute_weak_rowsums,
                      gDim, bDim,
                      A_nr_of_rows,
                      A_offd_nnz > 0,
                      CF_marker_dev,
                      A_diag_i,
                      A_diag_data,
                      Soc_diag_j,
                      A_offd_i,
                      A_offd_data,
                      Soc_offd_j,
                      rsWA );

   // AFF AFC
   hypre_NvtxPushRangeColor("Extract Submatrix", 2);
   hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, num_cpts_global, S, &AFC, &AFF);
   hypre_NvtxPopRange();

   W_nr_of_rows  = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(AFF));
   hypre_assert(A_nr_of_rows == W_nr_of_rows + hypre_ParCSRMatrixNumCols(AFC));

   rsW = hypre_TAlloc(HYPRE_Complex, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *new_end = HYPRE_THRUST_CALL( copy_if,
                                               rsWA,
                                               rsWA + A_nr_of_rows,
                                               CF_marker_dev,
                                               rsW,
                                               is_negative<HYPRE_Int>() );
   hypre_assert(new_end - rsW == W_nr_of_rows);
   hypre_TFree(rsWA, HYPRE_MEMORY_DEVICE);

   /* row sum of AFC, i.e., D_beta */
   rsFC = hypre_TAlloc(HYPRE_Complex, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixDiag(AFC), NULL, NULL, rsFC, 0, 1.0, "set");
   hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixOffd(AFC), NULL, NULL, rsFC, 0, 1.0, "add");

   /* collect off-processor rsFC */
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(AFF);
   hypre_ParCSRCommHandle *comm_handle;
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(AFF);
      comm_pkg = hypre_ParCSRMatrixCommPkg(AFF);
   }
   rsFC_offd = hypre_TAlloc(HYPRE_Complex, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(AFF)), HYPRE_MEMORY_DEVICE);
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int num_elmts_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   HYPRE_Complex *send_buf = hypre_TAlloc(HYPRE_Complex, num_elmts_send, HYPRE_MEMORY_DEVICE);
   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
   HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elmts_send,
                      rsFC,
                      send_buf );
   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg, HYPRE_MEMORY_DEVICE, send_buf, HYPRE_MEMORY_DEVICE, rsFC_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   hypre_TFree(send_buf, HYPRE_MEMORY_DEVICE);

   /* offd rows of AFF */
   if (num_procs > 1)
   {
      AFF_ext = hypre_ParCSRMatrixExtractBExtDevice(AFF, AFF, 1);
   }

   /* 5. Form matrix ~{A_FF}, (return twAFF in AFF data structure ) */
   HYPRE_Complex *AFF_diag_data_old = hypre_TAlloc(HYPRE_Complex, hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(AFF)),
                                                   HYPRE_MEMORY_DEVICE);
   HYPRE_THRUST_CALL( copy,
                      hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(AFF)),
                      hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(AFF)) + hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(AFF)),
                      AFF_diag_data_old );

   hypre_NvtxPushRangeColor("Compute interp matrix", 4);
   gDim = hypre_GetDefaultCUDAGridDimension(W_nr_of_rows, "warp", bDim);
   HYPRE_CUDA_LAUNCH( compute_twiaff_w,
                      gDim, bDim,
                      W_nr_of_rows,
                      hypre_ParCSRMatrixFirstRowIndex(AFF),
                      hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(AFF)),
                      hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(AFF)),
                      hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(AFF)),
                      AFF_diag_data_old,
                      hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(AFF)),
                      hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(AFF)),
                      hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(AFF)),
                      AFF_ext ? hypre_CSRMatrixI(AFF_ext)    : NULL,
                      AFF_ext ? hypre_CSRMatrixBigJ(AFF_ext) : NULL,
                      AFF_ext ? hypre_CSRMatrixData(AFF_ext) : NULL,
                      rsW,
                      rsFC,
                      rsFC_offd );
   hypre_TFree(rsW,               HYPRE_MEMORY_DEVICE);
   hypre_TFree(rsFC,              HYPRE_MEMORY_DEVICE);
   hypre_TFree(rsFC_offd,         HYPRE_MEMORY_DEVICE);
   hypre_TFree(AFF_diag_data_old, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixDestroy(AFF_ext);
   hypre_NvtxPopRange();

   /* 7. Perform matrix-matrix multiplication */
   hypre_NvtxPushRangeColor("Matrix-matrix mult", 3);
   W = hypre_ParCSRMatMatDevice(AFF, AFC);
   hypre_NvtxPopRange();

   /* 8. Construct P from matrix product W */
   P_diag_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)) +
                hypre_ParCSRMatrixNumCols(W);

   P_diag_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows+1, HYPRE_MEMORY_DEVICE);
   P_diag_j    = hypre_TAlloc(HYPRE_Int,     P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(HYPRE_Complex, P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_offd_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows+1, HYPRE_MEMORY_DEVICE);

   hypre_NvtxPushRangeColor("Extend matrix", 4);
   hypreDevice_extendWtoP( A_nr_of_rows,
                           W_nr_of_rows,
                           hypre_ParCSRMatrixNumCols(W),
                           CF_marker_dev,
                           hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(W)),
                           hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(W)),
                           P_diag_i,
                           P_diag_j,
                           P_diag_data,
                           hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(W)),
                           P_offd_i );
   hypre_TFree(CF_marker_dev, HYPRE_MEMORY_DEVICE);
   hypre_NvtxPopRange();

   // final P
   P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixGlobalNumCols(W),
                                hypre_ParCSRMatrixColStarts(A),
                                hypre_ParCSRMatrixColStarts(W),
                                hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(W)),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(W)));

   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;
   hypre_ParCSRMatrixOwnsColStarts(P) = 0;

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

   hypre_NvtxPushRangeColor("Truncation", 4);
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts );
   }
   hypre_NvtxPopRange();

   hypre_MatvecCommPkgCreate(P);

   for (i = 0; i < A_nr_of_rows; i++)
   {
      if (CF_marker[i] == -3)
      {
         CF_marker[i] = -1;
      }
   }

   *P_ptr = P;

   /* 9. Free memory */
   hypre_ParCSRMatrixDestroy(W);

   return hypre_error_flag;
}

//-----------------------------------------------------------------------
// S_*_j is the special j-array from device SoC
// -1: weak, -2: diag, >=0 (== A_diag_j) : strong
// add weak and the diagonal entries of F-rows
__global__
void compute_weak_rowsums( HYPRE_Int      nr_of_rows,
                           bool           has_offd,
                           HYPRE_Int     *CF_marker,
                           HYPRE_Int     *A_diag_i,
                           HYPRE_Complex *A_diag_a,
                           HYPRE_Int     *Soc_diag_j,
                           HYPRE_Int     *A_offd_i,
                           HYPRE_Complex *A_offd_a,
                           HYPRE_Int     *Soc_offd_j,
                           HYPRE_Real    *rs )
{
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>();

   if (row >= nr_of_rows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   HYPRE_Int ib, ie;

   if (lane == 0)
   {
      ib = read_only_load(CF_marker + row);
   }
   ib = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 0);

   if (ib >= 0)
   {
      return;
   }

   if (lane < 2)
   {
      ib = read_only_load(A_diag_i + row + lane);
   }
   ie = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 1);
   ib = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 0);

   HYPRE_Complex rl = 0.0;

   for (HYPRE_Int i = ib + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < ie); i += HYPRE_WARP_SIZE)
   {
      if (i < ie)
      {
         rl += read_only_load(&A_diag_a[i]) * (read_only_load(&Soc_diag_j[i]) < 0);
      }
   }

   if (has_offd)
   {
      if (lane < 2)
      {
         ib = read_only_load(A_offd_i + row + lane);
      }
      ie = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 1);
      ib = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 0);

      for (HYPRE_Int i = ib + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < ie); i += HYPRE_WARP_SIZE)
      {
         if (i < ie)
         {
            rl += read_only_load(&A_offd_a[i]) * (read_only_load(&Soc_offd_j[i]) < 0);
         }
      }
   }

   rl = warp_reduce_sum(rl);

   if (lane == 0)
   {
      rs[row] = rl;
   }
}

//-----------------------------------------------------------------------
__global__
void compute_aff_afc( HYPRE_Int      nr_of_rows,
                      HYPRE_Int     *AFF_diag_i,
                      HYPRE_Int     *AFF_diag_j,
                      HYPRE_Complex *AFF_diag_data,
                      HYPRE_Int     *AFF_offd_i,
                      HYPRE_Complex *AFF_offd_data,
                      HYPRE_Int     *AFC_diag_i,
                      HYPRE_Complex *AFC_diag_data,
                      HYPRE_Int     *AFC_offd_i,
                      HYPRE_Complex *AFC_offd_data,
                      HYPRE_Complex *rsW,
                      HYPRE_Complex *rsFC )
{
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>();

   if (row >= nr_of_rows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   HYPRE_Int p, q;

   HYPRE_Complex iscale, beta;

   if (lane == 0)
   {
      iscale = -1.0 / read_only_load(&rsW[row]);
      beta = read_only_load(&rsFC[row]);
   }
   iscale = __shfl_sync(HYPRE_WARP_FULL_MASK, iscale, 0);
   beta   = __shfl_sync(HYPRE_WARP_FULL_MASK, beta,   0);

   // AFF
   /* Diag part */
   if (lane < 2)
   {
      p = read_only_load(AFF_diag_i + row + lane);
   }
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   // do not assume diag is the first element of row
   for (HYPRE_Int j = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      if (j < q)
      {
         if (read_only_load(&AFF_diag_j[j]) == row)
         {
            AFF_diag_data[j] = beta * iscale;
         }
         else
         {
            AFF_diag_data[j] *= iscale;
         }
      }
   }

   /* offd part */
   if (lane < 2)
   {
      p = read_only_load(AFF_offd_i + row + lane);
   }
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      if (j < q)
      {
         AFF_offd_data[j] *= iscale;
      }
   }

   if (beta != 0.0)
   {
      beta = 1.0 / beta;
   }

   // AFC
   if (lane < 2)
   {
      p = read_only_load(AFC_diag_i + row + lane);
   }
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   /* Diag part */
   // do not assume diag is the first element of row
   for (HYPRE_Int j = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      if (j < q)
      {
         AFC_diag_data[j] *= beta;
      }
   }

   /* offd part */
   if (lane < 2)
   {
      p = read_only_load(AFC_offd_i + row + lane);
   }
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      if (j < q)
      {
         AFC_offd_data[j] *= beta;
      }
   }
}


//-----------------------------------------------------------------------
void
hypreDevice_extendWtoP( HYPRE_Int      P_nr_of_rows,
                        HYPRE_Int      W_nr_of_rows,
                        HYPRE_Int      W_nr_of_cols,
                        HYPRE_Int     *CF_marker,
                        HYPRE_Int      W_diag_nnz,
                        HYPRE_Int     *W_diag_i,
                        HYPRE_Int     *W_diag_j,
                        HYPRE_Complex *W_diag_data,
                        HYPRE_Int     *P_diag_i,
                        HYPRE_Int     *P_diag_j,
                        HYPRE_Complex *P_diag_data,
                        HYPRE_Int     *W_offd_i,
                        HYPRE_Int     *P_offd_i )
{
   // row index shift P --> W
   HYPRE_Int *PWoffset = hypre_TAlloc(HYPRE_Int, P_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_THRUST_CALL( transform,
                      CF_marker,
                      &CF_marker[P_nr_of_rows],
                      PWoffset,
                      is_nonnegative<HYPRE_Int>() );

   HYPRE_THRUST_CALL( exclusive_scan,
                      PWoffset,
                      &PWoffset[P_nr_of_rows+1],
                      PWoffset);

   // map F+C to (next) F
   HYPRE_Int *map2F = hypre_TAlloc(HYPRE_Int, P_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_THRUST_CALL( transform,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::counting_iterator<HYPRE_Int>(P_nr_of_rows + 1),
                      PWoffset,
                      map2F,
                      thrust::minus<HYPRE_Int>() );

   // P_diag_i
   HYPRE_THRUST_CALL( gather,
                      map2F,
                      map2F + P_nr_of_rows + 1,
                      W_diag_i,
                      P_diag_i );

   HYPRE_THRUST_CALL( transform,
                      P_diag_i,
                      P_diag_i + P_nr_of_rows + 1,
                      PWoffset,
                      P_diag_i,
                      thrust::plus<HYPRE_Int>() );

   // P_offd_i
   HYPRE_THRUST_CALL( gather,
                      map2F,
                      map2F + P_nr_of_rows + 1,
                      W_offd_i,
                      P_offd_i );

   hypre_TFree(map2F, HYPRE_MEMORY_DEVICE);

   // row index shift W --> P
   HYPRE_Int *WPoffset = hypre_TAlloc(HYPRE_Int, W_nr_of_rows, HYPRE_MEMORY_DEVICE);
   HYPRE_Int *new_end = HYPRE_THRUST_CALL( copy_if,
                                           PWoffset,
                                           PWoffset + P_nr_of_rows,
                                           CF_marker,
                                           WPoffset,
                                           is_negative<HYPRE_Int>() );
   hypre_assert(new_end - WPoffset == W_nr_of_rows);

   hypre_TFree(PWoffset, HYPRE_MEMORY_DEVICE);

   // elements shift
   HYPRE_Int *shift = hypreDevice_CsrRowPtrsToIndices(W_nr_of_rows, W_diag_nnz, W_diag_i);
   HYPRE_THRUST_CALL( gather,
                      shift,
                      shift + W_diag_nnz,
                      WPoffset,
                      shift);

   hypre_TFree(WPoffset, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( transform,
                      shift,
                      shift + W_diag_nnz,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      shift,
                      thrust::plus<HYPRE_Int>() );

   // P_diag_j and P_diag_data
   HYPRE_THRUST_CALL( scatter,
                      thrust::make_zip_iterator(thrust::make_tuple(W_diag_j, W_diag_data)),
                      thrust::make_zip_iterator(thrust::make_tuple(W_diag_j, W_diag_data)) + W_diag_nnz,
                      shift,
                      thrust::make_zip_iterator(thrust::make_tuple(P_diag_j, P_diag_data)) );

   hypre_TFree(shift, HYPRE_MEMORY_DEVICE);

   // fill the gap
   HYPRE_Int *PC_i = hypre_TAlloc(HYPRE_Int, W_nr_of_cols, HYPRE_MEMORY_DEVICE);
   new_end = HYPRE_THRUST_CALL( copy_if,
                                P_diag_i,
                                P_diag_i + P_nr_of_rows,
                                CF_marker,
                                PC_i,
                                is_nonnegative<HYPRE_Int>() );

   hypre_assert(new_end - PC_i == W_nr_of_cols);

   HYPRE_THRUST_CALL( scatter,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::counting_iterator<HYPRE_Int>(W_nr_of_cols),
                      PC_i,
                      P_diag_j );

   hypreDevice_ScatterConstant(P_diag_data, W_nr_of_cols, PC_i, 1.0);

   hypre_TFree(PC_i, HYPRE_MEMORY_DEVICE);
}


//-----------------------------------------------------------------------
// For Ext+i Interp, scale AFF from the left and the right
__global__
void compute_twiaff_w( HYPRE_Int      nr_of_rows,
                       HYPRE_Int      first_index,
                       HYPRE_Int     *AFF_diag_i,
                       HYPRE_Int     *AFF_diag_j,
                       HYPRE_Complex *AFF_diag_data,
                       HYPRE_Complex *AFF_diag_data_old,
                       HYPRE_Int     *AFF_offd_i,
                       HYPRE_Int     *AFF_offd_j,
                       HYPRE_Complex *AFF_offd_data,
                       HYPRE_Int     *AFF_ext_i,
                       HYPRE_BigInt  *AFF_ext_j,
                       HYPRE_Complex *AFF_ext_data,
                       HYPRE_Complex *rsW,
                       HYPRE_Complex *rsFC,
                       HYPRE_Complex *rsFC_offd )
{
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>();

   if (row >= nr_of_rows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();

   HYPRE_Int ib_diag, ie_diag, ib_offd, ie_offd;

   // diag
   if (lane < 2)
   {
      ib_diag = read_only_load(AFF_diag_i + row + lane);
   }
   ie_diag = __shfl_sync(HYPRE_WARP_FULL_MASK, ib_diag, 1);
   ib_diag = __shfl_sync(HYPRE_WARP_FULL_MASK, ib_diag, 0);

   HYPRE_Complex theta_i = 0.0;

   // do not assume diag is the first element of row
   // entire warp works on each j
   for (HYPRE_Int indj = ib_diag; indj < ie_diag; indj++)
   {
      HYPRE_Int j;

      if (lane == 0)
      {
         j = read_only_load(&AFF_diag_j[indj]);
      }
      j = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);

      if (j == row)
      {
         if (lane == 0)
         {
            AFF_diag_data[indj] = 1.0;
         }

         continue;
      }

      HYPRE_Int kb, ke;

      // find if there exists entry (j, row) in row j of diag
      if (lane < 2)
      {
         kb = read_only_load(AFF_diag_i + j + lane);
      }
      ke = __shfl_sync(HYPRE_WARP_FULL_MASK, kb, 1);
      kb = __shfl_sync(HYPRE_WARP_FULL_MASK, kb, 0);

      HYPRE_Int kmatch = -1;
      for (HYPRE_Int indk = kb + lane; __any_sync(HYPRE_WARP_FULL_MASK, indk < ke); indk += HYPRE_WARP_SIZE)
      {
         if (indk < ke && row == read_only_load(&AFF_diag_j[indk]))
         {
            kmatch = indk;
         }

         if (__any_sync(HYPRE_WARP_FULL_MASK, kmatch >= 0))
         {
            break;
         }
      }
      kmatch = warp_reduce_max(kmatch);

      if (lane == 0)
      {
         HYPRE_Complex vji = kmatch >= 0 ? read_only_load(&AFF_diag_data_old[kmatch]) : 0.0;
         HYPRE_Complex rsj = read_only_load(&rsFC[j]) + vji;
         HYPRE_Complex vij = read_only_load(&AFF_diag_data_old[indj]) / rsj;
         AFF_diag_data[indj] = vij;
         theta_i += vji * vij;
      }
   }

   // offd
   if (lane < 2)
   {
      ib_offd = read_only_load(AFF_offd_i + row + lane);
   }
   ie_offd = __shfl_sync(HYPRE_WARP_FULL_MASK, ib_offd, 1);
   ib_offd = __shfl_sync(HYPRE_WARP_FULL_MASK, ib_offd, 0);

   for (HYPRE_Int indj = ib_offd; indj < ie_offd; indj++)
   {
      HYPRE_Int j;

      if (lane == 0)
      {
         j = read_only_load(&AFF_offd_j[indj]);
      }
      j = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);

      HYPRE_Int kb, ke;

      if (lane < 2)
      {
         kb = read_only_load(AFF_ext_i + j + lane);
      }
      ke = __shfl_sync(HYPRE_WARP_FULL_MASK, kb, 1);
      kb = __shfl_sync(HYPRE_WARP_FULL_MASK, kb, 0);

      HYPRE_Int kmatch = -1;
      for (HYPRE_Int indk = kb + lane; __any_sync(HYPRE_WARP_FULL_MASK, indk < ke); indk += HYPRE_WARP_SIZE)
      {
         if (indk < ke && row + first_index == read_only_load(&AFF_ext_j[indk]))
         {
            kmatch = indk;
         }

         if (__any_sync(HYPRE_WARP_FULL_MASK, kmatch >= 0))
         {
            break;
         }
      }
      kmatch = warp_reduce_max(kmatch);

      if (lane == 0)
      {
         HYPRE_Complex vji = kmatch >= 0 ? read_only_load(&AFF_ext_data[kmatch]) : 0.0;
         HYPRE_Complex rsj = read_only_load(&rsFC_offd[j]) + vji;
         HYPRE_Complex vij = read_only_load(&AFF_offd_data[indj]) / rsj;
         AFF_offd_data[indj] = vij;
         theta_i += vji * vij;
      }
   }

   // scale row
   if (lane == 0)
   {
      theta_i = -1.0 / (theta_i + read_only_load(rsW + row));
   }
   theta_i = __shfl_sync(HYPRE_WARP_FULL_MASK, theta_i, 0);

   for (HYPRE_Int j = ib_diag + lane; __any_sync(HYPRE_WARP_FULL_MASK, j < ie_diag); j += HYPRE_WARP_SIZE)
   {
      if (j < ie_diag)
      {
         AFF_diag_data[j] *= theta_i;
      }
   }

   for (HYPRE_Int j = ib_offd + lane; __any_sync(HYPRE_WARP_FULL_MASK, j < ie_offd); j += HYPRE_WARP_SIZE)
   {
      if (j < ie_offd)
      {
         AFF_offd_data[j] *= theta_i;
      }
   }
}

#endif


/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Two-grid system solver
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "seq_mv/protos.h"

#if defined(HYPRE_USING_CUDA)
void hypreDevice_extendWtoP( HYPRE_Int P_nr_of_rows, HYPRE_Int W_nr_of_rows, HYPRE_Int W_nr_of_cols, HYPRE_Int *CF_marker, HYPRE_Int W_diag_nnz, HYPRE_Int *W_diag_i, HYPRE_Int *W_diag_j, HYPRE_Complex *W_diag_data, HYPRE_Int *P_diag_i, HYPRE_Int *P_diag_j, HYPRE_Complex *P_diag_data, HYPRE_Int *W_offd_i, HYPRE_Int *P_offd_i );

HYPRE_Int
hypre_MGRBuildPDevice(hypre_ParCSRMatrix *A,
                      HYPRE_Int          *CF_marker_host,
                      HYPRE_BigInt       *num_cpts_global,
                      HYPRE_Int          method,
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

  HYPRE_Int          *CF_marker_dev;
  hypre_ParCSRMatrix *A_FF, *A_FC, *P;
  hypre_CSRMatrix    *D_FF_inv, *W_diag, *W_offd;
  HYPRE_Int           W_nr_of_rows, P_diag_nnz, i;
  HYPRE_Complex      *rsFC, *rsWA, *rsW;
  HYPRE_Int          *P_diag_i, *P_diag_j, *P_offd_i;
  HYPRE_Complex      *P_diag_data;
  HYPRE_Complex      *diag;

  CF_marker_dev = hypre_TAlloc(HYPRE_Int, A_nr_of_rows, HYPRE_MEMORY_DEVICE);
  hypre_TMemcpy(CF_marker_dev, CF_marker_host, HYPRE_Int, A_nr_of_rows,
                HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

  hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker_host, num_cpts_global, NULL, &A_FC, &A_FF);
  HYPRE_Int local_nrows_wp = hypre_ParCSRMatrixNumRows(A_FF);
  diag = hypre_CTAlloc(HYPRE_Complex, local_nrows_wp, HYPRE_MEMORY_DEVICE);
  if (method == 1)
  {
    hypre_CSRMatrixExtractDiagonalDevice(hypre_ParCSRMatrixDiag(A_FF), diag, 3);
  }
  else if (method == 2)
  {
    hypre_CSRMatrixExtractDiagonalDevice(hypre_ParCSRMatrixDiag(A_FF), diag, 4);
  }

  // Doing extraneous work
  // TODO: no need to compute W for injection, i.e. W = 0
  D_FF_inv = hypre_CSRMatrixDiagMatrixFromVectorDevice(local_nrows_wp, diag);

  W_diag = hypre_CSRMatrixMultiplyDevice(D_FF_inv, hypre_ParCSRMatrixDiag(A_FC));
  W_offd = hypre_CSRMatrixMultiplyDevice(D_FF_inv, hypre_ParCSRMatrixOffd(A_FC));
  W_nr_of_rows = hypre_CSRMatrixNumRows(W_diag);

  if (method == 0)
  {
    hypre_CSRMatrixDropSmallEntriesDevice(W_diag, 1e-14, 0, 0);
    hypre_CSRMatrixDropSmallEntriesDevice(W_offd, 1e-14, 0, 0);
  }

  /* Construct P from matrix product W_diag */
  P_diag_nnz = hypre_CSRMatrixNumNonzeros(W_diag) + hypre_ParCSRMatrixNumCols(A_FC);

  P_diag_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows+1, HYPRE_MEMORY_DEVICE);
  P_diag_j    = hypre_TAlloc(HYPRE_Int,     P_diag_nnz,     HYPRE_MEMORY_DEVICE);
  P_diag_data = hypre_TAlloc(HYPRE_Complex, P_diag_nnz,     HYPRE_MEMORY_DEVICE);
  P_offd_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows+1, HYPRE_MEMORY_DEVICE);

  hypre_NvtxPushRangeColor("Extend matrix", 4);
  hypreDevice_extendWtoP( A_nr_of_rows,
                         W_nr_of_rows,
                         hypre_ParCSRMatrixNumCols(A_FC),
                         CF_marker_dev,
                         hypre_CSRMatrixNumNonzeros(W_diag),
                         hypre_CSRMatrixI(W_diag),
                         hypre_CSRMatrixJ(W_diag),
                         hypre_CSRMatrixData(W_diag),
                         P_diag_i,
                         P_diag_j,
                         P_diag_data,
                         hypre_CSRMatrixI(W_offd),
                         P_offd_i );
  hypre_TFree(CF_marker_dev, HYPRE_MEMORY_DEVICE);
  hypre_NvtxPopRange();

  // final P
  P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                              hypre_ParCSRMatrixGlobalNumRows(A),
                              hypre_ParCSRMatrixGlobalNumCols(A_FC),
                              hypre_ParCSRMatrixColStarts(A),
                              hypre_ParCSRMatrixColStarts(A_FC),
                              hypre_CSRMatrixNumCols(W_offd),
                              P_diag_nnz,
                              hypre_CSRMatrixNumNonzeros(W_offd));

  hypre_ParCSRMatrixOwnsRowStarts(P) = 0;
  hypre_ParCSRMatrixOwnsColStarts(P) = 0;

  hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(P))    = P_diag_i;
  hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(P))    = P_diag_j;
  hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(P)) = P_diag_data;

  hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(P))    = P_offd_i;
  hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(W_offd);
  hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(W_offd);
  hypre_CSRMatrixJ(W_offd)    = NULL;
  hypre_CSRMatrixData(W_offd) = NULL;

  hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(P)) = HYPRE_MEMORY_DEVICE;
  hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(P)) = HYPRE_MEMORY_DEVICE;

  hypre_ParCSRMatrixDeviceColMapOffd(P) = hypre_ParCSRMatrixDeviceColMapOffd(A_FC);
  hypre_ParCSRMatrixColMapOffd(P)       = hypre_ParCSRMatrixColMapOffd(A_FC);
  hypre_ParCSRMatrixDeviceColMapOffd(A_FC) = NULL;
  hypre_ParCSRMatrixColMapOffd(A_FC)       = NULL;

  hypre_ParCSRMatrixNumNonzeros(P)  = hypre_ParCSRMatrixNumNonzeros(A_FC) +
                                     hypre_ParCSRMatrixGlobalNumCols(A_FC);
  hypre_ParCSRMatrixDNumNonzeros(P) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);

  hypre_MatvecCommPkgCreate(P);

  *P_ptr = P;

  hypre_TFree(diag, HYPRE_MEMORY_DEVICE);
  hypre_ParCSRMatrixDestroy(A_FF);
  hypre_ParCSRMatrixDestroy(A_FC);
  hypre_CSRMatrixDestroy(W_diag);
  hypre_CSRMatrixDestroy(W_offd);

  return hypre_error_flag;
}

#endif

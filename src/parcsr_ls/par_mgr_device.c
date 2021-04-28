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
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA)
void hypreDevice_extendWtoP( HYPRE_Int P_nr_of_rows, HYPRE_Int W_nr_of_rows, HYPRE_Int W_nr_of_cols, HYPRE_Int *CF_marker, HYPRE_Int W_diag_nnz, HYPRE_Int *W_diag_i, HYPRE_Int *W_diag_j, HYPRE_Complex *W_diag_data, HYPRE_Int *P_diag_i, HYPRE_Int *P_diag_j, HYPRE_Complex *P_diag_data, HYPRE_Int *W_offd_i, HYPRE_Int *P_offd_i );

HYPRE_Int
hypre_MGRBuildPDevice(hypre_ParCSRMatrix *A,
                      HYPRE_Int          *CF_marker_host,
                      HYPRE_BigInt       *num_cpts_global,
                      HYPRE_Int          method,
                      hypre_ParCSRMatrix **P_ptr)
{
  MPI_Comm            comm = hypre_ParCSRMatrixComm(A);
  HYPRE_Int           num_procs;
  HYPRE_Int           A_nr_of_rows = hypre_ParCSRMatrixNumRows(A);

  HYPRE_Int          *CF_marker_dev;
  hypre_ParCSRMatrix *A_FF=NULL, *A_FC=NULL, *P=NULL;
  hypre_CSRMatrix    *D_FF_inv=NULL, *W_diag=NULL, *W_offd=NULL;
  HYPRE_Int           W_nr_of_rows, P_diag_nnz, nfpoints;
  HYPRE_Int          *P_diag_i=NULL, *P_diag_j=NULL, *P_offd_i=NULL;
  HYPRE_Complex      *P_diag_data=NULL, *diag=NULL;

  hypre_MPI_Comm_size(comm, &num_procs);

  CF_marker_dev = hypre_TAlloc(HYPRE_Int, A_nr_of_rows, HYPRE_MEMORY_DEVICE);
  hypre_TMemcpy(CF_marker_dev, CF_marker_host, HYPRE_Int, A_nr_of_rows,
                HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

  nfpoints = HYPRE_THRUST_CALL( count,
                  CF_marker_dev,
                  CF_marker_dev + A_nr_of_rows,
                  -1);

  if (method > 0)
  {
    hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker_host, num_cpts_global, NULL, &A_FC, &A_FF);
    diag = hypre_CTAlloc(HYPRE_Complex, nfpoints, HYPRE_MEMORY_DEVICE);
    if (method == 1)
    {
      hypre_CSRMatrixExtractDiagonalDevice(hypre_ParCSRMatrixDiag(A_FF), diag, 3);
    }
    else if (method == 2)
    {
      hypre_CSRMatrixExtractDiagonalDevice(hypre_ParCSRMatrixDiag(A_FF), diag, 4);
    }

    D_FF_inv = hypre_CSRMatrixDiagMatrixFromVectorDevice(nfpoints, diag);
    W_diag = hypre_CSRMatrixMultiplyDevice(D_FF_inv, hypre_ParCSRMatrixDiag(A_FC));
    W_offd = hypre_CSRMatrixMultiplyDevice(D_FF_inv, hypre_ParCSRMatrixOffd(A_FC));
    hypre_CSRMatrixDestroy(D_FF_inv);
  }
  else
  {
    hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker_host, num_cpts_global, NULL, &A_FC, NULL);
    W_diag = hypre_CSRMatrixCreate(nfpoints, 0, 0);
    hypre_CSRMatrixInitialize_v2(W_diag, 0, HYPRE_MEMORY_DEVICE);
  }
  W_nr_of_rows = hypre_CSRMatrixNumRows(W_diag);

  /* Construct P from matrix product W_diag */
  P_diag_nnz  = hypre_CSRMatrixNumNonzeros(W_diag) + hypre_ParCSRMatrixNumCols(A_FC);
  P_diag_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows+1, HYPRE_MEMORY_DEVICE);
  P_diag_j    = hypre_TAlloc(HYPRE_Int,     P_diag_nnz,     HYPRE_MEMORY_DEVICE);
  P_diag_data = hypre_TAlloc(HYPRE_Complex, P_diag_nnz,     HYPRE_MEMORY_DEVICE);
  if (method > 0)
  {
    P_offd_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows+1, HYPRE_MEMORY_DEVICE);
  }

  //hypre_NvtxPushRangeColor("Extend matrix", 4);
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
                         W_offd ? hypre_CSRMatrixI(W_offd) : NULL,
                         P_offd_i );
  hypre_TFree(CF_marker_dev, HYPRE_MEMORY_DEVICE);
  //hypre_NvtxPopRange();

  // final P
  P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                              hypre_ParCSRMatrixGlobalNumRows(A),
                              hypre_ParCSRMatrixGlobalNumCols(A_FC),
                              hypre_ParCSRMatrixColStarts(A),
                              hypre_ParCSRMatrixColStarts(A_FC),
                              W_offd ? hypre_CSRMatrixNumCols(W_offd) : 0,
                              P_diag_nnz,
                              W_offd ? hypre_CSRMatrixNumNonzeros(W_offd) : 0);

  hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(P)) = HYPRE_MEMORY_DEVICE;
  hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(P)) = HYPRE_MEMORY_DEVICE;

  hypre_ParCSRMatrixOwnsRowStarts(P) = 0;
  hypre_ParCSRMatrixOwnsColStarts(P) = 0;

  hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(P))    = P_diag_i;
  hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(P))    = P_diag_j;
  hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(P)) = P_diag_data;

  if (method > 0)
  {
    hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(P))    = P_offd_i;
    hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(W_offd);
    hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(W_offd);
    hypre_CSRMatrixJ(W_offd)    = NULL;
    hypre_CSRMatrixData(W_offd) = NULL;

    hypre_ParCSRMatrixDeviceColMapOffd(P) = hypre_ParCSRMatrixDeviceColMapOffd(A_FC);
    hypre_ParCSRMatrixColMapOffd(P)       = hypre_ParCSRMatrixColMapOffd(A_FC);
    hypre_ParCSRMatrixDeviceColMapOffd(A_FC) = NULL;
    hypre_ParCSRMatrixColMapOffd(A_FC)       = NULL;

    hypre_ParCSRMatrixNumNonzeros(P)  = hypre_ParCSRMatrixNumNonzeros(A_FC) +
                                       hypre_ParCSRMatrixGlobalNumCols(A_FC);
  }
  else
  {
    hypre_ParCSRMatrixNumNonzeros(P) = hypre_ParCSRMatrixGlobalNumCols(A_FC);
  }
  hypre_ParCSRMatrixDNumNonzeros(P) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);

  hypre_MatvecCommPkgCreate(P);

  *P_ptr = P;

  if (diag) hypre_TFree(diag, HYPRE_MEMORY_DEVICE);
  if (A_FF) hypre_ParCSRMatrixDestroy(A_FF);
  if (A_FC) hypre_ParCSRMatrixDestroy(A_FC);
  if (W_diag) hypre_CSRMatrixDestroy(W_diag);
  if (W_offd) hypre_CSRMatrixDestroy(W_offd);

  return hypre_error_flag;
}

#endif

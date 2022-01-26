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
template<typename T>
struct functor : public thrust::binary_function<T, T, T>
{
   T scale;

   functor(T scale_) { scale = scale_; }

   __host__ __device__
   T operator()(T &x, T &y) const
   {
      return x + scale * (y - hypre_abs(x));
   }
};

void hypreDevice_extendWtoP( HYPRE_Int P_nr_of_rows, HYPRE_Int W_nr_of_rows, HYPRE_Int W_nr_of_cols,
                             HYPRE_Int *CF_marker, HYPRE_Int W_diag_nnz, HYPRE_Int *W_diag_i, HYPRE_Int *W_diag_j,
                             HYPRE_Complex *W_diag_data, HYPRE_Int *P_diag_i, HYPRE_Int *P_diag_j, HYPRE_Complex *P_diag_data,
                             HYPRE_Int *W_offd_i, HYPRE_Int *P_offd_i );

HYPRE_Int
hypre_MGRBuildPDevice(hypre_ParCSRMatrix  *A,
                      HYPRE_Int           *CF_marker_host,
                      HYPRE_BigInt        *num_cpts_global,
                      HYPRE_Int            method,
                      hypre_ParCSRMatrix **P_ptr)
{
   MPI_Comm            comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int           num_procs, my_id;
   HYPRE_Int           A_nr_of_rows = hypre_ParCSRMatrixNumRows(A);

   HYPRE_Int          *CF_marker_dev;
   hypre_ParCSRMatrix *A_FF = NULL, *A_FC = NULL, *P = NULL;
   hypre_CSRMatrix    *W_diag = NULL, *W_offd = NULL;
   HYPRE_Int           W_nr_of_rows, P_diag_nnz, nfpoints;
   HYPRE_Int          *P_diag_i = NULL, *P_diag_j = NULL, *P_offd_i = NULL;
   HYPRE_Complex      *P_diag_data = NULL, *diag = NULL, *diag1 = NULL;
   HYPRE_BigInt        nC_global;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

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
         // extract diag inverse sqrt
         //        hypre_CSRMatrixExtractDiagonalDevice(hypre_ParCSRMatrixDiag(A_FF), diag, 3);

         // L1-Jacobi-type interpolation
         HYPRE_Complex scal = 1.0;
         diag1 = hypre_CTAlloc(HYPRE_Complex, nfpoints, HYPRE_MEMORY_DEVICE);
         hypre_CSRMatrixExtractDiagonalDevice(hypre_ParCSRMatrixDiag(A_FF), diag, 0);
         hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixDiag(A_FF), NULL, NULL, diag1, 1, 1.0, "set");
         hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixDiag(A_FC), NULL, NULL, diag1, 1, 1.0, "add");
         hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixOffd(A_FF), NULL, NULL, diag1, 1, 1.0, "add");
         hypre_CSRMatrixComputeRowSumDevice(hypre_ParCSRMatrixOffd(A_FC), NULL, NULL, diag1, 1, 1.0, "add");

         HYPRE_THRUST_CALL( transform, diag, diag + nfpoints, diag1, diag, functor<HYPRE_Complex>(scal));
         HYPRE_THRUST_CALL( transform, diag, diag + nfpoints, diag, 1.0 / _1);

         hypre_TFree(diag1, HYPRE_MEMORY_DEVICE);
      }
      else if (method == 2)
      {
         // extract diag inverse
         hypre_CSRMatrixExtractDiagonalDevice(hypre_ParCSRMatrixDiag(A_FF), diag, 2);
      }

      HYPRE_THRUST_CALL( transform, diag, diag + nfpoints, diag, thrust::negate<HYPRE_Complex>() );

      hypre_Vector *D_FF_inv = hypre_SeqVectorCreate(nfpoints);
      hypre_VectorData(D_FF_inv) = diag;
      hypre_SeqVectorInitialize_v2(D_FF_inv, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixDiagScaleDevice(hypre_ParCSRMatrixDiag(A_FC), D_FF_inv, NULL);
      hypre_CSRMatrixDiagScaleDevice(hypre_ParCSRMatrixOffd(A_FC), D_FF_inv, NULL);
      hypre_SeqVectorDestroy(D_FF_inv);
      W_diag = hypre_ParCSRMatrixDiag(A_FC);
      W_offd = hypre_ParCSRMatrixOffd(A_FC);
      nC_global = hypre_ParCSRMatrixGlobalNumCols(A_FC);
   }
   else
   {
      W_diag = hypre_CSRMatrixCreate(nfpoints, A_nr_of_rows - nfpoints, 0);
      W_offd = hypre_CSRMatrixCreate(nfpoints, 0, 0);
      hypre_CSRMatrixInitialize_v2(W_diag, 0, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixInitialize_v2(W_offd, 0, HYPRE_MEMORY_DEVICE);

      if (my_id == (num_procs - 1))
      {
         nC_global = num_cpts_global[1];
      }
      hypre_MPI_Bcast(&nC_global, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }

   W_nr_of_rows = hypre_CSRMatrixNumRows(W_diag);

   /* Construct P from matrix product W_diag */
   P_diag_nnz  = hypre_CSRMatrixNumNonzeros(W_diag) + hypre_CSRMatrixNumCols(W_diag);
   P_diag_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);
   P_diag_j    = hypre_TAlloc(HYPRE_Int,     P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(HYPRE_Complex, P_diag_nnz,     HYPRE_MEMORY_DEVICE);
   P_offd_i    = hypre_TAlloc(HYPRE_Int,     A_nr_of_rows + 1, HYPRE_MEMORY_DEVICE);

   //hypre_NvtxPushRangeColor("Extend matrix", 4);
   hypreDevice_extendWtoP( A_nr_of_rows,
                           W_nr_of_rows,
                           hypre_CSRMatrixNumCols(W_diag),
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
   //hypre_NvtxPopRange();

   // final P
   P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                nC_global,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                hypre_CSRMatrixNumCols(W_offd),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(W_offd) );

   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(P)) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(P)) = HYPRE_MEMORY_DEVICE;

   hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(W_offd);
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(W_offd);
   hypre_CSRMatrixJ(W_offd)    = NULL;
   hypre_CSRMatrixData(W_offd) = NULL;

   if (method > 0)
   {
      hypre_ParCSRMatrixDeviceColMapOffd(P)    = hypre_ParCSRMatrixDeviceColMapOffd(A_FC);
      hypre_ParCSRMatrixColMapOffd(P)          = hypre_ParCSRMatrixColMapOffd(A_FC);
      hypre_ParCSRMatrixDeviceColMapOffd(A_FC) = NULL;
      hypre_ParCSRMatrixColMapOffd(A_FC)       = NULL;
      hypre_ParCSRMatrixNumNonzeros(P)         = hypre_ParCSRMatrixNumNonzeros(
                                                    A_FC) + hypre_ParCSRMatrixGlobalNumCols(A_FC);
   }
   else
   {
      hypre_ParCSRMatrixNumNonzeros(P) = nC_global;
   }
   hypre_ParCSRMatrixDNumNonzeros(P) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);

   hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   if (A_FF)
   {
      hypre_ParCSRMatrixDestroy(A_FF);
   }
   if (A_FC)
   {
      hypre_ParCSRMatrixDestroy(A_FC);
   }

   if (method <= 0)
   {
      hypre_CSRMatrixDestroy(W_diag);
      hypre_CSRMatrixDestroy(W_offd);
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_MGRRelaxL1JacobiDevice( hypre_ParCSRMatrix *A,
                              hypre_ParVector    *f,
                              HYPRE_Int          *CF_marker_host,
                              HYPRE_Int           relax_points,
                              HYPRE_Real          relax_weight,
                              HYPRE_Real         *l1_norms,
                              hypre_ParVector    *u,
                              hypre_ParVector    *Vtemp )
{
   HYPRE_Int *CF_marker_dev = NULL;

   // Copy CF_marker_host to device
   if (CF_marker_host != NULL)
   {
      CF_marker_dev = hypre_TAlloc(HYPRE_Int, hypre_ParCSRMatrixNumRows(A), HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(CF_marker_dev, CF_marker_host, HYPRE_Int, hypre_ParCSRMatrixNumRows(A),
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   }

   hypre_BoomerAMGRelax(A, f, CF_marker_dev, 18, relax_points, relax_weight, 1.0, l1_norms, u, Vtemp,
                        NULL);

   hypre_TFree(CF_marker_dev, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif

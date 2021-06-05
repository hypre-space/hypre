/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
/* #if 0 */

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
   hypre_CSRMatrix *R_offd;

   /* arrays */
   HYPRE_Complex      *R_diag_a;
   HYPRE_Int          *R_diag_i;
   HYPRE_Int          *R_diag_j;
   HYPRE_Complex      *R_offd_a;
   HYPRE_Int          *R_offd_i;
   HYPRE_Int          *R_offd_j;
   HYPRE_BigInt       *col_map_offd_R;

   HYPRE_Int           i, j, j1, i1, ic,
                       num_cols_offd_R;
   HYPRE_Int           my_id, num_procs;
   HYPRE_BigInt        total_global_cpts;
   HYPRE_Int           nnz_diag, nnz_offd, cnt_diag, cnt_offd;
   HYPRE_BigInt       *send_buf_i;

   /* local size */
   HYPRE_Int n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt col_start = hypre_ParCSRMatrixFirstRowIndex(A);

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /*-------------- global number of C points and my start position */
   if (my_id == (num_procs -1))
   {
      total_global_cpts = num_cpts_global[1];
   }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs-1, comm);

   get AFF and ACF
   hypre_ParCSRMatrix *AFF, *ACF, *X, *X2, *Z, *Z2;
   // WM: TODO: Is the SoC matrix S what we want here, or does strong_thresholdR express something else?
   // That is, do we want this SoC used here to be independent of the regular S? Construct new SoC here for R?
   hypre_ParCSRMatrixGenerateFFCFDevice(A, CF_marker, num_cpts_global, S, &ACF, &AFF);
   /* WM: TODO: delete old code commented below */
   /* hypre_ParCSRMatrixExtractSubmatrixFC(A, CF_marker, num_cpts_global, "FF", &AFF, strong_thresholdR); */
   /* hypre_ParCSRMatrixExtractSubmatrixFC(A, CF_marker, num_cpts_global, "CF", &ACF, strong_thresholdR); */

   hypre_CSRMatrix *AFF_diag = hypre_ParCSRMatrixDiag(AFF);
   hypre_CSRMatrix *AFF_offd = hypre_ParCSRMatrixOffd(AFF);
   HYPRE_Complex   *AFF_diag_a = hypre_CSRMatrixData(AFF_diag);
   HYPRE_Int       *AFF_diag_i = hypre_CSRMatrixI(AFF_diag);
   HYPRE_Int       *AFF_diag_j = hypre_CSRMatrixJ(AFF_diag);
   HYPRE_Complex   *AFF_offd_a = hypre_CSRMatrixData(AFF_offd);
   HYPRE_Int       *AFF_offd_i = hypre_CSRMatrixI(AFF_offd);
   HYPRE_Int       *AFF_offd_j = hypre_CSRMatrixJ(AFF_offd);
   HYPRE_Int        n_fpts = hypre_CSRMatrixNumRows(AFF_diag);
   HYPRE_Int        n_cpts = n_fine - n_fpts;
   hypre_assert(n_cpts == hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(ACF)));

   /* map from F-pts to all points */
   /* WM: check correctness */
   HYPRE_Int       *Fmap = hypre_TAlloc(HYPRE_Int, n_fpts, HYPRE_MEMORY_DEVICE);
   HYPRE_THRUST_CALL(copy_if,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(n_fine),
                      CF_marker,
                      Fmap,
                      is_negative<HYPRE_Int>());
   /* WM: TODO: delete old code commented below */
   /* for (i = 0, j = 0; i < n_fine; i++) */
   /* { */
   /*    if (CF_marker[i] < 0) */
   /*    { */
   /*       Fmap[j++] = i; */
   /*    } */
   /* } */

   /* hypre_assert(j == n_fpts); */

   /* store inverses of diagonal entries of AFF */
   HYPRE_Complex *diag_entries = hypre_TAlloc(HYPRE_Complex, n_fpts, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixExtractDiagonalDevice( AFF_diag, diag_entries, 2);

   /* A_FF := I - D^{-1}*A_FF */
   for (i = 0; i < n_fpts; i++)
   {
      i1 = AFF_diag_i[i];

      /* make sure the first entry is diagonal */
      hypre_assert(AFF_diag_j[i1] == i);

      /* !!! store the inverse */
      HYPRE_Complex di = 1.0 / AFF_diag_a[i1];
      diag_entries[i] = di;
      di = -di;
      AFF_diag_a[i1] = 0.0;
      for (j = i1+1; j < AFF_diag_i[i+1]; j++)
      {
         AFF_diag_a[j] *= di;
      }
      if (num_procs > 1)
      {
         for (j = AFF_offd_i[i]; j < AFF_offd_i[i+1]; j++)
         {
            hypre_assert( hypre_ParCSRMatrixColMapOffd(AFF)[AFF_offd_j[j]] != \
                          i + hypre_ParCSRMatrixFirstRowIndex(AFF) );

            AFF_offd_a[j] *= di;
         }
      }
   }

   /* Z = Acf * (I + N + N^2 + ... + N^k) * D^{-1}
    * N = I - D^{-1} * A_FF (computed above)
    * the last D^{-1} will not be done here (but later)
    */
   if (NeumannDeg < 1)
   {
      Z = ACF;
   }
   else if (NeumannDeg == 1)
   {
      X = hypre_ParMatmul(ACF, AFF);
      hypre_ParCSRMatrixAdd(1.0, ACF, 1.0, X, &Z);
      hypre_ParCSRMatrixDestroy(X);
   }
   else
   {
      X = hypre_ParMatmul(AFF, AFF);
      hypre_ParCSRMatrixAdd(1.0, AFF, 1.0, X, &Z);
      for (i = 2; i < NeumannDeg; i++)
      {
         X2 = hypre_ParMatmul(X, AFF);
         hypre_ParCSRMatrixAdd(1.0, Z, 1.0, X2, &Z2);
         hypre_ParCSRMatrixDestroy(X);
         hypre_ParCSRMatrixDestroy(Z);
         Z = Z2;
         X = X2;
      }
      hypre_ParCSRMatrixDestroy(X);
      X = hypre_ParMatmul(ACF, Z);
      hypre_ParCSRMatrixDestroy(Z);
      hypre_ParCSRMatrixAdd(1.0, ACF, 1.0, X, &Z);
      hypre_ParCSRMatrixDestroy(X);
   }

   hypre_ParCSRMatrixDestroy(AFF);
   if (NeumannDeg >= 1)
   {
      hypre_ParCSRMatrixDestroy(ACF);
   }

   hypre_CSRMatrix *Z_diag = hypre_ParCSRMatrixDiag(Z);
   hypre_CSRMatrix *Z_offd = hypre_ParCSRMatrixOffd(Z);
   HYPRE_Complex   *Z_diag_a = hypre_CSRMatrixData(Z_diag);
   HYPRE_Int       *Z_diag_i = hypre_CSRMatrixI(Z_diag);
   HYPRE_Int       *Z_diag_j = hypre_CSRMatrixJ(Z_diag);
   HYPRE_Complex   *Z_offd_a = hypre_CSRMatrixData(Z_offd);
   HYPRE_Int       *Z_offd_i = hypre_CSRMatrixI(Z_offd);
   HYPRE_Int       *Z_offd_j = hypre_CSRMatrixJ(Z_offd);
   HYPRE_Int        num_cols_offd_Z = hypre_CSRMatrixNumCols(Z_offd);
   /*
   HYPRE_BigInt       *col_map_offd_Z  = hypre_ParCSRMatrixColMapOffd(Z);
   */
   /* send and recv diagonal entries (wrt Z) */
   HYPRE_Complex *diag_entries_offd = hypre_TAlloc(HYPRE_Complex, num_cols_offd_Z, HYPRE_MEMORY_HOST);
   hypre_ParCSRCommPkg *comm_pkg_Z = hypre_ParCSRMatrixCommPkg(Z);
   HYPRE_Int num_sends_Z = hypre_ParCSRCommPkgNumSends(comm_pkg_Z);
   HYPRE_Int num_elems_send_Z = hypre_ParCSRCommPkgSendMapStart(comm_pkg_Z, num_sends_Z);
   HYPRE_Complex *send_buf_Z = hypre_TAlloc(HYPRE_Complex, num_elems_send_Z, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_elems_send_Z; i++)
   {
      send_buf_Z[i] = diag_entries[hypre_ParCSRCommPkgSendMapElmt(comm_pkg_Z, i)];
   }
   comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg_Z, send_buf_Z, diag_entries_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* send and recv Fmap (wrt Z): global */
   HYPRE_BigInt *Fmap_offd_global = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_Z, HYPRE_MEMORY_HOST);
   send_buf_i = hypre_TAlloc(HYPRE_BigInt, num_elems_send_Z, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_elems_send_Z; i++)
   {
      send_buf_i[i] = Fmap[hypre_ParCSRCommPkgSendMapElmt(comm_pkg_Z, i)] + col_start;
   }
   comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg_Z, send_buf_i, Fmap_offd_global);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   nnz_diag = hypre_CSRMatrixNumNonzeros(Z_diag) + n_cpts;
   nnz_offd = hypre_CSRMatrixNumNonzeros(Z_offd);

   /*------------- allocate arrays */
   R_diag_i = hypre_CTAlloc(HYPRE_Int,  n_cpts+1, HYPRE_MEMORY_HOST);
   R_diag_j = hypre_CTAlloc(HYPRE_Int,  nnz_diag, HYPRE_MEMORY_HOST);
   R_diag_a = hypre_CTAlloc(HYPRE_Complex, nnz_diag, HYPRE_MEMORY_HOST);

   /* not in ``if num_procs > 1'',
    * allocation needed even for empty CSR */
   R_offd_i = hypre_CTAlloc(HYPRE_Int,  n_cpts+1, HYPRE_MEMORY_HOST);
   R_offd_j = hypre_CTAlloc(HYPRE_Int,  nnz_offd, HYPRE_MEMORY_HOST);
   R_offd_a = hypre_CTAlloc(HYPRE_Complex, nnz_offd, HYPRE_MEMORY_HOST);

   /* redundant */
   R_diag_i[0] = 0;
   R_offd_i[0] = 0;

   /* reset counters */
   cnt_diag = 0;
   cnt_offd = 0;

   for (i = 0, ic = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      for (j = Z_diag_i[ic]; j < Z_diag_i[ic+1]; j++)
      {
         j1 = Z_diag_j[j];
         R_diag_j[cnt_diag] = Fmap[j1];
         R_diag_a[cnt_diag++] = -Z_diag_a[j] * diag_entries[j1];
      }

      /* identity */
      R_diag_j[cnt_diag] = i;
      R_diag_a[cnt_diag++] = 1.0;

      for (j = Z_offd_i[ic]; j < Z_offd_i[ic+1]; j++)
      {
         j1 = Z_offd_j[j];
         R_offd_j[cnt_offd] = j1;
         R_offd_a[cnt_offd++] = -Z_offd_a[j] * diag_entries_offd[j1];
      }

      R_diag_i[ic+1] = cnt_diag;
      R_offd_i[ic+1] = cnt_offd;

      ic++;
   }

   hypre_assert(ic == n_cpts);
   hypre_assert(cnt_diag == nnz_diag);
   hypre_assert(cnt_offd == nnz_offd);

   num_cols_offd_R = num_cols_offd_Z;
   col_map_offd_R = Fmap_offd_global;

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

   R_offd = hypre_ParCSRMatrixOffd(R);
   hypre_CSRMatrixData(R_offd) = R_offd_a;
   hypre_CSRMatrixI(R_offd)    = R_offd_i;
   hypre_CSRMatrixJ(R_offd)    = R_offd_j;
   /* R does not own ColStarts, since A does */
   hypre_ParCSRMatrixOwnsColStarts(R) = 0;

   hypre_ParCSRMatrixColMapOffd(R) = col_map_offd_R;

   /* create CommPkg of R */
   hypre_ParCSRMatrixAssumedPartition(R) = hypre_ParCSRMatrixAssumedPartition(A);
   hypre_ParCSRMatrixOwnsAssumedPartition(R) = 0;
   hypre_MatvecCommPkgCreate(R);

   /* Filter small entries from R */
   if (filter_thresholdR > 0) {
      hypre_ParCSRMatrixDropSmallEntries(R, filter_thresholdR, -1);
   }

   *R_ptr = R;

   hypre_ParCSRMatrixDestroy(Z);
   hypre_TFree(Fmap, HYPRE_MEMORY_HOST);
   hypre_TFree(diag_entries, HYPRE_MEMORY_HOST);
   hypre_TFree(diag_entries_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buf_i, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buf_Z, HYPRE_MEMORY_HOST);

   return 0;
}

#endif // defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

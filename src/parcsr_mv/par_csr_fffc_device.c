/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

typedef thrust::tuple<HYPRE_Int, HYPRE_Int> Tuple;
//typedef thrust::tuple<HYPRE_Int, HYPRE_Int, HYPRe_Int> Tuple3;

/* transform from local F/C index to global F/C index,
 * where F index "x" are saved as "-x-1"
 */
struct FFFC_functor : public thrust::unary_function<Tuple, HYPRE_BigInt>
{
   HYPRE_BigInt CF_first[2];

   FFFC_functor(HYPRE_BigInt F_first_, HYPRE_BigInt C_first_)
   {
      CF_first[1] = F_first_;
      CF_first[0] = C_first_;
   }

   __host__ __device__
   HYPRE_BigInt operator()(const Tuple& t) const
   {
      const HYPRE_Int local_idx = thrust::get<0>(t);
      const HYPRE_Int cf_marker = thrust::get<1>(t);
      const HYPRE_Int s = cf_marker < 0;
      const HYPRE_Int m = 1 - 2*s;
      return m*(local_idx + CF_first[s] + s);
   }
};

/* this predicate selects A^s_{FF} */
template<typename T>
struct FF_pred : public thrust::unary_function<Tuple, bool>
{
   HYPRE_Int  option;
   HYPRE_Int *row_CF_marker;
   T         *col_CF_marker;

   FF_pred(HYPRE_Int option_, HYPRE_Int *row_CF_marker_, T *col_CF_marker_)
   {
      option = option_;
      row_CF_marker = row_CF_marker_;
      col_CF_marker = col_CF_marker_;
   }

   __host__ __device__
   bool operator()(const Tuple& t) const
   {
      const HYPRE_Int i = thrust::get<0>(t);
      const HYPRE_Int j = thrust::get<1>(t);

      if (option == 1)
      {
         /* A_{F,F} */
         return row_CF_marker[i] <   0 && (j == -2 || j >= 0 && col_CF_marker[j] < 0);
      }
      else
      {
         /* A_{F2, F} */
         return row_CF_marker[i] == -2 && (j == -2 || j >= 0 && col_CF_marker[j] < 0);
      }
   }
};

/* this predicate selects A^s_{FC} */
template<typename T>
struct FC_pred : public thrust::unary_function<Tuple, bool>
{
   HYPRE_Int *row_CF_marker;
   T         *col_CF_marker;

   FC_pred(HYPRE_Int *row_CF_marker_, T *col_CF_marker_)
   {
      row_CF_marker = row_CF_marker_;
      col_CF_marker = col_CF_marker_;
   }

   __host__ __device__
   bool operator()(const Tuple& t) const
   {
      const HYPRE_Int i = thrust::get<0>(t);
      const HYPRE_Int j = thrust::get<1>(t);

      return row_CF_marker[i] < 0 && (j >= 0 && col_CF_marker[j] >= 0);
   }
};

/* this predicate selects A^s_{CF} */
template<typename T>
struct CF_pred : public thrust::unary_function<Tuple, bool>
{
   HYPRE_Int *row_CF_marker;
   T         *col_CF_marker;

   CF_pred(HYPRE_Int *row_CF_marker_, T *col_CF_marker_)
   {
      row_CF_marker = row_CF_marker_;
      col_CF_marker = col_CF_marker_;
   }

   __host__ __device__
   bool operator()(const Tuple& t) const
   {
      const HYPRE_Int i = thrust::get<0>(t);
      const HYPRE_Int j = thrust::get<1>(t);

      return row_CF_marker[i] >= 0 && (j >= 0 && col_CF_marker[j] < 0);
   }
};

/* this predicate selects A^s_{CC} */
template<typename T>
struct CC_pred : public thrust::unary_function<Tuple, bool>
{
   HYPRE_Int *row_CF_marker;
   T         *col_CF_marker;

   CC_pred(HYPRE_Int *row_CF_marker_, T *col_CF_marker_)
   {
      row_CF_marker = row_CF_marker_;
      col_CF_marker = col_CF_marker_;
   }

   __host__ __device__
   bool operator()(const Tuple& t) const
   {
      const HYPRE_Int i = thrust::get<0>(t);
      const HYPRE_Int j = thrust::get<1>(t);

      return row_CF_marker[i] >= 0 && (j == -2 || j >= 0 && col_CF_marker[j] >= 0);
   }
};

/* this predicate selects A^s_{C,:} */
struct CX_pred : public thrust::unary_function<Tuple, bool>
{
   HYPRE_Int *row_CF_marker;

   CX_pred(HYPRE_Int *row_CF_marker_)
   {
      row_CF_marker = row_CF_marker_;
   }

   __host__ __device__
   bool operator()(const Tuple& t) const
   {
      const HYPRE_Int i = thrust::get<0>(t);
      const HYPRE_Int j = thrust::get<1>(t);

      return row_CF_marker[i] >= 0 && (j == -2 || j >= 0);
   }
};

/* this predicate selects A^s_{:,C} */
template<typename T>
struct XC_pred : public thrust::unary_function<Tuple, bool>
{
   T         *col_CF_marker;

   XC_pred(T *col_CF_marker_)
   {
      col_CF_marker = col_CF_marker_;
   }

   __host__ __device__
   bool operator()(const Tuple& t) const
   {
      const HYPRE_Int i = thrust::get<0>(t);
      const HYPRE_Int j = thrust::get<1>(t);

      return (j == -2 && col_CF_marker[i] >= 0) || (j >= 0 && col_CF_marker[j] >= 0);
   }
};

/* Option = 1:
 *    F is marked as -1, C is +1
 *    | AFF AFC |
 *    | ACF ACC |
 *
 * Option = 2 (for aggressive coarsening):
 *    F_2 is marked as -2 in CF_marker, F_1 as -1, and C_2 as +1
 *    | AF1F1 AF1F2 AF1C2 |
 *    | AF2F1 AF2F2 AF2C2 |
 *    | AC2F1 AC2F2 AC2C2 |
 *    F = F1 + F2
 *    AFC: A_{F, C2}
 *    AFF: A_{F2, F}
 *    ACF: A_{C2, F}
 *    ACC: A_{C2, C2}
 */

HYPRE_Int
hypre_ParCSRMatrixGenerateFFFCDevice_core( hypre_ParCSRMatrix  *A,
                                           HYPRE_Int           *CF_marker_host,
                                           HYPRE_BigInt        *cpts_starts,
                                           hypre_ParCSRMatrix  *S,
                                           hypre_ParCSRMatrix **AFC_ptr,
                                           hypre_ParCSRMatrix **AFF_ptr,
                                           hypre_ParCSRMatrix **ACF_ptr,
                                           hypre_ParCSRMatrix **ACC_ptr,
                                           HYPRE_Int            option )
{
   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int                num_sends     = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int                num_elem_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   //HYPRE_MemoryLocation     memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   /* diag part of A */
   hypre_CSRMatrix    *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int          *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int           A_diag_nnz = hypre_CSRMatrixNumNonzeros(A_diag);
   /* offd part of A */
   hypre_CSRMatrix    *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int          *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int           A_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);
   HYPRE_Int           num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   /* SoC */
   HYPRE_Int          *Soc_diag_j = S ? hypre_ParCSRMatrixSocDiagJ(S) : A_diag_j;
   HYPRE_Int          *Soc_offd_j = S ? hypre_ParCSRMatrixSocOffdJ(S) : A_offd_j;
   /* MPI size and rank */
   HYPRE_Int           my_id, num_procs;
   /* nF and nC */
   HYPRE_Int           n_local, nF_local, nC_local, nF2_local = 0;
   HYPRE_BigInt       *fpts_starts, *row_starts, *f2pts_starts = NULL;
   HYPRE_BigInt        nF_global, nC_global, nF2_global = 0;
   HYPRE_BigInt        F_first, C_first;
   HYPRE_Int          *CF_marker;
   /* work arrays */
   HYPRE_Int          *map2FC, *map2F2 = NULL, *itmp, *A_diag_ii, *A_offd_ii, *offd_mark;
   HYPRE_BigInt       *send_buf, *recv_buf;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   n_local    = hypre_ParCSRMatrixNumRows(A);
   row_starts = hypre_ParCSRMatrixRowStarts(A);

   if (my_id == (num_procs -1))
   {
      nC_global = cpts_starts[1];
   }
   hypre_MPI_Bcast(&nC_global, 1, HYPRE_MPI_BIG_INT, num_procs-1, comm);
   nC_local = (HYPRE_Int) (cpts_starts[1] - cpts_starts[0]);
   fpts_starts = hypre_TAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
   fpts_starts[0] = row_starts[0] - cpts_starts[0];
   fpts_starts[1] = row_starts[1] - cpts_starts[1];
   F_first = fpts_starts[0];
   C_first = cpts_starts[0];
   nF_local = n_local - nC_local;
   nF_global = hypre_ParCSRMatrixGlobalNumRows(A) - nC_global;

   map2FC     = hypre_TAlloc(HYPRE_Int,    n_local,         HYPRE_MEMORY_DEVICE);
   itmp       = hypre_TAlloc(HYPRE_Int,    n_local,         HYPRE_MEMORY_DEVICE);
   recv_buf   = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);

   hypre_MemoryLocation cf_memory_location;
   hypre_GetPointerLocation(CF_marker_host, &cf_memory_location);
   if (cf_memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_HOST))
   {
      CF_marker = hypre_TAlloc(HYPRE_Int, n_local, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(CF_marker, CF_marker_host, HYPRE_Int, n_local, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   }
   else
   {
      CF_marker = CF_marker_host;
   }

   if (option == 2)
   {
      nF2_local = HYPRE_THRUST_CALL( count,
                                     CF_marker,
                                     CF_marker + n_local,
                                     -2 );

      HYPRE_BigInt nF2_local_big = nF2_local;

      f2pts_starts = hypre_TAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
      hypre_MPI_Scan(&nF2_local_big, f2pts_starts + 1, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
      f2pts_starts[0] = f2pts_starts[1] - nF2_local_big;
      if (my_id == (num_procs -1))
      {
         nF2_global = f2pts_starts[1];
      }
      hypre_MPI_Bcast(&nF2_global, 1, HYPRE_MPI_BIG_INT, num_procs-1, comm);
   }

   /* map from all points (i.e, F+C) to F/C indices */
   HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,           is_negative<HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + n_local, is_negative<HYPRE_Int>()),
                      map2FC, /* F */
                      HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */

   HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,           is_nonnegative<HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + n_local, is_nonnegative<HYPRE_Int>()),
                      itmp, /* C */
                      HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */

   HYPRE_THRUST_CALL( scatter_if,
                      itmp,
                      itmp + n_local,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::make_transform_iterator(CF_marker, is_nonnegative<HYPRE_Int>()),
                      map2FC ); /* FC combined */

   hypre_TFree(itmp, HYPRE_MEMORY_DEVICE);

   if (option == 2)
   {
      map2F2 = hypre_TAlloc(HYPRE_Int, n_local, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL( exclusive_scan,
                         thrust::make_transform_iterator(CF_marker,           equal<HYPRE_Int>(-2)),
                         thrust::make_transform_iterator(CF_marker + n_local, equal<HYPRE_Int>(-2)),
                         map2F2, /* F2 */
                         HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */
   }

   /* send_buf: global F/C indices. Note F-pts "x" are saved as "-x-1" */
   send_buf = hypre_TAlloc(HYPRE_BigInt, num_elem_send, HYPRE_MEMORY_DEVICE);

   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   FFFC_functor functor(F_first, C_first);
   HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                      thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(map2FC, CF_marker)), functor),
                      send_buf );

   comm_handle = hypre_ParCSRCommHandleCreate_v2(21, comm_pkg, HYPRE_MEMORY_DEVICE, send_buf, HYPRE_MEMORY_DEVICE, recv_buf);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   hypre_TFree(send_buf, HYPRE_MEMORY_DEVICE);

   thrust::zip_iterator< thrust::tuple<HYPRE_Int*, HYPRE_Int*, HYPRE_Complex*> > new_end;

   A_diag_ii = hypre_TAlloc(HYPRE_Int, A_diag_nnz,      HYPRE_MEMORY_DEVICE);
   A_offd_ii = hypre_TAlloc(HYPRE_Int, A_offd_nnz,      HYPRE_MEMORY_DEVICE);
   offd_mark = hypre_TAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_DEVICE);

   hypreDevice_CsrRowPtrsToIndices_v2(n_local, A_diag_nnz, A_diag_i, A_diag_ii);
   hypreDevice_CsrRowPtrsToIndices_v2(n_local, A_offd_nnz, A_offd_i, A_offd_ii);

   if (AFF_ptr)
   {
      HYPRE_Int           AFF_diag_nnz, AFF_offd_nnz;
      HYPRE_Int          *AFF_diag_ii, *AFF_diag_i, *AFF_diag_j;
      HYPRE_Complex      *AFF_diag_a;
      HYPRE_Int          *AFF_offd_ii, *AFF_offd_i, *AFF_offd_j;
      HYPRE_Complex      *AFF_offd_a;
      hypre_ParCSRMatrix *AFF;
      hypre_CSRMatrix    *AFF_diag, *AFF_offd;
      HYPRE_BigInt       *col_map_offd_AFF;
      HYPRE_Int           num_cols_AFF_offd;

      /* AFF Diag */
      FF_pred<HYPRE_Int> AFF_pred_diag(option, CF_marker, CF_marker);
      AFF_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        AFF_pred_diag );

      AFF_diag_ii = hypre_TAlloc(HYPRE_Int,     AFF_diag_nnz, HYPRE_MEMORY_DEVICE);
      AFF_diag_j  = hypre_TAlloc(HYPRE_Int,     AFF_diag_nnz, HYPRE_MEMORY_DEVICE);
      AFF_diag_a  = hypre_TAlloc(HYPRE_Complex, AFF_diag_nnz, HYPRE_MEMORY_DEVICE);

      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)) + A_diag_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(AFF_diag_ii, AFF_diag_j, AFF_diag_a)),
                                   AFF_pred_diag );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AFF_diag_ii + AFF_diag_nnz );

      HYPRE_THRUST_CALL ( gather,
                          AFF_diag_j,
                          AFF_diag_j + AFF_diag_nnz,
                          map2FC,
                          AFF_diag_j );

      HYPRE_THRUST_CALL ( gather,
                          AFF_diag_ii,
                          AFF_diag_ii + AFF_diag_nnz,
                          option == 1 ? map2FC : map2F2,
                          AFF_diag_ii );

      AFF_diag_i = hypreDevice_CsrRowIndicesToPtrs(option == 1 ? nF_local : nF2_local, AFF_diag_nnz, AFF_diag_ii);
      hypre_TFree(AFF_diag_ii, HYPRE_MEMORY_DEVICE);

      /* AFF Offd */
      FF_pred<HYPRE_BigInt> AFF_pred_offd(option, CF_marker, recv_buf);
      AFF_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        AFF_pred_offd );

      AFF_offd_ii = hypre_TAlloc(HYPRE_Int,     AFF_offd_nnz, HYPRE_MEMORY_DEVICE);
      AFF_offd_j  = hypre_TAlloc(HYPRE_Int,     AFF_offd_nnz, HYPRE_MEMORY_DEVICE);
      AFF_offd_a  = hypre_TAlloc(HYPRE_Complex, AFF_offd_nnz, HYPRE_MEMORY_DEVICE);

      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(AFF_offd_ii, AFF_offd_j, AFF_offd_a)),
                                   AFF_pred_offd );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AFF_offd_ii + AFF_offd_nnz );

      HYPRE_THRUST_CALL ( gather,
                          AFF_offd_ii,
                          AFF_offd_ii + AFF_offd_nnz,
                          option == 1 ? map2FC : map2F2,
                          AFF_offd_ii );

      AFF_offd_i = hypreDevice_CsrRowIndicesToPtrs(option == 1 ? nF_local : nF2_local, AFF_offd_nnz, AFF_offd_ii);
      hypre_TFree(AFF_offd_ii, HYPRE_MEMORY_DEVICE);

      /* col_map_offd_AFF */
      HYPRE_Int *tmp_j = hypre_TAlloc(HYPRE_Int, hypre_max(AFF_offd_nnz, num_cols_A_offd), HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_j, AFF_offd_j, HYPRE_Int, AFF_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + AFF_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + AFF_offd_nnz );
      num_cols_AFF_offd = tmp_end - tmp_j;
      HYPRE_THRUST_CALL( fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_AFF_offd, tmp_j, 1);
      HYPRE_THRUST_CALL( exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j );
      HYPRE_THRUST_CALL( gather,
                         AFF_offd_j,
                         AFF_offd_j + AFF_offd_nnz,
                         tmp_j,
                         AFF_offd_j );
      col_map_offd_AFF = hypre_TAlloc(HYPRE_BigInt, num_cols_AFF_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *tmp_end_big = HYPRE_THRUST_CALL( copy_if,
                                                     thrust::make_transform_iterator(recv_buf, -_1-1),
                                                     thrust::make_transform_iterator(recv_buf, -_1-1) + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_AFF,
                                                     thrust::identity<HYPRE_Int>() );
      hypre_assert(tmp_end_big - col_map_offd_AFF == num_cols_AFF_offd);
      hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);

      AFF = hypre_ParCSRMatrixCreate(comm,
                                     option == 1 ? nF_global : nF2_global,
                                     nF_global,
                                     option == 1 ? fpts_starts : f2pts_starts,
                                     fpts_starts,
                                     num_cols_AFF_offd,
                                     AFF_diag_nnz,
                                     AFF_offd_nnz);

      AFF_diag = hypre_ParCSRMatrixDiag(AFF);
      hypre_CSRMatrixData(AFF_diag) = AFF_diag_a;
      hypre_CSRMatrixI(AFF_diag)    = AFF_diag_i;
      hypre_CSRMatrixJ(AFF_diag)    = AFF_diag_j;

      AFF_offd = hypre_ParCSRMatrixOffd(AFF);
      hypre_CSRMatrixData(AFF_offd) = AFF_offd_a;
      hypre_CSRMatrixI(AFF_offd)    = AFF_offd_i;
      hypre_CSRMatrixJ(AFF_offd)    = AFF_offd_j;

      hypre_CSRMatrixMemoryLocation(AFF_diag) = HYPRE_MEMORY_DEVICE;
      hypre_CSRMatrixMemoryLocation(AFF_offd) = HYPRE_MEMORY_DEVICE;

      hypre_ParCSRMatrixDeviceColMapOffd(AFF) = col_map_offd_AFF;
      hypre_ParCSRMatrixColMapOffd(AFF) = hypre_TAlloc(HYPRE_BigInt, num_cols_AFF_offd, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(AFF), col_map_offd_AFF, HYPRE_BigInt, num_cols_AFF_offd,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

      hypre_ParCSRMatrixSetNumNonzeros(AFF);
      hypre_ParCSRMatrixDNumNonzeros(AFF) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(AFF);
      hypre_MatvecCommPkgCreate(AFF);

      *AFF_ptr = AFF;
   }

   if (AFC_ptr)
   {
      HYPRE_Int           AFC_diag_nnz, AFC_offd_nnz;
      HYPRE_Int          *AFC_diag_ii, *AFC_diag_i, *AFC_diag_j;
      HYPRE_Complex      *AFC_diag_a;
      HYPRE_Int          *AFC_offd_ii, *AFC_offd_i, *AFC_offd_j;
      HYPRE_Complex      *AFC_offd_a;
      hypre_ParCSRMatrix *AFC;
      hypre_CSRMatrix    *AFC_diag, *AFC_offd;
      HYPRE_BigInt       *col_map_offd_AFC;
      HYPRE_Int           num_cols_AFC_offd;

      /* AFC Diag */
      FC_pred<HYPRE_Int> AFC_pred_diag(CF_marker, CF_marker);
      AFC_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        AFC_pred_diag );

      AFC_diag_ii = hypre_TAlloc(HYPRE_Int,     AFC_diag_nnz, HYPRE_MEMORY_DEVICE);
      AFC_diag_j  = hypre_TAlloc(HYPRE_Int,     AFC_diag_nnz, HYPRE_MEMORY_DEVICE);
      AFC_diag_a  = hypre_TAlloc(HYPRE_Complex, AFC_diag_nnz, HYPRE_MEMORY_DEVICE);

      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j, A_diag_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j, A_diag_a)) + A_diag_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(AFC_diag_ii, AFC_diag_j, AFC_diag_a)),
                                   AFC_pred_diag );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AFC_diag_ii + AFC_diag_nnz );

      HYPRE_THRUST_CALL ( gather,
                          AFC_diag_j,
                          AFC_diag_j + AFC_diag_nnz,
                          map2FC,
                          AFC_diag_j );

      HYPRE_THRUST_CALL ( gather,
                          AFC_diag_ii,
                          AFC_diag_ii + AFC_diag_nnz,
                          map2FC,
                          AFC_diag_ii );

      AFC_diag_i = hypreDevice_CsrRowIndicesToPtrs(nF_local, AFC_diag_nnz, AFC_diag_ii);
      hypre_TFree(AFC_diag_ii, HYPRE_MEMORY_DEVICE);

      /* AFC Offd */
      FC_pred<HYPRE_BigInt> AFC_pred_offd(CF_marker, recv_buf);
      AFC_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        AFC_pred_offd );

      AFC_offd_ii = hypre_TAlloc(HYPRE_Int,     AFC_offd_nnz, HYPRE_MEMORY_DEVICE);
      AFC_offd_j  = hypre_TAlloc(HYPRE_Int,     AFC_offd_nnz, HYPRE_MEMORY_DEVICE);
      AFC_offd_a  = hypre_TAlloc(HYPRE_Complex, AFC_offd_nnz, HYPRE_MEMORY_DEVICE);

      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(AFC_offd_ii, AFC_offd_j, AFC_offd_a)),
                                   AFC_pred_offd );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AFC_offd_ii + AFC_offd_nnz );

      HYPRE_THRUST_CALL ( gather,
                          AFC_offd_ii,
                          AFC_offd_ii + AFC_offd_nnz,
                          map2FC,
                          AFC_offd_ii );

      AFC_offd_i = hypreDevice_CsrRowIndicesToPtrs(nF_local, AFC_offd_nnz, AFC_offd_ii);
      hypre_TFree(AFC_offd_ii, HYPRE_MEMORY_DEVICE);

      /* col_map_offd_AFC */
      HYPRE_Int *tmp_j = hypre_TAlloc(HYPRE_Int, hypre_max(AFC_offd_nnz, num_cols_A_offd), HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_j, AFC_offd_j, HYPRE_Int, AFC_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + AFC_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + AFC_offd_nnz );
      num_cols_AFC_offd = tmp_end - tmp_j;
      HYPRE_THRUST_CALL( fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_AFC_offd, tmp_j, 1);
      HYPRE_THRUST_CALL( exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j);
      HYPRE_THRUST_CALL( gather,
                         AFC_offd_j,
                         AFC_offd_j + AFC_offd_nnz,
                         tmp_j,
                         AFC_offd_j );
      col_map_offd_AFC = hypre_TAlloc(HYPRE_BigInt, num_cols_AFC_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *tmp_end_big = HYPRE_THRUST_CALL( copy_if,
                                                     recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_AFC,
                                                     thrust::identity<HYPRE_Int>());
      hypre_assert(tmp_end_big - col_map_offd_AFC == num_cols_AFC_offd);
      hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);

      /* AFC */
      AFC = hypre_ParCSRMatrixCreate(comm,
                                     nF_global,
                                     nC_global,
                                     fpts_starts,
                                     cpts_starts,
                                     num_cols_AFC_offd,
                                     AFC_diag_nnz,
                                     AFC_offd_nnz);

      AFC_diag = hypre_ParCSRMatrixDiag(AFC);
      hypre_CSRMatrixData(AFC_diag) = AFC_diag_a;
      hypre_CSRMatrixI(AFC_diag)    = AFC_diag_i;
      hypre_CSRMatrixJ(AFC_diag)    = AFC_diag_j;

      AFC_offd = hypre_ParCSRMatrixOffd(AFC);
      hypre_CSRMatrixData(AFC_offd) = AFC_offd_a;
      hypre_CSRMatrixI(AFC_offd)    = AFC_offd_i;
      hypre_CSRMatrixJ(AFC_offd)    = AFC_offd_j;

      hypre_CSRMatrixMemoryLocation(AFC_diag) = HYPRE_MEMORY_DEVICE;
      hypre_CSRMatrixMemoryLocation(AFC_offd) = HYPRE_MEMORY_DEVICE;

      hypre_ParCSRMatrixDeviceColMapOffd(AFC) = col_map_offd_AFC;
      hypre_ParCSRMatrixColMapOffd(AFC) = hypre_TAlloc(HYPRE_BigInt, num_cols_AFC_offd, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(AFC), col_map_offd_AFC, HYPRE_BigInt, num_cols_AFC_offd,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

      hypre_ParCSRMatrixSetNumNonzeros(AFC);
      hypre_ParCSRMatrixDNumNonzeros(AFC) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(AFC);
      hypre_MatvecCommPkgCreate(AFC);

      *AFC_ptr = AFC;
   }

   if (ACF_ptr)
   {
      HYPRE_Int           ACF_diag_nnz, ACF_offd_nnz;
      HYPRE_Int          *ACF_diag_ii, *ACF_diag_i, *ACF_diag_j;
      HYPRE_Complex      *ACF_diag_a;
      HYPRE_Int          *ACF_offd_ii, *ACF_offd_i, *ACF_offd_j;
      HYPRE_Complex      *ACF_offd_a;
      hypre_ParCSRMatrix *ACF;
      hypre_CSRMatrix    *ACF_diag, *ACF_offd;
      HYPRE_BigInt       *col_map_offd_ACF;
      HYPRE_Int           num_cols_ACF_offd;

      /* ACF Diag */
      CF_pred<HYPRE_Int> ACF_pred_diag(CF_marker, CF_marker);
      ACF_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        ACF_pred_diag );

      ACF_diag_ii = hypre_TAlloc(HYPRE_Int,     ACF_diag_nnz, HYPRE_MEMORY_DEVICE);
      ACF_diag_j  = hypre_TAlloc(HYPRE_Int,     ACF_diag_nnz, HYPRE_MEMORY_DEVICE);
      ACF_diag_a  = hypre_TAlloc(HYPRE_Complex, ACF_diag_nnz, HYPRE_MEMORY_DEVICE);

      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j, A_diag_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j, A_diag_a)) + A_diag_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(ACF_diag_ii, ACF_diag_j, ACF_diag_a)),
                                   ACF_pred_diag );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == ACF_diag_ii + ACF_diag_nnz );

      HYPRE_THRUST_CALL ( gather,
                          ACF_diag_j,
                          ACF_diag_j + ACF_diag_nnz,
                          map2FC,
                          ACF_diag_j );

      HYPRE_THRUST_CALL ( gather,
                          ACF_diag_ii,
                          ACF_diag_ii + ACF_diag_nnz,
                          map2FC,
                          ACF_diag_ii );

      ACF_diag_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACF_diag_nnz, ACF_diag_ii);
      hypre_TFree(ACF_diag_ii, HYPRE_MEMORY_DEVICE);

      /* ACF Offd */
      CF_pred<HYPRE_BigInt> ACF_pred_offd(CF_marker, recv_buf);
      ACF_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        ACF_pred_offd );

      ACF_offd_ii = hypre_TAlloc(HYPRE_Int,     ACF_offd_nnz, HYPRE_MEMORY_DEVICE);
      ACF_offd_j  = hypre_TAlloc(HYPRE_Int,     ACF_offd_nnz, HYPRE_MEMORY_DEVICE);
      ACF_offd_a  = hypre_TAlloc(HYPRE_Complex, ACF_offd_nnz, HYPRE_MEMORY_DEVICE);

      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(ACF_offd_ii, ACF_offd_j, ACF_offd_a)),
                                   ACF_pred_offd );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == ACF_offd_ii + ACF_offd_nnz );

      HYPRE_THRUST_CALL ( gather,
                          ACF_offd_ii,
                          ACF_offd_ii + ACF_offd_nnz,
                          map2FC,
                          ACF_offd_ii );

      ACF_offd_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACF_offd_nnz, ACF_offd_ii);
      hypre_TFree(ACF_offd_ii, HYPRE_MEMORY_DEVICE);

      /* col_map_offd_ACF */
      HYPRE_Int *tmp_j = hypre_TAlloc(HYPRE_Int, hypre_max(ACF_offd_nnz, num_cols_A_offd), HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_j, ACF_offd_j, HYPRE_Int, ACF_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + ACF_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + ACF_offd_nnz );
      num_cols_ACF_offd = tmp_end - tmp_j;
      HYPRE_THRUST_CALL( fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_ACF_offd, tmp_j, 1);
      HYPRE_THRUST_CALL( exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j);
      HYPRE_THRUST_CALL( gather,
                         ACF_offd_j,
                         ACF_offd_j + ACF_offd_nnz,
                         tmp_j,
                         ACF_offd_j );
      col_map_offd_ACF = hypre_TAlloc(HYPRE_BigInt, num_cols_ACF_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *tmp_end_big = HYPRE_THRUST_CALL( copy_if,
                                                     thrust::make_transform_iterator(recv_buf, -_1-1),
                                                     thrust::make_transform_iterator(recv_buf, -_1-1) + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_ACF,
                                                     thrust::identity<HYPRE_Int>());
      hypre_assert(tmp_end_big - col_map_offd_ACF == num_cols_ACF_offd);
      hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);

      /* ACF */
      ACF = hypre_ParCSRMatrixCreate(comm,
                                     nC_global,
                                     nF_global,
                                     cpts_starts,
                                     fpts_starts,
                                     num_cols_ACF_offd,
                                     ACF_diag_nnz,
                                     ACF_offd_nnz);

      ACF_diag = hypre_ParCSRMatrixDiag(ACF);
      hypre_CSRMatrixData(ACF_diag) = ACF_diag_a;
      hypre_CSRMatrixI(ACF_diag)    = ACF_diag_i;
      hypre_CSRMatrixJ(ACF_diag)    = ACF_diag_j;

      ACF_offd = hypre_ParCSRMatrixOffd(ACF);
      hypre_CSRMatrixData(ACF_offd) = ACF_offd_a;
      hypre_CSRMatrixI(ACF_offd)    = ACF_offd_i;
      hypre_CSRMatrixJ(ACF_offd)    = ACF_offd_j;

      hypre_CSRMatrixMemoryLocation(ACF_diag) = HYPRE_MEMORY_DEVICE;
      hypre_CSRMatrixMemoryLocation(ACF_offd) = HYPRE_MEMORY_DEVICE;

      hypre_ParCSRMatrixDeviceColMapOffd(ACF) = col_map_offd_ACF;
      hypre_ParCSRMatrixColMapOffd(ACF) = hypre_TAlloc(HYPRE_BigInt, num_cols_ACF_offd, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(ACF), col_map_offd_ACF, HYPRE_BigInt, num_cols_ACF_offd,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

      hypre_ParCSRMatrixSetNumNonzeros(ACF);
      hypre_ParCSRMatrixDNumNonzeros(ACF) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(ACF);
      hypre_MatvecCommPkgCreate(ACF);

      *ACF_ptr = ACF;
   }

   if (ACC_ptr)
   {
      HYPRE_Int           ACC_diag_nnz, ACC_offd_nnz;
      HYPRE_Int          *ACC_diag_ii, *ACC_diag_i, *ACC_diag_j;
      HYPRE_Complex      *ACC_diag_a;
      HYPRE_Int          *ACC_offd_ii, *ACC_offd_i, *ACC_offd_j;
      HYPRE_Complex      *ACC_offd_a;
      hypre_ParCSRMatrix *ACC;
      hypre_CSRMatrix    *ACC_diag, *ACC_offd;
      HYPRE_BigInt       *col_map_offd_ACC;
      HYPRE_Int           num_cols_ACC_offd;

      /* ACC Diag */
      CC_pred<HYPRE_Int> ACC_pred_diag(CF_marker, CF_marker);
      ACC_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        ACC_pred_diag );

      ACC_diag_ii = hypre_TAlloc(HYPRE_Int,     ACC_diag_nnz, HYPRE_MEMORY_DEVICE);
      ACC_diag_j  = hypre_TAlloc(HYPRE_Int,     ACC_diag_nnz, HYPRE_MEMORY_DEVICE);
      ACC_diag_a  = hypre_TAlloc(HYPRE_Complex, ACC_diag_nnz, HYPRE_MEMORY_DEVICE);

      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)) + A_diag_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(ACC_diag_ii, ACC_diag_j, ACC_diag_a)),
                                   ACC_pred_diag );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == ACC_diag_ii + ACC_diag_nnz );

      HYPRE_THRUST_CALL ( gather,
                          ACC_diag_j,
                          ACC_diag_j + ACC_diag_nnz,
                          map2FC,
                          ACC_diag_j );

      HYPRE_THRUST_CALL ( gather,
                          ACC_diag_ii,
                          ACC_diag_ii + ACC_diag_nnz,
                          map2FC,
                          ACC_diag_ii );

      ACC_diag_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACC_diag_nnz, ACC_diag_ii);
      hypre_TFree(ACC_diag_ii, HYPRE_MEMORY_DEVICE);

      /* ACC Offd */
      CC_pred<HYPRE_BigInt> ACC_pred_offd(CF_marker, recv_buf);
      ACC_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        ACC_pred_offd );

      ACC_offd_ii = hypre_TAlloc(HYPRE_Int,     ACC_offd_nnz, HYPRE_MEMORY_DEVICE);
      ACC_offd_j  = hypre_TAlloc(HYPRE_Int,     ACC_offd_nnz, HYPRE_MEMORY_DEVICE);
      ACC_offd_a  = hypre_TAlloc(HYPRE_Complex, ACC_offd_nnz, HYPRE_MEMORY_DEVICE);

      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(ACC_offd_ii, ACC_offd_j, ACC_offd_a)),
                                   ACC_pred_offd );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == ACC_offd_ii + ACC_offd_nnz );

      HYPRE_THRUST_CALL ( gather,
                          ACC_offd_ii,
                          ACC_offd_ii + ACC_offd_nnz,
                          map2FC,
                          ACC_offd_ii );

      ACC_offd_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACC_offd_nnz, ACC_offd_ii);
      hypre_TFree(ACC_offd_ii, HYPRE_MEMORY_DEVICE);

      /* col_map_offd_ACC */
      HYPRE_Int *tmp_j = hypre_TAlloc(HYPRE_Int, hypre_max(ACC_offd_nnz, num_cols_A_offd), HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_j, ACC_offd_j, HYPRE_Int, ACC_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + ACC_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + ACC_offd_nnz );
      num_cols_ACC_offd = tmp_end - tmp_j;
      HYPRE_THRUST_CALL( fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_ACC_offd, tmp_j, 1);
      HYPRE_THRUST_CALL( exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j);
      HYPRE_THRUST_CALL( gather,
                         ACC_offd_j,
                         ACC_offd_j + ACC_offd_nnz,
                         tmp_j,
                         ACC_offd_j );
      col_map_offd_ACC = hypre_TAlloc(HYPRE_BigInt, num_cols_ACC_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *tmp_end_big = HYPRE_THRUST_CALL( copy_if,
                                                     recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_ACC,
                                                     thrust::identity<HYPRE_Int>());
      hypre_assert(tmp_end_big - col_map_offd_ACC == num_cols_ACC_offd);
      hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);

      /* ACC */
      ACC = hypre_ParCSRMatrixCreate(comm,
                                     nC_global,
                                     nC_global,
                                     cpts_starts,
                                     cpts_starts,
                                     num_cols_ACC_offd,
                                     ACC_diag_nnz,
                                     ACC_offd_nnz);

      ACC_diag = hypre_ParCSRMatrixDiag(ACC);
      hypre_CSRMatrixData(ACC_diag) = ACC_diag_a;
      hypre_CSRMatrixI(ACC_diag)    = ACC_diag_i;
      hypre_CSRMatrixJ(ACC_diag)    = ACC_diag_j;

      ACC_offd = hypre_ParCSRMatrixOffd(ACC);
      hypre_CSRMatrixData(ACC_offd) = ACC_offd_a;
      hypre_CSRMatrixI(ACC_offd)    = ACC_offd_i;
      hypre_CSRMatrixJ(ACC_offd)    = ACC_offd_j;

      hypre_CSRMatrixMemoryLocation(ACC_diag) = HYPRE_MEMORY_DEVICE;
      hypre_CSRMatrixMemoryLocation(ACC_offd) = HYPRE_MEMORY_DEVICE;

      hypre_ParCSRMatrixDeviceColMapOffd(ACC) = col_map_offd_ACC;
      hypre_ParCSRMatrixColMapOffd(ACC) = hypre_TAlloc(HYPRE_BigInt, num_cols_ACC_offd, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(ACC), col_map_offd_ACC, HYPRE_BigInt, num_cols_ACC_offd,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

      hypre_ParCSRMatrixSetNumNonzeros(ACC);
      hypre_ParCSRMatrixDNumNonzeros(ACC) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(ACC);
      hypre_MatvecCommPkgCreate(ACC);

      *ACC_ptr = ACC;
   }

   hypre_TFree(A_diag_ii, HYPRE_MEMORY_DEVICE);
   hypre_TFree(A_offd_ii, HYPRE_MEMORY_DEVICE);
   hypre_TFree(offd_mark, HYPRE_MEMORY_DEVICE);
   if (cf_memory_location == hypre_GetActualMemLocation(HYPRE_MEMORY_HOST))
   {
      hypre_TFree(CF_marker, HYPRE_MEMORY_DEVICE);
   }
   hypre_TFree(map2FC,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(map2F2,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(recv_buf,  HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

HYPRE_Int
hypre_ParCSRMatrixGenerateFFFCDevice( hypre_ParCSRMatrix  *A,
                                      HYPRE_Int           *CF_marker_host,
                                      HYPRE_BigInt        *cpts_starts,
                                      hypre_ParCSRMatrix  *S,
                                      hypre_ParCSRMatrix **AFC_ptr,
                                      hypre_ParCSRMatrix **AFF_ptr )
{
   return hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker_host, cpts_starts, S, AFC_ptr, AFF_ptr, NULL, NULL, 1);
}

HYPRE_Int
hypre_ParCSRMatrixGenerateFFFC3Device( hypre_ParCSRMatrix  *A,
                                       HYPRE_Int           *CF_marker_host,
                                       HYPRE_BigInt        *cpts_starts,
                                       hypre_ParCSRMatrix  *S,
                                       hypre_ParCSRMatrix **AFC_ptr,
                                       hypre_ParCSRMatrix **AFF_ptr)
{
   return hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker_host, cpts_starts, S, AFC_ptr, AFF_ptr, NULL, NULL, 2);
}

HYPRE_Int
hypre_ParCSRMatrixGenerateFFCFDevice( hypre_ParCSRMatrix  *A,
                                      HYPRE_Int           *CF_marker_host,
                                      HYPRE_BigInt        *cpts_starts,
                                      hypre_ParCSRMatrix  *S,
                                      hypre_ParCSRMatrix **ACF_ptr,
                                      hypre_ParCSRMatrix **AFF_ptr )
{
   return hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker_host, cpts_starts, S, NULL, AFF_ptr, ACF_ptr, NULL, 1);
}


HYPRE_Int
hypre_ParCSRMatrixGenerateCFDevice( hypre_ParCSRMatrix  *A,
                                    HYPRE_Int           *CF_marker_host,
                                    HYPRE_BigInt        *cpts_starts,
                                    hypre_ParCSRMatrix  *S,
                                    hypre_ParCSRMatrix **ACF_ptr)
{
   return hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker_host, cpts_starts, S, NULL, NULL, ACF_ptr, NULL, 1);
}

HYPRE_Int
hypre_ParCSRMatrixGenerateCCDevice( hypre_ParCSRMatrix  *A,
                                    HYPRE_Int           *CF_marker_host,
                                    HYPRE_BigInt        *cpts_starts,
                                    hypre_ParCSRMatrix  *S,
                                    hypre_ParCSRMatrix **ACC_ptr)
{
   return hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker_host, cpts_starts, S, NULL, NULL, NULL, ACC_ptr, 1);
}

HYPRE_Int
hypre_ParCSRMatrixGenerate1DCFDevice( hypre_ParCSRMatrix  *A,
                                      HYPRE_Int           *CF_marker_host,
                                      HYPRE_BigInt        *cpts_starts,
                                      hypre_ParCSRMatrix  *S,
                                      hypre_ParCSRMatrix **ACX_ptr,
                                      hypre_ParCSRMatrix **AXC_ptr )
{
   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int                num_sends     = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int                num_elem_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   //HYPRE_MemoryLocation     memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   /* diag part of A */
   hypre_CSRMatrix    *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int          *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int           A_diag_nnz = hypre_CSRMatrixNumNonzeros(A_diag);
   /* offd part of A */
   hypre_CSRMatrix    *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int          *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int           A_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);
   HYPRE_BigInt       *col_map_offd_A = hypre_ParCSRMatrixDeviceColMapOffd(A);
   HYPRE_Int           num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   /* SoC */
   HYPRE_Int          *Soc_diag_j = S ? hypre_ParCSRMatrixSocDiagJ(S) : A_diag_j;
   HYPRE_Int          *Soc_offd_j = S ? hypre_ParCSRMatrixSocOffdJ(S) : A_offd_j;
   /* MPI size and rank */
   HYPRE_Int           my_id, num_procs;
   /* nF and nC */
   HYPRE_Int           n_local, /*nF_local,*/ nC_local;
   HYPRE_BigInt       *fpts_starts, *row_starts;
   HYPRE_BigInt        /*nF_global,*/ nC_global;
   HYPRE_BigInt        F_first, C_first;
   HYPRE_Int          *CF_marker;
   /* work arrays */
   HYPRE_Int          *map2FC, *itmp, *A_diag_ii, *A_offd_ii, *offd_mark;
   HYPRE_BigInt       *send_buf, *recv_buf;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   n_local    = hypre_ParCSRMatrixNumRows(A);
   row_starts = hypre_ParCSRMatrixRowStarts(A);

   if (!col_map_offd_A)
   {
      col_map_offd_A = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(col_map_offd_A, hypre_ParCSRMatrixColMapOffd(A), HYPRE_BigInt, num_cols_A_offd,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixDeviceColMapOffd(A) = col_map_offd_A;
   }

   if (my_id == (num_procs -1))
   {
      nC_global = cpts_starts[1];
   }
   hypre_MPI_Bcast(&nC_global, 1, HYPRE_MPI_BIG_INT, num_procs-1, comm);
   nC_local = (HYPRE_Int) (cpts_starts[1] - cpts_starts[0]);
   fpts_starts = hypre_TAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
   fpts_starts[0] = row_starts[0] - cpts_starts[0];
   fpts_starts[1] = row_starts[1] - cpts_starts[1];
   F_first = fpts_starts[0];
   C_first = cpts_starts[0];
   /*
   nF_local = n_local - nC_local;
   nF_global = hypre_ParCSRMatrixGlobalNumRows(A) - nC_global;
   */

   CF_marker  = hypre_TAlloc(HYPRE_Int,    n_local,         HYPRE_MEMORY_DEVICE);
   map2FC     = hypre_TAlloc(HYPRE_Int,    n_local,         HYPRE_MEMORY_DEVICE);
   itmp       = hypre_TAlloc(HYPRE_Int,    n_local,         HYPRE_MEMORY_DEVICE);
   recv_buf   = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(CF_marker, CF_marker_host, HYPRE_Int, n_local, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   /* map from all points (i.e, F+C) to F/C indices */
   HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,           is_negative<HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + n_local, is_negative<HYPRE_Int>()),
                      map2FC, /* F */
                      HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */

   HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,           is_nonnegative<HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + n_local, is_nonnegative<HYPRE_Int>()),
                      itmp, /* C */
                      HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */

   HYPRE_THRUST_CALL( scatter_if,
                      itmp,
                      itmp + n_local,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::make_transform_iterator(CF_marker, is_nonnegative<HYPRE_Int>()),
                      map2FC ); /* FC combined */

   hypre_TFree(itmp, HYPRE_MEMORY_DEVICE);

   /* send_buf: global F/C indices. Note F-pts "x" are saved as "-x-1" */
   send_buf = hypre_TAlloc(HYPRE_BigInt, num_elem_send, HYPRE_MEMORY_DEVICE);

   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   FFFC_functor functor(F_first, C_first);
   HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                      thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(map2FC, CF_marker)), functor),
                      send_buf );

   comm_handle = hypre_ParCSRCommHandleCreate_v2(21, comm_pkg, HYPRE_MEMORY_DEVICE, send_buf, HYPRE_MEMORY_DEVICE, recv_buf);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   hypre_TFree(send_buf, HYPRE_MEMORY_DEVICE);

   thrust::zip_iterator< thrust::tuple<HYPRE_Int*, HYPRE_Int*, HYPRE_Complex*> > new_end;

   A_diag_ii = hypre_TAlloc(HYPRE_Int, A_diag_nnz,      HYPRE_MEMORY_DEVICE);
   A_offd_ii = hypre_TAlloc(HYPRE_Int, A_offd_nnz,      HYPRE_MEMORY_DEVICE);
   offd_mark = hypre_TAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_DEVICE);

   hypreDevice_CsrRowPtrsToIndices_v2(n_local, A_diag_nnz, A_diag_i, A_diag_ii);
   hypreDevice_CsrRowPtrsToIndices_v2(n_local, A_offd_nnz, A_offd_i, A_offd_ii);

   if (ACX_ptr)
   {
      HYPRE_Int           ACX_diag_nnz, ACX_offd_nnz;
      HYPRE_Int          *ACX_diag_ii, *ACX_diag_i, *ACX_diag_j;
      HYPRE_Complex      *ACX_diag_a;
      HYPRE_Int          *ACX_offd_ii, *ACX_offd_i, *ACX_offd_j;
      HYPRE_Complex      *ACX_offd_a;
      hypre_ParCSRMatrix *ACX;
      hypre_CSRMatrix    *ACX_diag, *ACX_offd;
      HYPRE_BigInt       *col_map_offd_ACX;
      HYPRE_Int           num_cols_ACX_offd;

      /* ACX Diag */
      CX_pred ACX_pred(CF_marker);
      ACX_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        ACX_pred );

      ACX_diag_ii = hypre_TAlloc(HYPRE_Int,     ACX_diag_nnz, HYPRE_MEMORY_DEVICE);
      ACX_diag_j  = hypre_TAlloc(HYPRE_Int,     ACX_diag_nnz, HYPRE_MEMORY_DEVICE);
      ACX_diag_a  = hypre_TAlloc(HYPRE_Complex, ACX_diag_nnz, HYPRE_MEMORY_DEVICE);

      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)) + A_diag_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(ACX_diag_ii, ACX_diag_j, ACX_diag_a)),
                                   ACX_pred );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == ACX_diag_ii + ACX_diag_nnz );

      HYPRE_THRUST_CALL ( gather,
                          ACX_diag_ii,
                          ACX_diag_ii + ACX_diag_nnz,
                          map2FC,
                          ACX_diag_ii );

      ACX_diag_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACX_diag_nnz, ACX_diag_ii);
      hypre_TFree(ACX_diag_ii, HYPRE_MEMORY_DEVICE);

      /* ACX Offd */
      ACX_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        ACX_pred );

      ACX_offd_ii = hypre_TAlloc(HYPRE_Int,     ACX_offd_nnz, HYPRE_MEMORY_DEVICE);
      ACX_offd_j  = hypre_TAlloc(HYPRE_Int,     ACX_offd_nnz, HYPRE_MEMORY_DEVICE);
      ACX_offd_a  = hypre_TAlloc(HYPRE_Complex, ACX_offd_nnz, HYPRE_MEMORY_DEVICE);

      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(ACX_offd_ii, ACX_offd_j, ACX_offd_a)),
                                   ACX_pred );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == ACX_offd_ii + ACX_offd_nnz );

      HYPRE_THRUST_CALL ( gather,
                          ACX_offd_ii,
                          ACX_offd_ii + ACX_offd_nnz,
                          map2FC,
                          ACX_offd_ii );

      ACX_offd_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACX_offd_nnz, ACX_offd_ii);
      hypre_TFree(ACX_offd_ii, HYPRE_MEMORY_DEVICE);

      /* col_map_offd_ACX */
      HYPRE_Int *tmp_j = hypre_TAlloc(HYPRE_Int, hypre_max(ACX_offd_nnz, num_cols_A_offd), HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_j, ACX_offd_j, HYPRE_Int, ACX_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + ACX_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + ACX_offd_nnz );
      num_cols_ACX_offd = tmp_end - tmp_j;
      HYPRE_THRUST_CALL( fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_ACX_offd, tmp_j, 1);
      HYPRE_THRUST_CALL( exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j);
      HYPRE_THRUST_CALL( gather,
                         ACX_offd_j,
                         ACX_offd_j + ACX_offd_nnz,
                         tmp_j,
                         ACX_offd_j );
      col_map_offd_ACX = hypre_TAlloc(HYPRE_BigInt, num_cols_ACX_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *tmp_end_big = HYPRE_THRUST_CALL( copy_if,
                                                     col_map_offd_A,
                                                     col_map_offd_A + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_ACX,
                                                     thrust::identity<HYPRE_Int>());
      hypre_assert(tmp_end_big - col_map_offd_ACX == num_cols_ACX_offd);
      hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);

      /* ACX */
      ACX = hypre_ParCSRMatrixCreate(comm,
                                     nC_global,
                                     hypre_ParCSRMatrixGlobalNumCols(A),
                                     cpts_starts,
                                     hypre_ParCSRMatrixColStarts(A),
                                     num_cols_ACX_offd,
                                     ACX_diag_nnz,
                                     ACX_offd_nnz);

      ACX_diag = hypre_ParCSRMatrixDiag(ACX);
      hypre_CSRMatrixData(ACX_diag) = ACX_diag_a;
      hypre_CSRMatrixI(ACX_diag)    = ACX_diag_i;
      hypre_CSRMatrixJ(ACX_diag)    = ACX_diag_j;

      ACX_offd = hypre_ParCSRMatrixOffd(ACX);
      hypre_CSRMatrixData(ACX_offd) = ACX_offd_a;
      hypre_CSRMatrixI(ACX_offd)    = ACX_offd_i;
      hypre_CSRMatrixJ(ACX_offd)    = ACX_offd_j;

      hypre_CSRMatrixMemoryLocation(ACX_diag) = HYPRE_MEMORY_DEVICE;
      hypre_CSRMatrixMemoryLocation(ACX_offd) = HYPRE_MEMORY_DEVICE;

      hypre_ParCSRMatrixDeviceColMapOffd(ACX) = col_map_offd_ACX;
      hypre_ParCSRMatrixColMapOffd(ACX) = hypre_TAlloc(HYPRE_BigInt, num_cols_ACX_offd, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(ACX), col_map_offd_ACX, HYPRE_BigInt, num_cols_ACX_offd,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

      hypre_ParCSRMatrixSetNumNonzeros(ACX);
      hypre_ParCSRMatrixDNumNonzeros(ACX) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(ACX);
      hypre_MatvecCommPkgCreate(ACX);

      *ACX_ptr = ACX;
   }

   if (AXC_ptr)
   {
      HYPRE_Int           AXC_diag_nnz, AXC_offd_nnz;
      HYPRE_Int          *AXC_diag_ii, *AXC_diag_i, *AXC_diag_j;
      HYPRE_Complex      *AXC_diag_a;
      HYPRE_Int          *AXC_offd_ii, *AXC_offd_i, *AXC_offd_j;
      HYPRE_Complex      *AXC_offd_a;
      hypre_ParCSRMatrix *AXC;
      hypre_CSRMatrix    *AXC_diag, *AXC_offd;
      HYPRE_BigInt       *col_map_offd_AXC;
      HYPRE_Int           num_cols_AXC_offd;

      /* AXC Diag */
      XC_pred<HYPRE_Int> AXC_pred_diag(CF_marker);
      AXC_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        AXC_pred_diag );

      AXC_diag_ii = hypre_TAlloc(HYPRE_Int,     AXC_diag_nnz, HYPRE_MEMORY_DEVICE);
      AXC_diag_j  = hypre_TAlloc(HYPRE_Int,     AXC_diag_nnz, HYPRE_MEMORY_DEVICE);
      AXC_diag_a  = hypre_TAlloc(HYPRE_Complex, AXC_diag_nnz, HYPRE_MEMORY_DEVICE);

      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)) + A_diag_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(AXC_diag_ii, AXC_diag_j, AXC_diag_a)),
                                   AXC_pred_diag );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AXC_diag_ii + AXC_diag_nnz );

      HYPRE_THRUST_CALL ( gather,
                          AXC_diag_j,
                          AXC_diag_j + AXC_diag_nnz,
                          map2FC,
                          AXC_diag_j );

      AXC_diag_i = hypreDevice_CsrRowIndicesToPtrs(n_local, AXC_diag_nnz, AXC_diag_ii);
      hypre_TFree(AXC_diag_ii, HYPRE_MEMORY_DEVICE);

      /* AXC Offd */
      XC_pred<HYPRE_BigInt> AXC_pred_offd(recv_buf);
      AXC_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        AXC_pred_offd );

      AXC_offd_ii = hypre_TAlloc(HYPRE_Int,     AXC_offd_nnz, HYPRE_MEMORY_DEVICE);
      AXC_offd_j  = hypre_TAlloc(HYPRE_Int,     AXC_offd_nnz, HYPRE_MEMORY_DEVICE);
      AXC_offd_a  = hypre_TAlloc(HYPRE_Complex, AXC_offd_nnz, HYPRE_MEMORY_DEVICE);

      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(AXC_offd_ii, AXC_offd_j, AXC_offd_a)),
                                   AXC_pred_offd );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AXC_offd_ii + AXC_offd_nnz );

      AXC_offd_i = hypreDevice_CsrRowIndicesToPtrs(n_local, AXC_offd_nnz, AXC_offd_ii);
      hypre_TFree(AXC_offd_ii, HYPRE_MEMORY_DEVICE);

      /* col_map_offd_AXC */
      HYPRE_Int *tmp_j = hypre_TAlloc(HYPRE_Int, hypre_max(AXC_offd_nnz, num_cols_A_offd), HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_j, AXC_offd_j, HYPRE_Int, AXC_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + AXC_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + AXC_offd_nnz );
      num_cols_AXC_offd = tmp_end - tmp_j;
      HYPRE_THRUST_CALL( fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_AXC_offd, tmp_j, 1);
      HYPRE_THRUST_CALL( exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j);
      HYPRE_THRUST_CALL( gather,
                         AXC_offd_j,
                         AXC_offd_j + AXC_offd_nnz,
                         tmp_j,
                         AXC_offd_j );
      col_map_offd_AXC = hypre_TAlloc(HYPRE_BigInt, num_cols_AXC_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *tmp_end_big = HYPRE_THRUST_CALL( copy_if,
                                                     recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_AXC,
                                                     thrust::identity<HYPRE_Int>());
      hypre_assert(tmp_end_big - col_map_offd_AXC == num_cols_AXC_offd);
      hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);

      /* AXC */
      AXC = hypre_ParCSRMatrixCreate(comm,
                                     hypre_ParCSRMatrixGlobalNumRows(A),
                                     nC_global,
                                     row_starts,
                                     cpts_starts,
                                     num_cols_AXC_offd,
                                     AXC_diag_nnz,
                                     AXC_offd_nnz);

      AXC_diag = hypre_ParCSRMatrixDiag(AXC);
      hypre_CSRMatrixData(AXC_diag) = AXC_diag_a;
      hypre_CSRMatrixI(AXC_diag)    = AXC_diag_i;
      hypre_CSRMatrixJ(AXC_diag)    = AXC_diag_j;

      AXC_offd = hypre_ParCSRMatrixOffd(AXC);
      hypre_CSRMatrixData(AXC_offd) = AXC_offd_a;
      hypre_CSRMatrixI(AXC_offd)    = AXC_offd_i;
      hypre_CSRMatrixJ(AXC_offd)    = AXC_offd_j;

      hypre_CSRMatrixMemoryLocation(AXC_diag) = HYPRE_MEMORY_DEVICE;
      hypre_CSRMatrixMemoryLocation(AXC_offd) = HYPRE_MEMORY_DEVICE;

      hypre_ParCSRMatrixDeviceColMapOffd(AXC) = col_map_offd_AXC;
      hypre_ParCSRMatrixColMapOffd(AXC) = hypre_TAlloc(HYPRE_BigInt, num_cols_AXC_offd, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(AXC), col_map_offd_AXC, HYPRE_BigInt, num_cols_AXC_offd,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

      hypre_ParCSRMatrixSetNumNonzeros(AXC);
      hypre_ParCSRMatrixDNumNonzeros(AXC) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(AXC);
      hypre_MatvecCommPkgCreate(AXC);

      *AXC_ptr = AXC;
   }

   hypre_TFree(A_diag_ii, HYPRE_MEMORY_DEVICE);
   hypre_TFree(A_offd_ii, HYPRE_MEMORY_DEVICE);
   hypre_TFree(offd_mark, HYPRE_MEMORY_DEVICE);
   hypre_TFree(CF_marker, HYPRE_MEMORY_DEVICE);
   hypre_TFree(map2FC,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(recv_buf,  HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

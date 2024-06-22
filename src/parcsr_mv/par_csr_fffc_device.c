/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

#if defined(HYPRE_USING_SYCL)
namespace thrust = std;
#endif

typedef thrust::tuple<HYPRE_Int, HYPRE_Int> Tuple;

/* transform from local F/C index to global F/C index,
 * where F index "x" are saved as "-x-1"
 */
#if defined(HYPRE_USING_SYCL)
struct FFFC_functor
#else
struct FFFC_functor : public thrust::unary_function<Tuple, HYPRE_BigInt>
#endif
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
      const HYPRE_Int m = 1 - 2 * s;
      return m * (local_idx + CF_first[s] + s);
   }
};

/* this predicate selects A^s_{FF} */
template<typename T>
#if defined(HYPRE_USING_SYCL)
struct FF_pred
#else
struct FF_pred : public thrust::unary_function<Tuple, bool>
#endif
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
         return row_CF_marker[i] <   0 && (j == -2 || (j >= 0 && col_CF_marker[j] < 0));
      }
      else
      {
         /* A_{F2, F} */
         return row_CF_marker[i] == -2 && (j == -2 || (j >= 0 && col_CF_marker[j] < 0));
      }
   }
};

/* this predicate selects A^s_{FC} */
template<typename T>
#if defined(HYPRE_USING_SYCL)
struct FC_pred
#else
struct FC_pred : public thrust::unary_function<Tuple, bool>
#endif
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
#if defined(HYPRE_USING_SYCL)
struct CF_pred
#else
struct CF_pred : public thrust::unary_function<Tuple, bool>
#endif
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
#if defined(HYPRE_USING_SYCL)
struct CC_pred
#else
struct CC_pred : public thrust::unary_function<Tuple, bool>
#endif
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

      return row_CF_marker[i] >= 0 && (j == -2 || (j >= 0 && col_CF_marker[j] >= 0));
   }
};

/* this predicate selects A^s_{C,:} */
#if defined(HYPRE_USING_SYCL)
struct CX_pred
#else
struct CX_pred : public thrust::unary_function<Tuple, bool>
#endif
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
#if defined(HYPRE_USING_SYCL)
struct XC_pred
#else
struct XC_pred : public thrust::unary_function<Tuple, bool>
#endif
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
                                           HYPRE_Int           *CF_marker,
                                           HYPRE_BigInt        *cpts_starts,
                                           hypre_ParCSRMatrix  *S,
                                           hypre_ParCSRMatrix **AFC_ptr,
                                           hypre_ParCSRMatrix **AFF_ptr,
                                           hypre_ParCSRMatrix **ACF_ptr,
                                           hypre_ParCSRMatrix **ACC_ptr,
                                           HYPRE_Int            option )
{
   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }
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
   HYPRE_BigInt        fpts_starts[2], *row_starts, f2pts_starts[2];
   HYPRE_BigInt        nF_global, nC_global, nF2_global = 0;
   HYPRE_BigInt        F_first, C_first;
   /* work arrays */
   HYPRE_Int          *map2FC, *map2F2 = NULL, *itmp, *A_diag_ii, *A_offd_ii, *offd_mark;
   HYPRE_BigInt       *send_buf, *recv_buf;

   hypre_GpuProfilingPushRange("ParCSRMatrixGenerateFFFC");

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   n_local    = hypre_ParCSRMatrixNumRows(A);
   row_starts = hypre_ParCSRMatrixRowStarts(A);

   if (my_id == (num_procs - 1))
   {
      nC_global = cpts_starts[1];
   }
   hypre_MPI_Bcast(&nC_global, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   nC_local = (HYPRE_Int) (cpts_starts[1] - cpts_starts[0]);
   fpts_starts[0] = row_starts[0] - cpts_starts[0];
   fpts_starts[1] = row_starts[1] - cpts_starts[1];
   F_first = fpts_starts[0];
   C_first = cpts_starts[0];
   nF_local = n_local - nC_local;
   nF_global = hypre_ParCSRMatrixGlobalNumRows(A) - nC_global;

   map2FC     = hypre_TAlloc(HYPRE_Int,    n_local,         HYPRE_MEMORY_DEVICE);
   itmp       = hypre_TAlloc(HYPRE_Int,    n_local,         HYPRE_MEMORY_DEVICE);
   recv_buf   = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);

   if (option == 2)
   {
#if defined(HYPRE_USING_SYCL)
      nF2_local = HYPRE_ONEDPL_CALL( std::count,
                                     CF_marker,
                                     CF_marker + n_local,
                                     -2 );
#else
      nF2_local = HYPRE_THRUST_CALL( count,
                                     CF_marker,
                                     CF_marker + n_local,
                                     -2 );
#endif

      HYPRE_BigInt nF2_local_big = nF2_local;

      hypre_MPI_Scan(&nF2_local_big, f2pts_starts + 1, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
      f2pts_starts[0] = f2pts_starts[1] - nF2_local_big;
      if (my_id == (num_procs - 1))
      {
         nF2_global = f2pts_starts[1];
      }
      hypre_MPI_Bcast(&nF2_global, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }

   /* map from all points (i.e, F+C) to F/C indices */
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,           is_negative<HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + n_local, is_negative<HYPRE_Int>()),
                      map2FC, /* F */
                      HYPRE_Int(0) );

   HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,           is_nonnegative<HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + n_local, is_nonnegative<HYPRE_Int>()),
                      itmp, /* C */
                      HYPRE_Int(0) );

   hypreSycl_scatter_if( itmp,
                         itmp + n_local,
                         oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                         CF_marker,
                         map2FC,
                         is_nonnegative<HYPRE_Int>() ); /* FC combined */
#else
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
#endif

   hypre_TFree(itmp, HYPRE_MEMORY_DEVICE);

   if (option == 2)
   {
      map2F2 = hypre_TAlloc(HYPRE_Int, n_local, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         oneapi::dpl::make_transform_iterator(CF_marker,           equal<HYPRE_Int>(-2)),
                         oneapi::dpl::make_transform_iterator(CF_marker + n_local, equal<HYPRE_Int>(-2)),
                         map2F2, /* F2 */
                         HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */
#else
      HYPRE_THRUST_CALL( exclusive_scan,
                         thrust::make_transform_iterator(CF_marker,           equal<HYPRE_Int>(-2)),
                         thrust::make_transform_iterator(CF_marker + n_local, equal<HYPRE_Int>(-2)),
                         map2F2, /* F2 */
                         HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */
#endif
   }

   /* send_buf: global F/C indices. Note F-pts "x" are saved as "-x-1" */
   send_buf = hypre_TAlloc(HYPRE_BigInt, num_elem_send, HYPRE_MEMORY_DEVICE);

   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   FFFC_functor functor(F_first, C_first);
#if defined(HYPRE_USING_SYCL)
   hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                     oneapi::dpl::make_transform_iterator(
                        oneapi::dpl::make_zip_iterator(map2FC, CF_marker), functor),
                     send_buf );
#else
   HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                      thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(map2FC, CF_marker)),
                                                      functor),
                      send_buf );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }
#endif

   comm_handle = hypre_ParCSRCommHandleCreate_v2(21, comm_pkg, HYPRE_MEMORY_DEVICE, send_buf,
                                                 HYPRE_MEMORY_DEVICE, recv_buf);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   hypre_TFree(send_buf, HYPRE_MEMORY_DEVICE);

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
#if defined(HYPRE_USING_SYCL)
      AFF_diag_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j) + A_diag_nnz,
                                        AFF_pred_diag );
#else
      AFF_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        AFF_pred_diag );
#endif

      AFF_diag_ii = hypre_TAlloc(HYPRE_Int,     AFF_diag_nnz, HYPRE_MEMORY_DEVICE);
      AFF_diag_j  = hypre_TAlloc(HYPRE_Int,     AFF_diag_nnz, HYPRE_MEMORY_DEVICE);
      AFF_diag_a  = hypre_TAlloc(HYPRE_Complex, AFF_diag_nnz, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a) + A_diag_nnz,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(AFF_diag_ii, AFF_diag_j, AFF_diag_a),
                                        AFF_pred_diag );

      hypre_assert( std::get<0>(new_end.base()) == AFF_diag_ii + AFF_diag_nnz );

      hypreSycl_gather( AFF_diag_j,
                        AFF_diag_j + AFF_diag_nnz,
                        map2FC,
                        AFF_diag_j );

      hypreSycl_gather( AFF_diag_ii,
                        AFF_diag_ii + AFF_diag_nnz,
                        option == 1 ? map2FC : map2F2,
                        AFF_diag_ii );

#else
      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
      auto new_end = HYPRE_THRUST_CALL( copy_if,
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
#endif

      AFF_diag_i = hypreDevice_CsrRowIndicesToPtrs(option == 1 ? nF_local : nF2_local, AFF_diag_nnz,
                                                   AFF_diag_ii);
      hypre_TFree(AFF_diag_ii, HYPRE_MEMORY_DEVICE);

      /* AFF Offd */
      FF_pred<HYPRE_BigInt> AFF_pred_offd(option, CF_marker, recv_buf);
#if defined(HYPRE_USING_SYCL)
      AFF_offd_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j) + A_offd_nnz,
                                        AFF_pred_offd );
#else
      AFF_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        AFF_pred_offd );
#endif

      AFF_offd_ii = hypre_TAlloc(HYPRE_Int,     AFF_offd_nnz, HYPRE_MEMORY_DEVICE);
      AFF_offd_j  = hypre_TAlloc(HYPRE_Int,     AFF_offd_nnz, HYPRE_MEMORY_DEVICE);
      AFF_offd_a  = hypre_TAlloc(HYPRE_Complex, AFF_offd_nnz, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a),
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a) + A_offd_nnz,
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                   oneapi::dpl::make_zip_iterator(AFF_offd_ii, AFF_offd_j, AFF_offd_a),
                                   AFF_pred_offd );

      hypre_assert( std::get<0>(new_end.base()) == AFF_offd_ii + AFF_offd_nnz );

      hypreSycl_gather( AFF_offd_ii,
                        AFF_offd_ii + AFF_offd_nnz,
                        option == 1 ? map2FC : map2F2,
                        AFF_offd_ii );
#else
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
#endif

      AFF_offd_i = hypreDevice_CsrRowIndicesToPtrs(option == 1 ? nF_local : nF2_local, AFF_offd_nnz,
                                                   AFF_offd_ii);
      hypre_TFree(AFF_offd_ii, HYPRE_MEMORY_DEVICE);

      /* col_map_offd_AFF */
      HYPRE_Int *tmp_j = hypre_TAlloc(HYPRE_Int, hypre_max(AFF_offd_nnz, num_cols_A_offd),
                                      HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_j, AFF_offd_j, HYPRE_Int, AFF_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::sort,
                         tmp_j,
                         tmp_j + AFF_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_ONEDPL_CALL( std::unique,
                                              tmp_j,
                                              tmp_j + AFF_offd_nnz );
      num_cols_AFF_offd = tmp_end - tmp_j;
      HYPRE_ONEDPL_CALL( std::fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_AFF_offd, tmp_j, (HYPRE_Int) 1);
      HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j,
                         0 );
      hypreSycl_gather( AFF_offd_j,
                        AFF_offd_j + AFF_offd_nnz,
                        tmp_j,
                        AFF_offd_j );
      col_map_offd_AFF = hypre_TAlloc(HYPRE_BigInt, num_cols_AFF_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *tmp_end_big = hypreSycl_copy_if( recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_AFF,
      [] (const auto & x) {return x;} );
      HYPRE_ONEDPL_CALL( std::transform,
                         col_map_offd_AFF,
                         col_map_offd_AFF + num_cols_AFF_offd,
                         col_map_offd_AFF,
      [] (auto const & x) { return -x - 1; } );
      hypre_assert(tmp_end_big - col_map_offd_AFF == num_cols_AFF_offd);
#else
      HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + AFF_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + AFF_offd_nnz );
      num_cols_AFF_offd = tmp_end - tmp_j;
      hypreDevice_IntFilln( offd_mark, num_cols_A_offd, 0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_AFF_offd, tmp_j, (HYPRE_Int) 1);
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
                                                     thrust::make_transform_iterator(recv_buf, -_1 - 1),
                                                     thrust::make_transform_iterator(recv_buf, -_1 - 1) + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_AFF,
                                                     thrust::identity<HYPRE_Int>() );
      hypre_assert(tmp_end_big - col_map_offd_AFF == num_cols_AFF_offd);
#endif
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
      hypre_ParCSRMatrixColMapOffd(AFF) = hypre_TAlloc(HYPRE_BigInt, num_cols_AFF_offd,
                                                       HYPRE_MEMORY_HOST);
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
#if defined(HYPRE_USING_SYCL)
      AFC_diag_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j) + A_diag_nnz,
                                        AFC_pred_diag );
#else
      AFC_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        AFC_pred_diag );
#endif

      AFC_diag_ii = hypre_TAlloc(HYPRE_Int,     AFC_diag_nnz, HYPRE_MEMORY_DEVICE);
      AFC_diag_j  = hypre_TAlloc(HYPRE_Int,     AFC_diag_nnz, HYPRE_MEMORY_DEVICE);
      AFC_diag_a  = hypre_TAlloc(HYPRE_Complex, AFC_diag_nnz, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j, A_diag_a),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j, A_diag_a) + A_diag_nnz,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(AFC_diag_ii, AFC_diag_j, AFC_diag_a),
                                        AFC_pred_diag );

      hypre_assert( std::get<0>(new_end.base()) == AFC_diag_ii + AFC_diag_nnz );

      hypreSycl_gather( AFC_diag_j,
                        AFC_diag_j + AFC_diag_nnz,
                        map2FC,
                        AFC_diag_j );

      hypreSycl_gather( AFC_diag_ii,
                        AFC_diag_ii + AFC_diag_nnz,
                        map2FC,
                        AFC_diag_ii );
#else
      auto new_end = HYPRE_THRUST_CALL( copy_if,
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
#endif

      AFC_diag_i = hypreDevice_CsrRowIndicesToPtrs(nF_local, AFC_diag_nnz, AFC_diag_ii);
      hypre_TFree(AFC_diag_ii, HYPRE_MEMORY_DEVICE);

      /* AFC Offd */
      FC_pred<HYPRE_BigInt> AFC_pred_offd(CF_marker, recv_buf);
#if defined(HYPRE_USING_SYCL)
      AFC_offd_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j) + A_offd_nnz,
                                        AFC_pred_offd );
#else
      AFC_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        AFC_pred_offd );
#endif

      AFC_offd_ii = hypre_TAlloc(HYPRE_Int,     AFC_offd_nnz, HYPRE_MEMORY_DEVICE);
      AFC_offd_j  = hypre_TAlloc(HYPRE_Int,     AFC_offd_nnz, HYPRE_MEMORY_DEVICE);
      AFC_offd_a  = hypre_TAlloc(HYPRE_Complex, AFC_offd_nnz, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a),
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a) + A_offd_nnz,
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                   oneapi::dpl::make_zip_iterator(AFC_offd_ii, AFC_offd_j, AFC_offd_a),
                                   AFC_pred_offd );

      hypre_assert( std::get<0>(new_end.base()) == AFC_offd_ii + AFC_offd_nnz );

      hypreSycl_gather( AFC_offd_ii,
                        AFC_offd_ii + AFC_offd_nnz,
                        map2FC,
                        AFC_offd_ii );
#else
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
#endif

      AFC_offd_i = hypreDevice_CsrRowIndicesToPtrs(nF_local, AFC_offd_nnz, AFC_offd_ii);
      hypre_TFree(AFC_offd_ii, HYPRE_MEMORY_DEVICE);

      /* col_map_offd_AFC */
      HYPRE_Int *tmp_j = hypre_TAlloc(HYPRE_Int, hypre_max(AFC_offd_nnz, num_cols_A_offd),
                                      HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_j, AFC_offd_j, HYPRE_Int, AFC_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::sort,
                         tmp_j,
                         tmp_j + AFC_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_ONEDPL_CALL( std::unique,
                                              tmp_j,
                                              tmp_j + AFC_offd_nnz );
      num_cols_AFC_offd = tmp_end - tmp_j;
      HYPRE_ONEDPL_CALL( std::fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_AFC_offd, tmp_j, (HYPRE_Int) 1);
      HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j,
                         0);
      hypreSycl_gather( AFC_offd_j,
                        AFC_offd_j + AFC_offd_nnz,
                        tmp_j,
                        AFC_offd_j );
      col_map_offd_AFC = hypre_TAlloc(HYPRE_BigInt, num_cols_AFC_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *tmp_end_big = hypreSycl_copy_if( recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_AFC,
      [] (const auto & x) {return x;});
#else
      HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + AFC_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + AFC_offd_nnz );
      num_cols_AFC_offd = tmp_end - tmp_j;
      hypreDevice_IntFilln( offd_mark, num_cols_A_offd, 0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_AFC_offd, tmp_j, (HYPRE_Int) 1);
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
#endif
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
      hypre_ParCSRMatrixColMapOffd(AFC) = hypre_TAlloc(HYPRE_BigInt, num_cols_AFC_offd,
                                                       HYPRE_MEMORY_HOST);
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
#if defined(HYPRE_USING_SYCL)
      ACF_diag_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j) + A_diag_nnz,
                                        ACF_pred_diag );
#else
      ACF_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        ACF_pred_diag );
#endif

      ACF_diag_ii = hypre_TAlloc(HYPRE_Int,     ACF_diag_nnz, HYPRE_MEMORY_DEVICE);
      ACF_diag_j  = hypre_TAlloc(HYPRE_Int,     ACF_diag_nnz, HYPRE_MEMORY_DEVICE);
      ACF_diag_a  = hypre_TAlloc(HYPRE_Complex, ACF_diag_nnz, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j, A_diag_a),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j, A_diag_a) + A_diag_nnz,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(ACF_diag_ii, ACF_diag_j, ACF_diag_a),
                                        ACF_pred_diag );

      hypre_assert( std::get<0>(new_end.base()) == ACF_diag_ii + ACF_diag_nnz );

      hypreSycl_gather( ACF_diag_j,
                        ACF_diag_j + ACF_diag_nnz,
                        map2FC,
                        ACF_diag_j );

      hypreSycl_gather( ACF_diag_ii,
                        ACF_diag_ii + ACF_diag_nnz,
                        map2FC,
                        ACF_diag_ii );
#else
      auto new_end = HYPRE_THRUST_CALL( copy_if,
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
#endif

      ACF_diag_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACF_diag_nnz, ACF_diag_ii);
      hypre_TFree(ACF_diag_ii, HYPRE_MEMORY_DEVICE);

      /* ACF Offd */
      CF_pred<HYPRE_BigInt> ACF_pred_offd(CF_marker, recv_buf);
#if defined(HYPRE_USING_SYCL)
      ACF_offd_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j) + A_offd_nnz,
                                        ACF_pred_offd );
#else
      ACF_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        ACF_pred_offd );
#endif

      ACF_offd_ii = hypre_TAlloc(HYPRE_Int,     ACF_offd_nnz, HYPRE_MEMORY_DEVICE);
      ACF_offd_j  = hypre_TAlloc(HYPRE_Int,     ACF_offd_nnz, HYPRE_MEMORY_DEVICE);
      ACF_offd_a  = hypre_TAlloc(HYPRE_Complex, ACF_offd_nnz, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a),
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a) + A_offd_nnz,
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                   oneapi::dpl::make_zip_iterator(ACF_offd_ii, ACF_offd_j, ACF_offd_a),
                                   ACF_pred_offd );

      hypre_assert( std::get<0>(new_end.base()) == ACF_offd_ii + ACF_offd_nnz );

      hypreSycl_gather( ACF_offd_ii,
                        ACF_offd_ii + ACF_offd_nnz,
                        map2FC,
                        ACF_offd_ii );
#else
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
#endif

      ACF_offd_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACF_offd_nnz, ACF_offd_ii);
      hypre_TFree(ACF_offd_ii, HYPRE_MEMORY_DEVICE);

      /* col_map_offd_ACF */
      HYPRE_Int *tmp_j = hypre_TAlloc(HYPRE_Int, hypre_max(ACF_offd_nnz, num_cols_A_offd),
                                      HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_j, ACF_offd_j, HYPRE_Int, ACF_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::sort,
                         tmp_j,
                         tmp_j + ACF_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_ONEDPL_CALL( std::unique,
                                              tmp_j,
                                              tmp_j + ACF_offd_nnz );
      num_cols_ACF_offd = tmp_end - tmp_j;
      HYPRE_ONEDPL_CALL( std::fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_ACF_offd, tmp_j, (HYPRE_Int) 1);
      HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j,
                         0);
      hypreSycl_gather( ACF_offd_j,
                        ACF_offd_j + ACF_offd_nnz,
                        tmp_j,
                        ACF_offd_j );
      col_map_offd_ACF = hypre_TAlloc(HYPRE_BigInt, num_cols_ACF_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *tmp_end_big = hypreSycl_copy_if( recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_ACF,
      [] (const auto & x) {return x;} );
      HYPRE_ONEDPL_CALL( std::transform,
                         col_map_offd_ACF,
                         col_map_offd_ACF + num_cols_ACF_offd,
                         col_map_offd_ACF,
      [] (const auto & x) {return -x - 1;} );
#else
      HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + ACF_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + ACF_offd_nnz );
      num_cols_ACF_offd = tmp_end - tmp_j;
      hypreDevice_IntFilln( offd_mark, num_cols_A_offd, 0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_ACF_offd, tmp_j, (HYPRE_Int) 1);
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
                                                     thrust::make_transform_iterator(recv_buf, -_1 - 1),
                                                     thrust::make_transform_iterator(recv_buf, -_1 - 1) + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_ACF,
                                                     thrust::identity<HYPRE_Int>());
#endif
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
      hypre_ParCSRMatrixColMapOffd(ACF) = hypre_TAlloc(HYPRE_BigInt, num_cols_ACF_offd,
                                                       HYPRE_MEMORY_HOST);
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
#if defined(HYPRE_USING_SYCL)
      ACC_diag_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j) + A_diag_nnz,
                                        ACC_pred_diag );
#else
      ACC_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        ACC_pred_diag );
#endif

      ACC_diag_ii = hypre_TAlloc(HYPRE_Int,     ACC_diag_nnz, HYPRE_MEMORY_DEVICE);
      ACC_diag_j  = hypre_TAlloc(HYPRE_Int,     ACC_diag_nnz, HYPRE_MEMORY_DEVICE);
      ACC_diag_a  = hypre_TAlloc(HYPRE_Complex, ACC_diag_nnz, HYPRE_MEMORY_DEVICE);

      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
#if defined(HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a) + A_diag_nnz,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(ACC_diag_ii, ACC_diag_j, ACC_diag_a),
                                        ACC_pred_diag );

      hypre_assert( std::get<0>(new_end.base()) == ACC_diag_ii + ACC_diag_nnz );

      hypreSycl_gather( ACC_diag_j,
                        ACC_diag_j + ACC_diag_nnz,
                        map2FC,
                        ACC_diag_j );

      hypreSycl_gather( ACC_diag_ii,
                        ACC_diag_ii + ACC_diag_nnz,
                        map2FC,
                        ACC_diag_ii );
#else
      auto new_end = HYPRE_THRUST_CALL( copy_if,
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
#endif

      ACC_diag_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACC_diag_nnz, ACC_diag_ii);
      hypre_TFree(ACC_diag_ii, HYPRE_MEMORY_DEVICE);

      /* ACC Offd */
      CC_pred<HYPRE_BigInt> ACC_pred_offd(CF_marker, recv_buf);
#if defined(HYPRE_USING_SYCL)
      ACC_offd_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j) + A_offd_nnz,
                                        ACC_pred_offd );
#else
      ACC_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        ACC_pred_offd );
#endif

      ACC_offd_ii = hypre_TAlloc(HYPRE_Int,     ACC_offd_nnz, HYPRE_MEMORY_DEVICE);
      ACC_offd_j  = hypre_TAlloc(HYPRE_Int,     ACC_offd_nnz, HYPRE_MEMORY_DEVICE);
      ACC_offd_a  = hypre_TAlloc(HYPRE_Complex, ACC_offd_nnz, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a),
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a) + A_offd_nnz,
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                   oneapi::dpl::make_zip_iterator(ACC_offd_ii, ACC_offd_j, ACC_offd_a),
                                   ACC_pred_offd );

      hypre_assert( std::get<0>(new_end.base()) == ACC_offd_ii + ACC_offd_nnz );

      hypreSycl_gather( ACC_offd_ii,
                        ACC_offd_ii + ACC_offd_nnz,
                        map2FC,
                        ACC_offd_ii );
#else
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
#endif

      ACC_offd_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACC_offd_nnz, ACC_offd_ii);
      hypre_TFree(ACC_offd_ii, HYPRE_MEMORY_DEVICE);

      /* col_map_offd_ACC */
      HYPRE_Int *tmp_j = hypre_TAlloc(HYPRE_Int, hypre_max(ACC_offd_nnz, num_cols_A_offd),
                                      HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_j, ACC_offd_j, HYPRE_Int, ACC_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::sort,
                         tmp_j,
                         tmp_j + ACC_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_ONEDPL_CALL( std::unique,
                                              tmp_j,
                                              tmp_j + ACC_offd_nnz );
      num_cols_ACC_offd = tmp_end - tmp_j;
      HYPRE_ONEDPL_CALL( std::fill_n,
                         offd_mark,
                         num_cols_A_offd,
                         0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_ACC_offd, tmp_j, (HYPRE_Int) 1);
      HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j,
                         0);
      hypreSycl_gather( ACC_offd_j,
                        ACC_offd_j + ACC_offd_nnz,
                        tmp_j,
                        ACC_offd_j );
      col_map_offd_ACC = hypre_TAlloc(HYPRE_BigInt, num_cols_ACC_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *tmp_end_big = hypreSycl_copy_if( recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_ACC,
      [] (const auto & x) {return x;} );
#else
      HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + ACC_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + ACC_offd_nnz );
      num_cols_ACC_offd = tmp_end - tmp_j;
      hypreDevice_IntFilln( offd_mark, num_cols_A_offd, 0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_ACC_offd, tmp_j, (HYPRE_Int) 1);
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
#endif
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
      hypre_ParCSRMatrixColMapOffd(ACC) = hypre_TAlloc(HYPRE_BigInt, num_cols_ACC_offd,
                                                       HYPRE_MEMORY_HOST);
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
   hypre_TFree(map2FC,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(map2F2,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(recv_buf,  HYPRE_MEMORY_DEVICE);

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGenerateFFFCDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixGenerateFFFCDevice( hypre_ParCSRMatrix  *A,
                                      HYPRE_Int           *CF_marker,
                                      HYPRE_BigInt        *cpts_starts,
                                      hypre_ParCSRMatrix  *S,
                                      hypre_ParCSRMatrix **AFC_ptr,
                                      hypre_ParCSRMatrix **AFF_ptr )
{
   return hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker, cpts_starts, S,
                                                    AFC_ptr, AFF_ptr,
                                                    NULL, NULL, 1);
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGenerateFFFC3Device
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixGenerateFFFC3Device( hypre_ParCSRMatrix  *A,
                                       HYPRE_Int           *CF_marker,
                                       HYPRE_BigInt        *cpts_starts,
                                       hypre_ParCSRMatrix  *S,
                                       hypre_ParCSRMatrix **AFC_ptr,
                                       hypre_ParCSRMatrix **AFF_ptr)
{
   return hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker, cpts_starts, S,
                                                    AFC_ptr, AFF_ptr,
                                                    NULL, NULL, 2);
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGenerateFFCFDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixGenerateFFCFDevice( hypre_ParCSRMatrix  *A,
                                      HYPRE_Int           *CF_marker,
                                      HYPRE_BigInt        *cpts_starts,
                                      hypre_ParCSRMatrix  *S,
                                      hypre_ParCSRMatrix **ACF_ptr,
                                      hypre_ParCSRMatrix **AFF_ptr )
{
   return hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker, cpts_starts, S,
                                                    NULL, AFF_ptr,
                                                    ACF_ptr, NULL, 1);
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGenerateCFDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixGenerateCFDevice( hypre_ParCSRMatrix  *A,
                                    HYPRE_Int           *CF_marker,
                                    HYPRE_BigInt        *cpts_starts,
                                    hypre_ParCSRMatrix  *S,
                                    hypre_ParCSRMatrix **ACF_ptr)
{
   return hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker, cpts_starts, S,
                                                    NULL, NULL,
                                                    ACF_ptr, NULL, 1);
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGenerateCCDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixGenerateCCDevice( hypre_ParCSRMatrix  *A,
                                    HYPRE_Int           *CF_marker,
                                    HYPRE_BigInt        *cpts_starts,
                                    hypre_ParCSRMatrix  *S,
                                    hypre_ParCSRMatrix **ACC_ptr)
{
   return hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker, cpts_starts, S,
                                                    NULL, NULL,
                                                    NULL, ACC_ptr, 1);
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGenerateCCCFDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixGenerateCCCFDevice( hypre_ParCSRMatrix  *A,
                                      HYPRE_Int           *CF_marker,
                                      HYPRE_BigInt        *cpts_starts,
                                      hypre_ParCSRMatrix  *S,
                                      hypre_ParCSRMatrix **ACF_ptr,
                                      hypre_ParCSRMatrix **ACC_ptr)
{
   return hypre_ParCSRMatrixGenerateFFFCDevice_core(A, CF_marker, cpts_starts, S,
                                                    NULL, NULL,
                                                    ACF_ptr, ACC_ptr, 1);
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGenerate1DCFDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixGenerate1DCFDevice( hypre_ParCSRMatrix  *A,
                                      HYPRE_Int           *CF_marker,
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
   HYPRE_BigInt        fpts_starts[2], *row_starts;
   HYPRE_BigInt        /*nF_global,*/ nC_global;
   HYPRE_BigInt        F_first, C_first;
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

   if (my_id == (num_procs - 1))
   {
      nC_global = cpts_starts[1];
   }
   hypre_MPI_Bcast(&nC_global, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   nC_local = (HYPRE_Int) (cpts_starts[1] - cpts_starts[0]);
   fpts_starts[0] = row_starts[0] - cpts_starts[0];
   fpts_starts[1] = row_starts[1] - cpts_starts[1];
   F_first = fpts_starts[0];
   C_first = cpts_starts[0];
   /*
   nF_local = n_local - nC_local;
   nF_global = hypre_ParCSRMatrixGlobalNumRows(A) - nC_global;
   */

   map2FC     = hypre_TAlloc(HYPRE_Int,    n_local,         HYPRE_MEMORY_DEVICE);
   itmp       = hypre_TAlloc(HYPRE_Int,    n_local,         HYPRE_MEMORY_DEVICE);
   recv_buf   = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   /* map from all points (i.e, F+C) to F/C indices */
   HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,           is_negative<HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + n_local, is_negative<HYPRE_Int>()),
                      map2FC, /* F */
                      HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */

   HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,           is_nonnegative<HYPRE_Int>()),
                      oneapi::dpl::make_transform_iterator(CF_marker + n_local, is_nonnegative<HYPRE_Int>()),
                      itmp, /* C */
                      HYPRE_Int(0) ); /* *MUST* pass init value since input and output types diff. */

   hypreSycl_scatter_if( itmp,
                         itmp + n_local,
                         oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                         CF_marker,
                         map2FC, /* FC combined */
                         is_nonnegative<HYPRE_Int>() );
#else
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
#endif

   hypre_TFree(itmp, HYPRE_MEMORY_DEVICE);

   /* send_buf: global F/C indices. Note F-pts "x" are saved as "-x-1" */
   send_buf = hypre_TAlloc(HYPRE_BigInt, num_elem_send, HYPRE_MEMORY_DEVICE);

   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   FFFC_functor functor(F_first, C_first);
#if defined(HYPRE_USING_SYCL)
   auto zip = oneapi::dpl::make_zip_iterator(map2FC, CF_marker);
   hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                     hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                     oneapi::dpl::make_transform_iterator(zip, functor),
                     send_buf );
#else
   HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                      thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(map2FC, CF_marker)),
                                                      functor),
                      send_buf );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
   /* RL: make sure send_buf is ready before issuing GPU-GPU MPI */
   if (hypre_GetGpuAwareMPI())
   {
      hypre_ForceSyncComputeStream(hypre_handle());
   }
#endif

   comm_handle = hypre_ParCSRCommHandleCreate_v2(21, comm_pkg, HYPRE_MEMORY_DEVICE, send_buf,
                                                 HYPRE_MEMORY_DEVICE, recv_buf);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   hypre_TFree(send_buf, HYPRE_MEMORY_DEVICE);

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
#if defined(HYPRE_USING_SYCL)
      ACX_diag_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j) + A_diag_nnz,
                                        ACX_pred );
#else
      ACX_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        ACX_pred );
#endif

      ACX_diag_ii = hypre_TAlloc(HYPRE_Int,     ACX_diag_nnz, HYPRE_MEMORY_DEVICE);
      ACX_diag_j  = hypre_TAlloc(HYPRE_Int,     ACX_diag_nnz, HYPRE_MEMORY_DEVICE);
      ACX_diag_a  = hypre_TAlloc(HYPRE_Complex, ACX_diag_nnz, HYPRE_MEMORY_DEVICE);

      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
#if defined(HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a) + A_diag_nnz,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(ACX_diag_ii, ACX_diag_j, ACX_diag_a),
                                        ACX_pred );

      hypre_assert( std::get<0>(new_end.base()) == ACX_diag_ii + ACX_diag_nnz );

      hypreSycl_gather( ACX_diag_ii,
                        ACX_diag_ii + ACX_diag_nnz,
                        map2FC,
                        ACX_diag_ii );
#else
      auto new_end = HYPRE_THRUST_CALL( copy_if,
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
#endif

      ACX_diag_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACX_diag_nnz, ACX_diag_ii);
      hypre_TFree(ACX_diag_ii, HYPRE_MEMORY_DEVICE);

      /* ACX Offd */
#if defined(HYPRE_USING_SYCL)
      ACX_offd_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j) + A_offd_nnz,
                                        ACX_pred );
#else
      ACX_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        ACX_pred );
#endif

      ACX_offd_ii = hypre_TAlloc(HYPRE_Int,     ACX_offd_nnz, HYPRE_MEMORY_DEVICE);
      ACX_offd_j  = hypre_TAlloc(HYPRE_Int,     ACX_offd_nnz, HYPRE_MEMORY_DEVICE);
      ACX_offd_a  = hypre_TAlloc(HYPRE_Complex, ACX_offd_nnz, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a),
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a) + A_offd_nnz,
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                   oneapi::dpl::make_zip_iterator(ACX_offd_ii, ACX_offd_j, ACX_offd_a),
                                   ACX_pred );

      hypre_assert( std::get<0>(new_end.base()) == ACX_offd_ii + ACX_offd_nnz );

      hypreSycl_gather( ACX_offd_ii,
                        ACX_offd_ii + ACX_offd_nnz,
                        map2FC,
                        ACX_offd_ii );
#else
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
#endif

      ACX_offd_i = hypreDevice_CsrRowIndicesToPtrs(nC_local, ACX_offd_nnz, ACX_offd_ii);
      hypre_TFree(ACX_offd_ii, HYPRE_MEMORY_DEVICE);

      /* col_map_offd_ACX */
      HYPRE_Int *tmp_j = hypre_TAlloc(HYPRE_Int, hypre_max(ACX_offd_nnz, num_cols_A_offd),
                                      HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_j, ACX_offd_j, HYPRE_Int, ACX_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::sort,
                         tmp_j,
                         tmp_j + ACX_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_ONEDPL_CALL( std::unique,
                                              tmp_j,
                                              tmp_j + ACX_offd_nnz );
#else
      HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + ACX_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + ACX_offd_nnz );
#endif
      num_cols_ACX_offd = tmp_end - tmp_j;
      hypreDevice_IntFilln( offd_mark, num_cols_A_offd, 0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_ACX_offd, tmp_j, (HYPRE_Int) 1);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j,
                         0 );
      hypreSycl_gather( ACX_offd_j,
                        ACX_offd_j + ACX_offd_nnz,
                        tmp_j,
                        ACX_offd_j );
      col_map_offd_ACX = hypre_TAlloc(HYPRE_BigInt, num_cols_ACX_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *tmp_end_big = hypreSycl_copy_if( col_map_offd_A,
                                                     col_map_offd_A + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_ACX,
      [] (const auto & x) {return x;} );
#else
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
#endif
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
      hypre_ParCSRMatrixColMapOffd(ACX) = hypre_TAlloc(HYPRE_BigInt, num_cols_ACX_offd,
                                                       HYPRE_MEMORY_HOST);
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
#if defined(HYPRE_USING_SYCL)
      AXC_diag_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j) + A_diag_nnz,
                                        AXC_pred_diag );
#else
      AXC_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                        AXC_pred_diag );
#endif

      AXC_diag_ii = hypre_TAlloc(HYPRE_Int,     AXC_diag_nnz, HYPRE_MEMORY_DEVICE);
      AXC_diag_j  = hypre_TAlloc(HYPRE_Int,     AXC_diag_nnz, HYPRE_MEMORY_DEVICE);
      AXC_diag_a  = hypre_TAlloc(HYPRE_Complex, AXC_diag_nnz, HYPRE_MEMORY_DEVICE);

      /* Notice that we cannot use Soc_diag_j in the first two arguments since the diagonal is marked as -2 */
#if defined(HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a),
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a) + A_diag_nnz,
                                        oneapi::dpl::make_zip_iterator(A_diag_ii, Soc_diag_j),
                                        oneapi::dpl::make_zip_iterator(AXC_diag_ii, AXC_diag_j, AXC_diag_a),
                                        AXC_pred_diag );

      hypre_assert( std::get<0>(new_end.base()) == AXC_diag_ii + AXC_diag_nnz );

      hypreSycl_gather( AXC_diag_j,
                        AXC_diag_j + AXC_diag_nnz,
                        map2FC,
                        AXC_diag_j );
#else
      auto new_end = HYPRE_THRUST_CALL( copy_if,
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
#endif

      AXC_diag_i = hypreDevice_CsrRowIndicesToPtrs(n_local, AXC_diag_nnz, AXC_diag_ii);
      hypre_TFree(AXC_diag_ii, HYPRE_MEMORY_DEVICE);

      /* AXC Offd */
      XC_pred<HYPRE_BigInt> AXC_pred_offd(recv_buf);
#if defined(HYPRE_USING_SYCL)
      AXC_offd_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                        oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j) + A_offd_nnz,
                                        AXC_pred_offd );
#else
      AXC_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                        AXC_pred_offd );
#endif

      AXC_offd_ii = hypre_TAlloc(HYPRE_Int,     AXC_offd_nnz, HYPRE_MEMORY_DEVICE);
      AXC_offd_j  = hypre_TAlloc(HYPRE_Int,     AXC_offd_nnz, HYPRE_MEMORY_DEVICE);
      AXC_offd_a  = hypre_TAlloc(HYPRE_Complex, AXC_offd_nnz, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a),
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j, A_offd_a) + A_offd_nnz,
                                   oneapi::dpl::make_zip_iterator(A_offd_ii, Soc_offd_j),
                                   oneapi::dpl::make_zip_iterator(AXC_offd_ii, AXC_offd_j, AXC_offd_a),
                                   AXC_pred_offd );

      hypre_assert( std::get<0>(new_end.base()) == AXC_offd_ii + AXC_offd_nnz );
#else
      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(AXC_offd_ii, AXC_offd_j, AXC_offd_a)),
                                   AXC_pred_offd );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AXC_offd_ii + AXC_offd_nnz );
#endif

      AXC_offd_i = hypreDevice_CsrRowIndicesToPtrs(n_local, AXC_offd_nnz, AXC_offd_ii);
      hypre_TFree(AXC_offd_ii, HYPRE_MEMORY_DEVICE);

      /* col_map_offd_AXC */
      HYPRE_Int *tmp_j = hypre_TAlloc(HYPRE_Int, hypre_max(AXC_offd_nnz, num_cols_A_offd),
                                      HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_j, AXC_offd_j, HYPRE_Int, AXC_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::sort,
                         tmp_j,
                         tmp_j + AXC_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_ONEDPL_CALL( std::unique,
                                              tmp_j,
                                              tmp_j + AXC_offd_nnz );
#else
      HYPRE_THRUST_CALL( sort,
                         tmp_j,
                         tmp_j + AXC_offd_nnz );
      HYPRE_Int *tmp_end = HYPRE_THRUST_CALL( unique,
                                              tmp_j,
                                              tmp_j + AXC_offd_nnz );
#endif
      num_cols_AXC_offd = tmp_end - tmp_j;
      hypreDevice_IntFilln( offd_mark, num_cols_A_offd, 0 );
      hypreDevice_ScatterConstant(offd_mark, num_cols_AXC_offd, tmp_j, (HYPRE_Int) 1);
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         offd_mark,
                         offd_mark + num_cols_A_offd,
                         tmp_j,
                         0 );
      hypreSycl_gather( AXC_offd_j,
                        AXC_offd_j + AXC_offd_nnz,
                        tmp_j,
                        AXC_offd_j );
      col_map_offd_AXC = hypre_TAlloc(HYPRE_BigInt, num_cols_AXC_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *tmp_end_big = hypreSycl_copy_if( recv_buf,
                                                     recv_buf + num_cols_A_offd,
                                                     offd_mark,
                                                     col_map_offd_AXC,
      [] (const auto & x) {return x;} );
#else
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
#endif
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
      hypre_ParCSRMatrixColMapOffd(AXC) = hypre_TAlloc(HYPRE_BigInt, num_cols_AXC_offd,
                                                       HYPRE_MEMORY_HOST);
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
   hypre_TFree(map2FC,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(recv_buf,  HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_GPU)

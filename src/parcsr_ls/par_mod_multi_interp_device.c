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
template<typename T>
struct tuple_plus
{
   __host__ __device__
   std::tuple<T, T> operator()( const std::tuple<T, T> & x1, const std::tuple<T, T> & x2) const
   {
      return std::make_tuple( std::get<0>(x1) + std::get<0>(x2),
                              std::get<1>(x1) + std::get<1>(x2) );
   }
};

struct local_equal_plus_constant
{
   HYPRE_BigInt _value;

   local_equal_plus_constant(HYPRE_BigInt value) : _value(value) {}

   __host__ __device__ HYPRE_BigInt operator()(HYPRE_BigInt /*x*/, HYPRE_BigInt y) const
   { return y + _value; }
};

/* transform from local C index to global C index */
struct globalC_functor
{
   HYPRE_BigInt C_first;

   globalC_functor(HYPRE_BigInt C_first_)
   {
      C_first = C_first_;
   }

   __host__ __device__
   HYPRE_BigInt operator()(const HYPRE_Int x) const
   {
      return ( (HYPRE_BigInt) x + C_first );
   }
};
#else
template<typename T>
struct tuple_plus : public
   thrust::binary_function<thrust::tuple<T, T>, thrust::tuple<T, T>, thrust::tuple<T, T> >
{
   __host__ __device__
   thrust::tuple<T, T> operator()( const thrust::tuple<T, T> & x1, const thrust::tuple<T, T> & x2)
   {
      return thrust::make_tuple( thrust::get<0>(x1) + thrust::get<0>(x2),
                                 thrust::get<1>(x1) + thrust::get<1>(x2) );
   }
};

template<typename T>
struct tuple_minus : public
   thrust::binary_function<thrust::tuple<T, T>, thrust::tuple<T, T>, thrust::tuple<T, T> >
{
   __host__ __device__
   thrust::tuple<T, T> operator()( const thrust::tuple<T, T> & x1, const thrust::tuple<T, T> & x2)
   {
      return thrust::make_tuple( thrust::get<0>(x1) - thrust::get<0>(x2),
                                 thrust::get<1>(x1) - thrust::get<1>(x2) );
   }
};

struct local_equal_plus_constant : public
   thrust::binary_function<HYPRE_BigInt, HYPRE_BigInt, HYPRE_BigInt>
{
   HYPRE_BigInt _value;

   local_equal_plus_constant(HYPRE_BigInt value) : _value(value) {}

   __host__ __device__ HYPRE_BigInt operator()(HYPRE_BigInt /*x*/, HYPRE_BigInt y)
   { return y + _value; }
};

/* transform from local C index to global C index */
struct globalC_functor : public thrust::unary_function<HYPRE_Int, HYPRE_BigInt>
{
   HYPRE_BigInt C_first;

   globalC_functor(HYPRE_BigInt C_first_)
   {
      C_first = C_first_;
   }

   __host__ __device__
   HYPRE_BigInt operator()(const HYPRE_Int x) const
   {
      return ( (HYPRE_BigInt) x + C_first );
   }
};
#endif

void hypre_modmp_init_fine_to_coarse( HYPRE_Int n_fine, HYPRE_Int *pass_marker, HYPRE_Int color,
                                      HYPRE_Int *fine_to_coarse );

void hypre_modmp_compute_num_cols_offd_fine_to_coarse( HYPRE_Int * pass_marker_offd,
                                                       HYPRE_Int color, HYPRE_Int num_cols_offd_A, HYPRE_Int & num_cols_offd,
                                                       HYPRE_Int ** fine_to_coarse_offd );

__global__ void hypreGPUKernel_cfmarker_masked_rowsum( hypre_DeviceItem &item, HYPRE_Int nrows,
                                                       HYPRE_Int *A_diag_i,
                                                       HYPRE_Int *A_diag_j, HYPRE_Complex *A_diag_data, HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j,
                                                       HYPRE_Complex *A_offd_data, HYPRE_Int *CF_marker, HYPRE_Int *dof_func, HYPRE_Int *dof_func_offd,
                                                       HYPRE_Complex *row_sums );

__global__ void hypreGPUKernel_generate_Pdiag_i_Poffd_i( hypre_DeviceItem &item,
                                                         HYPRE_Int num_points,
                                                         HYPRE_Int color,
                                                         HYPRE_Int *pass_order, HYPRE_Int *pass_marker, HYPRE_Int *pass_marker_offd, HYPRE_Int *S_diag_i,
                                                         HYPRE_Int *S_diag_j, HYPRE_Int *S_offd_i, HYPRE_Int *S_offd_j, HYPRE_Int *P_diag_i,
                                                         HYPRE_Int *P_offd_i );

__global__ void hypreGPUKernel_generate_Pdiag_j_Poffd_j( hypre_DeviceItem &item,
                                                         HYPRE_Int num_points,
                                                         HYPRE_Int color,
                                                         HYPRE_Int *pass_order, HYPRE_Int *pass_marker, HYPRE_Int *pass_marker_offd,
                                                         HYPRE_Int *fine_to_coarse, HYPRE_Int *fine_to_coarse_offd, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                                         HYPRE_Complex *A_diag_data, HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j, HYPRE_Complex *A_offd_data,
                                                         HYPRE_Int *Soc_diag_j, HYPRE_Int *Soc_offd_j, HYPRE_Int *P_diag_i, HYPRE_Int *P_offd_i,
                                                         HYPRE_Int *P_diag_j, HYPRE_Complex *P_diag_data, HYPRE_Int *P_offd_j, HYPRE_Complex *P_offd_data,
                                                         HYPRE_Complex *row_sums );

__global__ void hypreGPUKernel_insert_remaining_weights( hypre_DeviceItem &item, HYPRE_Int start,
                                                         HYPRE_Int stop,
                                                         HYPRE_Int *pass_order, HYPRE_Int *Pi_diag_i, HYPRE_Int *Pi_diag_j, HYPRE_Real *Pi_diag_data,
                                                         HYPRE_Int *P_diag_i, HYPRE_Int *P_diag_j, HYPRE_Real *P_diag_data, HYPRE_Int *Pi_offd_i,
                                                         HYPRE_Int *Pi_offd_j, HYPRE_Real *Pi_offd_data, HYPRE_Int *P_offd_i, HYPRE_Int *P_offd_j,
                                                         HYPRE_Real *P_offd_data );

__global__ void hypreGPUKernel_generate_Qdiag_j_Qoffd_j( hypre_DeviceItem &item,
                                                         HYPRE_Int num_points,
                                                         HYPRE_Int color,
                                                         HYPRE_Int *pass_order, HYPRE_Int *pass_marker, HYPRE_Int *pass_marker_offd,
                                                         HYPRE_Int *fine_to_coarse, HYPRE_Int *fine_to_coarse_offd, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                                         HYPRE_Complex *A_diag_data, HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j, HYPRE_Complex *A_offd_data,
                                                         HYPRE_Int *Soc_diag_j, HYPRE_Int *Soc_offd_j, HYPRE_Int *Q_diag_i, HYPRE_Int *Q_offd_i,
                                                         HYPRE_Int *Q_diag_j, HYPRE_Complex *Q_diag_data, HYPRE_Int *Q_offd_j, HYPRE_Complex *Q_offd_data,
                                                         HYPRE_Complex *w_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int *dof_func_offd );

__global__ void hypreGPUKernel_mutli_pi_rowsum( hypre_DeviceItem &item, HYPRE_Int num_points,
                                                HYPRE_Int *pass_order,
                                                HYPRE_Int *A_diag_i, HYPRE_Complex *A_diag_data, HYPRE_Int *Pi_diag_i, HYPRE_Complex *Pi_diag_data,
                                                HYPRE_Int *Pi_offd_i, HYPRE_Complex *Pi_offd_data, HYPRE_Complex *w_row_sum );

__global__ void hypreGPUKernel_pass_order_count( hypre_DeviceItem &item, HYPRE_Int num_points,
                                                 HYPRE_Int color,
                                                 HYPRE_Int *points_left, HYPRE_Int *pass_marker, HYPRE_Int *pass_marker_offd, HYPRE_Int *S_diag_i,
                                                 HYPRE_Int *S_diag_j, HYPRE_Int *S_offd_i, HYPRE_Int *S_offd_j, HYPRE_Int *diag_shifts );

__global__ void hypreGPUKernel_populate_big_P_offd_j( hypre_DeviceItem &item, HYPRE_Int start,
                                                      HYPRE_Int stop,
                                                      HYPRE_Int *pass_order, HYPRE_Int *P_offd_i, HYPRE_Int *P_offd_j, HYPRE_BigInt *col_map_offd_Pi,
                                                      HYPRE_BigInt *big_P_offd_j );

/*--------------------------------------------------------------------------
 * hypre_ParAMGBuildModMultipass
 * This routine implements Stuben's direct interpolation with multiple passes.
 * expressed with matrix matrix multiplications
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildModMultipassDevice( hypre_ParCSRMatrix  *A,
                                        HYPRE_Int           *CF_marker,
                                        hypre_ParCSRMatrix  *S,
                                        HYPRE_BigInt        *num_cpts_global,
                                        HYPRE_Real           trunc_factor,
                                        HYPRE_Int            P_max_elmts,
                                        HYPRE_Int            interp_type,
                                        HYPRE_Int            num_functions,
                                        HYPRE_Int           *dof_func,
                                        hypre_ParCSRMatrix **P_ptr )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MULTIPASS_INTERP] -= hypre_MPI_Wtime();
#endif

   hypre_assert( hypre_ParCSRMatrixMemoryLocation(A) == HYPRE_MEMORY_DEVICE );
   hypre_assert( hypre_ParCSRMatrixMemoryLocation(S) == HYPRE_MEMORY_DEVICE );

   MPI_Comm                comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;

   HYPRE_Int        n_fine          = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data     = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int       *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   HYPRE_Real      *A_offd_data     = hypre_CSRMatrixData(A_offd);
   HYPRE_Int        num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

   hypre_CSRMatrix *S_diag       = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i     = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j     = hypre_CSRMatrixJ(S_diag);
   hypre_CSRMatrix *S_offd       = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i     = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j     = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRMatrix **Pi;
   hypre_ParCSRMatrix  *P;
   hypre_CSRMatrix     *P_diag;
   HYPRE_Real          *P_diag_data;
   HYPRE_Int           *P_diag_i;
   HYPRE_Int           *P_diag_j;
   hypre_CSRMatrix     *P_offd;
   HYPRE_Real          *P_offd_data = NULL;
   HYPRE_Int           *P_offd_i;
   HYPRE_Int           *P_offd_j = NULL;
   HYPRE_BigInt        *col_map_offd_P = NULL;
   HYPRE_BigInt        *col_map_offd_P_host = NULL;
   HYPRE_Int            num_cols_offd_P = 0;
   HYPRE_Int            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int            num_elem_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   HYPRE_Int           *int_buf_data = NULL;
   HYPRE_Int            P_diag_size = 0, P_offd_size = 0;

   HYPRE_Int       *pass_starts;
   HYPRE_Int       *fine_to_coarse;
   HYPRE_Int       *points_left;
   HYPRE_Int       *pass_marker;
   HYPRE_Int       *pass_marker_offd = NULL;
   HYPRE_Int       *pass_order;

   HYPRE_Int        i;
   HYPRE_Int        num_passes, p, remaining;
   HYPRE_Int        pass_starts_p1, pass_starts_p2;
   HYPRE_BigInt     remaining_big; /* tmp variable for reducing global_remaining */
   HYPRE_BigInt     global_remaining;
   HYPRE_Int        cnt, cnt_old, cnt_rem, current_pass;

   HYPRE_BigInt     total_global_cpts;
   HYPRE_Int        my_id, num_procs;

   HYPRE_Int       *dof_func_offd = NULL;
   HYPRE_Real      *row_sums = NULL;

   hypre_GpuProfilingPushRange("Section1");

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (num_procs > 1)
   {
      if (my_id == num_procs - 1)
      {
         total_global_cpts = num_cpts_global[1];
      }
      hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      total_global_cpts = num_cpts_global[1];
   }

   if (!total_global_cpts)
   {
      *P_ptr = NULL;
      return hypre_error_flag;
   }

   hypre_BoomerAMGMakeSocFromSDevice(A, S);

   /* Generate pass marker array */
   /* contains pass numbers for each variable according to original order */
   pass_marker = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_DEVICE);
   /* contains row numbers according to new order, pass 1 followed by pass 2 etc */
   pass_order = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_DEVICE);
   /* F2C mapping */
   /* reverse of pass_order, keeps track where original numbers go */
   fine_to_coarse = hypre_TAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_DEVICE);
   /* contains row numbers of remaining points, auxiliary */
   points_left = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_DEVICE);
   P_diag_i = hypre_CTAlloc(HYPRE_Int, n_fine + 1, HYPRE_MEMORY_DEVICE);
   P_offd_i = hypre_CTAlloc(HYPRE_Int, n_fine + 1, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   /* Fpts; number of F pts */
   oneapi::dpl::counting_iterator<HYPRE_Int> count(0);
   HYPRE_Int *points_end = hypreSycl_copy_if( count,
                                              count + n_fine,
                                              CF_marker,
                                              points_left,
   [] (const auto & x) {return x != 1;} );
   remaining = points_end - points_left;

   /* Cpts; number of C pts */
   HYPRE_Int *pass_end = hypreSycl_copy_if( count,
                                            count + n_fine,
                                            CF_marker,
                                            pass_order,
                                            equal<HYPRE_Int>(1) );

   P_diag_size = cnt = pass_end - pass_order;

   /* mark C points pass-1; row nnz of C-diag = 1, C-offd = 0 */
   auto zip0 = oneapi::dpl::make_zip_iterator( pass_marker, P_diag_i, P_offd_i );
   hypreSycl_transform_if( zip0,
                           zip0 + n_fine,
                           CF_marker,
                           zip0,
   [] (const auto & x) {return std::make_tuple(HYPRE_Int(1), HYPRE_Int(1), HYPRE_Int(0));},
   equal<HYPRE_Int>(1) );

   HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(CF_marker,          equal<HYPRE_Int>(1)),
                      oneapi::dpl::make_transform_iterator(CF_marker + n_fine, equal<HYPRE_Int>(1)),
                      fine_to_coarse,
                      HYPRE_Int(0) );
#else
   /* Fpts; number of F pts */
   HYPRE_Int *points_end = HYPRE_THRUST_CALL( copy_if,
                                              thrust::make_counting_iterator(0),
                                              thrust::make_counting_iterator(n_fine),
                                              CF_marker,
                                              points_left,
                                              thrust::not1(equal<HYPRE_Int>(1)) );
   remaining = points_end - points_left;

   /* Cpts; number of C pts */
   HYPRE_Int *pass_end = HYPRE_THRUST_CALL( copy_if,
                                            thrust::make_counting_iterator(0),
                                            thrust::make_counting_iterator(n_fine),
                                            CF_marker,
                                            pass_order,
                                            equal<HYPRE_Int>(1) );

   P_diag_size = cnt = pass_end - pass_order;

   /* mark C points pass-1; row nnz of C-diag = 1, C-offd = 0 */
   HYPRE_THRUST_CALL( replace_if,
                      thrust::make_zip_iterator( thrust::make_tuple(pass_marker, P_diag_i, P_offd_i) ),
                      thrust::make_zip_iterator( thrust::make_tuple(pass_marker, P_diag_i, P_offd_i) ) + n_fine,
                      CF_marker,
                      equal<HYPRE_Int>(1),
                      thrust::make_tuple(HYPRE_Int(1), HYPRE_Int(1), HYPRE_Int(0)) );

   HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,          equal<HYPRE_Int>(1)),
                      thrust::make_transform_iterator(CF_marker + n_fine, equal<HYPRE_Int>(1)),
                      fine_to_coarse,
                      HYPRE_Int(0) );
#endif

   /* contains beginning for each pass in pass_order field, assume no more than 10 passes */
   pass_starts = hypre_CTAlloc(HYPRE_Int, 11, HYPRE_MEMORY_HOST);
   /* first pass is C */
   pass_starts[0] = 0;
   pass_starts[1] = cnt;

   /* communicate dof_func */
   if (num_procs > 1 && num_functions > 1)
   {
      int_buf_data = hypre_TAlloc(HYPRE_Int, num_elem_send, HYPRE_MEMORY_DEVICE);

      hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                        hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                        dof_func,
                        int_buf_data );
#else
      HYPRE_THRUST_CALL( gather,
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                         dof_func,
                         int_buf_data );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
      if (hypre_GetGpuAwareMPI())
      {
         hypre_ForceSyncComputeStream(hypre_handle());
      }
#endif

      dof_func_offd = hypre_TAlloc(HYPRE_Int, num_cols_offd_A, HYPRE_MEMORY_DEVICE);

      comm_handle = hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, HYPRE_MEMORY_DEVICE, int_buf_data,
                                                    HYPRE_MEMORY_DEVICE, dof_func_offd);
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /* communicate pass_marker */
   if (num_procs > 1)
   {
      if (!int_buf_data)
      {
         int_buf_data = hypre_CTAlloc(HYPRE_Int, num_elem_send, HYPRE_MEMORY_DEVICE);
      }

      hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                        hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                        pass_marker,
                        int_buf_data );
#else
      HYPRE_THRUST_CALL( gather,
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                         pass_marker,
                         int_buf_data );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
      if (hypre_GetGpuAwareMPI())
      {
         hypre_ForceSyncComputeStream(hypre_handle());
      }
#endif

      /* allocate one more see comments in hypre_modmp_compute_num_cols_offd_fine_to_coarse */
      pass_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A + 1, HYPRE_MEMORY_DEVICE);

      /* create a handle to start communication. 11: for integer */
      comm_handle = hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, HYPRE_MEMORY_DEVICE, int_buf_data,
                                                    HYPRE_MEMORY_DEVICE, pass_marker_offd);

      /* destroy the handle to finish communication */
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   current_pass = 1;
   num_passes = 1;
   /* color points according to pass number */
   remaining_big = remaining;
   hypre_MPI_Allreduce(&remaining_big, &global_remaining, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

   hypre_GpuProfilingPopRange();

   hypre_GpuProfilingPushRange("Section2");

   HYPRE_Int *points_left_old = hypre_TAlloc(HYPRE_Int, remaining, HYPRE_MEMORY_DEVICE);
   HYPRE_Int *diag_shifts     = hypre_TAlloc(HYPRE_Int, remaining, HYPRE_MEMORY_DEVICE);

   while (global_remaining > 0)
   {
      cnt_rem = 0;
      cnt_old = cnt;

      {
         dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
         dim3 gDim = hypre_GetDefaultDeviceGridDimension(remaining, "warp", bDim);

         /* output diag_shifts is 0/1 indicating if points_left_dev[i] is picked in this pass */
         HYPRE_GPU_LAUNCH( hypreGPUKernel_pass_order_count,
                           gDim, bDim,
                           remaining,
                           current_pass,
                           points_left,
                           pass_marker,
                           pass_marker_offd,
                           S_diag_i,
                           S_diag_j,
                           S_offd_i,
                           S_offd_j,
                           diag_shifts );

#if defined(HYPRE_USING_SYCL)
         cnt = HYPRE_ONEDPL_CALL( std::reduce,
                                  diag_shifts,
                                  diag_shifts + remaining,
                                  cnt_old,
                                  std::plus<HYPRE_Int>() );

         cnt_rem = remaining - (cnt - cnt_old);

         auto perm0 = oneapi::dpl::make_permutation_iterator(pass_marker, points_left);
         hypreSycl_transform_if( perm0,
                                 perm0 + remaining,
                                 diag_shifts,
                                 perm0,
         [current_pass = current_pass] (const auto & x) {return current_pass + 1;},
         [] (const auto & x) {return x;} );

         hypre_TMemcpy(points_left_old, points_left, HYPRE_Int, remaining, HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_DEVICE);

         HYPRE_Int *new_end;
         new_end = hypreSycl_copy_if( points_left_old,
                                      points_left_old + remaining,
                                      diag_shifts,
                                      pass_order + cnt_old,
         [] (const auto & x) {return x;} );

         hypre_assert(new_end - pass_order == cnt);

         new_end = hypreSycl_copy_if( points_left_old,
                                      points_left_old + remaining,
                                      diag_shifts,
                                      points_left,
         [] (const auto & x) {return !x;} );
#else
         cnt = HYPRE_THRUST_CALL( reduce,
                                  diag_shifts,
                                  diag_shifts + remaining,
                                  cnt_old,
                                  thrust::plus<HYPRE_Int>() );

         cnt_rem = remaining - (cnt - cnt_old);

         HYPRE_THRUST_CALL( replace_if,
                            thrust::make_permutation_iterator(pass_marker, points_left),
                            thrust::make_permutation_iterator(pass_marker, points_left + remaining),
                            diag_shifts,
                            thrust::identity<HYPRE_Int>(),
                            current_pass + 1 );

         hypre_TMemcpy(points_left_old, points_left, HYPRE_Int, remaining, HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_DEVICE);

         HYPRE_Int *new_end;
         new_end = HYPRE_THRUST_CALL( copy_if,
                                      points_left_old,
                                      points_left_old + remaining,
                                      diag_shifts,
                                      pass_order + cnt_old,
                                      thrust::identity<HYPRE_Int>() );

         hypre_assert(new_end - pass_order == cnt);

         new_end = HYPRE_THRUST_CALL( copy_if,
                                      points_left_old,
                                      points_left_old + remaining,
                                      diag_shifts,
                                      points_left,
                                      thrust::not1(thrust::identity<HYPRE_Int>()) );
#endif

         hypre_assert(new_end - points_left == cnt_rem);
      }

      remaining = cnt_rem;
      current_pass++;
      num_passes++;

      if (num_passes > 9)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, " Warning!!! too many passes! out of range!\n");
         break;
      }

      pass_starts[num_passes] = cnt;

      /* update pass_marker_offd */
      if (num_procs > 1)
      {
#if defined(HYPRE_USING_SYCL)
         hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                           hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                           pass_marker,
                           int_buf_data );
#else
         HYPRE_THRUST_CALL( gather,
                            hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                            hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                            pass_marker,
                            int_buf_data );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
         /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
         if (hypre_GetGpuAwareMPI())
         {
            hypre_ForceSyncComputeStream(hypre_handle());
         }
#endif

         /* create a handle to start communication. 11: for integer */
         comm_handle = hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, HYPRE_MEMORY_DEVICE, int_buf_data,
                                                       HYPRE_MEMORY_DEVICE, pass_marker_offd);

         /* destroy the handle to finish communication */
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      HYPRE_BigInt old_global_remaining = global_remaining;

      remaining_big = remaining;
      hypre_MPI_Allreduce(&remaining_big, &global_remaining, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

      /* if the number of remaining points does not change, we have a situation of isolated areas of
       * fine points that are not connected to any C-points, and the pass generation process breaks
       * down. Those points can be ignored, i.e. the corresponding rows in P will just be 0
       * and can be ignored for the algorithm. */
      if (old_global_remaining == global_remaining)
      {
         break;
      }

   } // while (global_remaining > 0)

   hypre_TFree(diag_shifts,     HYPRE_MEMORY_DEVICE);
   hypre_TFree(points_left_old, HYPRE_MEMORY_DEVICE);
   hypre_TFree(int_buf_data,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(points_left,     HYPRE_MEMORY_DEVICE);

   /* generate row sum of weak points and C-points to be ignored */
   row_sums = hypre_CTAlloc(HYPRE_Real, n_fine, HYPRE_MEMORY_DEVICE);

   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(n_fine, "warp", bDim);

      HYPRE_GPU_LAUNCH( hypreGPUKernel_cfmarker_masked_rowsum, gDim, bDim,
                        n_fine, A_diag_i, A_diag_j, A_diag_data,
                        A_offd_i, A_offd_j, A_offd_data,
                        CF_marker,
                        num_functions > 1 ? dof_func : NULL,
                        num_functions > 1 ? dof_func_offd : NULL,
                        row_sums );
   }

   hypre_GpuProfilingPopRange();

   hypre_GpuProfilingPushRange("MultipassPiDevice");

   Pi = hypre_CTAlloc(hypre_ParCSRMatrix*, num_passes, HYPRE_MEMORY_HOST);

   hypre_GenerateMultipassPiDevice(A, S, num_cpts_global, &pass_order[pass_starts[1]],
                                   pass_marker, pass_marker_offd,
                                   pass_starts[2] - pass_starts[1], 1, row_sums, &Pi[0]);

   hypre_GpuProfilingPopRange();

   if (interp_type == 8)
   {
      for (i = 1; i < num_passes - 1; i++)
      {
         hypre_GpuProfilingPushRange(std::string("MultipassPiDevice Loop" + std::to_string(i)).c_str());

         hypre_ParCSRMatrix *Q;
         HYPRE_BigInt *c_pts_starts = hypre_ParCSRMatrixRowStarts(Pi[i - 1]);

         hypre_GenerateMultipassPiDevice(A, S, c_pts_starts, &pass_order[pass_starts[i + 1]],
                                         pass_marker, pass_marker_offd,
                                         pass_starts[i + 2] - pass_starts[i + 1], i + 1, row_sums, &Q);

         hypre_GpuProfilingPopRange();
         Pi[i] = hypre_ParCSRMatMat(Q, Pi[i - 1]);

         hypre_ParCSRMatrixDestroy(Q);
      }
   }
   else if (interp_type == 9)
   {
      for (i = 1; i < num_passes - 1; i++)
      {
         hypre_GpuProfilingPushRange(std::string("MultiPiDevice Loop" + std::to_string(i)).c_str());
         HYPRE_BigInt *c_pts_starts = hypre_ParCSRMatrixRowStarts(Pi[i - 1]);

         hypre_GenerateMultiPiDevice(A, S, Pi[i - 1], c_pts_starts, &pass_order[pass_starts[i + 1]],
                                     pass_marker, pass_marker_offd,
                                     pass_starts[i + 2] - pass_starts[i + 1], i + 1,
                                     num_functions, dof_func, dof_func_offd, &Pi[i] );

         hypre_GpuProfilingPopRange();
      }
   }

   hypre_GpuProfilingPushRange("Section3");

   // We don't need the row sums anymore
   hypre_TFree(row_sums, HYPRE_MEMORY_DEVICE);

   /* populate P_diag_i/P_offd_i[i] with nnz of i-th row */
   for (i = 0; i < num_passes - 1; i++)
   {
      HYPRE_Int *Pi_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(Pi[i]));
      HYPRE_Int *Pi_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(Pi[i]));

      HYPRE_Int start = pass_starts[i + 1];
      HYPRE_Int stop  = pass_starts[i + 2];

#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::transform,
                         Pi_diag_i + 1,
                         Pi_diag_i + stop - start + 1,
                         Pi_diag_i,
                         oneapi::dpl::make_permutation_iterator( P_diag_i, pass_order + start ),
                         std::minus<HYPRE_Int>() );
      HYPRE_ONEDPL_CALL( std::transform,
                         Pi_offd_i + 1,
                         Pi_offd_i + stop - start + 1,
                         Pi_offd_i,
                         oneapi::dpl::make_permutation_iterator( P_offd_i, pass_order + start ),
                         std::minus<HYPRE_Int>() );
#else
      HYPRE_THRUST_CALL( transform,
                         thrust::make_zip_iterator(thrust::make_tuple(Pi_diag_i, Pi_offd_i)) + 1,
                         thrust::make_zip_iterator(thrust::make_tuple(Pi_diag_i, Pi_offd_i)) + stop - start + 1,
                         thrust::make_zip_iterator(thrust::make_tuple(Pi_diag_i, Pi_offd_i)),
                         thrust::make_permutation_iterator( thrust::make_zip_iterator(thrust::make_tuple(P_diag_i,
                                                                                                         P_offd_i)), pass_order + start ),
                         tuple_minus<HYPRE_Int>() );
#endif

      P_diag_size += hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(Pi[i]));
      P_offd_size += hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(Pi[i]));
   }

#if defined(HYPRE_USING_SYCL)
   /* WM: todo - this is a workaround since oneDPL's exclusive_scan gives incorrect results when doing the scan in place */
   auto zip2 = oneapi::dpl::make_zip_iterator( P_diag_i, P_offd_i );
   HYPRE_Int *P_diag_i_tmp = hypre_CTAlloc(HYPRE_Int, n_fine + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int *P_offd_i_tmp = hypre_CTAlloc(HYPRE_Int, n_fine + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      zip2,
                      zip2 + n_fine + 1,
                      oneapi::dpl::make_zip_iterator(P_diag_i_tmp, P_offd_i_tmp),
                      std::make_tuple(HYPRE_Int(0), HYPRE_Int(0)),
                      tuple_plus<HYPRE_Int>() );
   hypre_TMemcpy(P_diag_i, P_diag_i_tmp, HYPRE_Int, n_fine + 1, HYPRE_MEMORY_DEVICE,
                 HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(P_offd_i, P_offd_i_tmp, HYPRE_Int, n_fine + 1, HYPRE_MEMORY_DEVICE,
                 HYPRE_MEMORY_DEVICE);
   hypre_TFree(P_diag_i_tmp, HYPRE_MEMORY_DEVICE);
   hypre_TFree(P_offd_i_tmp, HYPRE_MEMORY_DEVICE);
#else
   HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_zip_iterator( thrust::make_tuple(P_diag_i, P_offd_i) ),
                      thrust::make_zip_iterator( thrust::make_tuple(P_diag_i, P_offd_i) ) + n_fine + 1,
                      thrust::make_zip_iterator( thrust::make_tuple(P_diag_i, P_offd_i) ),
                      thrust::make_tuple(HYPRE_Int(0), HYPRE_Int(0)),
                      tuple_plus<HYPRE_Int>() );
#endif

#ifdef HYPRE_DEBUG
   {
      HYPRE_Int tmp;
      hypre_TMemcpy(&tmp, &P_diag_i[n_fine], HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_assert(tmp == P_diag_size);
      hypre_TMemcpy(&tmp, &P_offd_i[n_fine], HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_assert(tmp == P_offd_size);
   }
#endif

   P_diag_j    = hypre_TAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(HYPRE_Real, P_diag_size, HYPRE_MEMORY_DEVICE);
   P_offd_j    = hypre_TAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_DEVICE);
   P_offd_data = hypre_TAlloc(HYPRE_Real, P_offd_size, HYPRE_MEMORY_DEVICE);

   /* insert weights for coarse points */
   {
#if defined(HYPRE_USING_SYCL)
      auto perm1 = oneapi::dpl::make_permutation_iterator( fine_to_coarse, pass_order );
      hypreSycl_scatter( perm1,
                         perm1 + pass_starts[1],
                         oneapi::dpl::make_permutation_iterator( P_diag_i, pass_order ),
                         P_diag_j );

      auto perm2 = oneapi::dpl::make_permutation_iterator( P_diag_i, pass_order );
      auto perm3 = oneapi::dpl::make_permutation_iterator( P_diag_data, perm2 );
      HYPRE_ONEDPL_CALL( std::transform,
                         perm3,
                         perm3 + pass_starts[1],
                         perm3,
      [] (const auto & x) {return 1.0;} );
#else
      HYPRE_THRUST_CALL( scatter,
                         thrust::make_permutation_iterator( fine_to_coarse, pass_order ),
                         thrust::make_permutation_iterator( fine_to_coarse, pass_order ) + pass_starts[1],
                         thrust::make_permutation_iterator( P_diag_i, pass_order ),
                         P_diag_j );

      HYPRE_THRUST_CALL( scatter,
                         thrust::make_constant_iterator<HYPRE_Real>(1.0),
                         thrust::make_constant_iterator<HYPRE_Real>(1.0) + pass_starts[1],
                         thrust::make_permutation_iterator( P_diag_i, pass_order ),
                         P_diag_data );
#endif
   }

   /* generate col_map_offd_P by combining all col_map_offd_Pi
    * and reompute indices if needed */

   /* insert remaining weights */
   for (p = 0; p < num_passes - 1; p++)
   {
      HYPRE_Int  *Pi_diag_i    = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(Pi[p]));
      HYPRE_Int  *Pi_offd_i    = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(Pi[p]));
      HYPRE_Int  *Pi_diag_j    = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(Pi[p]));
      HYPRE_Int  *Pi_offd_j    = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(Pi[p]));
      HYPRE_Real *Pi_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(Pi[p]));
      HYPRE_Real *Pi_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(Pi[p]));

      HYPRE_Int num_points = pass_starts[p + 2] - pass_starts[p + 1];

      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_points, "warp", bDim);

      pass_starts_p1 = pass_starts[p + 1];
      pass_starts_p2 = pass_starts[p + 2];
      HYPRE_GPU_LAUNCH( hypreGPUKernel_insert_remaining_weights, gDim, bDim,
                        pass_starts_p1, pass_starts_p2, pass_order,
                        Pi_diag_i, Pi_diag_j, Pi_diag_data,
                        P_diag_i, P_diag_j, P_diag_data,
                        Pi_offd_i, Pi_offd_j, Pi_offd_data,
                        P_offd_i, P_offd_j, P_offd_data );
   }

   /* Note that col indices in P_offd_j probably not consistent,
      this gets fixed after truncation */
   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                hypre_ParCSRMatrixRowStarts(A),
                                num_cpts_global,
                                num_cols_offd_P,
                                P_diag_size,
                                P_offd_size);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag)    = P_diag_i;
   hypre_CSRMatrixJ(P_diag)    = P_diag_j;

   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd)    = P_offd_i;
   hypre_CSRMatrixJ(P_offd)    = P_offd_j;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || P_max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, P_max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_offd_size = hypre_CSRMatrixNumNonzeros(P_offd);
   }

   hypre_GpuProfilingPopRange();

   num_cols_offd_P = 0;

   if (P_offd_size)
   {
      hypre_GpuProfilingPushRange("Section4");

      HYPRE_BigInt *big_P_offd_j = hypre_TAlloc(HYPRE_BigInt, P_offd_size, HYPRE_MEMORY_DEVICE);

      for (p = 0; p < num_passes - 1; p++)
      {
         HYPRE_BigInt *col_map_offd_Pi = hypre_ParCSRMatrixDeviceColMapOffd(Pi[p]);

         HYPRE_Int npoints = pass_starts[p + 2] - pass_starts[p + 1];
         dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
         dim3 gDim = hypre_GetDefaultDeviceGridDimension(npoints, "warp", bDim);

         pass_starts_p1 = pass_starts[p + 1];
         pass_starts_p2 = pass_starts[p + 2];
         HYPRE_GPU_LAUNCH( hypreGPUKernel_populate_big_P_offd_j, gDim, bDim,
                           pass_starts_p1,
                           pass_starts_p2,
                           pass_order,
                           P_offd_i,
                           P_offd_j,
                           col_map_offd_Pi,
                           big_P_offd_j );

      } // end num_passes for loop

      HYPRE_BigInt *tmp_P_offd_j = hypre_TAlloc(HYPRE_BigInt, P_offd_size, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_P_offd_j, big_P_offd_j, HYPRE_BigInt, P_offd_size, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::sort,
                         tmp_P_offd_j,
                         tmp_P_offd_j + P_offd_size );

      HYPRE_BigInt *new_end = HYPRE_ONEDPL_CALL( std::unique,
                                                 tmp_P_offd_j,
                                                 tmp_P_offd_j + P_offd_size );
#else
      HYPRE_THRUST_CALL( sort,
                         tmp_P_offd_j,
                         tmp_P_offd_j + P_offd_size );

      HYPRE_BigInt *new_end = HYPRE_THRUST_CALL( unique,
                                                 tmp_P_offd_j,
                                                 tmp_P_offd_j + P_offd_size );
#endif

      num_cols_offd_P = new_end - tmp_P_offd_j;
      col_map_offd_P = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_P, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(col_map_offd_P, tmp_P_offd_j, HYPRE_BigInt, num_cols_offd_P, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_DEVICE);
      hypre_TFree(tmp_P_offd_j, HYPRE_MEMORY_DEVICE);

      // PB: It seems we still need this on the host??
      col_map_offd_P_host = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_P, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(col_map_offd_P_host, col_map_offd_P, HYPRE_BigInt, num_cols_offd_P, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                         col_map_offd_P,
                         col_map_offd_P + num_cols_offd_P,
                         big_P_offd_j,
                         big_P_offd_j + P_offd_size,
                         P_offd_j );
#else
      HYPRE_THRUST_CALL( lower_bound,
                         col_map_offd_P,
                         col_map_offd_P + num_cols_offd_P,
                         big_P_offd_j,
                         big_P_offd_j + P_offd_size,
                         P_offd_j );
#endif

      hypre_TFree(big_P_offd_j, HYPRE_MEMORY_DEVICE);

      hypre_GpuProfilingPopRange();
   } // if (P_offd_size)

   hypre_GpuProfilingPushRange("Section5");

   hypre_ParCSRMatrixColMapOffd(P)       = col_map_offd_P_host;
   hypre_ParCSRMatrixDeviceColMapOffd(P) = col_map_offd_P;
   hypre_CSRMatrixNumCols(P_offd)        = num_cols_offd_P;

   hypre_CSRMatrixMemoryLocation(P_diag) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(P_offd) = HYPRE_MEMORY_DEVICE;

   hypre_MatvecCommPkgCreate(P);

   for (i = 0; i < num_passes - 1; i++)
   {
      hypre_ParCSRMatrixDestroy(Pi[i]);
   }

   hypre_TFree(Pi,               HYPRE_MEMORY_HOST);
   hypre_TFree(dof_func_offd,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(pass_starts,      HYPRE_MEMORY_HOST);
   hypre_TFree(pass_marker,      HYPRE_MEMORY_DEVICE);
   hypre_TFree(pass_marker_offd, HYPRE_MEMORY_DEVICE);
   hypre_TFree(pass_order,       HYPRE_MEMORY_DEVICE);
   hypre_TFree(fine_to_coarse,   HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::replace_if,
                      CF_marker,
                      CF_marker + n_fine,
                      equal<HYPRE_Int>(-3),
                      static_cast<HYPRE_Int>(-1) );
#else
   HYPRE_THRUST_CALL( replace_if,
                      CF_marker,
                      CF_marker + n_fine,
                      equal<HYPRE_Int>(-3),
                      static_cast<HYPRE_Int>(-1) );
#endif

   *P_ptr = P;

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

HYPRE_Int
hypre_GenerateMultipassPiDevice( hypre_ParCSRMatrix  *A,
                                 hypre_ParCSRMatrix  *S,
                                 HYPRE_BigInt        *c_pts_starts,
                                 HYPRE_Int           *pass_order,
                                 HYPRE_Int           *pass_marker,
                                 HYPRE_Int           *pass_marker_offd,
                                 HYPRE_Int            num_points, /* |F| */
                                 HYPRE_Int            color, /* C-color */
                                 HYPRE_Real          *row_sums,
                                 hypre_ParCSRMatrix **P_ptr)
{
   MPI_Comm                comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;

   hypre_CSRMatrix *A_diag      = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i    = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j    = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int        n_fine      = hypre_CSRMatrixNumRows(A_diag);

   hypre_CSRMatrix *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int       *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   HYPRE_Real      *A_offd_data     = hypre_CSRMatrixData(A_offd);
   HYPRE_Int        num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

   hypre_CSRMatrix *S_diag   = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd   = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   HYPRE_Int *Soc_diag_j = hypre_ParCSRMatrixSocDiagJ(S);
   HYPRE_Int *Soc_offd_j = hypre_ParCSRMatrixSocOffdJ(S);

   HYPRE_BigInt    *col_map_offd_P     = NULL;
   HYPRE_BigInt    *col_map_offd_P_dev = NULL;
   HYPRE_Int        num_cols_offd_P;
   HYPRE_Int        nnz_diag, nnz_offd;

   hypre_ParCSRMatrix *P;
   hypre_CSRMatrix    *P_diag;
   HYPRE_Real         *P_diag_data;
   HYPRE_Int          *P_diag_i; /*at first counter of nonzero cols for each row,
                                   finally will be pointer to start of row */
   HYPRE_Int          *P_diag_j;
   hypre_CSRMatrix    *P_offd;
   HYPRE_Real         *P_offd_data = NULL;
   HYPRE_Int          *P_offd_i; /*at first counter of nonzero cols for each row,
                                   finally will be pointer to start of row */
   HYPRE_Int          *P_offd_j = NULL;

   HYPRE_Int       *fine_to_coarse;
   HYPRE_Int       *fine_to_coarse_offd = NULL;
   HYPRE_BigInt     f_pts_starts[2];
   HYPRE_Int        my_id, num_procs;
   HYPRE_BigInt     total_global_fpts;
   HYPRE_BigInt     total_global_cpts;
   HYPRE_BigInt    *big_convert_offd = NULL;
   HYPRE_BigInt    *big_buf_data = NULL;

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   fine_to_coarse = hypre_TAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_DEVICE);

   hypre_modmp_init_fine_to_coarse(n_fine, pass_marker, color, fine_to_coarse);

   if (num_procs > 1)
   {
      HYPRE_BigInt big_Fpts = num_points;

      hypre_MPI_Scan(&big_Fpts, f_pts_starts + 1, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

      f_pts_starts[0] = f_pts_starts[1] - big_Fpts;

      if (my_id == num_procs - 1)
      {
         total_global_fpts = f_pts_starts[1];
         total_global_cpts = c_pts_starts[1];
      }
      hypre_MPI_Bcast(&total_global_fpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      f_pts_starts[0] = 0;
      f_pts_starts[1] = num_points;
      total_global_fpts = f_pts_starts[1];
      total_global_cpts = c_pts_starts[1];
   }

   num_cols_offd_P = 0;

   if (num_procs > 1)
   {
      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      HYPRE_Int num_elem_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

      big_convert_offd = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_A, HYPRE_MEMORY_DEVICE);
      big_buf_data     = hypre_TAlloc(HYPRE_BigInt, num_elem_send,   HYPRE_MEMORY_DEVICE);

      globalC_functor functor(c_pts_starts[0]);

#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                        hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                        oneapi::dpl::make_transform_iterator(fine_to_coarse, functor),
                        big_buf_data );
#else
      HYPRE_THRUST_CALL( gather,
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                         thrust::make_transform_iterator(fine_to_coarse, functor),
                         big_buf_data );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure big_buf_data is ready before issuing GPU-GPU MPI */
      if (hypre_GetGpuAwareMPI())
      {
         hypre_ForceSyncComputeStream(hypre_handle());
      }
#endif

      comm_handle = hypre_ParCSRCommHandleCreate_v2(21, comm_pkg, HYPRE_MEMORY_DEVICE, big_buf_data,
                                                    HYPRE_MEMORY_DEVICE, big_convert_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);

      // This will allocate fine_to_coarse_offd
      hypre_modmp_compute_num_cols_offd_fine_to_coarse( pass_marker_offd, color, num_cols_offd_A,
                                                        num_cols_offd_P, &fine_to_coarse_offd );

      //FIXME: Clean this up when we don't need the host pointer anymore
      col_map_offd_P     = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_P, HYPRE_MEMORY_HOST);
      col_map_offd_P_dev = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_P, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      HYPRE_BigInt *col_map_end = hypreSycl_copy_if( big_convert_offd,
                                                     big_convert_offd + num_cols_offd_A,
                                                     pass_marker_offd,
                                                     col_map_offd_P_dev,
                                                     equal<HYPRE_Int>(color) );
#else
      HYPRE_BigInt *col_map_end = HYPRE_THRUST_CALL( copy_if,
                                                     big_convert_offd,
                                                     big_convert_offd + num_cols_offd_A,
                                                     pass_marker_offd,
                                                     col_map_offd_P_dev,
                                                     equal<HYPRE_Int>(color) );
#endif

      hypre_assert(num_cols_offd_P == col_map_end - col_map_offd_P_dev);

      //FIXME: Clean this up when we don't need the host pointer anymore
      hypre_TMemcpy(col_map_offd_P, col_map_offd_P_dev, HYPRE_BigInt, num_cols_offd_P, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);

      hypre_TFree(big_convert_offd, HYPRE_MEMORY_DEVICE);
      hypre_TFree(big_buf_data,     HYPRE_MEMORY_DEVICE);
   }

   P_diag_i = hypre_TAlloc(HYPRE_Int, num_points + 1, HYPRE_MEMORY_DEVICE);
   P_offd_i = hypre_TAlloc(HYPRE_Int, num_points + 1, HYPRE_MEMORY_DEVICE);

   /* generate P_diag_i and P_offd_i */
   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_points, "warp", bDim);

      HYPRE_GPU_LAUNCH( hypreGPUKernel_generate_Pdiag_i_Poffd_i, gDim, bDim,
                        num_points, color, pass_order, pass_marker, pass_marker_offd,
                        S_diag_i, S_diag_j, S_offd_i, S_offd_j,
                        P_diag_i, P_offd_i );

      hypre_Memset(P_diag_i + num_points, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);
      hypre_Memset(P_offd_i + num_points, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      /* WM: todo - this is a workaround since oneDPL's exclusive_scan gives incorrect results when doing the scan in place */
      auto zip3 = oneapi::dpl::make_zip_iterator( P_diag_i, P_offd_i );
      HYPRE_Int *P_diag_i_tmp = hypre_CTAlloc(HYPRE_Int, num_points + 1, HYPRE_MEMORY_DEVICE);
      HYPRE_Int *P_offd_i_tmp = hypre_CTAlloc(HYPRE_Int, num_points + 1, HYPRE_MEMORY_DEVICE);
      HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         zip3,
                         zip3 + num_points + 1,
                         oneapi::dpl::make_zip_iterator( P_diag_i_tmp, P_offd_i_tmp ),
                         std::make_tuple(HYPRE_Int(0), HYPRE_Int(0)),
                         tuple_plus<HYPRE_Int>() );
      hypre_TMemcpy(P_diag_i, P_diag_i_tmp, HYPRE_Int, num_points + 1, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(P_offd_i, P_offd_i_tmp, HYPRE_Int, num_points + 1, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_DEVICE);
      hypre_TFree(P_diag_i_tmp, HYPRE_MEMORY_DEVICE);
      hypre_TFree(P_offd_i_tmp, HYPRE_MEMORY_DEVICE);
#else
      HYPRE_THRUST_CALL( exclusive_scan,
                         thrust::make_zip_iterator( thrust::make_tuple(P_diag_i, P_offd_i) ),
                         thrust::make_zip_iterator( thrust::make_tuple(P_diag_i, P_offd_i) ) + num_points + 1,
                         thrust::make_zip_iterator( thrust::make_tuple(P_diag_i, P_offd_i) ),
                         thrust::make_tuple(HYPRE_Int(0), HYPRE_Int(0)),
                         tuple_plus<HYPRE_Int>() );
#endif

      hypre_TMemcpy(&nnz_diag, &P_diag_i[num_points], HYPRE_Int, 1, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(&nnz_offd, &P_offd_i[num_points], HYPRE_Int, 1, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);
   }

   /* generate P_diag_j/data and P_offd_j/data */
   P_diag_j    = hypre_TAlloc(HYPRE_Int,  nnz_diag, HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(HYPRE_Real, nnz_diag, HYPRE_MEMORY_DEVICE);
   P_offd_j    = hypre_TAlloc(HYPRE_Int,  nnz_offd, HYPRE_MEMORY_DEVICE);
   P_offd_data = hypre_TAlloc(HYPRE_Real, nnz_offd, HYPRE_MEMORY_DEVICE);

   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_points, "warp", bDim);

      HYPRE_GPU_LAUNCH( hypreGPUKernel_generate_Pdiag_j_Poffd_j, gDim, bDim,
                        num_points,
                        color,
                        pass_order,
                        pass_marker,
                        pass_marker_offd,
                        fine_to_coarse,
                        fine_to_coarse_offd,
                        A_diag_i,
                        A_diag_j,
                        A_diag_data,
                        A_offd_i,
                        A_offd_j,
                        A_offd_data,
                        Soc_diag_j,
                        Soc_offd_j,
                        P_diag_i,
                        P_offd_i,
                        P_diag_j,
                        P_diag_data,
                        P_offd_j,
                        P_offd_data,
                        row_sums );
   }

   hypre_TFree(fine_to_coarse,      HYPRE_MEMORY_DEVICE);
   hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_DEVICE);

   P = hypre_ParCSRMatrixCreate(comm,
                                total_global_fpts,
                                total_global_cpts,
                                f_pts_starts,
                                c_pts_starts,
                                num_cols_offd_P,
                                nnz_diag,
                                nnz_offd);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag)    = P_diag_i;
   hypre_CSRMatrixJ(P_diag)    = P_diag_j;

   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd)    = P_offd_i;
   hypre_CSRMatrixJ(P_offd)    = P_offd_j;

   hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
   hypre_ParCSRMatrixDeviceColMapOffd(P) = col_map_offd_P_dev;

   hypre_CSRMatrixMemoryLocation(P_diag) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(P_offd) = HYPRE_MEMORY_DEVICE;

   hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   return hypre_error_flag;
}

HYPRE_Int
hypre_GenerateMultiPiDevice( hypre_ParCSRMatrix  *A,
                             hypre_ParCSRMatrix  *S,
                             hypre_ParCSRMatrix  *P,
                             HYPRE_BigInt        *c_pts_starts,
                             HYPRE_Int           *pass_order,
                             HYPRE_Int           *pass_marker,
                             HYPRE_Int           *pass_marker_offd,
                             HYPRE_Int            num_points,
                             HYPRE_Int            color,
                             HYPRE_Int            num_functions,
                             HYPRE_Int           *dof_func,
                             HYPRE_Int           *dof_func_offd,
                             hypre_ParCSRMatrix **Pi_ptr )
{
   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;

   hypre_CSRMatrix *A_diag      = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i    = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j    = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int        n_fine      = hypre_CSRMatrixNumRows(A_diag);

   hypre_CSRMatrix *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int       *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   HYPRE_Real      *A_offd_data     = hypre_CSRMatrixData(A_offd);
   HYPRE_Int        num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

   hypre_CSRMatrix *S_diag   = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd   = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   HYPRE_Int *Soc_diag_j = hypre_ParCSRMatrixSocDiagJ(S);
   HYPRE_Int *Soc_offd_j = hypre_ParCSRMatrixSocOffdJ(S);

   HYPRE_BigInt    *col_map_offd_Q     = NULL;
   HYPRE_BigInt    *col_map_offd_Q_dev = NULL;
   HYPRE_Int        num_cols_offd_Q;

   hypre_ParCSRMatrix *Pi;
   hypre_CSRMatrix    *Pi_diag;
   HYPRE_Int          *Pi_diag_i;
   HYPRE_Real         *Pi_diag_data;
   hypre_CSRMatrix    *Pi_offd;
   HYPRE_Int          *Pi_offd_i;
   HYPRE_Real         *Pi_offd_data;

   HYPRE_Int           nnz_diag, nnz_offd;

   hypre_ParCSRMatrix *Q;
   hypre_CSRMatrix    *Q_diag;
   HYPRE_Real         *Q_diag_data;
   HYPRE_Int          *Q_diag_i; /*at first counter of nonzero cols for each row,
                                   finally will be pointer to start of row */
   HYPRE_Int          *Q_diag_j;
   hypre_CSRMatrix    *Q_offd;
   HYPRE_Real         *Q_offd_data = NULL;
   HYPRE_Int          *Q_offd_i; /*at first counter of nonzero cols for each row,
                                  finally will be pointer to start of row */
   HYPRE_Int          *Q_offd_j = NULL;

   HYPRE_Int       *fine_to_coarse;
   HYPRE_Int       *fine_to_coarse_offd = NULL;
   HYPRE_BigInt     f_pts_starts[2];
   HYPRE_Int        my_id, num_procs;
   HYPRE_BigInt     total_global_fpts;
   HYPRE_BigInt     total_global_cpts;
   HYPRE_BigInt    *big_convert_offd = NULL;
   HYPRE_BigInt    *big_buf_data = NULL;
   HYPRE_Real      *w_row_sum;

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   fine_to_coarse = hypre_TAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_DEVICE);

   hypre_modmp_init_fine_to_coarse(n_fine, pass_marker, color, fine_to_coarse);

   if (num_procs > 1)
   {
      HYPRE_BigInt big_Fpts = num_points;

      hypre_MPI_Scan(&big_Fpts, f_pts_starts + 1, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

      f_pts_starts[0] = f_pts_starts[1] - big_Fpts;

      if (my_id == num_procs - 1)
      {
         total_global_fpts = f_pts_starts[1];
         total_global_cpts = c_pts_starts[1];
      }
      hypre_MPI_Bcast(&total_global_fpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      f_pts_starts[0] = 0;
      f_pts_starts[1] = num_points;
      total_global_fpts = f_pts_starts[1];
      total_global_cpts = c_pts_starts[1];
   }

   num_cols_offd_Q = 0;

   if (num_procs > 1)
   {
      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      HYPRE_Int num_elem_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

      big_convert_offd = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_A, HYPRE_MEMORY_DEVICE);
      big_buf_data     = hypre_TAlloc(HYPRE_BigInt, num_elem_send,   HYPRE_MEMORY_DEVICE);

      globalC_functor functor(c_pts_starts[0]);

#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                        hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                        oneapi::dpl::make_transform_iterator(fine_to_coarse, functor),
                        big_buf_data );
#else
      HYPRE_THRUST_CALL( gather,
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                         thrust::make_transform_iterator(fine_to_coarse, functor),
                         big_buf_data );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
      /* RL: make sure big_buf_data is ready before issuing GPU-GPU MPI */
      if (hypre_GetGpuAwareMPI())
      {
         hypre_ForceSyncComputeStream(hypre_handle());
      }
#endif

      comm_handle = hypre_ParCSRCommHandleCreate_v2(21, comm_pkg, HYPRE_MEMORY_DEVICE, big_buf_data,
                                                    HYPRE_MEMORY_DEVICE, big_convert_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);

      // This will allocate fine_to_coarse_offd_dev
      hypre_modmp_compute_num_cols_offd_fine_to_coarse( pass_marker_offd, color, num_cols_offd_A,
                                                        num_cols_offd_Q, &fine_to_coarse_offd );

      //FIXME: PB: It seems we need the host value too?!?!
      col_map_offd_Q     = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_Q, HYPRE_MEMORY_HOST);
      col_map_offd_Q_dev = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_Q, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      HYPRE_BigInt *col_map_end = hypreSycl_copy_if( big_convert_offd,
                                                     big_convert_offd + num_cols_offd_A,
                                                     pass_marker_offd,
                                                     col_map_offd_Q_dev,
                                                     equal<HYPRE_Int>(color) );
#else
      HYPRE_BigInt *col_map_end = HYPRE_THRUST_CALL( copy_if,
                                                     big_convert_offd,
                                                     big_convert_offd + num_cols_offd_A,
                                                     pass_marker_offd,
                                                     col_map_offd_Q_dev,
                                                     equal<HYPRE_Int>(color) );
#endif

      hypre_assert(num_cols_offd_Q == col_map_end - col_map_offd_Q_dev);

      //FIXME: PB: It seems like we're required to have a host version of this??
      hypre_TMemcpy(col_map_offd_Q, col_map_offd_Q_dev, HYPRE_BigInt, num_cols_offd_Q, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);

      hypre_TFree(big_convert_offd, HYPRE_MEMORY_DEVICE );
      hypre_TFree(big_buf_data, HYPRE_MEMORY_DEVICE);
   }

   Q_diag_i = hypre_TAlloc(HYPRE_Int, num_points + 1, HYPRE_MEMORY_DEVICE);
   Q_offd_i = hypre_TAlloc(HYPRE_Int, num_points + 1, HYPRE_MEMORY_DEVICE);

   /* generate Q_diag_i and Q_offd_i */
   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_points, "warp", bDim);

      HYPRE_GPU_LAUNCH( hypreGPUKernel_generate_Pdiag_i_Poffd_i, gDim, bDim,
                        num_points, color, pass_order, pass_marker, pass_marker_offd,
                        S_diag_i, S_diag_j, S_offd_i, S_offd_j,
                        Q_diag_i, Q_offd_i );

      hypre_Memset(Q_diag_i + num_points, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);
      hypre_Memset(Q_offd_i + num_points, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
      /* WM: todo - this is a workaround since oneDPL's exclusive_scan gives incorrect results when doing the scan in place */
      auto zip4 = oneapi::dpl::make_zip_iterator( Q_diag_i, Q_offd_i );
      HYPRE_Int *Q_diag_i_tmp = hypre_CTAlloc(HYPRE_Int, num_points + 1, HYPRE_MEMORY_DEVICE);
      HYPRE_Int *Q_offd_i_tmp = hypre_CTAlloc(HYPRE_Int, num_points + 1, HYPRE_MEMORY_DEVICE);
      HYPRE_ONEDPL_CALL( std::exclusive_scan,
                         zip4,
                         zip4 + num_points + 1,
                         oneapi::dpl::make_zip_iterator( Q_diag_i_tmp, Q_offd_i_tmp ),
                         std::make_tuple(HYPRE_Int(0), HYPRE_Int(0)),
                         tuple_plus<HYPRE_Int>() );
      hypre_TMemcpy(Q_diag_i, Q_diag_i_tmp, HYPRE_Int, num_points + 1, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(Q_offd_i, Q_offd_i_tmp, HYPRE_Int, num_points + 1, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_DEVICE);
      hypre_TFree(Q_diag_i_tmp, HYPRE_MEMORY_DEVICE);
      hypre_TFree(Q_offd_i_tmp, HYPRE_MEMORY_DEVICE);
#else
      HYPRE_THRUST_CALL( exclusive_scan,
                         thrust::make_zip_iterator( thrust::make_tuple(Q_diag_i, Q_offd_i) ),
                         thrust::make_zip_iterator( thrust::make_tuple(Q_diag_i, Q_offd_i) ) + num_points + 1,
                         thrust::make_zip_iterator( thrust::make_tuple(Q_diag_i, Q_offd_i) ),
                         thrust::make_tuple(HYPRE_Int(0), HYPRE_Int(0)),
                         tuple_plus<HYPRE_Int>() );
#endif

      hypre_TMemcpy(&nnz_diag, &Q_diag_i[num_points], HYPRE_Int, 1, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(&nnz_offd, &Q_offd_i[num_points], HYPRE_Int, 1, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);
   }

   /* generate P_diag_j/data and P_offd_j/data */
   Q_diag_j    = hypre_TAlloc(HYPRE_Int,  nnz_diag,   HYPRE_MEMORY_DEVICE);
   Q_diag_data = hypre_TAlloc(HYPRE_Real, nnz_diag,   HYPRE_MEMORY_DEVICE);
   Q_offd_j    = hypre_TAlloc(HYPRE_Int,  nnz_offd,   HYPRE_MEMORY_DEVICE);
   Q_offd_data = hypre_TAlloc(HYPRE_Real, nnz_offd,   HYPRE_MEMORY_DEVICE);
   w_row_sum   = hypre_TAlloc(HYPRE_Real, num_points, HYPRE_MEMORY_DEVICE);

   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_points, "warp", bDim);

      HYPRE_GPU_LAUNCH( hypreGPUKernel_generate_Qdiag_j_Qoffd_j, gDim, bDim,
                        num_points,
                        color,
                        pass_order,
                        pass_marker,
                        pass_marker_offd,
                        fine_to_coarse,
                        fine_to_coarse_offd,
                        A_diag_i,
                        A_diag_j,
                        A_diag_data,
                        A_offd_i,
                        A_offd_j,
                        A_offd_data,
                        Soc_diag_j,
                        Soc_offd_j,
                        Q_diag_i,
                        Q_offd_i,
                        Q_diag_j,
                        Q_diag_data,
                        Q_offd_j,
                        Q_offd_data,
                        w_row_sum,
                        num_functions,
                        dof_func,
                        dof_func_offd );
   }

   hypre_TFree(fine_to_coarse,      HYPRE_MEMORY_DEVICE);
   hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_DEVICE);

   Q = hypre_ParCSRMatrixCreate(comm,
                                total_global_fpts,
                                total_global_cpts,
                                f_pts_starts,
                                c_pts_starts,
                                num_cols_offd_Q,
                                nnz_diag,
                                nnz_offd);

   Q_diag = hypre_ParCSRMatrixDiag(Q);
   hypre_CSRMatrixData(Q_diag) = Q_diag_data;
   hypre_CSRMatrixI(Q_diag)    = Q_diag_i;
   hypre_CSRMatrixJ(Q_diag)    = Q_diag_j;

   Q_offd = hypre_ParCSRMatrixOffd(Q);
   hypre_CSRMatrixData(Q_offd) = Q_offd_data;
   hypre_CSRMatrixI(Q_offd)    = Q_offd_i;
   hypre_CSRMatrixJ(Q_offd)    = Q_offd_j;

   hypre_ParCSRMatrixColMapOffd(Q) = col_map_offd_Q;
   hypre_ParCSRMatrixDeviceColMapOffd(Q) = col_map_offd_Q_dev;

   hypre_CSRMatrixMemoryLocation(Q_diag) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(Q_offd) = HYPRE_MEMORY_DEVICE;

   hypre_MatvecCommPkgCreate(Q);

   Pi = hypre_ParCSRMatMat(Q, P);

   Pi_diag = hypre_ParCSRMatrixDiag(Pi);
   Pi_diag_data = hypre_CSRMatrixData(Pi_diag);
   Pi_diag_i = hypre_CSRMatrixI(Pi_diag);
   Pi_offd = hypre_ParCSRMatrixOffd(Pi);
   Pi_offd_data = hypre_CSRMatrixData(Pi_offd);
   Pi_offd_i = hypre_CSRMatrixI(Pi_offd);

   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_points, "warp", bDim);

      HYPRE_GPU_LAUNCH( hypreGPUKernel_mutli_pi_rowsum, gDim, bDim,
                        num_points, pass_order, A_diag_i, A_diag_data,
                        Pi_diag_i, Pi_diag_data, Pi_offd_i, Pi_offd_data,
                        w_row_sum );
   }

   hypre_TFree(w_row_sum, HYPRE_MEMORY_DEVICE);

   hypre_ParCSRMatrixDestroy(Q);

   *Pi_ptr = Pi;

   return hypre_error_flag;
}

void hypre_modmp_init_fine_to_coarse( HYPRE_Int  n_fine,
                                      HYPRE_Int *pass_marker,
                                      HYPRE_Int  color,
                                      HYPRE_Int *fine_to_coarse )
{
   // n_fine == pass_marker.size()
   // Host code this is replacing:
   // n_cpts = 0;
   // for (HYPRE_Int i=0; i < n_fine; i++)
   //  {
   //    if (pass_marker[i] == color)
   //      fine_to_coarse[i] = n_cpts++;
   //    else
   //      fine_to_coarse[i] = -1;
   //  }

   if (n_fine == 0)
   {
      return;
   }

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(pass_marker,          equal<HYPRE_Int>(color)),
                      oneapi::dpl::make_transform_iterator(pass_marker + n_fine, equal<HYPRE_Int>(color)),
                      fine_to_coarse,
                      HYPRE_Int(0) );

   hypreSycl_transform_if( fine_to_coarse,
                           fine_to_coarse + n_fine,
                           pass_marker,
                           fine_to_coarse,
   [] (const auto & x) {return -1;},
   [color = color] (const auto & x) {return x != color;} );
#else
   HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(pass_marker,          equal<HYPRE_Int>(color)),
                      thrust::make_transform_iterator(pass_marker + n_fine, equal<HYPRE_Int>(color)),
                      fine_to_coarse,
                      HYPRE_Int(0) );

   HYPRE_THRUST_CALL( replace_if,
                      fine_to_coarse,
                      fine_to_coarse + n_fine,
                      pass_marker,
                      thrust::not1(equal<HYPRE_Int>(color)),
                      -1 );
#endif
}

void
hypre_modmp_compute_num_cols_offd_fine_to_coarse( HYPRE_Int  *pass_marker_offd,
                                                  HYPRE_Int   color,
                                                  HYPRE_Int   num_cols_offd_A,
                                                  HYPRE_Int  &num_cols_offd,
                                                  HYPRE_Int **fine_to_coarse_offd_ptr )
{
   // We allocate with a "+1" because the host version of this code incremented the counter
   // even on the last match, so we create an extra entry the exclusive_scan will reflect this
   // and we can read off the last entry and only do 1 kernel call and 1 memcpy
   // RL: this trick requires pass_marker_offd has 1 more space allocated too
   HYPRE_Int *fine_to_coarse_offd = hypre_TAlloc(HYPRE_Int, num_cols_offd_A + 1, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      oneapi::dpl::make_transform_iterator(pass_marker_offd,
                                                           equal<HYPRE_Int>(color)),
                      oneapi::dpl::make_transform_iterator(pass_marker_offd + num_cols_offd_A + 1,
                                                           equal<HYPRE_Int>(color)),
                      fine_to_coarse_offd,
                      HYPRE_Int(0) );
#else
   HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(pass_marker_offd,                       equal<HYPRE_Int>(color)),
                      thrust::make_transform_iterator(pass_marker_offd + num_cols_offd_A + 1, equal<HYPRE_Int>(color)),
                      fine_to_coarse_offd,
                      HYPRE_Int(0) );
#endif

   hypre_TMemcpy( &num_cols_offd, fine_to_coarse_offd + num_cols_offd_A, HYPRE_Int, 1,
                  HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   *fine_to_coarse_offd_ptr = fine_to_coarse_offd;
}

__global__
void hypreGPUKernel_cfmarker_masked_rowsum( hypre_DeviceItem    &item,
                                            HYPRE_Int      nrows,
                                            HYPRE_Int     *A_diag_i,
                                            HYPRE_Int     *A_diag_j,
                                            HYPRE_Complex *A_diag_data,
                                            HYPRE_Int     *A_offd_i,
                                            HYPRE_Int     *A_offd_j,
                                            HYPRE_Complex *A_offd_data,
                                            HYPRE_Int     *CF_marker,
                                            HYPRE_Int     *dof_func,
                                            HYPRE_Int     *dof_func_offd,
                                            HYPRE_Complex *row_sums )
{
   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows || read_only_load(&CF_marker[row_i]) >= 0)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p = 0;
   HYPRE_Int q = 0;
   HYPRE_Int func_i = dof_func ? read_only_load(&dof_func[row_i]) : 0;

   // A_diag part
   if (lane < 2)
   {
      p = read_only_load(A_diag_i + row_i + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   HYPRE_Complex row_sum_i = 0.0;

   // exclude diagonal: do not assume it is the first entry
   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      HYPRE_Int col = read_only_load(&A_diag_j[j]);

      if (row_i != col)
      {
         HYPRE_Int func_j = dof_func ? read_only_load(&dof_func[col]) : 0;

         if (func_i == func_j)
         {
            HYPRE_Complex value = read_only_load(&A_diag_data[j]);
            row_sum_i += value;
         }
      }
   }

   // A_offd part
   if (lane < 2)
   {
      p = read_only_load(A_offd_i + row_i + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      HYPRE_Int func_j = 0;
      if (dof_func_offd)
      {
         HYPRE_Int col = read_only_load(&A_offd_j[j]);
         func_j = read_only_load(&dof_func_offd[col]);
      }

      if (func_i == func_j)
      {
         HYPRE_Complex value = read_only_load(&A_offd_data[j]);
         row_sum_i += value;
      }
   }

   row_sum_i = warp_reduce_sum(item, row_sum_i);

   if (lane == 0)
   {
      row_sums[row_i] = row_sum_i;
   }
}

__global__
void hypreGPUKernel_mutli_pi_rowsum( hypre_DeviceItem    &item,
                                     HYPRE_Int      num_points,
                                     HYPRE_Int     *pass_order,
                                     HYPRE_Int     *A_diag_i,
                                     HYPRE_Complex *A_diag_data,
                                     HYPRE_Int     *Pi_diag_i,
                                     HYPRE_Complex *Pi_diag_data,
                                     HYPRE_Int     *Pi_offd_i,
                                     HYPRE_Complex *Pi_offd_data,
                                     HYPRE_Complex *w_row_sum )
{
   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= num_points)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p_diag = 0, q_diag = 0, p_offd = 0, q_offd = 0;
   HYPRE_Real row_sum_C = 0.0;

   // Pi_diag
   if (lane < 2)
   {
      p_diag = read_only_load(Pi_diag_i + row_i + lane);
   }
   q_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag, 0);

   for (HYPRE_Int j = p_diag + lane; j < q_diag; j += HYPRE_WARP_SIZE)
   {
      row_sum_C += read_only_load(&Pi_diag_data[j]);
   }

   // Pi_offd
   if (lane < 2)
   {
      p_offd = read_only_load(Pi_offd_i + row_i + lane);
   }
   q_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd, 0);

   for (HYPRE_Int j = p_offd + lane; j < q_offd; j += HYPRE_WARP_SIZE)
   {
      row_sum_C += read_only_load(&Pi_offd_data[j]);
   }

   row_sum_C = warp_reduce_sum(item, row_sum_C);

   if ( lane == 0 )
   {
      const HYPRE_Int i1 = read_only_load(&pass_order[row_i]);
      const HYPRE_Int j1 = read_only_load(&A_diag_i[i1]);
      //XXX RL: rely on diagonal is the first of row [FIX?]
      const HYPRE_Real diagonal = read_only_load(&A_diag_data[j1]);
      const HYPRE_Real value = row_sum_C * diagonal;
      row_sum_C += read_only_load(&w_row_sum[row_i]);

      if ( value != 0.0 )
      {
         row_sum_C /= value;
      }
   }

   row_sum_C = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, row_sum_C, 0);

   // Pi_diag
   for (HYPRE_Int j = p_diag + lane; j < q_diag; j += HYPRE_WARP_SIZE)
   {
      Pi_diag_data[j] *= -row_sum_C;
   }

   // Pi_offd
   for (HYPRE_Int j = p_offd + lane; j < q_offd; j += HYPRE_WARP_SIZE)
   {
      Pi_offd_data[j] *= -row_sum_C;
   }
}

__global__
void hypreGPUKernel_generate_Pdiag_i_Poffd_i( hypre_DeviceItem &item,
                                              HYPRE_Int  num_points,
                                              HYPRE_Int  color,
                                              HYPRE_Int *pass_order,
                                              HYPRE_Int *pass_marker,
                                              HYPRE_Int *pass_marker_offd,
                                              HYPRE_Int *S_diag_i,
                                              HYPRE_Int *S_diag_j,
                                              HYPRE_Int *S_offd_i,
                                              HYPRE_Int *S_offd_j,
                                              HYPRE_Int *P_diag_i,
                                              HYPRE_Int *P_offd_i )
{
   /*
    nnz_diag = 0;
    nnz_offd = 0;
    for (i=0; i < num_points; i++)
    {
      i1 = pass_order[i];
      for (j=S_diag_i[i1]; j < S_diag_i[i1+1]; j++)
      {
         j1 = S_diag_j[j];
         if (pass_marker[j1] == color)
         {
             P_diag_i[i]++;
             nnz_diag++;
         }
      }
      for (j=S_offd_i[i1]; j < S_offd_i[i1+1]; j++)
      {
         j1 = S_offd_j[j];
         if (pass_marker_offd[j1] == color)
         {
             P_offd_i[i]++;
             nnz_offd++;
         }
      }
    }
   */

   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= num_points)
   {
      return;
   }

   HYPRE_Int i1 = read_only_load(&pass_order[row_i]);
   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p = 0;
   HYPRE_Int q = 0;
   HYPRE_Int diag_increment = 0;
   HYPRE_Int offd_increment = 0;

   // S_diag
   if (lane < 2)
   {
      p = read_only_load(S_diag_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      const HYPRE_Int j1 = read_only_load(&S_diag_j[j]);
      const HYPRE_Int marker = read_only_load(&pass_marker[j1]);

      diag_increment += marker == color;
   }

   diag_increment = warp_reduce_sum(item, diag_increment);

   // Increment P_diag_i, but then we need to also do a block reduction
   // on diag_increment to log the total nnz_diag for the block
   // Then after the kernel, we'll accumulate nnz_diag for each block
   if (lane == 0)
   {
      P_diag_i[row_i] = diag_increment;
   }

   // S_offd
   if (lane < 2)
   {
      p = read_only_load(S_offd_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      const HYPRE_Int j1 = read_only_load(&S_offd_j[j]);
      const HYPRE_Int marker = read_only_load(&pass_marker_offd[j1]);

      offd_increment += marker == color;
   }

   offd_increment = warp_reduce_sum(item, offd_increment);

   // Increment P_offd_i, but then we need to also do a block reduction
   // on offd_increment to log the total nnz_offd for the block
   // Then after the kernel, we'll accumulate nnz_offd for each block
   if (lane == 0)
   {
      P_offd_i[row_i] = offd_increment;
   }
}

__global__
void hypreGPUKernel_generate_Pdiag_j_Poffd_j( hypre_DeviceItem    &item,
                                              HYPRE_Int      num_points,
                                              HYPRE_Int      color,
                                              HYPRE_Int     *pass_order,
                                              HYPRE_Int     *pass_marker,
                                              HYPRE_Int     *pass_marker_offd,
                                              HYPRE_Int     *fine_to_coarse,
                                              HYPRE_Int     *fine_to_coarse_offd,
                                              HYPRE_Int     *A_diag_i,
                                              HYPRE_Int     *A_diag_j,
                                              HYPRE_Complex *A_diag_data,
                                              HYPRE_Int     *A_offd_i,
                                              HYPRE_Int     *A_offd_j,
                                              HYPRE_Complex *A_offd_data,
                                              HYPRE_Int     *Soc_diag_j,
                                              HYPRE_Int     *Soc_offd_j,
                                              HYPRE_Int     *P_diag_i,
                                              HYPRE_Int     *P_offd_i,
                                              HYPRE_Int     *P_diag_j,
                                              HYPRE_Complex *P_diag_data,
                                              HYPRE_Int     *P_offd_j,
                                              HYPRE_Complex *P_offd_data,
                                              HYPRE_Complex *row_sums )
{
   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= num_points)
   {
      return;
   }

   HYPRE_Int i1 = read_only_load(&pass_order[row_i]);
   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p_diag_A = 0, q_diag_A, p_diag_P = 0, q_diag_P;
   HYPRE_Int k;
   HYPRE_Complex row_sum_C = 0.0, diagonal = 0.0;

   // S_diag
   if (lane < 2)
   {
      p_diag_A = read_only_load(A_diag_i + i1 + lane);
      p_diag_P = read_only_load(P_diag_i + row_i + lane);
   }
   q_diag_A = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag_A, 1);
   p_diag_A = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag_A, 0);
   q_diag_P = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag_P, 1);
   p_diag_P = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag_P, 0);

   k = p_diag_P;
   for (HYPRE_Int j = p_diag_A + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < q_diag_A);
        j += HYPRE_WARP_SIZE)
   {
      HYPRE_Int equal = 0;
      HYPRE_Int sum = 0;
      HYPRE_Int j1 = -1;

      if ( j < q_diag_A )
      {
         j1 = read_only_load(&Soc_diag_j[j]);
         equal = j1 > -1 && read_only_load(&pass_marker[j1]) == color;
      }

      HYPRE_Int pos = warp_prefix_sum(item, lane, equal, sum);

      if (equal)
      {
         P_diag_j[k + pos] = read_only_load(&fine_to_coarse[j1]);
         HYPRE_Complex val = read_only_load(&A_diag_data[j]);
         P_diag_data[k + pos] = val;
         row_sum_C += val;
      }

      if (j1 == -2)
      {
         diagonal = read_only_load(&A_diag_data[j]);
      }

      k += sum;
   }

   hypre_device_assert(k == q_diag_P);

   // S_offd
   HYPRE_Int p_offd_A = 0, q_offd_A, p_offd_P = 0, q_offd_P;

   if (lane < 2)
   {
      p_offd_A = read_only_load(A_offd_i + i1 + lane);
      p_offd_P = read_only_load(P_offd_i + row_i + lane);
   }
   q_offd_A = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd_A, 1);
   p_offd_A = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd_A, 0);
   q_offd_P = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd_P, 1);
   p_offd_P = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd_P, 0);

   k = p_offd_P;
   for (HYPRE_Int j = p_offd_A + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < q_offd_A);
        j += HYPRE_WARP_SIZE)
   {
      HYPRE_Int equal = 0;
      HYPRE_Int sum = 0;
      HYPRE_Int j1 = -1;

      if ( j < q_offd_A )
      {
         j1 = read_only_load(&Soc_offd_j[j]);
         equal = j1 > -1 && read_only_load(&pass_marker_offd[j1]) == color;
      }

      HYPRE_Int pos = warp_prefix_sum(item, lane, equal, sum);

      if (equal)
      {
         P_offd_j[k + pos] = read_only_load(&fine_to_coarse_offd[j1]);
         HYPRE_Complex val = read_only_load(&A_offd_data[j]);
         P_offd_data[k + pos] = val;
         row_sum_C += val;
      }

      k += sum;
   }

   hypre_device_assert(k == q_offd_P);

   row_sum_C = warp_reduce_sum(item, row_sum_C);
   diagonal = warp_reduce_sum(item, diagonal);
   HYPRE_Complex value = row_sum_C * diagonal;
   HYPRE_Complex row_sum_i = 0.0;

   if (lane == 0)
   {
      row_sum_i = read_only_load(&row_sums[i1]);

      if (value)
      {
         row_sum_i /= value;
         row_sums[i1] = row_sum_i;
      }
   }

   row_sum_i = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, row_sum_i, 0);

   for (HYPRE_Int j = p_diag_P + lane; j < q_diag_P; j += HYPRE_WARP_SIZE)
   {
      P_diag_data[j] = -P_diag_data[j] * row_sum_i;
   }

   for (HYPRE_Int j = p_offd_P + lane; j < q_offd_P; j += HYPRE_WARP_SIZE)
   {
      P_offd_data[j] = -P_offd_data[j] * row_sum_i;
   }
}

__global__
void hypreGPUKernel_insert_remaining_weights( hypre_DeviceItem &item,
                                              HYPRE_Int   start,
                                              HYPRE_Int   stop,
                                              HYPRE_Int  *pass_order,
                                              HYPRE_Int  *Pi_diag_i,
                                              HYPRE_Int  *Pi_diag_j,
                                              HYPRE_Real *Pi_diag_data,
                                              HYPRE_Int  *P_diag_i,
                                              HYPRE_Int  *P_diag_j,
                                              HYPRE_Real *P_diag_data,
                                              HYPRE_Int  *Pi_offd_i,
                                              HYPRE_Int  *Pi_offd_j,
                                              HYPRE_Real *Pi_offd_data,
                                              HYPRE_Int  *P_offd_i,
                                              HYPRE_Int  *P_offd_j,
                                              HYPRE_Real *P_offd_data )
{
   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= stop - start)
   {
      return;
   }

   HYPRE_Int i1 = read_only_load(&pass_order[row_i + start]);
   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p = 0;
   HYPRE_Int q = 0;
   HYPRE_Int i2;

   // P_diag
   if (lane < 2)
   {
      p = read_only_load(P_diag_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   i2 = read_only_load(&Pi_diag_i[row_i]) - p;
   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      P_diag_j[j] = Pi_diag_j[j + i2];
      P_diag_data[j] = Pi_diag_data[j + i2];
   }

   // P_offd
   if (lane < 2)
   {
      p = read_only_load(P_offd_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   i2 = read_only_load(&Pi_offd_i[row_i]) - p;
   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      P_offd_j[j] = Pi_offd_j[j + i2];
      P_offd_data[j] = Pi_offd_data[j + i2];
   }
}


__global__
void hypreGPUKernel_generate_Qdiag_j_Qoffd_j( hypre_DeviceItem    &item,
                                              HYPRE_Int      num_points,
                                              HYPRE_Int      color,
                                              HYPRE_Int     *pass_order,
                                              HYPRE_Int     *pass_marker,
                                              HYPRE_Int     *pass_marker_offd,
                                              HYPRE_Int     *fine_to_coarse,
                                              HYPRE_Int     *fine_to_coarse_offd,
                                              HYPRE_Int     *A_diag_i,
                                              HYPRE_Int     *A_diag_j,
                                              HYPRE_Complex *A_diag_data,
                                              HYPRE_Int     *A_offd_i,
                                              HYPRE_Int     *A_offd_j,
                                              HYPRE_Complex *A_offd_data,
                                              HYPRE_Int     *Soc_diag_j,
                                              HYPRE_Int     *Soc_offd_j,
                                              HYPRE_Int     *Q_diag_i,
                                              HYPRE_Int     *Q_offd_i,
                                              HYPRE_Int     *Q_diag_j,
                                              HYPRE_Complex *Q_diag_data,
                                              HYPRE_Int     *Q_offd_j,
                                              HYPRE_Complex *Q_offd_data,
                                              HYPRE_Complex *w_row_sum,
                                              HYPRE_Int      num_functions,
                                              HYPRE_Int     *dof_func,
                                              HYPRE_Int     *dof_func_offd )
{
   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= num_points)
   {
      return;
   }

   HYPRE_Int i1 = read_only_load(&pass_order[row_i]);
   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p_diag_A = 0, q_diag_A, p_diag_P = 0;
#ifdef HYPRE_DEBUG
   HYPRE_Int q_diag_P;
#endif
   HYPRE_Int k;
   HYPRE_Complex w_row_sum_i = 0.0;
   HYPRE_Int dof_func_i1 = -1;

   if (num_functions > 1)
   {
      if (lane == 0)
      {
         dof_func_i1 = read_only_load(&dof_func[i1]);
      }
      dof_func_i1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, dof_func_i1, 0);
   }

   // S_diag
#ifdef HYPRE_DEBUG
   if (lane < 2)
   {
      p_diag_A = read_only_load(A_diag_i + i1 + lane);
      p_diag_P = read_only_load(Q_diag_i + row_i + lane);
   }
   q_diag_A = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag_A, 1);
   p_diag_A = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag_A, 0);
   q_diag_P = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag_P, 1);
   p_diag_P = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag_P, 0);
#else
   if (lane < 2)
   {
      p_diag_A = read_only_load(A_diag_i + i1 + lane);
   }
   q_diag_A = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag_A, 1);
   p_diag_A = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag_A, 0);
   if (lane == 0)
   {
      p_diag_P = read_only_load(Q_diag_i + row_i);
   }
   p_diag_P = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag_P, 0);
#endif

   k = p_diag_P;
   for (HYPRE_Int j = p_diag_A + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < q_diag_A);
        j += HYPRE_WARP_SIZE)
   {
      HYPRE_Int equal = 0;
      HYPRE_Int sum = 0;
      HYPRE_Int j1 = -1;

      if ( j < q_diag_A )
      {
         j1 = read_only_load(&Soc_diag_j[j]);
         equal = j1 > -1 && read_only_load(&pass_marker[j1]) == color;
      }

      HYPRE_Int pos = warp_prefix_sum(item, lane, equal, sum);

      if (equal)
      {
         Q_diag_j[k + pos] = read_only_load(&fine_to_coarse[j1]);
         Q_diag_data[k + pos] = read_only_load(&A_diag_data[j]);
      }
      else if (j < q_diag_A && j1 != -2)
      {
         if (num_functions > 1)
         {
            const HYPRE_Int col = read_only_load(&A_diag_j[j]);
            if ( dof_func_i1 == read_only_load(&dof_func[col]) )
            {
               w_row_sum_i += read_only_load(&A_diag_data[j]);
            }
         }
         else
         {
            w_row_sum_i += read_only_load(&A_diag_data[j]);
         }
      }

      k += sum;
   }

#ifdef HYPRE_DEBUG
   hypre_device_assert(k == q_diag_P);
#endif

   // S_offd
   HYPRE_Int p_offd_A = 0, q_offd_A, p_offd_P = 0;
#ifdef HYPRE_DEBUG
   HYPRE_Int q_offd_P;
#endif

#ifdef HYPRE_DEBUG
   if (lane < 2)
   {
      p_offd_A = read_only_load(A_offd_i + i1 + lane);
      p_offd_P = read_only_load(Q_offd_i + row_i + lane);
   }
   q_offd_A = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd_A, 1);
   p_offd_A = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd_A, 0);
   q_offd_P = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd_P, 1);
   p_offd_P = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd_P, 0);
#else
   if (lane < 2)
   {
      p_offd_A = read_only_load(A_offd_i + i1 + lane);
   }
   q_offd_A = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd_A, 1);
   p_offd_A = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd_A, 0);
   if (lane == 0)
   {
      p_offd_P = read_only_load(Q_offd_i + row_i);
   }
   p_offd_P = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd_P, 0);
#endif

   k = p_offd_P;
   for (HYPRE_Int j = p_offd_A + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < q_offd_A);
        j += HYPRE_WARP_SIZE)
   {
      HYPRE_Int equal = 0;
      HYPRE_Int sum = 0;
      HYPRE_Int j1 = -1;

      if ( j < q_offd_A )
      {
         j1 = read_only_load(&Soc_offd_j[j]);
         equal = j1 > -1 && read_only_load(&pass_marker_offd[j1]) == color;
      }

      HYPRE_Int pos = warp_prefix_sum(item, lane, equal, sum);

      if (equal)
      {
         Q_offd_j[k + pos] = read_only_load(&fine_to_coarse_offd[j1]);
         Q_offd_data[k + pos] = read_only_load(&A_offd_data[j]);
      }
      else if (j < q_offd_A)
      {
         if (num_functions > 1)
         {
            const HYPRE_Int col = read_only_load(&A_offd_j[j]);
            if ( dof_func_i1 == read_only_load(&dof_func_offd[col]) )
            {
               w_row_sum_i += read_only_load(&A_offd_data[j]);
            }
         }
         else
         {
            w_row_sum_i += read_only_load(&A_offd_data[j]);
         }
      }

      k += sum;
   }

#ifdef HYPRE_DEBUG
   hypre_device_assert(k == q_offd_P);
#endif

   w_row_sum_i = warp_reduce_sum(item, w_row_sum_i);

   if (lane == 0)
   {
      w_row_sum[row_i] = w_row_sum_i;
   }
}

__global__
void hypreGPUKernel_pass_order_count( hypre_DeviceItem &item,
                                      HYPRE_Int  num_points,
                                      HYPRE_Int  color,
                                      HYPRE_Int *points_left,
                                      HYPRE_Int *pass_marker,
                                      HYPRE_Int *pass_marker_offd,
                                      HYPRE_Int *S_diag_i,
                                      HYPRE_Int *S_diag_j,
                                      HYPRE_Int *S_offd_i,
                                      HYPRE_Int *S_offd_j,
                                      HYPRE_Int *diag_shifts )
{
   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= num_points)
   {
      return;
   }

   HYPRE_Int i1 = read_only_load(&points_left[row_i]);
   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p = 0;
   HYPRE_Int q = 0;
   hypre_int brk = 0;

   // S_diag
   if (lane < 2)
   {
      p = read_only_load(S_diag_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      if (j < q)
      {
         HYPRE_Int j1 = read_only_load(&S_diag_j[j]);
         if ( read_only_load(&pass_marker[j1]) == color )
         {
            brk = 1;
         }
      }

      brk = warp_any_sync(item, HYPRE_WARP_FULL_MASK, brk);

      if (brk)
      {
         break;
      }
   }

   if (brk)
   {
      // Only one thread can increment because of the break
      // so we just need to increment by 1
      if (lane == 0)
      {
         diag_shifts[row_i] = 1;
      }

      return;
   }

   // S_offd
   if (lane < 2)
   {
      p = read_only_load(S_offd_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      if (j < q)
      {
         HYPRE_Int j1 = read_only_load(&S_offd_j[j]);
         if ( read_only_load(&pass_marker_offd[j1]) == color )
         {
            brk = 1;
         }
      }

      brk = warp_any_sync(item, HYPRE_WARP_FULL_MASK, brk);

      if (brk)
      {
         break;
      }
   }

   // Only one thread can increment because of the break
   // so we just need to increment by 1
   if (lane == 0)
   {
      diag_shifts[row_i] = (brk != 0);
   }
}

__global__
void hypreGPUKernel_populate_big_P_offd_j( hypre_DeviceItem   &item,
                                           HYPRE_Int     start,
                                           HYPRE_Int     stop,
                                           HYPRE_Int    *pass_order,
                                           HYPRE_Int    *P_offd_i,
                                           HYPRE_Int    *P_offd_j,
                                           HYPRE_BigInt *col_map_offd_Pi,
                                           HYPRE_BigInt *big_P_offd_j )
{
   HYPRE_Int i = hypre_gpu_get_grid_warp_id<1, 1>(item) + start;

   if (i >= stop)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int i1 = read_only_load(&pass_order[i]);
   HYPRE_Int p = 0;
   HYPRE_Int q = 0;

   if (lane < 2)
   {
      p = read_only_load(P_offd_i + i1 + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      HYPRE_Int col = read_only_load(&P_offd_j[j]);
      big_P_offd_j[j] = read_only_load(&col_map_offd_Pi[col]);
   }
}

#endif // defined(HYPRE_USING_GPU)

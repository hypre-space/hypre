/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_SYCL_UTILS_HPP
#define HYPRE_SYCL_UTILS_HPP

#if defined(HYPRE_USING_SYCL)

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>

#define HYPRE_SYCL_CALL( EXPR )                                     \
  try {                                                             \
    EXPR;                                                           \
  }                                                                 \
  catch (sycl::exception const &ex) {                           \
    hypre_printf("SYCL ERROR (code = %s) at %s:%d\n", ex.what(),    \
                   __FILE__, __LINE__);                             \
    assert(0); exit(1);						    \
  }                                                                 \
  catch(std::runtime_error const& ex) {                             \
    hypre_printf("STD ERROR (code = %s) at %s:%d\n", ex.what(),     \
                   __FILE__, __LINE__);                             \
    assert(0); exit(1);						    \
  }

// HYPRE_WARP_BITSHIFT is just log2 of HYPRE_WARP_SIZE
#define HYPRE_SUBGROUP_SIZE       32
//#define HYPRE_WARP_BITSHIFT   5 // abb needs to be removed
#define HYPRE_MAX_NUM_SUBGROUPS   (64 * 64 * 32)
#define HYPRE_FLT_LARGE       1e30
#define HYPRE_1D_BLOCK_SIZE   512
#define HYPRE_MAX_NUM_STREAMS 10

struct hypre_SyclData
{
   oneapi::mkl::rng::philox4x32x10*  onemklrng_generator=nullptr;
   oneapi::mkl::index_base           cusparse_mat_descr;
   sycl::queue*                      sycl_queues[HYPRE_MAX_NUM_STREAMS] = {};
   sycl::device                      sycl_device;

   /* by default, hypre puts GPU computations in this queue
    * Do not be confused with the default (null) SYCL queue */
   HYPRE_Int                         sycl_compute_queue_num;
   /* work space for hypre's SYCL reducer */
   void                             *sycl_reduce_buffer;
   /* the device buffers needed to do MPI communication for struct comm */
   HYPRE_Complex*                    struct_comm_recv_buffer;
   HYPRE_Complex*                    struct_comm_send_buffer;
   HYPRE_Int                         struct_comm_recv_buffer_size;
   HYPRE_Int                         struct_comm_send_buffer_size;
   /* device spgemm options */
   HYPRE_Int                         spgemm_use_onemklsparse;
   HYPRE_Int                         spgemm_num_passes;
   HYPRE_Int                         spgemm_rownnz_estimate_method;
   HYPRE_Int                         spgemm_rownnz_estimate_nsamples;
   float                             spgemm_rownnz_estimate_mult_factor;
   char                              spgemm_hash_type;
};

#define hypre_SyclDataSyclDevice(data)                     ((data) -> sycl_device)
#define hypre_SyclDataSyclComputeQueueNum(data)           ((data) -> sycl_compute_queue_num)
#define hypre_SyclDataSyclReduceBuffer(data)               ((data) -> sycl_reduce_buffer)
#define hypre_SyclDataStructCommRecvBuffer(data)           ((data) -> struct_comm_recv_buffer)
#define hypre_SyclDataStructCommSendBuffer(data)           ((data) -> struct_comm_send_buffer)
#define hypre_SyclDataStructCommRecvBufferSize(data)       ((data) -> struct_comm_recv_buffer_size)
#define hypre_SyclDataStructCommSendBufferSize(data)       ((data) -> struct_comm_send_buffer_size)
#define hypre_SyclDataSpgemmUseOnemklsparse(data)          ((data) -> spgemm_use_onemklsparse)
#define hypre_SyclDataSpgemmNumPasses(data)                ((data) -> spgemm_num_passes)
#define hypre_SyclDataSpgemmRownnzEstimateMethod(data)     ((data) -> spgemm_rownnz_estimate_method)
#define hypre_SyclDataSpgemmRownnzEstimateNsamples(data)   ((data) -> spgemm_rownnz_estimate_nsamples)
#define hypre_SyclDataSpgemmRownnzEstimateMultFactor(data) ((data) -> spgemm_rownnz_estimate_mult_factor)
#define hypre_SyclDataSpgemmHashType(data)                 ((data) -> spgemm_hash_type)

hypre_SyclData* hypre_SyclDataCreate();
void hypre_SyclDataDestroy(hypre_SyclData* data);

oneapi::mkl::rng::philox4x32x10* hypre_SyclDataonemklrngGenerator(hypre_SyclData *data);
// oneapi::mkl::index_base hypre_SyclDataCusparseMatDescr(hypre_SyclData *data);
sycl::queue *hypre_SyclDataSyclQueue(hypre_SyclData *data, HYPRE_Int i);
sycl::queue *hypre_SyclDataSyclComputeQueue(hypre_SyclData *data);

// Data structure and accessor routines for Cuda Sparse Triangular Matrices
struct hypre_CsrsvData
{
   csrsv2Info_t info_L;
   csrsv2Info_t info_U;
   hypre_int    BufferSize;
   char        *Buffer;
};

#define hypre_CsrsvDataInfoL(data)      ((data) -> info_L)
#define hypre_CsrsvDataInfoU(data)      ((data) -> info_U)
#define hypre_CsrsvDataBufferSize(data) ((data) -> BufferSize)
#define hypre_CsrsvDataBuffer(data)     ((data) -> Buffer)

#endif //#if defined(HYPRE_USING_SYCL)

#if defined(HYPRE_USING_SYCL)

// for includes of PSTL algorithms
#define PSTL_USE_PARALLEL_POLICIES 0

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/functional>

#include <dpct/dpl_extras/algorithm.h> // dpct::remove_if, iota, remove_copy_if, copy_if

#include <algorithm>
#include <functional>
#include <iterator>

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 * macro for launching SYCL kernels, DPL, onemkl::blas::sparse, onemkl::rng calls
 *                    NOTE: IN HYPRE'S DEFAULT QUEUE
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 */

#ifdef HYPRE_DEBUG                                                                                                   \

#define HYPRE_SYCL_3D_LAUNCH(kernel_name, gridsize, blocksize, ...)                                                  \
{                                                                                                                    \
   if ( gridsize[2] == 0 || gridsize[1] == 0 || gridsize[0] == 0 ||                                                  \
        blocksize[2] == 0 || blocksize[1] == 0 || blocksize[0] == 0 )                                                \
   {                                                                                                                 \
     hypre_printf("Error %s %d: Invalid SYCL 3D launch parameters grid/block (%d, %d, %d) (%d, %d, %d)\n",           \
		  __FILE__, __LINE__,					                                             \
		  gridsize[0], gridsize[1], gridsize[2], blocksize[0], blocksize[1], blocksize[2]);                  \
     assert(0); exit(1);                                                                                             \
   }                                                                                                                 \
   else                                                                                                              \
   {                                                                                                                 \
      hypre_HandleSyclComputeQueue(hypre_handle())->submit([&] (sycl::handler& cgh) [[cl::intel_reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] { \
	   cgh.parallel_for(sycl::nd_range<3>(gridSize*blocksize, blocksize), [=] (sycl::nd_item<3> item) {  \
	      (kernel_name)(item, __VA_ARGS__);                                                                      \
         });                                                                                                         \
      }).wait_and_throw();                                                                                           \
   }                                                                                                                 \
}

#define HYPRE_SYCL_2D_LAUNCH(kernel_name, gridsize, blocksize, ...)                                                  \
{                                                                                                                    \
   if ( gridsize[1] == 0 || gridsize[0] == 0 || blocksize[1] == 0 || blocksize[0] == 0 )                             \
   {                                                                                                                 \
     hypre_printf("Error %s %d: Invalid SYCL 2D launch parameters grid/block (%d, %d) (%d, %d)\n",                   \
		  __FILE__, __LINE__,					                                             \
		  gridsize[0], gridsize[1], blocksize[0], blocksize[1]);                                             \
     assert(0); exit(1);                                                                                             \
   }                                                                                                                 \
   else                                                                                                              \
   {                                                                                                                 \
      hypre_HandleSyclComputeQueue(hypre_handle())->submit([&] (sycl::handler& cgh) [[cl::intel_reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] { \
	   cgh.parallel_for(sycl::nd_range<2>(gridSize*blocksize, blocksize), [=] (sycl::nd_item<2> item) {  \
	      (kernel_name)(item, __VA_ARGS__);                                                                      \
         });                                                                                                         \
      }).wait_and_throw();                                                                                           \
   }                                                                                                                 \
}

#define HYPRE_SYCL_1D_LAUNCH(kernel_name, gridsize, blocksize, ...)                                                  \
{                                                                                                                    \
   if ( gridsize[0] == 0 || blocksize[0] == 0 )                                                                      \
   {                                                                                                                 \
     hypre_printf("Error %s %d: Invalid SYCL 1D launch parameters grid/block (%d) (%d)\n",                           \
		  __FILE__, __LINE__,					                                             \
		  gridsize[0], blocksize[0]);                              	      			             \
     assert(0); exit(1);                                                                                             \
   }                                                                                                                 \
   else                                                                                                              \
   {                                                                                                                 \
      hypre_HandleSyclComputeQueue(hypre_handle())->submit([&] (sycl::handler& cgh) [[cl::intel_reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] { \
	   cgh.parallel_for(sycl::nd_range<1>(gridSize*blocksize, blocksize), [=] (sycl::nd_item<1> item) {  \
	      (kernel_name)(item, __VA_ARGS__);                                                                      \
         });                                                                                                         \
      }).wait_and_throw();                                                                                           \      
   }                                                                                                                 \
}

#else

#define HYPRE_SYCL_3D_LAUNCH(kernel_name, gridsize, blocksize, ...)                                                  \
{                                                                                                                    \
   if ( gridsize[2] == 0 || gridsize[1] == 0 || gridsize[0] == 0 ||                                                  \
        blocksize[2] == 0 || blocksize[1] == 0 || blocksize[0] == 0 )                                                \
   {                                                                                                                 \
     hypre_printf("Error %s %d: Invalid SYCL 3D launch parameters grid/block (%d, %d, %d) (%d, %d, %d)\n",           \
		  __FILE__, __LINE__,					                                             \
		  gridsize[0], gridsize[1], gridsize[2], blocksize[0], blocksize[1], blocksize[2]);                  \
     assert(0); exit(1);                                                                                             \
   }                                                                                                                 \
   else                                                                                                              \
   {                                                                                                                 \
      hypre_HandleSyclComputeQueue(hypre_handle())->submit([&] (sycl::handler& cgh) [[cl::intel_reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] { \
	  cgh.parallel_for(sycl::nd_range<3>(gridSize*blocksize, blocksize), [=] (sycl::nd_item<3> item) { \
	      (kernel_name)(item, __VA_ARGS__);				\
         });                                                                                                         \
      });                                                                                                            \
   }                                                                                                                 \
}

#define HYPRE_SYCL_2D_LAUNCH(kernel_name, gridsize, blocksize, ...)                                                  \
{                                                                                                                    \
   if ( gridsize[1] == 0 || gridsize[0] == 0 || blocksize[1] == 0 || blocksize[0] == 0 )                             \
   {                                                                                                                 \
     hypre_printf("Error %s %d: Invalid SYCL 2D launch parameters grid/block (%d, %d) (%d, %d)\n",                   \
		  __FILE__, __LINE__,					                                             \
		  gridsize[0], gridsize[1], blocksize[0], blocksize[1]);                                             \
     assert(0); exit(1);                                                                                             \
   }                                                                                                                 \
   else                                                                                                              \
   {                                                                                                                 \
      hypre_HandleSyclComputeQueue(hypre_handle())->submit([&] (sycl::handler& cgh) [[cl::intel_reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] { \
	   cgh.parallel_for(sycl::nd_range<2>(gridSize*blocksize, blocksize), [=] (sycl::nd_item<2> item) {  \
	      (kernel_name)(item, __VA_ARGS__);                                                                      \
         });                                                                                                         \
      });                                                                                                            \
   }                                                                                                                 \
}

#define HYPRE_SYCL_1D_LAUNCH(kernel_name, gridsize, blocksize, ...)                                                  \
{                                                                                                                    \
   if ( gridsize[0] == 0 || blocksize[0] == 0 )                                                                      \
   {                                                                                                                 \
     hypre_printf("Error %s %d: Invalid SYCL 1D launch parameters grid/block (%d) (%d)\n",                           \
		  __FILE__, __LINE__,					                                             \
		  gridsize[0], blocksize[0]);                              	      			             \
     assert(0); exit(1);                                                                                             \
   }                                                                                                                 \
   else                                                                                                              \
   {                                                                                                                 \
      hypre_HandleSyclComputeQueue(hypre_handle())->submit([&] (sycl::handler& cgh) [[cl::intel_reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] { \
	   cgh.parallel_for(sycl::nd_range<1>(gridSize*blocksize, blocksize), [=] (sycl::nd_item<1> item) {  \
	      (kernel_name)(item, __VA_ARGS__);                                                                      \
         });                                                                                                         \
      });                                                                                                            \
   }                                                                                                                 \
}

#endif // HYPRE_DEBUG

#define HYPRE_ONEDPL_CALL(func_name, ...)                                                                            \
  func_name(oneapi::dpl::execution::make_device_policy(*hypre_HandleSyclComputeQueue(hypre_handle())), __VA_ARGS__);

/* function for SYCL parallelized std::iota */
template <typename T>
struct iota_sequence_fun {
  using result_type = T;
  iota_sequence_fun(T _init, T _step) : init(_init), step(_step) {}

  template <typename _T> result_type operator()(_T &&i) const {
    return static_cast<T>(init + step * i);
  }

private:
  const T init;
  const T step;
};

template <class Iter, class T>
void sycl_iota(Iter first, Iter last, T init=0) {
  using DiffSize = typename std::iterator_traits<Iter>::difference_type;
  std::transform( oneapi::dpl::execution::make_device_policy(*hypre_HandleSyclComputeQueue(hypre_handle())),
		  oneapi::dpl::counting_iterator<DiffSize>(0),
		  oneapi::dpl::counting_iterator<DiffSize>(std::distance(first, last)),
		  first,
		  iota_sequence_fun<T>(init, T(1)) );
}

/* return the number of work-items in current work-group */
template <hypre_int dim>
static __inline__ __attribute__((always_inline))
hypre_int hypre_sycl_get_num_workitems(sycl::nd_item<dim>& item)
{
  return item.get_group().get_local_linear_range(); 
}

/* return the flattened or linearlized work-item id in current work-group (not global)*/
template <hypre_int dim>
static __inline__ __attribute__((always_inline))
hypre_int hypre_sycl_get_workitem_id(sycl::nd_item<dim>& item)
{
  return item.get_local_linear_id();
}

/* return the number of sub-groups in current work-group */
template <hypre_int dim>
static __inline__ __attribute__((always_inline))
hypre_int hypre_sycl_get_num_subgroups(sycl::nd_item<dim>& item)
{
  return item.get_sub_group().get_group_range().get(0);
}

/* return the sub_group id in work-group */
template <hypre_int dim>
static __inline__ __attribute__((always_inline))
hypre_int hypre_sycl_get_subgroup_id(sycl::nd_item<dim>& item)
{
  return item.get_sub_group().get_group_linear_id();
}

/* return the work-item lane id in a sub_group */
template <hypre_int dim>
static __inline__ __attribute__((always_inline))
hypre_int hypre_sycl_get_lane_id(sycl::nd_item<dim>& item)
{
  return hypre_sycl_get_workitem_id<dim>(item) &
    (item.get_sub_group().get_local_range().get(0)-1);
}

/* return the num of work_groups in nd_range */
template <hypre_int dim>
static __inline__ __attribute__((always_inline))
hypre_int hypre_sycl_get_num_workgroups(sycl::nd_item<dim>& item)
{
  // return item.get_group().get_group_linear_range(); // API available in SYCL 2020
  
  switch (dim)
  {
  case 1:
    return (item.get_group_range(0));
  case 2:
    return (item.get_group_range(0) * item.get_group_range(1));
  case 3:
    return (item.get_group_range(0) * item.get_group_range(1) * item.get_group_range(2));
  }

  return -1;
}

/* return the flattened or linearlized work-group id in nd_range */
template <hypre_int dim>
static __inline__ __attribute__((always_inline))
hypre_int hypre_sycl_get_workgroup_id(sycl::nd_item<dim>& item)
{
  return item.get_group_linear_id();
}

/* return the number of work-items in global iteration space*/
template <hypre_int dim>
static __inline__ __attribute__((always_inline))
hypre_int hypre_sycl_get_global_num_workitems(sycl::nd_item<dim>& item)
{
  switch (dim)
  {
  case 1:
    return (item.get_global_range(0));
  case 2:
    return (item.get_global_range(0) * item.get_global_range(1));
  case 3:
    return (item.get_global_range(0) * item.get_global_range(1) * item.get_global_range(2));
  }

  return -1;
}

/* return the flattened work-item id in global iteration space */
template <hypre_int dim>
static __inline__ __attribute__((always_inline))
hypre_int hypre_sycl_get_global_workitem_id(sycl::nd_item<dim>& item)
{
  return item.get_global_linear_id();
}

/* return the number of sub-groups in global iteration space */
template <hypre_int dim>
static __inline__ __attribute__((always_inline))
hypre_int hypre_sycl_get_global_num_subgroups(sycl::nd_item<dim>& item)
{
  return hypre_sycl_get_num_workgroups<dim>(item) * hypre_sycl_get_num_subgroups<dim>(item);
}

/* return the flattened sub-group id in global iteration space */
template <hypre_int dim>
static __inline__ __attribute__((always_inline))
hypre_int hypre_sycl_get_global_subgroup_id(sycl::nd_item<dim>& item)
{
  return hypre_sycl_get_workgroup_id<dim>(item) * hypre_sycl_get_num_subgroups<dim>(item) +
    hypre_sycl_get_subgroup_id<dim>(item);
}

// #if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 600
// static __device__ __forceinline__
// hypre_double atomicAdd(hypre_double* address, hypre_double val)
// {
//     hypre_ulonglongint* address_as_ull = (hypre_ulonglongint*) address;
//     hypre_ulonglongint old = *address_as_ull, assumed;

//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                         __double_as_longlong(val +
// 					     __longlong_as_double(assumed)));

//     // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//     } while (assumed != old);

//     return __longlong_as_double(old);
// }
// #endif

template <typename T>
static __inline__ __attribute__((always_inline))
T read_only_load(const T *ptr)
{
   return __ldg( ptr );
}

/* exclusive prefix scan */
template <typename T>
static __inline__ __attribute__((always_inline))
T warp_prefix_sum(hypre_int lane_id, T in, T &all_sum, sycl::nd_item<1>& item)
{
  sycl::ONEAPI::sub_group SG = item.get_sub_group();
  hypre_int hypre_warp_size = SG.get_local_range().get(0);
#pragma unroll
  for (hypre_int d = 2; d <=hypre_warp_size; d <<= 1)
  {
    T t = SG.shuffle_up(in, d >> 1);
    if ( (lane_id & (d - 1)) == (d - 1) )
    {
      in += t;
    }
  }

  all_sum = SG.shuffle(in, hypre_warp_size-1);

  if (lane_id == hypre_warp_size-1)
  {
    in = 0;
  }

#pragma unroll
  for (hypre_int d = hypre_warp_size/2; d > 0; d >>= 1)
  {
    T t = SG.shuffle_xor(in, d);

    if ( (lane_id & (d - 1)) == (d - 1))
    {
      if ( (lane_id & ((d << 1) - 1)) == ((d << 1) - 1) )
      {
	in += t;
      }
      else
      {
	in = t;
      }
    }
  }
  return in;
}

template <typename T> static __inline__ __attribute__((always_inline))
T warp_reduce_sum(T in, sycl::nd_item<1>& item)
{
  sycl::ONEAPI::sub_group SG = item.get_sub_group();
  //sycl::ONEAPI::reduce(SG, in, std::plus<T>());
#pragma unroll
  for (hypre_int d = SG.get_local_range().get(0)/2; d > 0; d >>= 1)
  {
    in += SG.shuffle_down(in, d);
  }
  return in;
}

template <typename T> static __inline__ __attribute__((always_inline))
T warp_allreduce_sum(T in, sycl::nd_item<1>& item)
{
  sycl::ONEAPI::sub_group SG = item.get_sub_group();
  //sycl::ONEAPI::reduce(SG, in, std::bit_xor<T>());
#pragma unroll
  for (hypre_int d = SG.get_local_range().get(0)/2; d > 0; d >>= 1)
  {
    in += SG.shuffle_xor(in, d);
  }
  return in;
}

template <typename T> static __inline__ __attribute__((always_inline))
T warp_reduce_max(T in, sycl::nd_item<1>& item)
{
  sycl::ONEAPI::sub_group SG = item.get_sub_group();
#pragma unroll
  for (hypre_int d = SG.get_local_range().get(0)/2; d > 0; d >>= 1)
  {
    in = sycl::max(in, SG.shuffle_down(in, d));
  }
  return in;
}

template <typename T> static __inline__ __attribute__((always_inline))
T warp_allreduce_max(T in, sycl::nd_item<1>& item)
{
  sycl::ONEAPI::sub_group SG = item.get_sub_group();
#pragma unroll
  for (hypre_int d = SG.get_local_range().get(0)/2; d > 0; d >>= 1)
  {
    in = sycl::max(in, SG.shuffle_xor(in, d));
  }
  return in;
}

template <typename T> static __inline__ __attribute__((always_inline))
T warp_reduce_min(T in, sycl::nd_item<1>& item)
{
  sycl::ONEAPI::sub_group SG = item.get_sub_group();
#pragma unroll
  for (hypre_int d = SG.get_local_range().get(0)/2; d > 0; d >>= 1)
  {
    in = sycl::min(in, SG.shuffle_down(in, d));
  }
  return in;
}

template <typename T> static __inline__ __attribute__((always_inline))
T warp_allreduce_min(T in, sycl::nd_item<1>& item)
{
  sycl::ONEAPI::sub_group SG = item.get_sub_group();
#pragma unroll
  for (hypre_int d = SG.get_local_range().get(0)/2; d > 0; d >>= 1)
  {
    in = sycl::min(in, SG.shuffle_xor(in, d));
  }
  return in;
}

static __inline__ __attribute__((always_inline))
hypre_int next_power_of_2(hypre_int n)
{
  if (n <= 0)
  {
    return 0;
  }

  /* if n is power of 2, return itself */
  if ( (n & (n - 1)) == 0 )
  {
    return n;
  }

  n |= (n >>  1);
  n |= (n >>  2);
  n |= (n >>  4);
  n |= (n >>  8);
  n |= (n >> 16);
  n ^= (n >>  1);
  n  = (n <<  1);

  return n;
}

template <typename T>
struct absolute_value {
  T operator()(const T &x) const
  {
    return x < T(0) ? -x : x;
  }
};

template <typename T>
struct is_negative {
   bool operator()(const T &x)
   {
      return (x < 0);
   }
};

template <typename T>
struct is_positive {
   bool operator()(const T &x)
   {
      return (x > 0);
   }
};

template <typename T>
struct is_nonnegative {
   bool operator()(const T &x)
   {
      return (x >= 0);
   }
};

template <typename T>
struct in_range {
   T low, up;

   in_range(T low_, T up_) { low = low_; up = up_; }

   bool operator()(const T &x)
   {
      return (x >= low && x <= up);
   }
};

template <typename T>
struct out_of_range {
   T low, up;

   out_of_range(T low_, T up_) { low = low_; up = up_; }

   bool operator()(const T &x)
   {
      return (x < low || x > up);
   }
};

template <typename T>
struct less_than {
   T val;

   less_than(T val_) { val = val_; }

   bool operator()(const T &x)
   {
      return (x < val);
   }
};

template <typename T>
struct equal {
   T val;

   equal(T val_) { val = val_; }

   bool operator()(const T &x)
   {
      return (x == val);
   }
};

/* hypre_Sycl_utils.cpp */
sycl::range<1> hypre_GetDefaultSYCLWorkgroupDimension();

sycl::range<1> hypre_GetDefaultSYCLGridDimension(HYPRE_Int n,
						 const char *granularity,
						 sycl::range<1> bDim);

template <typename T1, typename T2, typename T3> HYPRE_Int hypreDevice_StableSortByTupleKey(HYPRE_Int N, T1 *keys1, T2 *keys2, T3 *vals, HYPRE_Int opt);

template <typename T1, typename T2, typename T3, typename T4> HYPRE_Int hypreDevice_StableSortTupleByTupleKey(HYPRE_Int N, T1 *keys1, T2 *keys2, T3 *vals1, T4 *vals2, HYPRE_Int opt);

template <typename T1, typename T2, typename T3> HYPRE_Int hypreDevice_ReduceByTupleKey(HYPRE_Int N, T1 *keys1_in,  T2 *keys2_in,  T3 *vals_in, T1 *keys1_out, T2 *keys2_out, T3 *vals_out);

template <typename T>
HYPRE_Int hypreDevice_CsrRowPtrsToIndicesWithRowNum(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr, T *d_row_num, T *d_row_ind);

template <typename T>
HYPRE_Int hypreDevice_ScatterConstant(T *x, HYPRE_Int n, HYPRE_Int *map, T v);

HYPRE_Int hypreDevice_GetRowNnz(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int *d_diag_ia, HYPRE_Int *d_offd_ia, HYPRE_Int *d_rownnz);

HYPRE_Int hypreDevice_CopyParCSRRows(HYPRE_Int nrows, HYPRE_Int *d_row_indices, HYPRE_Int job, HYPRE_Int has_offd, HYPRE_Int first_col, HYPRE_Int *d_col_map_offd_A, HYPRE_Int *d_diag_i, HYPRE_Int *d_diag_j, HYPRE_Complex *d_diag_a, HYPRE_Int *d_offd_i, HYPRE_Int *d_offd_j, HYPRE_Complex *d_offd_a, HYPRE_Int *d_ib, HYPRE_BigInt *d_jb, HYPRE_Complex *d_ab);

HYPRE_Int hypreDevice_IntegerReduceSum(HYPRE_Int m, HYPRE_Int *d_i);

HYPRE_Int hypreDevice_IntegerInclusiveScan(HYPRE_Int n, HYPRE_Int *d_i);

HYPRE_Int hypreDevice_IntegerExclusiveScan(HYPRE_Int n, HYPRE_Int *d_i);

HYPRE_Int* hypreDevice_CsrRowPtrsToIndices(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr);

HYPRE_Int hypreDevice_CsrRowPtrsToIndices_v2(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ptr, HYPRE_Int *d_row_ind);

HYPRE_Int* hypreDevice_CsrRowIndicesToPtrs(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ind);

HYPRE_Int hypreDevice_CsrRowIndicesToPtrs_v2(HYPRE_Int nrows, HYPRE_Int nnz, HYPRE_Int *d_row_ind, HYPRE_Int *d_row_ptr);

HYPRE_Int hypreDevice_GenScatterAdd(HYPRE_Real *x, HYPRE_Int ny, HYPRE_Int *map, HYPRE_Real *y, char *work);

HYPRE_Int hypreDevice_BigToSmallCopy(HYPRE_Int *tgt, const HYPRE_BigInt *src, HYPRE_Int size);

int hypre_CachingMallocDevice(void **ptr, size_t nbytes);

int hypre_CachingMallocManaged(void **ptr, size_t nbytes);

int hypre_CachingFreeDevice(void *ptr);

int hypre_CachingFreeManaged(void *ptr);

#endif // #if defined(HYPRE_USING_SYCL)

#if defined(HYPRE_USING_CUSPARSE)

cudaDataType hypre_HYPREComplexToCudaDataType();

cusparseIndexType_t hypre_HYPREIntToCusparseIndexType();

#endif // #if defined(HYPRE_USING_CUSPARSE)

#endif /* #ifndef HYPRE_SYCL_UTILS_HPP */

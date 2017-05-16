/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for the BoxLoop
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * BoxLoop macros:
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_NEWBOXLOOP_HEADER
#define HYPRE_NEWBOXLOOP_HEADER

extern "C++" {
#include <RAJA/RAJA.hxx>
}
using namespace RAJA;

typedef struct hypre_Boxloop_struct
{
	HYPRE_Int lsize0,lsize1,lsize2;
	HYPRE_Int strides0,strides1,strides2;
	HYPRE_Int bstart0,bstart1,bstart2;
	HYPRE_Int bsize0,bsize1,bsize2;
} hypre_Boxloop;

#define BLOCKSIZE 256

#if defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_MANAGED)
#include <cuda.h>
#include <cuda_runtime.h>

#define AxCheckError(err) CheckError(err, __FUNCTION__, __LINE__)
inline void CheckError(cudaError_t const err, char const* const fun, const HYPRE_Int line)
{
    if (err)
    {
        printf("CUDA Error Code[%d]: %s\n%s() Line:%d\n", err, cudaGetErrorString(err), fun, line);
		HYPRE_Int *p = NULL; *p = 1;
    }
}

#define hypre_exec_policy cuda_exec<BLOCKSIZE>
#define hypre_reduce_policy  cuda_reduce_atomic<BLOCKSIZE>
#define hypre_fence() \
cudaError err = cudaGetLastError();\
if ( cudaSuccess != err ) {\
printf("\n ERROR zypre_newBoxLoop: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
}\
AxCheckError(cudaDeviceSynchronize());

#elif defined(HYPRE_USE_OPENMP)
   #define hypre_exec_policy      omp_for_exec
   #define hypre_reduce_policy omp_reduce
   #define hypre_fence() 
#elif defined(HYPRE_USING_OPENMP_ACC)
   #define hypre_exec_policy      omp_parallel_for_acc
   #define hypre_reduce_policy omp_acc_reduce
#else 
   #define hypre_exec_policy   seq_exec
   #define hypre_reduce_policy seq_reduce
   #define hypre_fence()
#endif

#define zypre_BoxLoopIncK(k,box,i)					\
{									\
   HYPRE_Int idx = idx_local;						\
   local_idx  = idx % box.lsize0;					\
   idx        = idx / box.lsize0;					\
   i += (local_idx*box.strides0 + box.bstart0) * hypre_boxD##k;		\
   hypre_boxD##k *= hypre_max(0, box.bsize0 + 1);			\
   local_idx  = idx % box.lsize1;					\
   idx        = idx / box.lsize1;					\
   i += (local_idx*box.strides1 + box.bstart1) * hypre_boxD##k;		\
   hypre_boxD##k *= hypre_max(0, box.bsize1 + 1);			\
   local_idx  = idx % box.lsize2;					\
   idx  = idx / box.lsize2;					\
   i += (local_idx*box.strides2 + box.bstart2) * hypre_boxD##k;		\
   hypre_boxD##k *= hypre_max(0, box.bsize2 + 1);			\
}


#define zypre_BoxLoopCUDAInit(ndim,loop_size)				\
  HYPRE_Int hypre__tot = 1;						\
  for (HYPRE_Int i = 0;i < ndim;i ++)					\
      hypre__tot *= loop_size[i];


#define zypre_BoxLoopCUDADeclare()										\
	HYPRE_Int local_idx;												\
	HYPRE_Int idx_local = idx;

#define zypre_newBoxLoop0Begin(ndim, loop_size)			\
{									\
   zypre_BoxLoopCUDAInit(ndim,loop_size);					\
   forall< hypre_exec_policy >(0, hypre__tot, [=] RAJA_DEVICE (HYPRE_Int idx) \
   {


#define zypre_newBoxLoop0End()					\
	});											\
	hypre_fence();      \
}

#define zypre_BoxLoopDataDeclareK(k,ndim,loop_size,dbox,start,stride)	\
	hypre_Boxloop databox##k;					\
	databox##k.lsize0 = loop_size[0];				\
	databox##k.strides0 = stride[0];				\
	databox##k.bstart0  = start[0] - dbox->imin[0];			\
	databox##k.bsize0   = dbox->imax[0]-dbox->imin[0];		\
	if (ndim > 1)							\
	{								\
	    databox##k.lsize1 = loop_size[1];				\
	    databox##k.strides1 = stride[1];				\
	    databox##k.bstart1  = start[1] - dbox->imin[1];		\
	    databox##k.bsize1   = dbox->imax[1]-dbox->imin[1];   	\
	}								\
	else						        	\
	{							       	\
		databox##k.lsize1 = 1;				       	\
		databox##k.strides1 = 0;		       		\
		databox##k.bstart1  = 0;	       			\
		databox##k.bsize1   = 0;		       		\
	}								\
	if (ndim == 3)							\
	{								\
	    databox##k.lsize2 = loop_size[2];				\
	    databox##k.strides2 = stride[2];				\
	    databox##k.bstart2  = start[2] - dbox->imin[2];		\
	    databox##k.bsize2   = dbox->imax[2]-dbox->imin[2];   	\
	}								\
	else								\
	{								\
	    databox##k.lsize2 = 1;					\
	    databox##k.strides2 = 0;					\
	    databox##k.bstart2  = 0;					\
	    databox##k.bsize2   = 0;					\
	}

#define zypre_newBoxLoop1Begin(ndim, loop_size,				\
			       dbox1, start1, stride1, i1)		\
{    														\
    zypre_BoxLoopCUDAInit(ndim,loop_size);						\
    zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);	\
    forall< hypre_exec_policy >(0, hypre__tot, [=] RAJA_DEVICE (HYPRE_Int idx) \
    {									\
      zypre_BoxLoopCUDADeclare();					\
      HYPRE_Int hypre_boxD1 = 1;					\
      HYPRE_Int i1 = 0;							\
      zypre_BoxLoopIncK(1,databox1,i1);

      
#define zypre_newBoxLoop1End(i1)				\
	});											\
    hypre_fence();\
}
	
#define zypre_newBoxLoop2Begin(ndim, loop_size,				\
                                dbox1, start1, stride1, i1,	\
                                dbox2, start2, stride2, i2)	\
{    														\
    zypre_BoxLoopCUDAInit(ndim,loop_size);						\
    zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);	\
    zypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);	\
    forall< hypre_exec_policy >(0, hypre__tot, [=] RAJA_DEVICE (HYPRE_Int idx) \
    {									\
        zypre_BoxLoopCUDADeclare()					\
        HYPRE_Int hypre_boxD1 = 1,hypre_boxD2 = 1;			\
		HYPRE_Int i1 = 0, i2 = 0;							\
	local_idx  = idx_local % databox1.lsize0;			\
	idx_local  = idx_local / databox1.lsize0;			\
	i1 += (local_idx*databox1.strides0 + databox1.bstart0) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize0 + 1);		\
	i2 += (local_idx*databox2.strides0 + databox2.bstart0) * hypre_boxD2; \
	hypre_boxD2 *= hypre_max(0, databox2.bsize0 + 1);		\
	local_idx  = idx_local % databox1.lsize1;			\
	idx_local  = idx_local / databox1.lsize1;			\
	i1 += (local_idx*databox1.strides1 + databox1.bstart1) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize1 + 1);		\
	i2 += (local_idx*databox2.strides1 + databox2.bstart1) * hypre_boxD2; \
	hypre_boxD2 *= hypre_max(0, databox2.bsize1 + 1);		\
	local_idx  = idx_local % databox1.lsize2;			\
	idx_local  = idx_local / databox1.lsize2;			\
	i1 += (local_idx*databox1.strides2 + databox1.bstart2) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize2 + 1);		\
	i2 += (local_idx*databox2.strides2 + databox2.bstart2) * hypre_boxD2; \
	hypre_boxD2 *= hypre_max(0, databox2.bsize2 + 1);		\
	


#define zypre_newBoxLoop2End(i1, i2)			\
	});											\
    hypre_fence();\
}

#define zypre_newBoxLoop3Begin(ndim, loop_size,				\
			       dbox1, start1, stride1, i1,		\
			       dbox2, start2, stride2, i2,		\
			       dbox3, start3, stride3, i3)		\
  {									\
  zypre_BoxLoopCUDAInit(ndim,loop_size);						\
        zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1); \
        zypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2); \
        zypre_BoxLoopDataDeclareK(3,ndim,loop_size,dbox3,start3,stride3); \
        forall< hypre_exec_policy >(0, hypre__tot, [=] RAJA_DEVICE (HYPRE_Int idx) \
	{								\
	  zypre_BoxLoopCUDADeclare();					\
	  HYPRE_Int hypre_boxD1 = 1,hypre_boxD2 = 1,hypre_boxD3 = 1; \
	  HYPRE_Int i1 = 0, i2 = 0, i3 = 0;				\
	  local_idx  = idx_local % databox1.lsize0;				\
	  idx_local  = idx_local / databox1.lsize0;				\
	  i1 += (local_idx*databox1.strides0 + databox1.bstart0) * hypre_boxD1;	\
	  hypre_boxD1 *= hypre_max(0, databox1.bsize0 + 1);			\
	  i2 += (local_idx*databox2.strides0 + databox2.bstart0) * hypre_boxD2;	\
	  hypre_boxD2 *= hypre_max(0, databox2.bsize0 + 1);			\
	  i3 += (local_idx*databox3.strides0 + databox3.bstart0) * hypre_boxD3;	\
	  hypre_boxD3 *= hypre_max(0, databox3.bsize0 + 1);			\
	  local_idx  = idx_local % databox1.lsize1;				\
	  idx_local  = idx_local / databox1.lsize1;				\
	  i1 += (local_idx*databox1.strides1 + databox1.bstart1) * hypre_boxD1;	\
	  hypre_boxD1 *= hypre_max(0, databox1.bsize1 + 1);			\
	  i2 += (local_idx*databox2.strides1 + databox2.bstart1) * hypre_boxD2;	\
	  hypre_boxD2 *= hypre_max(0, databox2.bsize1 + 1);			\
	  i3 += (local_idx*databox3.strides1 + databox3.bstart1) * hypre_boxD3;	\
	  hypre_boxD3 *= hypre_max(0, databox3.bsize1 + 1);			\
	  local_idx  = idx_local % databox1.lsize2;				\
	  idx_local  = idx_local / databox1.lsize2;				\
	  i1 += (local_idx*databox1.strides2 + databox1.bstart2) * hypre_boxD1;	\
	  hypre_boxD1 *= hypre_max(0, databox1.bsize2 + 1);			\
	  i2 += (local_idx*databox2.strides2 + databox2.bstart2) * hypre_boxD2;	\
	  hypre_boxD2 *= hypre_max(0, databox2.bsize2 + 1);			\
	  i3 += (local_idx*databox3.strides2 + databox3.bstart2) * hypre_boxD3;	\
	  hypre_boxD3 *= hypre_max(0, databox3.bsize2 + 1);			\
	  

#define zypre_newBoxLoop3End(i1, i2, i3)			\
	});											\
	hypre_fence();							\
}

#define zypre_newBoxLoop4Begin(ndim, loop_size,				\
			       dbox1, start1, stride1, i1,		\
			       dbox2, start2, stride2, i2,		\
			       dbox3, start3, stride3, i3,		\
			       dbox4, start4, stride4, i4)		\
{								       \
 zypre_BoxLoopCUDAInit(ndim,loop_size);					       \
     zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1); \
     zypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2); \
     zypre_BoxLoopDataDeclareK(3,ndim,loop_size,dbox3,start3,stride3); \
     zypre_BoxLoopDataDeclareK(4,ndim,loop_size,dbox4,start4,stride4); \
     forall< hypre_exec_policy >(0, hypre__tot, [=] RAJA_DEVICE (HYPRE_Int idx) \
     {									\
         zypre_BoxLoopCUDADeclare();					\
		 HYPRE_Int hypre_boxD1 = 1,hypre_boxD2 = 1,hypre_boxD3 = 1,hypre_boxD4 = 1; \
	 HYPRE_Int i1 = 0, i2 = 0, i3 = 0,i4 = 0;			\
	 local_idx  = idx_local % databox1.lsize0;			\
	 idx_local  = idx_local / databox1.lsize0;			\
	 i1 += (local_idx*databox1.strides0 + databox1.bstart0) * hypre_boxD1; \
	 hypre_boxD1 *= hypre_max(0, databox1.bsize0 + 1);		\
	 i2 += (local_idx*databox2.strides0 + databox2.bstart0) * hypre_boxD2; \
	 hypre_boxD2 *= hypre_max(0, databox2.bsize0 + 1);		\
	 i3 += (local_idx*databox3.strides0 + databox3.bstart0) * hypre_boxD3; \
	 hypre_boxD3 *= hypre_max(0, databox3.bsize0 + 1);		\
	 i4 += (local_idx*databox4.strides0 + databox4.bstart0) * hypre_boxD4; \
	 hypre_boxD4 *= hypre_max(0, databox4.bsize0 + 1);		\
	 local_idx  = idx_local % databox1.lsize1;			\
	 idx_local  = idx_local / databox1.lsize1;			\
	 i1 += (local_idx*databox1.strides1 + databox1.bstart1) * hypre_boxD1; \
	 hypre_boxD1 *= hypre_max(0, databox1.bsize1 + 1);		\
	 i2 += (local_idx*databox2.strides1 + databox2.bstart1) * hypre_boxD2; \
	 hypre_boxD2 *= hypre_max(0, databox2.bsize1 + 1);		\
	 i3 += (local_idx*databox3.strides1 + databox3.bstart1) * hypre_boxD3; \
	 hypre_boxD3 *= hypre_max(0, databox3.bsize1 + 1);		\
	 i4 += (local_idx*databox4.strides1 + databox4.bstart1) * hypre_boxD4; \
	 hypre_boxD4 *= hypre_max(0, databox4.bsize1 + 1);		\
	 local_idx  = idx_local % databox1.lsize2;			\
	 idx_local  = idx_local / databox1.lsize2;			\
	 i1 += (local_idx*databox1.strides2 + databox1.bstart2) * hypre_boxD1; \
	 hypre_boxD1 *= hypre_max(0, databox1.bsize2 + 1);		\
	 i2 += (local_idx*databox2.strides2 + databox2.bstart2) * hypre_boxD2; \
	 hypre_boxD2 *= hypre_max(0, databox2.bsize2 + 1);		\
	 i3 += (local_idx*databox3.strides2 + databox3.bstart2) * hypre_boxD3; \
	 hypre_boxD3 *= hypre_max(0, databox3.bsize2 + 1);		\
	 i4 += (local_idx*databox4.strides2 + databox4.bstart2) * hypre_boxD4; \
	 hypre_boxD4 *= hypre_max(0, databox4.bsize2 + 1);		\
	 
#define zypre_newBoxLoop4End(i1, i2, i3, i4)	\
  });						\
  hypre_fence();				\
}
#define zypre_BasicBoxLoopDataDeclareK(k,ndim,loop_size,stride)		\
	hypre_Boxloop databox##k;					\
	databox##k.lsize0 = loop_size[0];				\
	databox##k.strides0 = stride[0];				\
	if (ndim > 1)							\
	{								\
	    databox##k.lsize1 = loop_size[1];				\
	    databox##k.strides1 = stride[1];				\
	}								\
	else						        	\
	{							       	\
		databox##k.lsize1 = 1;				       	\
		databox##k.strides1 = 0;		       		\
	}								\
	if (ndim == 3)							\
	{								\
	    databox##k.lsize2 = loop_size[2];				\
	    databox##k.strides2 = stride[2];				\
	}								\
	else								\
	{								\
	    databox##k.lsize2 = 1;					\
	    databox##k.strides2 = 0;					\
	}

#define zypre_newBasicBoxLoop2Begin(ndim, loop_size,			\
				    stride1, i1,			\
				    stride2, i2)			\
{    		       				                	\
    zypre_BoxLoopCUDAInit(ndim,loop_size);		        	\
    zypre_BasicBoxLoopDataDeclareK(1,ndim,loop_size,stride1);	\
    zypre_BasicBoxLoopDataDeclareK(2,ndim,loop_size,stride2);	\
    forall< hypre_exec_policy >(0, hypre__tot, [=] RAJA_DEVICE (HYPRE_Int idx) \
    {									\
        zypre_BoxLoopCUDADeclare()					\
	HYPRE_Int i1 = 0, i2 = 0;					\
	local_idx  = idx_local % databox1.lsize0;			\
	idx_local  = idx_local / databox1.lsize0;			\
	i1 += local_idx*databox1.strides0;		\
	i2 += local_idx*databox2.strides0; \
	local_idx  = idx_local % databox1.lsize1;			\
	idx_local  = idx_local / databox1.lsize1;			\
	i1 += local_idx*databox1.strides1; \
	i2 += local_idx*databox2.strides1; \
	local_idx  = idx_local % databox1.lsize2;			\
	idx_local  = idx_local / databox1.lsize2;			\
	i1 += local_idx*databox1.strides2; \
	i2 += local_idx*databox2.strides2; \
	
#define MAX_BLOCK BLOCKSIZE

extern "C++" {
#if defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_MANAGED)
template<class T>
class ReduceMult   
{
public:
  /*!
   * \brief Constructor takes initial reduction value (default constructor
   * is disabled).
   *
   * Note: Constructor only executes on the host.
   */
  explicit ReduceMult(T init_val)
  {
    m_is_copy_host = false;
    m_myID = getCudaReductionId();
    getCudaReductionTallyBlock(m_myID,
                               (void **)&m_tally_host,
                               (void **)&m_tally_device);
    m_tally_host->tally = init_val;
  }

  /*!
   * \brief Initialize shared memory on device, request shared memory on host.
   *
   * Copy constructor executes on both host and device.
   * On host requests dynamic shared memory and gets offset into dynamic
   * shared memory if in forall.
   * On device initializes dynamic shared memory to appropriate value.
   */
  RAJA_HOST_DEVICE
  ReduceMult(const ReduceMult<T> &other)
  {
    *this = other;
#if defined(__CUDA_ARCH__)
    m_is_copy_device = true;
    m_finish_reduction = !other.m_is_copy_device;
    extern __shared__ unsigned char sd_block[];
    T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

    HYPRE_Int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    // initialize shared memory
    T val = static_cast<T>(0);
    for (HYPRE_Int i = BLOCKSIZE / 2; i > 0; i /= 2) {
      // this descends all the way to 1
      if (threadId < i) {
        sd[threadId + i] = val;
      }
    }
    if (threadId < 1) {
      sd[threadId] = val;
    }

    __syncthreads();
#else
    m_is_copy_host = true;
    m_smem_offset = getCudaSharedmemOffset(m_myID, BLOCKSIZE, sizeof(T));
#endif
  }

  /*!
   * \brief Finish reduction on device and free memory on host.
   *
   * Destruction on host releases the device memory chunk for
   * reduction id and id itself for others to use.
   * Destruction on device completes the reduction.
   *
   * Note: destructor executes on both host and device.
   */
  RAJA_HOST_DEVICE ~ReduceMult<T>()
  {
#if defined(__CUDA_ARCH__)
    if (m_finish_reduction) {
      extern __shared__ unsigned char sd_block[];
      T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

      HYPRE_Int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      T temp = 1;
      __syncthreads();

      for (HYPRE_Int i = BLOCKSIZE / 2; i >= WARP_SIZE; i /= 2) {
        if (threadId < i) {
          sd[threadId] *= sd[threadId + i];
        }
        __syncthreads();
      }

      if (threadId < WARP_SIZE) {
        temp = sd[threadId];
        for (HYPRE_Int i = WARP_SIZE / 2; i > 0; i /= 2) {
          temp *= HIDDEN::shfl_xor<T>(temp, i);
        }
      }

      // one thread adds to tally
      if (threadId == 0) {
        _atomicAdd<T>(&(m_tally_device->tally), temp);
      }
    }
#else
    if (!m_is_copy_host) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
#endif

    
  }

  /*!
   * \brief Operator that returns reduced sum value.
   *
   * Note: accessor only executes on host.
   */
  operator T()
  {
    beforeCudaReadTallyBlock<true>(m_myID);
    return m_tally_host->tally;
  }

  /*!
   * \brief Operator that returns reduced sum value.
   *
   * Note: accessor only executes on host.
   */
  T get() { return operator T(); }

  /*!
   * \brief Operator that adds value to sum.
   *
   * Note: only operates on device.
   */
  RAJA_DEVICE ReduceMult<T> const &
  operator*=(T val) const
  {
    extern __shared__ unsigned char sd_block[];
    T *sd = reinterpret_cast<T *>(&sd_block[m_smem_offset]);

    HYPRE_Int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    sd[threadId] *= val;

    return *this;
  }

private:
  /*!
   * \brief Default constructor is declared private and not implemented.
   */
  ReduceMult<T>();

  /*!
   * \brief Pointer to host tally block cache slot for this reduction variable.
   */
  CudaReductionTallyTypeAtomic<T> *m_tally_host = nullptr;

  /*!
   * \brief Pointer to device tally block slot for this reduction variable.
   */
  CudaReductionTallyTypeAtomic<T> *m_tally_device = nullptr;

  /*!
   * \brief My cuda reduction variable ID.
   */
  HYPRE_Int m_myID = -1;

  /*!
   * \brief Byte offset into dynamic shared memory.
   */
  HYPRE_Int m_smem_offset = -1;

  /*!
   * \brief If this variable is a copy or not; only original may release memory 
   *        or perform finalization.
   */
  bool m_is_copy_host = false;
  bool m_is_copy_device = false;
  bool m_finish_reduction = false;

  // Sanity checks for block size and template type size
  static constexpr bool powerOfTwoCheck = (!(BLOCKSIZE & (BLOCKSIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCKSIZE >= 32) && (BLOCKSIZE <= 1024));
  static constexpr bool sizeofcheck =
      ((sizeof(T) <= sizeof(CudaReductionDummyDataType))
       && (sizeof(CudaReductionTallyType<T>)
           <= sizeof(CudaReductionDummyTallyType))
       && (sizeof(CudaReductionBlockType<T>)
           <= sizeof(CudaReductionDummyBlockType)));
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck,
      "Error: type must be of size <= " 
      RAJA_STRINGIFY_MACRO(RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};
#elif defined(HYPRE_USING_OPENMP)
    template <typename T>
    class ReduceMult
    {
        using my_type = ReduceMult;
        
    public:
        //
        // Constructor takes default value (default ctor is disabled).
        //
        explicit ReduceMult(T init_val, T initializer = 1)
        : m_parent(NULL), m_val(init_val), m_custom_init(initializer)
        {
        }
        
        //
        // Copy ctor.
        //
        ReduceMult(const ReduceMult& other) :
        m_parent(other.m_parent ? other.m_parent : &other),
        m_val(other.m_custom_init),
        m_custom_init(other.m_custom_init)
        {
        }
        
        //
        // Destruction releases the shared memory block chunk for reduction id
        // and id itself for others to use.
        //
        ~ReduceMult()
        {
            if (m_parent) {
#pragma omp critical
                {
                    *m_parent *= m_val;
                }
            }
        }
        
        //
        // Operator that returns reduced sum value.
        //
        operator T()
        {
            return m_val;
        }
        
        //
        // Method that returns sum value.
        //
        T get() { return operator T(); }
        
        //
        // += operator that adds value to sum for current thread.
        //
        const ReduceMult& operator*=(T rhs) const
        {
            this->m_val *= rhs;
            return *this;
        }
        
        ReduceMult& operator*=(T rhs)
        {
            this->m_val *= rhs;
            return *this;
        }
        
    private:
        //
        // Default ctor is declared private and not implemented.
        //
        ReduceMult();
        
        const my_type * m_parent;
        
        mutable T m_val;
        T m_custom_init;
        
    };
#else
    template <typename T>
    class ReduceMult
    {
        using my_type = ReduceMult;
        
    public:
        //
        // Constructor takes default value (default ctor is disabled).
        //
        explicit ReduceMult(T init_m_val, T initializer = 1) :
        m_parent(NULL),
        m_val(init_m_val),
        m_custom_init(initializer)
        {
        }
        
        //
        // Copy ctor.
        //
        ReduceMult(const ReduceMult& other) :
        m_parent(other.m_parent ? other.m_parent : &other),
        m_val(other.m_custom_init),
        m_custom_init(other.m_custom_init)
        {
        }
        
        //
        // Destruction releases the shared memory block chunk for reduction id
        // and id itself for others to use.
        //
        ~ReduceMult()
        {
            if (m_parent) {
                *m_parent *= m_val;
            }
        }
        
        //
        // Operator that returns reduced sum value.
        //
        operator T()
        {
            return m_val;
        }
        
        //
        // Method that returns reduced sum value.
        //
        T get() { return operator T(); }
        
        //
        // += operator that adds value to sum.
        //
        ReduceMult& operator*=(T rhs)
        {
            this->m_val *= rhs;
            return *this;
        }
        
        const ReduceMult& operator*=(T rhs) const
        {
            this->m_val *= rhs;
            return *this;
        }
        
    private:
        //
        // Default ctor is declared private and not implemented.
        //
        ReduceMult();
        
        const my_type * m_parent;
        
        mutable T m_val;
        T m_custom_init;
    };
#endif
}


#define zypre_newBoxLoop1ReductionBegin(ndim, loop_size,		\
					dbox1, start1, stride1, i1,sum) \
{									\
   HYPRE_Real sum_tmp;							\
   {									\
      ReduceSum< hypre_reduce_policy, HYPRE_Real> sum(0.0);				\
      zypre_newBoxLoop1Begin(ndim, loop_size, dbox1, start1, stride1,i1)	\
      {

#define zypre_newBoxLoop1ReductionEnd(i1,sum)				\
      }									\
      zypre_newBoxLoop1End(i1);					\
      hypre_fence();						\
      sum_tmp = (HYPRE_Real)(sum);				\
   }								\
   sum += sum_tmp; \
}
		    
#define zypre_newBoxLoop2ReductionBegin(ndim, loop_size,		\
					dbox1, start1, stride1, i1,	\
					dbox2, start2, stride2, i2,sum)	\
{									\
   HYPRE_Real sum_tmp;							\
   {									\
      ReduceSum< hypre_reduce_policy, HYPRE_Real> sum(0.0);				\
      zypre_newBoxLoop2Begin(ndim, loop_size, \
			     dbox1, start1, stride1,i1,\
			     dbox2, start2, stride2,i2)	\
      {

#define zypre_newBoxLoop2ReductionEnd(i1,i2,sum)			\
      }									\
      zypre_newBoxLoop2End(i1,i2);					\
      hypre_fence();							\
      sum_tmp = (HYPRE_Real)(sum);					\
   }								\
   sum += sum_tmp; \
}

#define zypre_newBoxLoop1ReductionMult(ndim, loop_size,				\
				       dbox1, start1, stride1, i1,xp,sum) \
{									\
   ReduceMult<HYPRE_Real> local_result_raja(1.0);				\
   zypre_newBoxLoop1Begin(ndim, loop_size, dbox1, start1, stride1, i1) \
   {									\
       local_result_raja *= xp[i1];					\
   }									\
   zypre_newBoxLoop1End(i1)						\
   hypre_fence();							\
   sum *= (HYPRE_Real)(local_result_raja);				\
}


#define hypre_LoopBegin(size,idx)					\
{									\
   forall< hypre_exec_policy >(0, size, [=] RAJA_DEVICE (HYPRE_Int idx)	\
   {

#define hypre_LoopEnd()					\
   });							\
   hypre_fence();		\
}
  
#define zypre_BoxBoundaryCopyBegin(ndim, loop_size, stride1, i1, idx) 	\
{									\
    zypre_BoxLoopCUDAInit(ndim,loop_size);				\
    hypre_Boxloop databox1;						\
    databox1.lsize0 = loop_size[0];					\
    databox1.lsize1 = loop_size[1];					\
    databox1.lsize2 = loop_size[2];					\
    databox1.strides0 = stride1[0];					\
    databox1.strides1 = stride1[1];					\
    databox1.strides2 = stride1[2];					\
    forall< hypre_exec_policy >(0, hypre__tot, [=] RAJA_DEVICE (HYPRE_Int idx) \
    {									\
        zypre_BoxLoopCUDADeclare()					\
        HYPRE_Int i1 = 0;							\
        local_idx  = idx_local % databox1.lsize0;			\
        idx_local  = idx_local / databox1.lsize0;			\
        i1 += local_idx*databox1.strides0;				\
        local_idx  = idx_local % databox1.lsize1;			\
        idx_local  = idx_local / databox1.lsize1;			\
        i1 += local_idx*databox1.strides1;				\
        local_idx  = idx_local % databox1.lsize2;			\
        idx_local  = idx_local / databox1.lsize2;			\
        i1 += local_idx*databox1.strides2;				\
		
#define zypre_BoxBoundaryCopyEnd()				\
	});											\
	hypre_fence();							\
}

#define zypre_BoxDataExchangeBegin(ndim, loop_size,				\
                                   stride1, i1,	\
                                   stride2, i2)	\
{    														\
    zypre_BoxLoopCUDAInit(ndim,loop_size);					\
    hypre_Boxloop databox1,databox2;					\
    databox1.lsize0 = loop_size[0];					\
    databox1.lsize1 = loop_size[1];					\
    databox1.lsize2 = loop_size[2];					\
    databox1.strides0 = stride1[0];					\
    databox1.strides1 = stride1[1];					\
    databox1.strides2 = stride1[2];					\
    databox2.lsize0 = loop_size[0];					\
    databox2.lsize1 = loop_size[1];					\
    databox2.lsize2 = loop_size[2];					\
    databox2.strides0 = stride2[0];					\
    databox2.strides1 = stride2[1];					\
    databox2.strides2 = stride2[2];					\
    forall< hypre_exec_policy >(0, hypre__tot, [=] RAJA_DEVICE (HYPRE_Int idx) \
    {									\
        zypre_BoxLoopCUDADeclare()					\
	HYPRE_Int i1 = 0, i2 = 0;					\
	local_idx  = idx_local % databox1.lsize0;			\
	idx_local  = idx_local / databox1.lsize0;			\
	i1 += local_idx*databox1.strides0;				\
	i2 += local_idx*databox2.strides0;				\
	local_idx  = idx_local % databox1.lsize1;			\
	idx_local  = idx_local / databox1.lsize1;			\
	i1 += local_idx*databox1.strides1;				\
	i2 += local_idx*databox2.strides1;				\
	local_idx  = idx_local % databox1.lsize2;			\
	idx_local  = idx_local / databox1.lsize2;			\
	i1 += local_idx*databox1.strides2;				\
	i2 += local_idx*databox2.strides2;



#define zypre_BoxDataExchangeEnd()				\
	});											\
	hypre_fence();							\
}

#define zypre_newBoxLoop0For()

#define zypre_newBoxLoop1For(i1)

#define zypre_newBoxLoop2For(i1, i2) 
 
#define zypre_newBoxLoop3For(i1, i2, i3)

#define zypre_newBoxLoop4For(i1, i2, i3, i4)

#define zypre_newBoxLoopSetOneBlock()

#define hypre_newBoxLoopGetIndex(index)					\
  index[0] = hypre__i; index[1] = hypre__j; index[2] = hypre__k

#define hypre_BoxLoopGetIndex    zypre_BoxLoopGetIndex
#define hypre_BoxLoopSetOneBlock zypre_newBoxLoopSetOneBlock
#define hypre_BoxLoopBlock()       0
#define hypre_BoxLoop0Begin      zypre_newBoxLoop0Begin
#define hypre_BoxLoop0For        zypre_newBoxLoop0For
#define hypre_BoxLoop0End        zypre_newBoxLoop0End
#define hypre_BoxLoop1Begin      zypre_newBoxLoop1Begin
#define hypre_BoxLoop1For        zypre_newBoxLoop1For
#define hypre_BoxLoop1End        zypre_newBoxLoop1End
#define hypre_BoxLoop2Begin      zypre_newBoxLoop2Begin
#define hypre_BoxLoop2For        zypre_newBoxLoop2For
#define hypre_BoxLoop2End        zypre_newBoxLoop2End
#define hypre_BoxLoop3Begin      zypre_newBoxLoop3Begin
#define hypre_BoxLoop3For        zypre_newBoxLoop3For
#define hypre_BoxLoop3End        zypre_newBoxLoop3End
#define hypre_BoxLoop4Begin      zypre_newBoxLoop4Begin
#define hypre_BoxLoop4For        zypre_newBoxLoop4For
#define hypre_BoxLoop4End        zypre_newBoxLoop4End

#define hypre_newBoxLoop1ReductionBegin zypre_newBoxLoop1ReductionBegin
#define hypre_newBoxLoop1ReductionEnd   zypre_newBoxLoop1ReductionEnd
#define hypre_newBoxLoop2ReductionBegin zypre_newBoxLoop2ReductionBegin
#define hypre_newBoxLoop2ReductionEnd   zypre_newBoxLoop2ReductionEnd
#define hypre_newBoxLoop1ReductionMult zypre_newBoxLoop1ReductionMult
#define hypre_BoxBoundaryCopyBegin zypre_BoxBoundaryCopyBegin
#define hypre_BoxBoundaryCopyEnd zypre_BoxBoundaryCopyEnd
#define hypre_BoxDataExchangeBegin zypre_BoxDataExchangeBegin
#define hypre_BoxDataExchangeEnd zypre_BoxDataExchangeEnd

#define hypre_BasicBoxLoop2Begin zypre_newBasicBoxLoop2Begin
#endif

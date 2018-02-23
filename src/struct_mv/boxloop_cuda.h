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

#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#define HYPRE_LAMBDA [=] __host__  __device__

typedef struct hypre_Boxloop_struct
{
	HYPRE_Int lsize0,lsize1,lsize2;
	HYPRE_Int strides0,strides1,strides2;
	HYPRE_Int bstart0,bstart1,bstart2;
	HYPRE_Int bsize0,bsize1,bsize2;
} hypre_Boxloop;

#define BLOCKSIZE 512
#define WARP_SIZE 32
#define BLOCK_SIZE 512

#if 0
#define hypre_fence()
/*printf("\n hypre_newBoxLoop in %s(%d) function %s\n",__FILE__,__LINE__,__FUNCTION__);*/
#else
#define hypre_fence() \
{		      \
  cudaError err = cudaGetLastError();		\
  if ( cudaSuccess != err )			\
  {									\
    printf("\n ERROR hypre_newBoxLoop: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
    HYPRE_Int *p = NULL; *p = 1;\
  }									\
  hypre_CheckErrorDevice(cudaDeviceSynchronize());				\
} 
#endif

#define hypre_reduce_policy  cuda_reduce<BLOCKSIZE>

extern "C++" {
template <typename LOOP_BODY>
__global__ void forall_kernel(LOOP_BODY loop_body, HYPRE_Int length)
{
	HYPRE_Int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < length)
		loop_body(idx);
}

template<typename LOOP_BODY>
void BoxLoopforall (HYPRE_Int policy, HYPRE_Int length, LOOP_BODY loop_body)
{
  
  if (policy == HYPRE_MEMORY_HOST)
  {
    HYPRE_Int idx;
#pragma omp parallel for 
    for (idx = 0;idx < length;idx++)
    { 
      loop_body(idx);
    }
    
  }
  else if (policy == HYPRE_MEMORY_DEVICE)
  {    
     size_t const blockSize = BLOCKSIZE;
     size_t gridSize  = (length + blockSize - 1) / blockSize;
     if (gridSize == 0) gridSize = 1;	
     //hypre_printf("length= %d, blocksize = %d, gridsize = %d\n",length,blockSize,gridSize);
     forall_kernel<<<gridSize, blockSize>>>(loop_body,length);
  }
  else if (policy == 2)
  {
  }
}
}


#define hypre_BoxLoopIncK(k,box,hypre__i)					\
   HYPRE_Int hypre_boxD##k = 1;						\
   HYPRE_Int hypre__i = 0;							\
   hypre__i += (hypre_IndexD(local_idx, 0)*box.strides0 + box.bstart0) * hypre_boxD##k;	\
   hypre_boxD##k *= hypre_max(0, box.bsize0 + 1);			\
   hypre__i += (hypre_IndexD(local_idx, 1)*box.strides1 + box.bstart1) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize1 + 1);			\
   hypre__i += (hypre_IndexD(local_idx, 2)*box.strides2 + box.bstart2) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize2 + 1);

#define hypre_newBoxLoopInit(ndim,loop_size)				\
  HYPRE_Int hypre__tot = 1;						\
  for (HYPRE_Int hypre_d = 0;hypre_d < ndim;hypre_d ++)			\
    hypre__tot *= loop_size[hypre_d];

#define hypre_BasicBoxLoopInit(ndim,loop_size)	\
  HYPRE_Int hypre__tot = 1;						\
  for (HYPRE_Int hypre_d = 0;hypre_d < ndim;hypre_d ++)			\
    hypre__tot *= loop_size[hypre_d];					\

#define hypre_newBoxLoopDeclare(box)\
  hypre_Index local_idx;						\
  HYPRE_Int idx_local = idx;						\
  hypre_IndexD(local_idx, 0)  = idx_local % box.lsize0;			\
  idx_local = idx_local / box.lsize0;					\
  hypre_IndexD(local_idx, 1)  = idx_local % box.lsize1;			\
  idx_local = idx_local / box.lsize1;					\
  hypre_IndexD(local_idx, 2)  = idx_local % box.lsize2;\

#define hypre_newBoxLoop0Begin(ndim, loop_size)				\
{									\
   hypre_newBoxLoopInit(ndim,loop_size);				\
   BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx) \
   {

#define hypre_newBoxLoop0End()					\
    });									\
    hypre_fence();							\
}

#define hypre_BoxLoopDataDeclareK(k,ndim,loop_size,dbox,start,stride)	\
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
	    databox##k.bsize1   = dbox->imax[1]-dbox->imin[1];	\
	}								\
	else						        	\
	{							       	\
		databox##k.lsize1 = 1;				       	\
		databox##k.strides1 = 0;		       		\
		databox##k.bstart1  = 0;	       			\
		databox##k.bsize1   = 0;		       		\
	}								\
	if (ndim == 3)							\
	{							      	\
	      databox##k.lsize2 = loop_size[2];				\
	      databox##k.strides2 = stride[2];				\
	      databox##k.bstart2  = start[2] - dbox->imin[2];		\
	      databox##k.bsize2   = dbox->imax[2]-dbox->imin[2];	\
	}				                        	\
	else						        	\
	{							       	\
		databox##k.lsize2 = 1;				       	\
		databox##k.strides2 = 0;		       		\
		databox##k.bstart2  = 0;	       			\
		databox##k.bsize2   = 0;		       		\
	}

#define hypre_newBoxLoop1Begin(ndim, loop_size,				\
			       dbox1, start1, stride1, i1)		\
{									\
    hypre_newBoxLoopInit(ndim,loop_size);				\
    hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);	\
    BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx) \
    {									\
      hypre_newBoxLoopDeclare(databox1);				\
      hypre_BoxLoopIncK(1,databox1,i1);
      
#define hypre_newBoxLoop1End(i1)				\
    });									\
    hypre_fence();							\
}
	
#define hypre_newBoxLoop2Begin(ndim, loop_size,				\
			       dbox1, start1, stride1, i1,		\
			       dbox2, start2, stride2, i2)		\
{									\
    hypre_newBoxLoopInit(ndim,loop_size);						\
    hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);	\
    hypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);	\
    BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx) \
    {									\
       hypre_newBoxLoopDeclare(databox1);				\
       hypre_BoxLoopIncK(1,databox1,i1);				\
       hypre_BoxLoopIncK(2,databox2,i2);

#define hypre_newBoxLoop2End(i1, i2)			\
    });							\
    hypre_fence();					\
}

#define hypre_newBoxLoop3Begin(ndim, loop_size,				\
			       dbox1, start1, stride1, i1,		\
			       dbox2, start2, stride2, i2,		\
			       dbox3, start3, stride3, i3)		\
{									\
    hypre_newBoxLoopInit(ndim,loop_size);						\
    hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);	\
    hypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);	\
    hypre_BoxLoopDataDeclareK(3,ndim,loop_size,dbox3,start3,stride3);	\
    BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx) \
    {									\
        hypre_newBoxLoopDeclare(databox1);				\
	hypre_BoxLoopIncK(1,databox1,i1);				\
	hypre_BoxLoopIncK(2,databox2,i2);				\
	hypre_BoxLoopIncK(3,databox3,i3);
	

#define hypre_newBoxLoop3End(i1, i2,i3)			\
    });									\
    hypre_fence();							\
}

#define hypre_newBoxLoop4Begin(ndim, loop_size,				\
			       dbox1, start1, stride1, i1,		\
			       dbox2, start2, stride2, i2,		\
			       dbox3, start3, stride3, i3,		\
			       dbox4, start4, stride4, i4)		\
{								       \
     hypre_newBoxLoopInit(ndim,loop_size);			       \
     hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1); \
     hypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2); \
     hypre_BoxLoopDataDeclareK(3,ndim,loop_size,dbox3,start3,stride3); \
     hypre_BoxLoopDataDeclareK(4,ndim,loop_size,dbox4,start4,stride4); \
     BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx) \
     {									\
        hypre_newBoxLoopDeclare(databox1);				\
	hypre_BoxLoopIncK(1,databox1,i1);				\
	hypre_BoxLoopIncK(2,databox2,i2);				\
	hypre_BoxLoopIncK(3,databox3,i3);				\
	hypre_BoxLoopIncK(4,databox4,i4);
		
#define hypre_newBoxLoop4End(i1, i2, i3, i4)	\
    });						\
    hypre_fence();				\
}

#define zypre_BasicBoxLoopDataDeclareK(k,ndim,loop_size,stride)		\
	hypre_Boxloop databox##k;					\
	databox##k.lsize0   = loop_size[0];				\
	databox##k.strides0 = stride[0];				\
	databox##k.bstart0  = 0;					\
	databox##k.bsize0   = 0;					\
	if (ndim > 1)							\
	{								\
	   databox##k.lsize1   = loop_size[1];				\
	   databox##k.strides1 = stride[1];				\
	   databox##k.bstart1  = 0;					\
	   databox##k.bsize1   = 0;					\
	}								\
	else						        	\
	{							       	\
	   databox##k.lsize1   = 1;				       	\
	   databox##k.strides1 = 0;					\
	   databox##k.bstart1  = 0;					\
	   databox##k.bsize1   = 0;					\
	}								\
	if (ndim == 3)							\
	{								\
	   databox##k.lsize2   = loop_size[2];				\
	   databox##k.strides2 = stride[2];				\
	   databox##k.bstart2  = 0;				        \
	   databox##k.bsize2   = 0;			                \
	}								\
	else								\
	{								\
	    databox##k.lsize2   = 1;					\
	    databox##k.strides2 = 0;					\
	    databox##k.bstart2  = 0;					\
	    databox##k.bsize2   = 0;					\
	}

#define zypre_newBasicBoxLoop1Begin(ndim, loop_size,			\
				    stride1, i1)			\
{    		       				                	\
    hypre_BasicBoxLoopInit(ndim,loop_size);		        	\
    zypre_BasicBoxLoopDataDeclareK(1,ndim,loop_size,stride1);	\
    BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx) \
    {									\
        hypre_newBoxLoopDeclare(databox1);					\
        hypre_BoxLoopIncK(1,databox1,i1);					\
	
#define zypre_newBasicBoxLoop2Begin(ndim, loop_size,			\
				    stride1, i1,			\
				    stride2, i2)			\
{    		       				                	\
    hypre_BasicBoxLoopInit(ndim,loop_size);		        	\
    zypre_BasicBoxLoopDataDeclareK(1,ndim,loop_size,stride1);	\
    zypre_BasicBoxLoopDataDeclareK(2,ndim,loop_size,stride2);	\
    BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx) \
    {									\
        hypre_newBoxLoopDeclare(databox1);					\
        hypre_BoxLoopIncK(1,databox1,i1);					\
        hypre_BoxLoopIncK(2,databox2,i2);				\


#define hypre_LoopBegin(size,idx)					\
{									\
   BoxLoopforall(hypre_exec_policy,size,HYPRE_LAMBDA (HYPRE_Int idx)	\
   {

#define hypre_LoopEnd()					\
   });							\
   hypre_fence();					\
}

#define MAX_BLOCK BLOCKSIZE

extern "C++" {
template<typename T>
__device__ __forceinline__ T hypre_shfl_xor(T var, HYPRE_Int laneMask)
{
  const HYPRE_Int int_sizeof_T = 
      (sizeof(T) + sizeof(HYPRE_Int) - 1) / sizeof(HYPRE_Int);
  union {
    T var;
    HYPRE_Int arr[int_sizeof_T];
  } Tunion;
  Tunion.var = var;

  for(HYPRE_Int i = 0; i < int_sizeof_T; ++i) {
    Tunion.arr[i] =
#ifndef CUDART_VERSION
      #error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 9000)
      __shfl_xor_sync(0xFFFFFFFF, Tunion.arr[i], laneMask);
#elif (CUDART_VERSION <= 8000)
      __shfl_xor(Tunion.arr[i], laneMask);
#endif
  }
  return Tunion.var;
}

#define RAJA_MAX(a, b) (((b) > (a)) ? (b) : (a))

template <typename T>
class ReduceSum
{
public:
   //
   // Constructor takes initial reduction value (default ctor is disabled).
   // Ctor only executes on the host.
   //
  explicit ReduceSum( T init_val,HYPRE_Int location )
   {
      data_location = location;
      
      m_is_copy = false;

      m_init_val = init_val;
      m_reduced_val = static_cast<T>(0);

      m_myID = getCudaReductionId();

      if (data_location == HYPRE_MEMORY_DEVICE)
      {
	 m_blockdata = getCudaReductionMemBlock(m_myID) ;
	 m_blockoffset = 1;
      
	 // Entire shared memory block must be initialized to zero so
	 // sum reduction is correct.
	 size_t len = RAJA_CUDA_REDUCE_BLOCK_LENGTH;
	 cudaMemset(&m_blockdata[m_blockoffset], 0,
		    sizeof(CudaReductionBlockDataType)*len); 

	 m_max_grid_size = m_blockdata;
	 m_max_grid_size[0] = 0;

	 cudaDeviceSynchronize();
      }
      else if (data_location == HYPRE_MEMORY_HOST)
      {
	 m_blockdata = getCPUReductionMemBlock(m_myID);
	 HYPRE_Int nthreads = omp_get_max_threads();
	 #pragma omp parallel for schedule(static, 1)
	 for (HYPRE_Int i = 0; i < nthreads; ++i ) {
	    m_blockdata[i*s_block_offset] = 0 ;
	 }
      }
   }

   //
   // Copy ctor executes on both host and device.
   //
   __host__ __device__ 
   ReduceSum( const ReduceSum< T >& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor executes on both host and device.
   // Destruction on host releases the unique id for others to use. 
   //
   __host__ __device__ 
   ~ReduceSum< T >()
   {
      if (!m_is_copy) {
	{
#if defined( __CUDA_ARCH__ )
#else
	   releaseCudaReductionId(m_myID);
#endif
	}
      }
   }

   //
   // Operator to retrieve reduced sum value (before object is destroyed).
   // Accessor only operates on host.
   //
   __host__ __device__
   operator T()
   {
     
     if (data_location == HYPRE_MEMORY_DEVICE) 
     {
        cudaDeviceSynchronize() ;
	m_blockdata[m_blockoffset] = static_cast<T>(0);

	size_t grid_size = m_max_grid_size[0];
	for (size_t i=1; i <= grid_size; ++i) {
	   m_blockdata[m_blockoffset] += m_blockdata[m_blockoffset+i];
	}
	m_reduced_val = m_init_val + static_cast<T>(m_blockdata[m_blockoffset]);
     }
     else if (data_location == HYPRE_MEMORY_HOST)
     {
#if defined( __CUDA_ARCH__ )
#else
		 T tmp_reduced_val = static_cast<T>(0);
	HYPRE_Int nthreads = omp_get_max_threads();
	for ( HYPRE_Int i = 0; i < nthreads; ++i ) {
	   tmp_reduced_val += static_cast<T>(m_blockdata[i*s_block_offset]);
	}
	m_reduced_val = m_init_val + tmp_reduced_val;
#endif
     }
     return m_reduced_val;
   }

   //
   // += operator to accumulate arg value in the proper shared
   // memory block location.
   //
   __host__ __device__
   ReduceSum< T > operator+=(T val) const
   {  
#if defined( __CUDA_ARCH__ )
      if (data_location == HYPRE_MEMORY_DEVICE)
      {	
        __shared__ T sd[BLOCK_SIZE];

	if ( blockIdx.x  + blockIdx.y  + blockIdx.z +
	     threadIdx.x + threadIdx.y + threadIdx.z == 0 ) {
           HYPRE_Int numBlock = gridDim.x * gridDim.y * gridDim.z ;
           m_max_grid_size[0] = RAJA_MAX( numBlock,  m_max_grid_size[0] );
	}

       // initialize shared memory
	for ( HYPRE_Int i = BLOCK_SIZE / 2; i > 0; i /=2 ) {     
          // this descends all the way to 1
           if ( threadIdx.x < i ) {
	      sd[threadIdx.x + i] = m_reduced_val;  
	   }
	}
	__syncthreads();

	sd[threadIdx.x] = val;

	T temp = 0;
	__syncthreads();

	for (HYPRE_Int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
	   if (threadIdx.x < i) {
	      sd[threadIdx.x] += sd[threadIdx.x + i];
	   }
	   __syncthreads();
	}

	if (threadIdx.x < WARP_SIZE) {
	   temp = sd[threadIdx.x];
	   for (HYPRE_Int i = WARP_SIZE / 2; i > 0; i /= 2) {
	      temp += hypre_shfl_xor(temp, i);
	   }
	}

	// one thread adds to gmem, we skip m_blockdata[m_blockoffset]
	// because we will be accumlating into this
	if (threadIdx.x == 0) {
	   HYPRE_Int blockID = m_blockoffset + 1 + blockIdx.x +
	     blockIdx.y*gridDim.x +
	     blockIdx.z*gridDim.x*gridDim.y ;
	   m_blockdata[blockID] += temp ;
	}
      }
#else
      if (data_location == HYPRE_MEMORY_HOST)
      {
         HYPRE_Int tid = omp_get_thread_num();
		 m_blockdata[tid*s_block_offset] += val;
      }
#endif
      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceSum< T >();

   bool m_is_copy;
   HYPRE_Int m_myID;
   HYPRE_Int data_location;

   T m_init_val;
   T m_reduced_val;
   static const HYPRE_Int s_block_offset = 
      COHERENCE_BLOCK_SIZE/sizeof(CudaReductionBlockDataType);
   CudaReductionBlockDataType* m_blockdata ;
   
   HYPRE_Int m_blockoffset;

   CudaReductionBlockDataType* m_max_grid_size;
};

}
#define hypre_newBoxLoopGetIndex(index)\
  index[0] = hypre_IndexD(local_idx, 0); index[1] = hypre_IndexD(local_idx, 1); index[2] = hypre_IndexD(local_idx, 2);
  
#define hypre_BoxLoopGetIndex    zypre_BoxLoopGetIndex

#define hypre_BoxLoopSetOneBlock() ; 
#define hypre_BoxLoopBlock()       0

#define hypre_BoxLoop0Begin      hypre_newBoxLoop0Begin
#define hypre_BoxLoop0For        hypre_newBoxLoop0For
#define hypre_BoxLoop0End        hypre_newBoxLoop0End
#define hypre_BoxLoop1Begin      hypre_newBoxLoop1Begin
#define hypre_BoxLoop1For        hypre_newBoxLoop1For
#define hypre_BoxLoop1End        hypre_newBoxLoop1End
#define hypre_BoxLoop2Begin      hypre_newBoxLoop2Begin
#define hypre_BoxLoop2For        hypre_newBoxLoop2For
#define hypre_BoxLoop2End        hypre_newBoxLoop2End
#define hypre_BoxLoop3Begin      hypre_newBoxLoop3Begin
#define hypre_BoxLoop3For        hypre_newBoxLoop3For
#define hypre_BoxLoop3End        hypre_newBoxLoop3End
#define hypre_BoxLoop4Begin      hypre_newBoxLoop4Begin
#define hypre_BoxLoop4For        hypre_newBoxLoop4For
#define hypre_BoxLoop4End        hypre_newBoxLoop4End

#define hypre_BasicBoxLoop1Begin zypre_newBasicBoxLoop1Begin 
#define hypre_BasicBoxLoop2Begin zypre_newBasicBoxLoop2Begin 
#endif

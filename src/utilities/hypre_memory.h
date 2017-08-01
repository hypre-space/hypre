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
 * Header file for memory management utilities
 *
 * The abstract memory model has a Host (think CPU) and a Device (think GPU) and
 * three basic types of memory management utilities:
 *
 *    1. Malloc(..., location) 
 *             location=LOCATION_DEVICE - malloc memory on the device
 *             location=LOCATION_HOST   - malloc memory on the host
 *    2. MemCopy(..., method)
 *             method=HOST_TO_DEVICE    - copy from host to device
 *             method=DEVICE_TO_HOST    - copy from device to host
 *             method=DEVICE_TO_DEVICE  - copy from device to device
 *    3. SetExecutionMode
 *             location=LOCATION_DEVICE - execute on the device
 *             location=LOCATION_HOST   - execute on the host
 *
 * Although the abstract model does not explicitly reflect a managed memory
 * model (i.e., unified memory), it can support it.  Here is a summary of how
 * the abstract model would be mapped to specific hardware scenarios:
 *
 *    Not using a device, not using managed memory
 *       Malloc(..., location) 
 *             location=LOCATION_DEVICE - host malloc          e.g., malloc
 *             location=LOCATION_HOST   - host malloc          e.g., malloc
 *       MemCopy(..., method)
 *             method=HOST_TO_DEVICE    - copy from host to host e.g., memcpy
 *             method=DEVICE_TO_HOST    - copy from host to host e.g., memcpy
 *             method=DEVICE_TO_DEVICE  - copy from host to host e.g., memcpy
 *       SetExecutionMode
 *             location=LOCATION_DEVICE - execute on the device
 *             location=LOCATION_HOST   - execute on the host    
 *
 *    Using a device, not using managed memory
 *       Malloc(..., location) 
 *             location=LOCATION_DEVICE - device malloc        e.g., cudaMalloc
 *             location=LOCATION_HOST   - host malloc          e.g., malloc
 *       MemCopy(..., method)
 *             method=HOST_TO_DEVICE    - copy from host to device e.g., cudaMemcpy
 *             method=DEVICE_TO_HOST    - copy from device to host e.g., cudaMemcpy
 *             method=DEVICE_TO_DEVICE  - copy from device to device e.g., cudaMemcpy
 *       SetExecutionMode
 *             location=LOCATION_DEVICE - execute on the device
 *             location=LOCATION_HOST   - execute on the host 
 *
 *    Using a device, using managed memory
 *       Malloc(..., location) 
 *             location=LOCATION_DEVICE - managed malloc        e.g., cudaMallocManaged
 *             location=LOCATION_HOST   - host malloc          e.g., malloc
 *       MemCopy(..., method)
 *             method=HOST_TO_DEVICE    - no-op
 *             method=DEVICE_TO_HOST    - no-op
 *             method=DEVICE_TO_DEVICE  - copy from device to device e.g., cudaMemcpy
 *       SetExecutionMode
 *             location=LOCATION_DEVICE - execute on the device
 *             location=LOCATION_HOST   - execute on the host 
 *
 * Questions:
 *
 *    1. Pinned memory?
 *    2. Need to allocate some host-only memory in a unified memory setting?
 *
 *****************************************************************************/

#ifndef hypre_MEMORY_HEADER
#define hypre_MEMORY_HEADER

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Interface prototypes (only for the no-device setting for starters)
 *--------------------------------------------------------------------------*/

#define hypre_HostTAlloc(type, count) \
( (type *)hypre_MAlloc((size_t)(sizeof(type) * (count))) )

#define hypre_HostCTAlloc(type, count) \
( (type *)hypre_HostfCAlloc((size_t)(count), (size_t)sizeof(type)) )

#define hypre_TReAlloc(ptr, type, count) \
( (type *)hypre_HostReAlloc((char *)ptr, (size_t)(sizeof(type) * (count))) )

#define hypre_HostTFree(ptr) \
( hypre_Free((char *)ptr), ptr = NULL )

#define hypre_DeviceTAlloc(type, count)   hypre_TAlloc(type, (count))
#define hypre_DeviceCTAlloc(type, count)  hypre_CTAlloc(type, (count))
#define hypre_DeviceTReAlloc(type, count) hypre_TReAlloc(type, (count))
#define hypre_DeviceTFree(ptr)            hypre_TFree(ptr)

#define hypre_MemCopyToDevice(ptrHost, ptrDevice, type, count) \
memcpy(ptrDevice, ptrHost, sizeof(type)*(count))

#define hypre_MemCopyFromDevice(ptrHost, ptrDevice, type, count) \
memcpy(ptrHost, ptrDevice, sizeof(type)*(count))

/*--------------------------------------------------------------------------
 * NEW INTERFACE
 *--------------------------------------------------------------------------*/
#define LOCATION_DEVICE ( 0)
#define LOCATION_HOST   ( 1)
#define LOCATION_UNSET  (-1)

#define HOST_TO_DEVICE   (0)
#define DEVICE_TO_HOST   (1)
#define DEVICE_TO_DEVICE (2)

extern HYPRE_Int hypre_exec_policy;
  
#if defined(HYPRE_MEMORY_GPU)
/* Using a device, not using managed memory */

#define hypre_TAlloc(type, count, location)\
({\
   type *ptr;\
   if (location==LOCATION_DEVICE)\
   {\
      ptr = hypre_DeviceTAlloc(type,count);\
   }\
   else if (location==LOCATION_HOST)\
   {\
      ptr = hypre_HostTAlloc(type,count);\
   }\
   ptr;})

#define hypre_CTAlloc(type, count, location)\
({\
   type *ptr;\
   if (location==LOCATION_DEVICE)\
   {\
      ptr = hypre_DeviceCTAlloc(type,count);\
   }\
   else if (location==LOCATION_HOST)\
   {\
      ptr = hypre_HostCTAlloc(type,count);\
   }\
   ptr;})
  
#define hypre_TReAlloc(type, count, location)\
({\
   type *ptr;\
   if (location==LOCATION_DEVICE)\
   {\
      ptr = hypre_DeviceTReAlloc(type,count);\
   }\
   else if (location==LOCATION_HOST)\
   {\
      ptr = hypre_HostTReAlloc(type,count);\
   }\
   ptr;})

#define hypre_TFree(type, count, location)\
({\
   if (location==LOCATION_DEVICE)\
   {\
      hypre_DeviceTFree(type);\
   }\
   else if (location==LOCATION_HOST)\
   {\
      hypre_HostTFree(type);\
   }\
})

#define hypre_memory_copy(ptrTo, ptrFrom, type, count, method)\
   if (method==HOST_TO_DEVICE)\
   {\
      cudaMemcpy(ptrTo, ptrFrom, sizeof(type)*count, cudaMemcpyHostToDevice);\
   }\
   else if (method==DEVICE_TO_HOST)\
   {\
      cudaMemcpy(ptrTo, ptrFrom, sizeof(type)*count, cudaMemcpyDeviceToHost);\
   }\
   else if (method==DEVICE_TO_DEVICE)\
   {\
      cudaMemcpy(ptrTo, ptrFrom, sizeof(type)*count, cudaMemcpyDeviceToDevice);\
   }

#define hypre_set_execution_mode(location)\
     hypre_exec_policy = location;

#elif defined(HYPRE_USE_MANAGED)
/* Using a device, not using managed memory */

#define hypre_TAlloc(type, count, location)\
({\
   type *ptr;\
   if (location==LOCATION_DEVICE)\
   {\
      cudaMallocManaged((void**)&ptr, sizeof(type)*count, cudaMemAttachGlobal);\
   }\
   else if (location==LOCATION_HOST)\
   {\
      ptr = hypre_HostTAlloc(type,count);\
   }\
   ptr;})

#define hypre_CTAlloc(type, count, location)\
({\
   type *ptr;\
   if (location==LOCATION_DEVICE)\
   {\
      cudaMallocManaged((void**)&ptr, sizeof(type)*count, cudaMemAttachGlobal);\
      cudaMemset(ptr, 0, sizeof(type)*count);\
   }\
   else if (location==LOCATION_HOST)\
   {\
      ptr = hypre_HostCTAlloc(type,count);\
   }\
   ptr;})
  
#define hypre_TReAlloc(type, count, location)\
({\
   type *ptr;\
   if (location==LOCATION_DEVICE)\
   {\
      ptr = hypre_UMTReAlloc(type,count);\
   }\
   else if (location==LOCATION_HOST)\
   {\
      ptr = hypre_HostTReAlloc(type,count);\
   }\
   ptr;})

#define hypre_TFree(type, count, location)\
({\
   if (location==LOCATION_DEVICE)\
   {\
      hypre_DeviceTFree(type);\
   }\
   else if (location==LOCATION_HOST)\
   {\
      hypre_HostTFree(type);\
   }\
})

#define hypre_memory_copy(ptrTo, ptrFrom, type, count, method)\
   if (method==HOST_TO_DEVICE)\
   {\
   	  if (ptrTo != ptrFrom)\
      {\
         cudaMemcpy(ptrTo, ptrFrom, sizeof(type)*count, cudaMemcpyDefault);\
      }\
      else\
      {\
         ptrTo = ptrFrom;\
      }\
   }\
   else if (method==DEVICE_TO_HOST)\
   {\
      if (ptrTo != ptrFrom)\
      {\
         cudaMemcpy(ptrTo, ptrFrom, sizeof(type)*count, cudaMemcpyDefault);\
      }\
      else\
      {\
         ptrTo = ptrFrom;\
      }\
   }\
   else if (method==DEVICE_TO_DEVICE)\
   {\
      if (ptrTo != ptrFrom)\
      {\
         cudaMemcpy(ptrTo, ptrFrom, sizeof(type)*count, cudaMemcpyDeviceToDevice);\
      }\
      else\
      {\
         ptrTo = ptrFrom;\
      }\
   }

#define hypre_set_execution_mode(location)\
     hypre_exec_policy = location;

#elif defined(HYPRE_USE_OMP45)
/* Using OpenMP-4.5 */

#define hypre_TAlloc(type, count, location)\
({\
   type *ptr;\
   HYPRE_Int device_num = omp_get_default_device();\
   if (location==LOCATION_DEVICE)\
   {\
      hypre_omp45_offload(device_num, ptr, type, 0, count, "enter", "alloc");\
   }\
   else if (location==LOCATION_HOST)\
   {\
      ptr = hypre_HostTAlloc(type,count);\
   }\
   ptr;})

#define hypre_CTAlloc(type, count, location)\
({\
   type *ptr;\
   HYPRE_Int device_num = omp_get_default_device();\
   if (location==LOCATION_DEVICE)\
   {\
      hypre_omp45_offload(device_num, ptr, type, 0, count, "enter", "to");\
   }\
   else if (location==LOCATION_HOST)\
   {\
      ptr = hypre_HostCTAlloc(type,count);\
   }\
   ptr;})
 
/* how to implement realloc fro omp4.5 */
#define hypre_TReAlloc(type, count, location)\
({\
   type *ptr;\
   HYPRE_Int device_num = omp_get_default_device();\
   if (location==LOCATION_DEVICE)\
   {\
   }\
   else if (location==LOCATION_HOST)\
   {\
      ptr = hypre_HostTReAlloc(type,count);\
   }\
   ptr;})

#define hypre_TFree(type, count, location)\
({\
   HYPRE_Int device_num = omp_get_default_device();\
   if (location==LOCATION_DEVICE)\
   {\
      hypre_omp45_offload(device_num, ptr, type, 0, count, "exit", "delete");\
   }\
   else if (location==LOCATION_HOST)\
   {\
      hypre_HostTFree(type);\
   }\
})

/* how to implement device to device copy for Openmp4.5 */
#define hypre_memory_copy(ptrTo, ptrFrom, type, count, method)\
   HYPRE_Int device_num = omp_get_default_device();\
   if (method==HOST_TO_DEVICE)\
   {\
      if (ptrTo != ptrFrom)\
      {\
         Memcpy(ptrTo, ptrFrom, sizeof(type)*count);\
      }\
      hypre_omp45_offload(device_num, ptrTo, ptrFrom, type, 0, count, "update", "to");\
   }\
   else if (method==DEVICE_TO_HOST)\
   {\
      if (ptrTo != ptrFrom)\
      {\
         Memcpy(ptrTo, ptrFrom, sizeof(type)*count);\
      }\
      hypre_omp45_offload(device_num, ptrTo, ptrFrom, type, 0, count, "update", "from");\
   }\
   else if (method==DEVICE_TO_DEVICE)\
   {\
      if (ptrTo != ptrFrom)\
      {\
      }\
   }

#define hypre_set_execution_mode(location)\
   if (location==LOCATION_DEVICE)\
   {\
      HYPRE_OMP45_OFFLOAD_ON();\
   }\
   else if (location==LOCATION_HOST)\
   {\
      HYPRE_OMP45_OFFLOAD_OFF();\
   }

#else
/* Not using a device, not using managed memory */

#define hypre_TAlloc(type, count, location) hypre_HostTAlloc(type,count);
#define hypre_CTAlloc(type, count, location) hypre_HostCTAlloc(type,count);
#define hypre_TReAlloc(type, count, location) hypre_HostTReAlloc(type,count);
#define hypre_TFree(type, count, location) hypre_HostTFree(type);

#define hypre_memory_copy(ptrTo, ptrFrom, type, count, method)\
   if (ptrTo != ptrFrom)\
   {\
      memcpy(ptrTo, ptrFrom, sizeof(type)*count);\
   }\
   else\
   {\
      ptrTo = ptrFrom;\
   }

#define hypre_set_execution_mode(location) ;

#endif
/*--------------------------------------------------------------------------
 * CURRENT CODE
 *--------------------------------------------------------------------------*/

#if defined(HYPRE_MEMORY_GPU) || defined(HYPRE_USE_MANAGED)
#ifdef __cplusplus
extern "C++" {
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef __cplusplus
}
#endif
#define HYPRE_CUDA_GLOBAL __host__ __device__
  
#if defined(HYPRE_MEMORY_GPU)
#define hypre_DeviceTAlloc(type, count) \
  ({									\
    type * ptr;								\
    cudaError_t cudaerr = cudaMalloc((void**)&ptr,sizeof(type)*(count)); \
    if ( cudaerr != cudaSuccess ) {					\
      printf("\n ERROR hypre_DataTAlloc %lu : %s in %s(%d) function %s\n",sizeof(type)*(count),cudaGetErrorString(cudaerr),__FILE__,__LINE__,__FUNCTION__); \
      HYPRE_Int *p = NULL; *p = 1;						\
    }									\
    ptr;})
	
#define hypre_DeviceCTAlloc(type, count) \
	({								   \
	type * ptr;						   \
	cudaError_t cudaerr = cudaMalloc((void**)&ptr,sizeof(type)*(count)); \
	if ( cudaerr != cudaSuccess ) {										\
		printf("\n hypre_DataCTAlloc %lu : %s in %s(%d) function %s\n",sizeof(type)*(count),cudaGetErrorString(cudaerr),__FILE__,__LINE__,__FUNCTION__); \
		HYPRE_Int *p = NULL; *p = 1;\
	}		\
	cudaMemset(ptr,0,sizeof(type)*(count));	   \
	ptr;})									   \
	
#define hypre_DeviceTReAlloc(ptr, type, count) {type *newptr;				\
	                                         cudaMalloc((void**)&,sizeof(type)*(count), cudaMemAttachGlobal);	\
											 memcpy(newptr, ptr, sizeof(type)*(count)); \
											 cudaFree(ptr);				\
											 ptr = newptr;}
#else
 #define hypre_DeviceTAlloc(type, count) \
	({																	\
	type * ptr;															\
	cudaError_t cudaerr = cudaMallocManaged((void**)&ptr,sizeof(type)*(count), cudaMemAttachGlobal);\
	if ( cudaerr != cudaSuccess ) {										\
		printf("\n ERROR hypre_DataTAlloc %lu : %s in %s(%d) function %s\n",sizeof(type)*(count),cudaGetErrorString(cudaerr),__FILE__,__LINE__,__FUNCTION__); \
		HYPRE_Int *p = NULL; *p = 1;\
	}\
	ptr;})
	
#define hypre_DeviceCTAlloc(type, count) \
	({								   \
	type * ptr;						   \
	cudaError_t cudaerr = cudaMallocManaged((void**)&ptr,sizeof(type)*(count), cudaMemAttachGlobal); \
	if ( cudaerr != cudaSuccess ) {										\
		printf("\n hypre_DataCTAlloc %lu : %s in %s(%d) function %s\n",sizeof(type)*(count),cudaGetErrorString(cudaerr),__FILE__,__LINE__,__FUNCTION__); \
		HYPRE_Int *p = NULL; *p = 1;\
	}		\
	cudaMemset(ptr,0,sizeof(type)*(count));	   \
	ptr;})									   \
	
#define hypre_DeviceTReAlloc(ptr, type, count) {type *newptr;				\
	                                      cudaMallocManaged((void**)&ptr,sizeof(type)*(count), cudaMemAttachGlobal);	\
					      memcpy(newptr, ptr, sizeof(type)*(count)); \
					      cudaFree(ptr);		\
					      ptr = newptr;} 
#endif
  
#define hypre_DeviceTFree(ptr) \
	{											\
		cudaError_t cudaerr = cudaFree(ptr);							\
		if ( cudaerr != cudaSuccess ) {									\
			printf("\n CudaFree : %s in %s(%d) function %s\n",cudaGetErrorString(cudaerr),__FILE__,__LINE__,__FUNCTION__); \
			HYPRE_Int *p = NULL; *p = 1;										\
		}																\
	}																	\
	

#define hypre_DataCopyToData(ptrH,ptrD,type,count)						\
	{cudaError_t cudaerr = cudaMemcpy(ptrD, ptrH, sizeof(type)*count, cudaMemcpyHostToDevice); \
if ( cudaerr != cudaSuccess ) {										\
		printf("\n hypre_DataCopyToData %lu : %s in %s(%d) function %s\n",sizeof(type)*(count),cudaGetErrorString(cudaerr),__FILE__,__LINE__,__FUNCTION__); \
		HYPRE_Int *p = NULL; *p = 1;\
}							  \
	}
	
	
#define hypre_DataCopyFromData(ptrH,ptrD,type,count)						\
	{cudaError_t cudaerr = cudaMemcpy(ptrH, ptrD, sizeof(type)*count, cudaMemcpyDeviceToHost); \
	if ( cudaerr != cudaSuccess ) {										\
		printf("\n hypre_DataCTAlloc %lu : %s in %s(%d) function %s\n",sizeof(type)*(count),cudaGetErrorString(cudaerr),__FILE__,__LINE__,__FUNCTION__); \
		HYPRE_Int *p = NULL; *p = 1;\
	}\
	}

#define hypre_DeviceMemset(ptr,value,type,count)	\
	cudaMemset(ptr,value,count*sizeof(type));
	
#define hypre_UMTAlloc(type, count)				\
  ({									\
      type * ptr;								\
      cudaMallocManaged((void**)&ptr,sizeof(type)*(count), cudaMemAttachGlobal); \
      ptr;								\
  })
	
#define hypre_UMCTAlloc(type, count)					\
  ({									\
    type * ptr;								\
    cudaMallocManaged((void**)&ptr,sizeof(type)*(count), cudaMemAttachGlobal); \
    cudaMemset(ptr,0,sizeof(type)*(count));				\
    ptr;})								\
  
  
#define hypre_UMTReAlloc(type, count)\
  ({							 \
    type * ptr;								\
    type *newptr;							\
    cudaMallocManaged((void**)&newptr,sizeof(type)*(count), cudaMemAttachGlobal); \
    cudaFree(ptr);							\
    ptr = newptr;							\
    ptr;})								\
  
#define hypre_UMTFree(ptr) \
      cudaFree(ptr)

#define hypre_TAlloc(type, count) \
( (type *)hypre_MAlloc((size_t)(sizeof(type) * (count))) )

#define hypre_CTAlloc(type, count) \
( (type *)hypre_CAlloc((size_t)(count), (size_t)sizeof(type)) )

#define hypre_TReAlloc(ptr, type, count) \
( (type *)hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count))) )

#define hypre_TFree(ptr) \
( hypre_Free((char *)ptr), ptr = NULL )
  
  //#define hypre_TAlloc(type, count)  hypre_UMTAlloc(type, count)
  //#define hypre_CTAlloc(type, count) hypre_UMCTAlloc(type, count)
  //#define hypre_TReAlloc(ptr, type, count) hypre_UMTReAlloc(type, count)
  //#define hypre_TFree(ptr) hypre_UMTFree(ptr)

#define hypre_SharedTAlloc(type, count) hypre_TAlloc(type, (count))
#define hypre_SharedCTAlloc(type, count) hypre_CTAlloc(type, (count))
#define hypre_SharedTReAlloc(type, count) hypre_TReAlloc(type, (count))
#define hypre_SharedTFree(ptr) hypre_TFree(ptr)
#else
#define HYPRE_CUDA_GLOBAL 

/*--------------------------------------------------------------------------
 * Use standard memory routines
 *--------------------------------------------------------------------------*/

#define hypre_TAlloc(type, count) \
( (type *)hypre_MAlloc((size_t)(sizeof(type) * (count))) )

#define hypre_CTAlloc(type, count) \
( (type *)hypre_CAlloc((size_t)(count), (size_t)sizeof(type)) )

#define hypre_TReAlloc(ptr, type, count) \
( (type *)hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count))) )

#define hypre_TFree(ptr) \
( hypre_Free((char *)ptr), ptr = NULL )

#define hypre_SharedTAlloc(type, count) hypre_TAlloc(type, (count))
#define hypre_SharedCTAlloc(type, count) hypre_CTAlloc(type, (count))
#define hypre_SharedTReAlloc(type, count) hypre_TReAlloc(type, (count))
#define hypre_SharedTFree(ptr) hypre_TFree(ptr)

#define hypre_DeviceTAlloc(type, count) hypre_TAlloc(type, (count))
#define hypre_DeviceCTAlloc(type, count) hypre_CTAlloc(type, (count))
#define hypre_DeviceTReAlloc(type, count) hypre_TReAlloc(type, (count))
#define hypre_DeviceTFree(ptr) hypre_TFree(ptr)
#define hypre_DataCopyToData(ptrH,ptrD,type,count) memcpy(ptrD, ptrH, sizeof(type)*(count))
#define hypre_DataCopyFromData(ptrH,ptrD,type,count) memcpy(ptrH, ptrD, sizeof(type)*(count))
#define hypre_DeviceMemset(ptr,value,type,count)	memset(ptr,value,count*sizeof(type))
#define hypre_UMTAlloc(type, count) hypre_TAlloc(type, (count))
#define hypre_UMCTAlloc(type, count) hypre_CTAlloc(type, (count))
#define hypre_UMTReAlloc(type, count) hypre_TReAlloc(type, (count))
#define hypre_UMTFree(ptr) hypre_TFree(ptr)
#endif
  
#define hypre_PinnedTAlloc(type, count)\
( (type *)hypre_MAllocPinned((size_t)(sizeof(type) * (count))) )

#define hypre_HostTAlloc(type, count) \
( (type *)hypre_MAllocHost((size_t)(sizeof(type) * (count))) )

#define hypre_HostCTAlloc(type, count) \
( (type *)hypre_CAllocHost((size_t)(count), (size_t)sizeof(type)) )

#define hypre_HostTReAlloc(ptr, type, count) \
( (type *)hypre_ReAllocHost((char *)ptr, (size_t)(sizeof(type) * (count))) )

#define hypre_HostTFree(ptr) \
( hypre_FreeHost((char *)ptr), ptr = NULL )

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* hypre_memory.c */
HYPRE_Int hypre_OutOfMemory ( size_t size );
char *hypre_MAlloc ( size_t size );
char *hypre_CAlloc ( size_t count , size_t elt_size );
char *hypre_MAllocPinned( size_t size );
char *hypre_ReAlloc ( char *ptr , size_t size );
void hypre_Free ( char *ptr );
char *hypre_CAllocHost( size_t count,size_t elt_size );
char *hypre_MAllocHost( size_t size );
char *hypre_ReAllocHost( char   *ptr,size_t  size );
void hypre_FreeHost( char *ptr );
char *hypre_SharedMAlloc ( size_t size );
char *hypre_SharedCAlloc ( size_t count , size_t elt_size );
char *hypre_SharedReAlloc ( char *ptr , size_t size );
void hypre_SharedFree ( char *ptr );
HYPRE_Real *hypre_IncrementSharedDataPtr ( HYPRE_Real *ptr , size_t size );

#ifdef __cplusplus
}
#endif

#endif


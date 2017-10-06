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
 *****************************************************************************/

#ifndef hypre_MEMORY_HEADER
#define hypre_MEMORY_HEADER

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#define hypre_InitMemoryDebug(id)
#define hypre_FinalizeMemoryDebug()

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

#elif defined(HYPRE_USE_OMP45) /* ifdef at L29 */

#include "omp.h"

/* CPU memory management */
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

/* stringification:
 * _Pragma(string-literal), so we need to cast argument to a string
 * The three dots as last argument of the macro tells compiler that this is a variadic macro. 
 * I.e. this is a macro that receives variable number of arguments. 
 */
#define HYPRE_STR(s...) #s
#define HYPRE_XSTR(s...) HYPRE_STR(s)

/* OpenMP 4.5 GPU memory management */
/* empty */
#define HYPRE_CUDA_GLOBAL 
/* stats */
#define HYPRE_Long long
extern HYPRE_Int hypre__global_offload;
extern HYPRE_Long hypre__offload_to_count;
extern HYPRE_Long hypre__offload_from_count;
extern HYPRE_Long hypre__offload_to_byte_count;
extern HYPRE_Long hypre__offload_from_byte_count;

extern HYPRE_Long hypre__update_to_count;
extern HYPRE_Long hypre__update_from_count;
extern HYPRE_Long hypre__update_to_byte_count;
extern HYPRE_Long hypre__update_from_byte_count;

/* DEBUG MODE: check if offloading has effect */
#define HYPRE_OFFLOAD_DEBUG

#ifdef HYPRE_OFFLOAD_DEBUG
/* if we ``enter'' an address, it should not exist in device [o.w NO EFFECT] 
   if we ``exit'' or ''update'' an address, it should exist in device [o.w ERROR]
   hypre__offload_flag: 0 == OK; 1 == WRONG
 * */
#define HYPRE_OFFLOAD_FLAG(devnum, hptr, type) \
   HYPRE_Int hypre__offload_flag = \
   (type[1] == 'n') == omp_target_is_present(hptr, devnum);
#else 
#define HYPRE_OFFLOAD_FLAG(...) \
   HYPRE_Int hypre__offload_flag = 0;
#endif

/* OMP 4.5 offloading macro */
#define hypre_omp45_offload(devnum, hptr, datatype, offset, size, type1, type2) \
{\
   /* devnum: device number \
    * hptr: host poiter \
    * datatype: e.g., int, float, double, ... \
    * type1: ``e(n)ter'', ''e(x)it'', or ``u(p)date'' \
    * type2: ``(a)lloc'', ``(t)o'', ``(d)elete'', ''(f)rom'' \
    */ \
   datatype *hypre__offload_hptr = hptr; \
   /* if hypre__global_offload ==    0, or
    *    hptr (host pointer)   == NULL, 
    *    this offload will be IGNORED */ \
   if (hypre__global_offload && hypre__offload_hptr != NULL) { \
      /* offloading offset and size (in datatype) */ \
      HYPRE_Int hypre__offload_offset = offset, hypre__offload_size = size; \
      /* in HYPRE_OFFLOAD_DEBUG mode, we test if this offload has effect */ \
      HYPRE_OFFLOAD_FLAG(devnum, hypre__offload_hptr, type1) \
      if (hypre__offload_flag) { \
         printf("[!NO Effect! %s %d] device %d target: %6s %6s, data %p, [%d:%d]\n", __FILE__, __LINE__, devnum, type1, type2, (void *)hypre__offload_hptr, hypre__offload_offset, hypre__offload_size); exit(0); \
      } else { \
         HYPRE_Int offload_bytes = size * sizeof(datatype); \
         /* printf("[            %s %d] device %d target: %6s %6s, data %p, [%d:%d]\n", __FILE__, __LINE__, devnum, type1, type2, (void *)hypre__offload_hptr, hypre__offload_offset, hypre__offload_size); */ \
         if (type1[1] == 'n' && type2[0] == 't') { \
            /* enter to */\
            hypre__offload_to_count ++; hypre__offload_to_byte_count += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target enter data map(to:hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'n' && type2[0] == 'a') { \
            /* enter alloc */ \
            hypre__offload_to_count ++; hypre__offload_to_byte_count += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target enter data map(alloc:hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'x' && type2[0] == 'd') { \
            /* exit delete */\
            hypre__offload_from_count ++; hypre__offload_from_byte_count += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target exit data map(delete:hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'x' && type2[0] == 'f') {\
            /* exit from */ \
            hypre__offload_from_count ++; hypre__offload_from_byte_count += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target exit data map(from:hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'p' && type2[0] == 't') { \
            /* update to */ \
            hypre__offload_to_count ++; hypre__offload_to_byte_count += offload_bytes; \
            hypre__update_to_count ++; hypre__update_to_byte_count += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target update to(hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'p' && type2[0] == 'f') {\
            /* update from */ \
            hypre__offload_from_count ++; hypre__offload_from_byte_count += offload_bytes; \
            hypre__update_from_count ++; hypre__update_from_byte_count += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target update from(hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else {\
            printf("error: unrecognized offloading type combination!\n"); exit(-1); \
         } \
      } \
   } \
}

/* OMP offloading switch */
#define HYPRE_OMP45_OFFLOAD_ON() \
{ \
   fprintf(stdout, "Hypre OMP 4.5 Mapping/Offloading has been turned on\n"); \
   hypre__global_offload = 1; \
}

#define HYPRE_OMP45_OFFLOAD_OFF() \
{ \
   fprintf(stdout, "Hypre OMP 4.5 Mapping/Offloading has been turned off\n"); \
   hypre__global_offload = 0; \
}

/* NOTE:
 * if HYPRE_OMP45_OFFLOAD is turned off, then the associated 
 * device memory operations in the following macros will have no effect
 */ 

/* Device TAlloc */
#define hypre_DeviceTAlloc(type, count) \
({\
   type *ptr = hypre_TAlloc(type, count); \
   HYPRE_Int device_num = omp_get_default_device(); \
   hypre_omp45_offload(device_num, ptr, type, 0, count, "enter", "alloc"); \
   ptr; \
 })

/* Device CTAlloc */
#define hypre_DeviceCTAlloc(type, count) \
({\
   type *ptr = hypre_CTAlloc(type, count); \
   HYPRE_Int device_num = omp_get_default_device(); \
   hypre_omp45_offload(device_num, ptr, type, 0, count, "enter", "to"); \
   ptr; \
 })

/* Device TFree */
#define hypre_DeviceTFree(ptr, type, count) \
{\
   HYPRE_Int device_num = omp_get_default_device(); \
   hypre_omp45_offload(device_num, ptr, type, 0, count, "exit", "delete"); \
   hypre_TFree(ptr); \
}

/* DataCopyToData: HostToDevice 
 * src:  [from] a CPU ptr 
 * dest: [to]   a mapped CPU ptr */
#define hypre_DataCopyToData(src, dest, type, count) \
{\
   HYPRE_Int device_num = omp_get_default_device(); \
   /* CPU memcpy */ \
   if (dest != src) { \
      memcpy(dest, src, sizeof(type)*count); \
   } \
   /* update to device */ \
   hypre_omp45_offload(device_num, dest, type, 0, count, "update", "to"); \
}

/* DataCopyFromData: DeviceToHost 
 * src:  [from] a mapped CPU ptr
 * dest: [to]   a CPU ptr */
#define hypre_DataCopyFromData(dest, src, type, count) \
{\
   HYPRE_Int device_num = omp_get_default_device(); \
   /* update from device */ \
   hypre_omp45_offload(device_num, src, type, 0, count, "update", "from"); \
   /* CPU memcpy */ \
   if (dest != src) { \
      memcpy(dest, src, sizeof(type)*count); \
   } \
}

/* DeviceMemset 
 * memset: [to] a mapped CPU ptr
 * memset host memory first and the update the device memory */
#define hypre_DeviceMemset(ptr, value, type, count) \
{\
   HYPRE_Int device_num = omp_get_default_device(); \
   /* host memset */ \
   memset(ptr, value, (count)*sizeof(type)); \
   /* update to device */ \
   hypre_omp45_offload(device_num, ptr, type, 0, count, "update", "to"); \
}

/* UMTAlloc TODO */
#define hypre_UMTAlloc(type, count) hypre_TAlloc(type, count)

#define hypre_UMCTAlloc(type, count) hypre_CTAlloc(type, count)

#define hypre_UMTFree(ptr) hypre_TFree(ptr)

#define hypre_InitMemoryDebug(id)

#define hypre_FinalizeMemoryDebug()

#else /* ifdef at L29 */
#define HYPRE_CUDA_GLOBAL 

/*--------------------------------------------------------------------------
 * Use "Debug Malloc Library", dmalloc
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_MEMORY_DMALLOC

#define hypre_InitMemoryDebug(id)    hypre_InitMemoryDebugDML(id)
#define hypre_FinalizeMemoryDebug()  hypre_FinalizeMemoryDebugDML()

#define hypre_TAlloc(type, count) \
( (type *)hypre_MAllocDML((size_t)(sizeof(type) * (count)),\
                          __FILE__, __LINE__) )

#define hypre_CTAlloc(type, count) \
( (type *)hypre_CAllocDML((size_t)(count), (size_t)sizeof(type),\
                          __FILE__, __LINE__) )

#define hypre_TReAlloc(ptr, type, count) \
( (type *)hypre_ReAllocDML((char *)ptr,\
                           (size_t)(sizeof(type) * (count)),\
                           __FILE__, __LINE__) )

#define hypre_TFree(ptr) \
( hypre_FreeDML((char *)ptr, __FILE__, __LINE__), ptr = NULL )

/*--------------------------------------------------------------------------
 * Use standard memory routines
 *--------------------------------------------------------------------------*/

#else

#define hypre_InitMemoryDebug(id)
#define hypre_FinalizeMemoryDebug()  

#define hypre_TAlloc(type, count) \
( (type *)hypre_MAlloc((size_t)(sizeof(type) * (count))) )

#define hypre_CTAlloc(type, count) \
( (type *)hypre_CAlloc((size_t)(count), (size_t)sizeof(type)) )

#define hypre_TReAlloc(ptr, type, count) \
( (type *)hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count))) )

#define hypre_TFree(ptr) \
( hypre_Free((char *)ptr), ptr = NULL )

#endif

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

/* memory_dmalloc.c */
HYPRE_Int hypre_InitMemoryDebugDML( HYPRE_Int id );
HYPRE_Int hypre_FinalizeMemoryDebugDML( void );
char *hypre_MAllocDML( HYPRE_Int size , char *file , HYPRE_Int line );
char *hypre_CAllocDML( HYPRE_Int count , HYPRE_Int elt_size , char *file , HYPRE_Int line );
char *hypre_ReAllocDML( char *ptr , HYPRE_Int size , char *file , HYPRE_Int line );
void hypre_FreeDML( char *ptr , char *file , HYPRE_Int line );

#ifdef __cplusplus
}
#endif

#endif


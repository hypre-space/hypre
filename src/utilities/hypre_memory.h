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
 *       MemoryCopy(..., locTo,locFrom)
 *             locTo=LOCATION_HOST,   locFrom=LOCATION_DEVICE  - copy from host to host e.g., memcpy
 *             locTo=LOCATION_DEVICE, locFrom=LOCATION_HOST    - copy from host to host e.g., memcpy
 *             locTo=LOCATION_DEVICE, locFrom=LOCATION_DEVICE  - copy from host to host e.g., memcpy
 *       SetExecutionMode
 *             location=LOCATION_DEVICE - execute on the host
 *             location=LOCATION_HOST   - execute on the host    
 *
 *    Using a device, not using managed memory
 *       Malloc(..., location) 
 *             location=LOCATION_DEVICE - device malloc        e.g., cudaMalloc
 *             location=LOCATION_HOST   - host malloc          e.g., malloc
 *       MemoryCopy(..., locTo,locFrom)
 *             locTo=LOCATION_HOST,   locFrom=LOCATION_DEVICE  - copy from device to host e.g., cudaMemcpy
 *             locTo=LOCATION_DEVICE, locFrom=LOCATION_HOST    - copy from host to device e.g., cudaMemcpy
 *             locTo=LOCATION_DEVICE, locFrom=LOCATION_DEVICE  - copy from device to device e.g., cudaMemcpy
 *       SetExecutionMode
 *             location=LOCATION_DEVICE - execute on the device
 *             location=LOCATION_HOST   - execute on the host 
 *
 *    Using a device, using managed memory
 *       Malloc(..., location) 
 *             location=LOCATION_DEVICE - managed malloc        e.g., cudaMallocManaged
 *             location=LOCATION_HOST   - host malloc          e.g., malloc
 *       MemoryCopy(..., locTo,locFrom)
 *             locTo=LOCATION_HOST,   locFrom=LOCATION_DEVICE  - copy from device to host e.g., cudaMallocManaged
 *             locTo=LOCATION_DEVICE, locFrom=LOCATION_HOST    - copy from host to device e.g., cudaMallocManaged
 *             locTo=LOCATION_DEVICE, locFrom=LOCATION_DEVICE  - copy from device to device e.g., cudaMallocManaged
 *       SetExecutionMode
 *             location=LOCATION_DEVICE - execute on the device
 *             location=LOCATION_HOST   - execute on the host 
 *
 * Questions:
 *
 *    1. Pinned memory, prefetch?
 *
 *****************************************************************************/

#ifndef hypre_MEMORY_HEADER
#define hypre_MEMORY_HEADER

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HYPRE_MEMORY_DEVICE ( 0)
#define HYPRE_MEMORY_HOST   ( 1)
#define HYPRE_MEMORY_SHARED ( 2)
#define HYPRE_MEMORY_UNSET  (-1)

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
#else
#define HYPRE_CUDA_GLOBAL 
#endif

#define hypre_InitMemoryDebug(id)
#define hypre_FinalizeMemoryDebug()

#define hypre_TAlloc(type, count, location) \
( (type *)hypre_MAlloc((size_t)(sizeof(type) * (count)), location) )

#define hypre_CTAlloc(type, count, location) \
( (type *)hypre_CAlloc((size_t)(count), (size_t)sizeof(type), location) )

#define hypre_TReAlloc(ptr, type, count, location) \
( (type *)hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count)), location) )

#define hypre_TFree(ptr,location) \
( hypre_Free((char *)ptr, location), ptr = NULL )

#define hypre_TMemcpy(dst, src, type, count, locdst, locsrc) \
(hypre_Memcpy((char *)(dst),(char *)(src),(size_t)(sizeof(type) * (count)),locdst, locsrc))

#define hypre_DeviceMemset(ptr,value,type,count)	memset(ptr,value,count*sizeof(type))
  
#define hypre_PinnedTAlloc(type, count)\
( (type *)hypre_MAllocPinned((size_t)(sizeof(type) * (count))) )

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* hypre_memory.c */
HYPRE_Int hypre_OutOfMemory ( size_t size );
char *hypre_MAlloc( size_t size , HYPRE_Int location );
char *hypre_CAlloc( size_t count ,  size_t elt_size , HYPRE_Int location);
char *hypre_MAllocPinned( size_t size );
char *hypre_ReAlloc( char *ptr ,  size_t size , HYPRE_Int location);
void hypre_Free( char *ptr , HYPRE_Int location );
char *hypre_CAllocHost( size_t count,size_t elt_size );
char *hypre_MAllocHost( size_t size );
char *hypre_ReAllocHost( char   *ptr,size_t  size );
void hypre_FreeHost( char *ptr );
char *hypre_SharedMAlloc ( size_t size );
char *hypre_SharedCAlloc ( size_t count , size_t elt_size );
char *hypre_SharedReAlloc ( char *ptr , size_t size );
void hypre_SharedFree ( char *ptr );
void hypre_Memcpy( char *dst, char *src, size_t size, HYPRE_Int locdst, HYPRE_Int locsrc );
void hypre_MemcpyAsync( char *dst, char *src, size_t size, HYPRE_Int locdst, HYPRE_Int locsrc );	
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


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
 *    1. prefetch?
 *
 *****************************************************************************/

#ifndef hypre_MEMORY_HEADER
#define hypre_MEMORY_HEADER

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HYPRE_MEMORY_UNSET         (-1)
#define HYPRE_MEMORY_DEVICE        ( 0)
#define HYPRE_MEMORY_HOST          ( 1)
#define HYPRE_MEMORY_SHARED        ( 2)
#define HYPRE_MEMORY_HOST_PINNED   ( 3)

/*==================================================================
 *       default def of memory location selected based memory env
 *   +-------------------------------------------------------------+
 *   |                           |          HYPRE_MEMORY_*         |
 *   |        MEM \ LOC          | HOST | DEVICE | SHARED | PINNED |
 *   |---------------------------+---------------+-----------------|
 *   | HYPRE_USING_HOST_MEMORY   | HOST | HOST   | HOST   | HOST   |
 *   |---------------------------+---------------+-------- --------|
 *   | HYPRE_USING_DEVICE_MEMORY | HOST | DEVICE | DEVICE | PINNED |
 *   |---------------------------+---------------+-----------------|
 *   | HYPRE_USING_UNIFIED_MEMORY| HOST | DEVICE | SHARED | PINNED |
 *   +-------------------------------------------------------------+
 *==================================================================*/

#if defined(HYPRE_USING_HOST_MEMORY)

/* default memory model without device (host only) */
#define HYPRE_MEMORY_HOST_ACT         HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_DEVICE_ACT       HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_SHARED_ACT       HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_HOST_PINNED_ACT  HYPRE_MEMORY_HOST

#elif defined(HYPRE_USING_DEVICE_MEMORY)

/* default memory model with device and without unified memory */
#define HYPRE_MEMORY_HOST_ACT         HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_DEVICE_ACT       HYPRE_MEMORY_DEVICE
#define HYPRE_MEMORY_SHARED_ACT       HYPRE_MEMORY_DEVICE
#define HYPRE_MEMORY_HOST_PINNED_ACT  HYPRE_MEMORY_HOST_PINNED

#elif defined(HYPRE_USING_UNIFIED_MEMORY)

/* default memory model with device and with unified memory */
#define HYPRE_MEMORY_HOST_ACT         HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_DEVICE_ACT       HYPRE_MEMORY_DEVICE
#define HYPRE_MEMORY_SHARED_ACT       HYPRE_MEMORY_SHARED
#define HYPRE_MEMORY_HOST_PINNED_ACT  HYPRE_MEMORY_HOST_PINNED

#else

/* default */
#define HYPRE_MEMORY_HOST_ACT         HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_DEVICE_ACT       HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_SHARED_ACT       HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_HOST_PINNED_ACT  HYPRE_MEMORY_HOST

#endif

/* the above definitions might be overridden to customize a memory location */
/* #undef  HYPRE_MEMORY_HOST_ACT */
/* #undef  HYPRE_MEMORY_DEVICE_ACT */
/* #undef  HYPRE_MEMORY_SHARED_ACT */
/* #undef  HYPRE_MEMORY_PINNED_ACT */
/* #define HYPRE_MEMORY_HOST_ACT    HYPRE_MEMORY_? */
/* #define HYPRE_MEMORY_DEVICE_ACT  HYPRE_MEMORY_? */
/* #define HYPRE_MEMORY_SHARED_ACT  HYPRE_MEMORY_? */
/* #define HYPRE_MEMORY_PINNED_ACT  HYPRE_MEMORY_? */

#define HYPRE_MEM_PAD_LEN 1

/*
#if defined(HYPRE_USING_CUDA)
#define HYPRE_CUDA_GLOBAL __host__ __device__
#else
#define HYPRE_CUDA_GLOBAL
#endif
*/

/* OpenMP 4.5 */
#if defined(HYPRE_USING_DEVICE_OPENMP)

#include "omp.h"

/* stringification:
 * _Pragma(string-literal), so we need to cast argument to a string
 * The three dots as last argument of the macro tells compiler that this is a variadic macro.
 * I.e. this is a macro that receives variable number of arguments.
 */
#define HYPRE_STR(s...) #s
#define HYPRE_XSTR(s...) HYPRE_STR(s)

/* OpenMP 4.5 device memory management */
extern HYPRE_Int hypre__global_offload;
extern HYPRE_Int hypre__offload_device_num;
extern HYPRE_Int hypre__offload_host_num;

/* stats */
extern size_t hypre__target_allc_count;
extern size_t hypre__target_free_count;
extern size_t hypre__target_allc_bytes;
extern size_t hypre__target_free_bytes;
extern size_t hypre__target_htod_count;
extern size_t hypre__target_dtoh_count;
extern size_t hypre__target_htod_bytes;
extern size_t hypre__target_dtoh_bytes;

/* DEBUG MODE: check if offloading has effect
 * (it is turned on when configured with --enable-debug) */

#ifdef HYPRE_OMP45_DEBUG
/* if we ``enter'' an address, it should not exist in device [o.w NO EFFECT]
   if we ``exit'' or ''update'' an address, it should exist in device [o.w ERROR]
hypre__offload_flag: 0 == OK; 1 == WRONG
 */
#define HYPRE_OFFLOAD_FLAG(devnum, hptr, type) \
   HYPRE_Int hypre__offload_flag = (type[1] == 'n') == omp_target_is_present(hptr, devnum);
#else
#define HYPRE_OFFLOAD_FLAG(...) \
   HYPRE_Int hypre__offload_flag = 0; /* non-debug mode, always OK */
#endif

/* OMP 4.5 offloading macro */
#define hypre_omp45_offload(devnum, hptr, datatype, offset, count, type1, type2) \
{\
   /* devnum: device number \
    * hptr: host poiter \
    * datatype \
    * type1: ``e(n)ter'', ''e(x)it'', or ``u(p)date'' \
    * type2: ``(a)lloc'', ``(t)o'', ``(d)elete'', ''(f)rom'' \
    */ \
   datatype *hypre__offload_hptr = (datatype *) hptr; \
   /* if hypre__global_offload ==    0, or
    *    hptr (host pointer)   == NULL,
    *    this offload will be IGNORED */ \
   if (hypre__global_offload && hypre__offload_hptr != NULL) { \
      /* offloading offset and size (in datatype) */ \
      size_t hypre__offload_offset = offset, hypre__offload_size = count; \
      /* in HYPRE_OMP45_DEBUG mode, we test if this offload has effect */ \
      HYPRE_OFFLOAD_FLAG(devnum, hypre__offload_hptr, type1) \
      if (hypre__offload_flag) { \
         printf("[!NO Effect! %s %d] device %d target: %6s %6s, data %p, [%ld:%ld]\n", __FILE__, __LINE__, devnum, type1, type2, (void *)hypre__offload_hptr, hypre__offload_offset, hypre__offload_size); exit(0); \
      } else { \
         size_t offload_bytes = count * sizeof(datatype); \
         /* printf("[            %s %d] device %d target: %6s %6s, data %p, [%d:%d]\n", __FILE__, __LINE__, devnum, type1, type2, (void *)hypre__offload_hptr, hypre__offload_offset, hypre__offload_size); */ \
         if (type1[1] == 'n' && type2[0] == 't') { \
            /* enter to */\
            hypre__target_allc_count ++; \
            hypre__target_allc_bytes += offload_bytes; \
            hypre__target_htod_count ++; \
            hypre__target_htod_bytes += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target enter data map(to:hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'n' && type2[0] == 'a') { \
            /* enter alloc */ \
            hypre__target_allc_count ++; \
            hypre__target_allc_bytes += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target enter data map(alloc:hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'x' && type2[0] == 'd') { \
            /* exit delete */\
            hypre__target_free_count ++; \
            hypre__target_free_bytes += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target exit data map(delete:hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'x' && type2[0] == 'f') {\
            /* exit from */ \
            hypre__target_free_count ++; \
            hypre__target_free_bytes += offload_bytes; \
            hypre__target_dtoh_count ++; \
            hypre__target_dtoh_bytes += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target exit data map(from:hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'p' && type2[0] == 't') { \
            /* update to */ \
            hypre__target_htod_count ++; \
            hypre__target_htod_bytes += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target update to(hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'p' && type2[0] == 'f') {\
            /* update from */ \
            hypre__target_dtoh_count ++; \
            hypre__target_dtoh_bytes += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target update from(hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else {\
            printf("error: unrecognized offloading type combination!\n"); exit(-1); \
         } \
      } \
   } \
}

#endif /*  #if defined(HYPRE_USING_DEVICE_OPENMP) */

/*
#define hypre_InitMemoryDebug(id)
#define hypre_FinalizeMemoryDebug()
*/
//#define TRACK_MEMORY_ALLOCATIONS
#if defined(TRACK_MEMORY_ALLOCATIONS)

typedef struct {
  char *file;
  size_t size;
  void *end;
  HYPRE_Int line;
  HYPRE_Int type;} pattr_t;

pattr_t *patpush(void *ptr, pattr_t *ss);

#define hypre_TAlloc(type, count, location) \
  ( (type *)hypre_MAllocIns((size_t)(sizeof(type) * (count)), location,__FILE__,__LINE__) )

#define hypre_CTAlloc(type, count, location) \
  ( (type *)hypre_CAllocIns((size_t)(count), (size_t)sizeof(type), location,__FILE__,__LINE__) )

#define hypre_TReAlloc(ptr, type, count, location) \
  ( (type *)hypre_ReAllocIns((char *)ptr, (size_t)(sizeof(type) * (count)), location,__FILE__,__LINE__) )

void assert_check(void *ptr, char *file, HYPRE_Int line);

void assert_check_host(void *ptr, char *file, HYPRE_Int line);


#define ASSERT_MANAGED(ptr)\
  ( assert_check((ptr),__FILE__,__LINE__))

#define ASSERT_HOST(ptr)\
  ( assert_check_host((ptr),__FILE__,__LINE__))

#else

#if 0

/* These Allocs are with printfs, for debug */
#define hypre_TAlloc(type, count, location) \
(\
 /*printf("[%s:%d] MALLOC %ld B\n", __FILE__,__LINE__, (size_t)(sizeof(type) * (count))) ,*/ \
 (type *) hypre_MAlloc((size_t)(sizeof(type) * (count)), location) \
)

#define hypre_CTAlloc(type, count, location) \
(\
{\
 /* if (location == HYPRE_MEMORY_DEVICE) printf("[%s:%d] CTALLOC %.3f MB\n", __FILE__,__LINE__, (size_t)(sizeof(type) * (count))/1024.0/1024.0); */ \
 (type *) hypre_CAlloc((size_t)(count), (size_t)sizeof(type), location); \
}\
)

#define hypre_TReAlloc(ptr, type, count, location) \
(\
 /* printf("[%s:%d] TReALLOC %ld B\n", __FILE__,__LINE__, (size_t)(sizeof(type) * (count))) , */ \
 (type *)hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count)), location) \
)

#else

#define hypre_TAlloc(type, count, location) \
( (type *) hypre_MAlloc((size_t)(sizeof(type) * (count)), location) )

#define hypre_CTAlloc(type, count, location) \
( (type *) hypre_CAlloc((size_t)(count), (size_t)sizeof(type), location) )

#define hypre_TReAlloc(ptr, type, count, location) \
( (type *) hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count)), location) )

#endif

#endif

#define hypre_TFree(ptr,location) \
( hypre_Free((char *)ptr, location), ptr = NULL )

#define hypre_TMemcpy(dst, src, type, count, locdst, locsrc) \
(hypre_Memcpy((char *)(dst),(char *)(src),(size_t)(sizeof(type) * (count)),locdst, locsrc))

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* hypre_memory.c */
#if 0
char *hypre_CAllocIns( size_t count ,  size_t elt_size , HYPRE_Int location,char *file, HYPRE_Int line);
char *hypre_ReAllocIns( char *ptr ,  size_t size , HYPRE_Int location,char *file, HYPRE_Int line);
char *hypre_MAllocIns( size_t size , HYPRE_Int location,char *file,HYPRE_Int line);
char *hypre_MAllocPinned( size_t size );
#else
void * hypre_MAlloc(size_t size, HYPRE_Int location);
void * hypre_CAlloc( size_t count, size_t elt_size, HYPRE_Int location);
void * hypre_ReAlloc(void *ptr, size_t size, HYPRE_Int location);
void   hypre_Memcpy(void *dst, void *src, size_t size, HYPRE_Int loc_dst, HYPRE_Int loc_src);
void * hypre_Memset(void *ptr, HYPRE_Int value, size_t num, HYPRE_Int location);
void   hypre_Free(void *ptr, HYPRE_Int location);
#endif
/*
char *hypre_CAllocHost( size_t count,size_t elt_size );
char *hypre_MAllocHost( size_t size );
char *hypre_ReAllocHost( char   *ptr,size_t  size );
void hypre_FreeHost( char *ptr );
char *hypre_SharedMAlloc ( size_t size );
char *hypre_SharedCAlloc ( size_t count , size_t elt_size );
char *hypre_SharedReAlloc ( char *ptr , size_t size );
void hypre_SharedFree ( char *ptr );
void hypre_MemcpyAsync( char *dst, char *src, size_t size, HYPRE_Int locdst, HYPRE_Int locsrc );
HYPRE_Real *hypre_IncrementSharedDataPtr ( HYPRE_Real *ptr , size_t size );
*/

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


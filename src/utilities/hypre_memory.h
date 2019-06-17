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

#define HYPRE_EXEC_UNSET           (-1)
#define HYPRE_EXEC_DEVICE          ( 0)
#define HYPRE_EXEC_HOST            ( 1)

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
//#define HYPRE_MEMORY_SHARED_ACT       HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_HOST_PINNED_ACT  HYPRE_MEMORY_HOST_PINNED

#else

/* default */
#define HYPRE_MEMORY_HOST_ACT         HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_DEVICE_ACT       HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_SHARED_ACT       HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_HOST_PINNED_ACT  HYPRE_MEMORY_HOST

#endif

/* the above definitions might be overridden to customize
 * memory locations */

/* #undef  HYPRE_MEMORY_HOST_ACT */
/* #undef  HYPRE_MEMORY_DEVICE_ACT */
/* #undef  HYPRE_MEMORY_SHARED_ACT */
/* #undef  HYPRE_MEMORY_PINNED_ACT */
/* #define HYPRE_MEMORY_HOST_ACT    HYPRE_MEMORY_? */
/* #define HYPRE_MEMORY_DEVICE_ACT  HYPRE_MEMORY_? */
/* #define HYPRE_MEMORY_SHARED_ACT  HYPRE_MEMORY_? */
/* #define HYPRE_MEMORY_PINNED_ACT  HYPRE_MEMORY_? */

/*-------------------------------------------------------
 * hypre_GetActualMemLocation
 *   return actual location based on the selected memory model
 *-------------------------------------------------------*/
static inline HYPRE_Int
hypre_GetActualMemLocation(HYPRE_Int location)
{
   if (location == HYPRE_MEMORY_HOST)
   {
      return HYPRE_MEMORY_HOST_ACT;
   }

   if (location == HYPRE_MEMORY_DEVICE)
   {
      return HYPRE_MEMORY_DEVICE_ACT;
   }

   if (location == HYPRE_MEMORY_SHARED)
   {
      return HYPRE_MEMORY_SHARED_ACT;
   }

   if (location == HYPRE_MEMORY_HOST_PINNED)
   {
      return HYPRE_MEMORY_HOST_PINNED_ACT;
   }

   return HYPRE_MEMORY_UNSET;
}

/*---------------------------------------------------
 * hypre_GetExecPolicy
 * Return execution policy based on memory locations
 *---------------------------------------------------*/
/* for unary operation */
static inline HYPRE_Int
hypre_GetExecPolicy1(HYPRE_Int location)
{
   HYPRE_Int exec = HYPRE_EXEC_UNSET;

   location = hypre_GetActualMemLocation(location);

   switch (location)
   {
      case HYPRE_MEMORY_HOST :
      case HYPRE_MEMORY_HOST_PINNED :
         exec = HYPRE_EXEC_HOST;
         break;
      case HYPRE_MEMORY_DEVICE :
         exec = HYPRE_EXEC_DEVICE;
         break;
      case HYPRE_MEMORY_SHARED :
         exec = HYPRE_EXEC_HOST;
         break;
   }

   return exec;
}

/* for binary operation */
static inline HYPRE_Int
hypre_GetExecPolicy2(HYPRE_Int location1,
                     HYPRE_Int location2)
{
   location1 = hypre_GetActualMemLocation(location1);
   location2 = hypre_GetActualMemLocation(location2);

   /* HOST_PINNED has the same exec policy as HOST */
   if (location1 == HYPRE_MEMORY_HOST_PINNED)
   {
      location1 = HYPRE_MEMORY_HOST;
   }

   if (location2 == HYPRE_MEMORY_HOST_PINNED)
   {
      location2 = HYPRE_MEMORY_HOST;
   }

   /* no policy for these combinations */
   if ( (location1 == HYPRE_MEMORY_HOST && location2 == HYPRE_MEMORY_DEVICE) ||
        (location2 == HYPRE_MEMORY_HOST && location1 == HYPRE_MEMORY_DEVICE) )
   {
      return HYPRE_EXEC_UNSET;
   }

   /* policy for S-S can be HOST or DEVICE. Choose HOST by default */
   if (location1 == HYPRE_MEMORY_SHARED && location2 == HYPRE_MEMORY_SHARED)
   {
      return HYPRE_EXEC_HOST;
   }

   if (location1 == HYPRE_MEMORY_HOST || location2 == HYPRE_MEMORY_HOST)
   {
      return HYPRE_EXEC_HOST;
   }

   if (location1 == HYPRE_MEMORY_DEVICE || location2 == HYPRE_MEMORY_DEVICE)
   {
      return HYPRE_EXEC_DEVICE;
   }

   return HYPRE_EXEC_UNSET;
}

#define HYPRE_MEM_PAD_LEN 1


#if 0
/* These Allocs are with printfs, for debug */
#define hypre_TAlloc(type, count, location) \
(\
{\
 if (location == HYPRE_MEMORY_DEVICE) printf("[%s:%d] TALLOC %.3f MB\n", __FILE__,__LINE__, (size_t)(sizeof(type) * (count))/1024.0/1024.0); \
 (type *) hypre_MAlloc((size_t)(sizeof(type) * (count)), location); \
}\
)

#define hypre_CTAlloc(type, count, location) \
(\
{\
 if (location == HYPRE_MEMORY_DEVICE) printf("[%s:%d] CTALLOC %.3f MB\n", __FILE__,__LINE__, (size_t)(sizeof(type) * (count))/1024.0/1024.0); \
 (type *) hypre_CAlloc((size_t)(count), (size_t)sizeof(type), location); \
}\
)

#define hypre_TReAlloc(ptr, type, count, location) \
(\
{\
 if (location == HYPRE_MEMORY_DEVICE) printf("[%s:%d] TReALLOC %p, %.3f MB\n", __FILE__,__LINE__, ptr, (size_t)(sizeof(type) * (count))/1024.0/1024.0); \
 (type *)hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count)), location); \
}\
)

#define hypre_TMemcpy(dst, src, type, count, locdst, locsrc) \
( \
{ \
  printf("[%s:%d] TMemcpy %d to %d %.3f MB\n", __FILE__,__LINE__, locsrc, locdst, (size_t)(sizeof(type) * (count))/1024.0/1024.0); \
  hypre_Memcpy((void *)(dst), (void *)(src), (size_t)(sizeof(type) * (count)), locdst, locsrc); \
} \
)

#else

#define hypre_TAlloc(type, count, location) \
( (type *) hypre_MAlloc((size_t)(sizeof(type) * (count)), location) )

#define hypre_CTAlloc(type, count, location) \
( (type *) hypre_CAlloc((size_t)(count), (size_t)sizeof(type), location) )

#define hypre_TReAlloc(ptr, type, count, location) \
( (type *) hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count)), location) )

#define hypre_TMemcpy(dst, src, type, count, locdst, locsrc) \
(hypre_Memcpy((void *)(dst), (void *)(src), (size_t)(sizeof(type) * (count)), locdst, locsrc))

#endif

#define hypre_TFree(ptr, location) \
( hypre_Free((void *)ptr, location), ptr = NULL )

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


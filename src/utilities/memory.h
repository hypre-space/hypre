/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
 *****************************************************************************/

#ifndef hypre_MEMORY_HEADER
#define hypre_MEMORY_HEADER

#include <stdio.h>
#include <stdlib.h>

#if defined(HYPRE_USING_UMPIRE)
#include "umpire/interface/umpire.h"
#define HYPRE_UMPIRE_POOL_NAME_MAX_LEN 1024
#endif

/* stringification:
 * _Pragma(string-literal), so we need to cast argument to a string
 * The three dots as last argument of the macro tells compiler that this is a variadic macro.
 * I.e. this is a macro that receives variable number of arguments.
 */
#define HYPRE_STR(...) #__VA_ARGS__
#define HYPRE_XSTR(...) HYPRE_STR(__VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum _hypre_MemoryLocation
{
   hypre_MEMORY_UNDEFINED = -1,
   hypre_MEMORY_HOST          ,
   hypre_MEMORY_HOST_PINNED   ,
   hypre_MEMORY_DEVICE        ,
   hypre_MEMORY_UNIFIED
} hypre_MemoryLocation;

/*-------------------------------------------------------
 * hypre_GetActualMemLocation
 *   return actual location based on the selected memory model
 *-------------------------------------------------------*/
static inline hypre_MemoryLocation
hypre_GetActualMemLocation(HYPRE_MemoryLocation location)
{
   if (location == HYPRE_MEMORY_HOST)
   {
      return hypre_MEMORY_HOST;
   }

   if (location == HYPRE_MEMORY_DEVICE)
   {
#if defined(HYPRE_USING_HOST_MEMORY)
      return hypre_MEMORY_HOST;
#elif defined(HYPRE_USING_DEVICE_MEMORY)
      return hypre_MEMORY_DEVICE;
#elif defined(HYPRE_USING_UNIFIED_MEMORY)
      return hypre_MEMORY_UNIFIED;
#else
#error Wrong HYPRE memory setting.
#endif
   }

   return hypre_MEMORY_UNDEFINED;
}

#ifdef HYPRE_USING_MEMORY_TRACKER

typedef struct
{
   char                  _action[16];
   void                 *_ptr;
   size_t                _nbytes;
   hypre_MemoryLocation  _memory_location;
   char                  _filename[256];
   char                  _function[256];
   HYPRE_Int             _line;
   size_t                _pair;
} hypre_MemoryTrackerEntry;

typedef struct
{
   size_t actual_size;
   size_t alloced_size;
   size_t prev_end;
   hypre_MemoryTrackerEntry *data;
} hypre_MemoryTracker;

/* These Allocs are with memory tracker, for debug */
#define hypre_TAlloc(type, count, location)                                                                                           \
(                                                                                                                                     \
{                                                                                                                                     \
   void *ptr = hypre_MAlloc((size_t)(sizeof(type) * (count)), location);                                                              \
   hypre_MemoryTrackerInsert("malloc", ptr, sizeof(type)*(count), hypre_GetActualMemLocation(location), __FILE__, __func__, __LINE__);\
   (type *) ptr;                                                                                                                      \
}                                                                                                                                     \
)

#define _hypre_TAlloc(type, count, location)                                                                                          \
(                                                                                                                                     \
{                                                                                                                                     \
   void *ptr = _hypre_MAlloc((size_t)(sizeof(type) * (count)), location);                                                             \
   hypre_MemoryTrackerInsert("malloc", ptr, sizeof(type)*(count), location, __FILE__, __func__, __LINE__);                            \
   (type *) ptr;                                                                                                                      \
}                                                                                                                                     \
)

#define hypre_CTAlloc(type, count, location)                                                                                          \
(                                                                                                                                     \
{                                                                                                                                     \
   void *ptr = hypre_CAlloc((size_t)(count), (size_t)sizeof(type), location);                                                         \
   hypre_MemoryTrackerInsert("calloc", ptr, sizeof(type)*(count), hypre_GetActualMemLocation(location), __FILE__, __func__, __LINE__);\
   (type *) ptr;                                                                                                                      \
}                                                                                                                                     \
)

#define hypre_TReAlloc(ptr, type, count, location)                                                                                         \
(                                                                                                                                          \
{                                                                                                                                          \
   hypre_MemoryTrackerInsert("rfree", ptr, (size_t) -1, hypre_GetActualMemLocation(location), __FILE__, __func__, __LINE__);               \
   void *new_ptr = hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count)), location);                                                 \
   hypre_MemoryTrackerInsert("rmalloc", new_ptr, sizeof(type)*(count), hypre_GetActualMemLocation(location), __FILE__, __func__, __LINE__);\
   (type *) new_ptr;                                                                                                                       \
}                                                                                                                                          \
)

#define hypre_TReAlloc_v2(ptr, old_type, old_count, new_type, new_count, location)                                                                 \
(                                                                                                                                                  \
{                                                                                                                                                  \
   hypre_MemoryTrackerInsert("rfree", ptr, sizeof(old_type)*(old_count), hypre_GetActualMemLocation(location), __FILE__, __func__, __LINE__);      \
   void *new_ptr = hypre_ReAlloc_v2((char *)ptr, (size_t)(sizeof(old_type)*(old_count)), (size_t)(sizeof(new_type)*(new_count)), location);        \
   hypre_MemoryTrackerInsert("rmalloc", new_ptr, sizeof(new_type)*(new_count), hypre_GetActualMemLocation(location), __FILE__, __func__, __LINE__);\
   (new_type *) new_ptr;                                                                                                                           \
}                                                                                                                                                  \
)

#define hypre_TMemcpy(dst, src, type, count, locdst, locsrc)                                     \
(                                                                                                \
{                                                                                                \
   hypre_Memcpy((void *)(dst), (void *)(src), (size_t)(sizeof(type) * (count)), locdst, locsrc); \
}                                                                                                \
)

#define hypre_TFree(ptr, location)                                                                                          \
(                                                                                                                           \
{                                                                                                                           \
   hypre_MemoryTrackerInsert("free", ptr, (size_t) -1, hypre_GetActualMemLocation(location), __FILE__, __func__, __LINE__); \
   hypre_Free((void *)ptr, location);                                                                                       \
   ptr = NULL;                                                                                                              \
}                                                                                                                           \
)

#define _hypre_TFree(ptr, location)                                                                             \
(                                                                                                               \
{                                                                                                               \
   hypre_MemoryTrackerInsert("free", ptr, (size_t) -1, location, __FILE__, __func__, __LINE__);                 \
   _hypre_Free((void *)ptr, location);                                                                          \
   ptr = NULL;                                                                                                  \
}                                                                                                               \
)

#else /* #ifdef HYPRE_USING_MEMORY_TRACKER */

#define hypre_TAlloc(type, count, location) \
( (type *) hypre_MAlloc((size_t)(sizeof(type) * (count)), location) )

#define _hypre_TAlloc(type, count, location) \
( (type *) _hypre_MAlloc((size_t)(sizeof(type) * (count)), location) )

#define hypre_CTAlloc(type, count, location) \
( (type *) hypre_CAlloc((size_t)(count), (size_t)sizeof(type), location) )

#define hypre_TReAlloc(ptr, type, count, location) \
( (type *) hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count)), location) )

#define hypre_TReAlloc_v2(ptr, old_type, old_count, new_type, new_count, location) \
( (new_type *) hypre_ReAlloc_v2((char *)ptr, (size_t)(sizeof(old_type)*(old_count)), (size_t)(sizeof(new_type)*(new_count)), location) )

#define hypre_TMemcpy(dst, src, type, count, locdst, locsrc) \
(hypre_Memcpy((void *)(dst), (void *)(src), (size_t)(sizeof(type) * (count)), locdst, locsrc))

#define hypre_TFree(ptr, location) \
( hypre_Free((void *)ptr, location), ptr = NULL )

#define _hypre_TFree(ptr, location) \
( _hypre_Free((void *)ptr, location), ptr = NULL )

#endif /* #ifdef HYPRE_USING_MEMORY_TRACKER */


/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* memory.c */
void * hypre_Memset(void *ptr, HYPRE_Int value, size_t num, HYPRE_MemoryLocation location);
void   hypre_MemPrefetch(void *ptr, size_t size, HYPRE_MemoryLocation location);
void * hypre_MAlloc(size_t size, HYPRE_MemoryLocation location);
void * hypre_CAlloc( size_t count, size_t elt_size, HYPRE_MemoryLocation location);
void   hypre_Free(void *ptr, HYPRE_MemoryLocation location);
void   hypre_Memcpy(void *dst, void *src, size_t size, HYPRE_MemoryLocation loc_dst, HYPRE_MemoryLocation loc_src);
void * hypre_ReAlloc(void *ptr, size_t size, HYPRE_MemoryLocation location);
void * hypre_ReAlloc_v2(void *ptr, size_t old_size, size_t new_size, HYPRE_MemoryLocation location);

void * _hypre_MAlloc(size_t size, hypre_MemoryLocation location);
void   _hypre_Free(void *ptr, hypre_MemoryLocation location);

HYPRE_ExecutionPolicy hypre_GetExecPolicy1(HYPRE_MemoryLocation location);
HYPRE_ExecutionPolicy hypre_GetExecPolicy2(HYPRE_MemoryLocation location1, HYPRE_MemoryLocation location2);

HYPRE_Int hypre_GetPointerLocation(const void *ptr, hypre_MemoryLocation *memory_location);
HYPRE_Int hypre_PrintMemoryTracker();
HYPRE_Int hypre_SetCubMemPoolSize( hypre_uint bin_growth, hypre_uint min_bin, hypre_uint max_bin, size_t max_cached_bytes );
HYPRE_Int hypre_umpire_host_pooled_allocate(void **ptr, size_t nbytes);
HYPRE_Int hypre_umpire_host_pooled_free(void *ptr);
void *hypre_umpire_host_pooled_realloc(void *ptr, size_t size);
HYPRE_Int hypre_umpire_device_pooled_allocate(void **ptr, size_t nbytes);
HYPRE_Int hypre_umpire_device_pooled_free(void *ptr);
HYPRE_Int hypre_umpire_um_pooled_allocate(void **ptr, size_t nbytes);
HYPRE_Int hypre_umpire_um_pooled_free(void *ptr);
HYPRE_Int hypre_umpire_pinned_pooled_allocate(void **ptr, size_t nbytes);
HYPRE_Int hypre_umpire_pinned_pooled_free(void *ptr);

#ifdef HYPRE_USING_MEMORY_TRACKER
hypre_MemoryTracker * hypre_MemoryTrackerCreate();
void hypre_MemoryTrackerDestroy(hypre_MemoryTracker *tracker);
void hypre_MemoryTrackerInsert(const char *action, void *ptr, size_t nbytes, hypre_MemoryLocation memory_location, const char *filename, const char *function, HYPRE_Int line);
HYPRE_Int hypre_PrintMemoryTracker();
#endif

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


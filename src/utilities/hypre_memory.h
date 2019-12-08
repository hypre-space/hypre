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

/* stringification:
 * _Pragma(string-literal), so we need to cast argument to a string
 * The three dots as last argument of the macro tells compiler that this is a variadic macro.
 * I.e. this is a macro that receives variable number of arguments.
 */
#define HYPRE_STR(...) #__VA_ARGS__
#define HYPRE_XSTR(...) HYPRE_STR(__VA_ARGS__)

//#define HYPRE_USING_MEMORY_TRACKER

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

#ifdef HYPRE_USING_CUB_ALLOCATOR
#ifdef __cplusplus
extern "C++" {
#endif
#include "cub/util_allocator.cuh"
#ifdef __cplusplus
}
#endif
#endif

#ifdef HYPRE_USING_MEMORY_TRACKER

#ifdef __cplusplus
extern "C++" {
#endif

#include <vector>

struct hypre_memory_tracker_t
{
   char      _action[16];
   void     *_ptr;
   size_t    _nbytes;
   HYPRE_Int _memory_location;
   char      _filename[256];
   char      _function[256];
   HYPRE_Int _line;

   hypre_memory_tracker_t(const char *action, void *ptr, size_t nbytes, HYPRE_Int memory_location, const char *filename, const char *function, HYPRE_Int line)
   {
      sprintf(_action, "%s", action);
      _ptr = ptr;
      _nbytes = nbytes;
      _memory_location = memory_location;
      sprintf(_filename, "%s", filename);
      sprintf(_function, "%s", function);
      _line = line;
   }

   void print()
   {
      printf("%8s  %16p  %10ld  %d  %32s  %64s      %d\n",
            _action, _ptr, _nbytes, _memory_location, _filename, _function, _line);

   }
};

extern std::vector<hypre_memory_tracker_t> hypre_memory_tracker;

static inline void hypre_MemoryTrackerInsert(struct hypre_memory_tracker_t const &memory)
{
   if (memory._memory_location != HYPRE_MEMORY_HOST)
   {
      hypre_memory_tracker.push_back(memory);
   }
}

#ifdef __cplusplus
}
#endif

/* These Allocs are with printfs, for debug */
#define hypre_TAlloc(type, count, location)                                                                                           \
(                                                                                                                                     \
{                                                                                                                                     \
   void *ptr = hypre_MAlloc((size_t)(sizeof(type) * (count)), location);                                                              \
   hypre_MemoryTrackerInsert( hypre_memory_tracker_t("malloc", ptr, sizeof(type)*(count), location, __FILE__, __func__, __LINE__) );  \
   (type *) ptr;                                                                                                                      \
}                                                                                                                                     \
)

#define hypre_CTAlloc(type, count, location)                                                                                          \
(                                                                                                                                     \
{                                                                                                                                     \
   void *ptr = hypre_CAlloc((size_t)(count), (size_t)sizeof(type), location);                                                         \
   hypre_MemoryTrackerInsert( hypre_memory_tracker_t("calloc", ptr, sizeof(type)*(count), location, __FILE__, __func__, __LINE__) );  \
   (type *) ptr;                                                                                                                      \
}                                                                                                                                     \
)

#define hypre_TReAlloc(ptr, type, count, location)                                      \
(                                                                                       \
{                                                                                       \
   (type *) hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count)), location);     \
}                                                                                       \
)

#define hypre_TReAlloc_v2(ptr, old_type, old_count, new_type, new_count, location)                                                                \
(                                                                                                                                                 \
{                                                                                                                                                 \
   hypre_MemoryTrackerInsert( hypre_memory_tracker_t("rfree", ptr, sizeof(old_type)*(old_count), location, __FILE__, __func__, __LINE__) );       \
   void *new_ptr = hypre_ReAlloc_v2((char *)ptr, (size_t)(sizeof(old_type)*(old_count)), (size_t)(sizeof(new_type)*(new_count)), location);       \
   hypre_MemoryTrackerInsert( hypre_memory_tracker_t("rmalloc", new_ptr, sizeof(new_type)*(new_count), location, __FILE__, __func__, __LINE__) ); \
   (new_type *) new_ptr;                                                                                                                          \
}                                                                                                                                                 \
)

#define hypre_TMemcpy(dst, src, type, count, locdst, locsrc)                                     \
(                                                                                                \
{                                                                                                \
   hypre_Memcpy((void *)(dst), (void *)(src), (size_t)(sizeof(type) * (count)), locdst, locsrc); \
}                                                                                                \
)

#define hypre_TFree(ptr, location)                                                                              \
(                                                                                                               \
{                                                                                                               \
   hypre_MemoryTrackerInsert( hypre_memory_tracker_t("free", ptr, 0, location, __FILE__, __func__, __LINE__) ); \
   hypre_Free((void *)ptr, location);                                                                           \
   ptr = NULL;                                                                                                  \
}                                                                                                               \
)

#else /* #ifdef HYPRE_USING_MEMORY_TRACKER */

#define hypre_TAlloc(type, count, location) \
( (type *) hypre_MAlloc((size_t)(sizeof(type) * (count)), location) )

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

#endif /* #ifdef HYPRE_USING_MEMORY_TRACKER */


/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* hypre_memory.c */
void * hypre_MAlloc(size_t size, HYPRE_Int location);
void * hypre_CAlloc( size_t count, size_t elt_size, HYPRE_Int location);
void * hypre_ReAlloc(void *ptr, size_t size, HYPRE_Int location);
void * hypre_ReAlloc_v2(void *ptr, size_t old_size, size_t new_size, HYPRE_Int location);
void   hypre_Memcpy(void *dst, void *src, size_t size, HYPRE_Int loc_dst, HYPRE_Int loc_src);
void * hypre_Memset(void *ptr, HYPRE_Int value, size_t num, HYPRE_Int location);
void   hypre_Free(void *ptr, HYPRE_Int location);
HYPRE_Int hypre_GetMemoryLocation(const void *ptr, HYPRE_Int *memory_location);
HYPRE_Int hypre_PrintMemoryTracker();
HYPRE_Int hypre_SetCubMemPoolSize( hypre_uint bin_growth, hypre_uint min_bin, hypre_uint max_bin, size_t max_cached_bytes );

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


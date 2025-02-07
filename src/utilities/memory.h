/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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

#if defined(HYPRE_USING_UNIFIED_MEMORY) && defined(HYPRE_USING_DEVICE_OPENMP)
//#pragma omp requires unified_shared_memory
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
   hypre_MEMORY_HOST,
   hypre_MEMORY_HOST_PINNED,
   hypre_MEMORY_DEVICE,
   hypre_MEMORY_UNIFIED,
   hypre_NUM_MEMORY_LOCATION
} hypre_MemoryLocation;

/*-------------------------------------------------------
 * hypre_GetActualMemLocation
 *   return actual location based on the selected memory model
 *-------------------------------------------------------*/
static inline HYPRE_MAYBE_UNUSED_FUNC hypre_MemoryLocation
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


#if !defined(HYPRE_USING_MEMORY_TRACKER)

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

#define _hypre_TMemcpy(dst, src, type, count, locdst, locsrc) \
(_hypre_Memcpy((void *)(dst), (void *)(src), (size_t)(sizeof(type) * (count)), locdst, locsrc))

#define hypre_TFree(ptr, location) \
( hypre_Free((void *)ptr, location), ptr = NULL )

#define _hypre_TFree(ptr, location) \
( _hypre_Free((void *)ptr, location), ptr = NULL )

#endif /* #if !defined(HYPRE_USING_MEMORY_TRACKER) */

#ifdef __cplusplus
}
#endif

#endif /* hypre_MEMORY_HEADER */

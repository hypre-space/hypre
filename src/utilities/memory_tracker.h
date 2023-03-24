/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_MEMORY_TRACKER_HEADER
#define hypre_MEMORY_TRACKER_HEADER

#if defined(HYPRE_USING_MEMORY_TRACKER)

extern size_t hypre_total_bytes[hypre_MEMORY_UNIFIED + 1];
extern size_t hypre_peak_bytes[hypre_MEMORY_UNIFIED + 1];
extern size_t hypre_current_bytes[hypre_MEMORY_UNIFIED + 1];
extern HYPRE_Int hypre_memory_tracker_print;
extern char hypre_memory_tracker_filename[HYPRE_MAX_FILE_NAME_LEN];

typedef enum _hypre_MemoryTrackerEvent
{
   HYPRE_MEMORY_EVENT_ALLOC = 0,
   HYPRE_MEMORY_EVENT_FREE,
   HYPRE_MEMORY_EVENT_COPY,
   HYPRE_MEMORY_NUM_EVENTS,
} hypre_MemoryTrackerEvent;

typedef enum _hypre_MemcpyType
{
   hypre_MEMCPY_H2H = 0,
   hypre_MEMCPY_D2H,
   hypre_MEMCPY_H2D,
   hypre_MEMCPY_D2D,
   hypre_MEMCPY_NUM_TYPES,
} hypre_MemcpyType;

typedef struct
{
   size_t                index;
   size_t                time_step;
   char                  action[16];
   void                 *ptr;
   void                 *ptr2;
   size_t                nbytes;
   hypre_MemoryLocation  memory_location;
   hypre_MemoryLocation  memory_location2;
   char                  filename[HYPRE_MAX_FILE_NAME_LEN];
   char                  function[256];
   HYPRE_Int             line;
   size_t                pair;
} hypre_MemoryTrackerEntry;

typedef struct
{
   size_t                     head;
   size_t                     actual_size;
   size_t                     alloced_size;
   hypre_MemoryTrackerEntry  *data;
   /* Free Queue is sorted based on (ptr, time_step) ascendingly */
   hypre_MemoryTrackerEntry  *sorted_data;
   /* compressed sorted_data with the same ptr */
   size_t                     sorted_data_compressed_len;
   size_t                    *sorted_data_compressed_offset;
   hypre_MemoryTrackerEntry **sorted_data_compressed;
} hypre_MemoryTrackerQueue;

typedef struct
{
   size_t                   curr_time_step;
   hypre_MemoryTrackerQueue queue[HYPRE_MEMORY_NUM_EVENTS];
} hypre_MemoryTracker;

extern hypre_MemoryTracker *_hypre_memory_tracker;

#define hypre_TAlloc(type, count, location)                                                         \
(                                                                                                   \
{                                                                                                   \
   void *ptr = hypre_MAlloc((size_t)(sizeof(type) * (count)), location);                            \
                                                                                                    \
   hypre_MemoryLocation alocation = hypre_GetActualMemLocation(location);                           \
   hypre_MemoryTrackerInsert1("malloc", ptr, sizeof(type)*(count), alocation,                       \
                              __FILE__, __func__, __LINE__);                                        \
   (type *) ptr;                                                                                    \
}                                                                                                   \
)

#define hypre_CTAlloc(type, count, location)                                                        \
(                                                                                                   \
{                                                                                                   \
   void *ptr = hypre_CAlloc((size_t)(count), (size_t)sizeof(type), location);                       \
                                                                                                    \
   hypre_MemoryLocation alocation = hypre_GetActualMemLocation(location);                           \
   hypre_MemoryTrackerInsert1("calloc", ptr, sizeof(type)*(count), alocation,                       \
                              __FILE__, __func__, __LINE__);                                        \
   (type *) ptr;                                                                                    \
}                                                                                                   \
)

#define hypre_TReAlloc(ptr, type, count, location)                                                  \
(                                                                                                   \
{                                                                                                   \
   void *new_ptr = hypre_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count)), location);          \
                                                                                                    \
   hypre_MemoryLocation alocation = hypre_GetActualMemLocation(location);                           \
   hypre_MemoryTrackerInsert1("rfree", ptr, (size_t) -1, alocation,                                 \
                              __FILE__, __func__, __LINE__);                                        \
   hypre_MemoryTrackerInsert1("rmalloc", new_ptr, sizeof(type)*(count), alocation,                  \
                              __FILE__, __func__, __LINE__);                                        \
   (type *) new_ptr;                                                                                \
}                                                                                                   \
)

#define hypre_TReAlloc_v2(ptr, old_type, old_count, new_type, new_count, location)                  \
(                                                                                                   \
{                                                                                                   \
   void *new_ptr = hypre_ReAlloc_v2((char *)ptr, (size_t)(sizeof(old_type)*(old_count)),            \
                                    (size_t)(sizeof(new_type)*(new_count)), location);              \
                                                                                                    \
   hypre_MemoryLocation alocation = hypre_GetActualMemLocation(location);                           \
   hypre_MemoryTrackerInsert1("rfree", ptr, sizeof(old_type)*(old_count), alocation,                \
                              __FILE__, __func__, __LINE__);                                        \
   hypre_MemoryTrackerInsert1("rmalloc", new_ptr, sizeof(new_type)*(new_count), alocation,          \
                              __FILE__, __func__, __LINE__);                                        \
   (new_type *) new_ptr;                                                                            \
}                                                                                                   \
)

#define hypre_TMemcpy(dst, src, type, count, locdst, locsrc)                                        \
(                                                                                                   \
{                                                                                                   \
   hypre_Memcpy((void *)(dst), (void *)(src), (size_t)(sizeof(type) * (count)), locdst, locsrc);    \
                                                                                                    \
   hypre_MemoryLocation alocation_dst = hypre_GetActualMemLocation(locdst);                         \
   hypre_MemoryLocation alocation_src = hypre_GetActualMemLocation(locsrc);                         \
   hypre_MemoryTrackerInsert2("memcpy", (void *) (dst), (void *) (src), sizeof(type)*(count),       \
                              alocation_dst, alocation_src,                                         \
                              __FILE__, __func__, __LINE__);                                        \
}                                                                                                   \
)

#define hypre_TFree(ptr, location)                                                                  \
(                                                                                                   \
{                                                                                                   \
   hypre_Free((void *)ptr, location);                                                               \
                                                                                                    \
   hypre_MemoryLocation alocation = hypre_GetActualMemLocation(location);                           \
   hypre_MemoryTrackerInsert1("free", ptr, (size_t) -1, alocation,                                  \
                              __FILE__, __func__, __LINE__);                                        \
   ptr = NULL;                                                                                      \
}                                                                                                   \
)

#define _hypre_TAlloc(type, count, location)                                                        \
(                                                                                                   \
{                                                                                                   \
   void *ptr = _hypre_MAlloc((size_t)(sizeof(type) * (count)), location);                           \
                                                                                                    \
   hypre_MemoryTrackerInsert1("malloc", ptr, sizeof(type)*(count), location,                        \
                              __FILE__, __func__, __LINE__);                                        \
   (type *) ptr;                                                                                    \
}                                                                                                   \
)

#define _hypre_TFree(ptr, location)                                                                 \
(                                                                                                   \
{                                                                                                   \
   _hypre_Free((void *)ptr, location);                                                              \
                                                                                                    \
   hypre_MemoryTrackerInsert1("free", ptr, (size_t) -1, location,                                   \
                             __FILE__, __func__, __LINE__);                                         \
   ptr = NULL;                                                                                      \
}                                                                                                   \
)

#endif /* #if defined(HYPRE_USING_MEMORY_TRACKER) */
#endif /* #ifndef hypre_MEMORY_TRACKER_HEADER */


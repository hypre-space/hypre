/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Memory tracker
 * Do NOT use hypre_T* in this file since we don't want to track them,
 * Do NOT use hypre_printf, hypre_fprintf, which have hypre_TAlloc/Free
 * endless for-loop otherwise
 *--------------------------------------------------------------------------*/

#include "_hypre_utilities.h"

#if defined(HYPRE_USING_MEMORY_TRACKER)

hypre_MemoryTracker *_hypre_memory_tracker = NULL;

/* accessor to the global ``_hypre_memory_tracker'' */
hypre_MemoryTracker*
hypre_memory_tracker(void)
{
#ifdef HYPRE_USING_OPENMP
   #pragma omp critical
#endif
   {
      if (!_hypre_memory_tracker)
      {
         _hypre_memory_tracker = hypre_MemoryTrackerCreate();
      }
   }

   return _hypre_memory_tracker;
}

size_t hypre_total_bytes[hypre_NUM_MEMORY_LOCATION];
size_t hypre_peak_bytes[hypre_NUM_MEMORY_LOCATION];
size_t hypre_current_bytes[hypre_NUM_MEMORY_LOCATION];
HYPRE_Int hypre_memory_tracker_print = 0;
char hypre_memory_tracker_filename[HYPRE_MAX_FILE_NAME_LEN] = "HypreMemoryTrack.log";

char *hypre_basename(const char *name)
{
   const char *base = name;
   while (*name)
   {
      if (*name++ == '/')
      {
         base = name;
      }
   }
   return (char *) base;
}

hypre_MemcpyType
hypre_GetMemcpyType(hypre_MemoryLocation dst,
                    hypre_MemoryLocation src)
{
   HYPRE_Int d = 0, s = 0;

   if      (dst == hypre_MEMORY_HOST   || dst == hypre_MEMORY_HOST_PINNED) { d = 0; }
   else if (dst == hypre_MEMORY_DEVICE || dst == hypre_MEMORY_UNIFIED)     { d = 1; }

   if      (src == hypre_MEMORY_HOST   || src == hypre_MEMORY_HOST_PINNED) { s = 0; }
   else if (src == hypre_MEMORY_DEVICE || src == hypre_MEMORY_UNIFIED)     { s = 1; }

   if (d == 0 && s == 0) { return hypre_MEMCPY_H2H; }
   if (d == 0 && s == 1) { return hypre_MEMCPY_D2H; }
   if (d == 1 && s == 0) { return hypre_MEMCPY_H2D; }
   if (d == 1 && s == 1) { return hypre_MEMCPY_D2D; }

   return hypre_MEMCPY_NUM_TYPES;
}

hypre_int
hypre_MemoryTrackerQueueCompSort(const void *e1,
                                 const void *e2)
{
   void *p1 = ((hypre_MemoryTrackerEntry *) e1) -> ptr;
   void *p2 = ((hypre_MemoryTrackerEntry *) e2) -> ptr;

   if (p1 < p2) { return -1; }
   if (p1 > p2) { return  1; }

   size_t t1 = ((hypre_MemoryTrackerEntry *) e1) -> time_step;
   size_t t2 = ((hypre_MemoryTrackerEntry *) e2) -> time_step;

   if (t1 < t2) { return -1; }
   if (t1 > t2) { return  1; }

   return 0;
}


hypre_int
hypre_MemoryTrackerQueueCompSearch(const void *e1,
                                   const void *e2)
{
   void *p1 = ((hypre_MemoryTrackerEntry **) e1)[0] -> ptr;
   void *p2 = ((hypre_MemoryTrackerEntry **) e2)[0] -> ptr;

   if (p1 < p2) { return -1; }
   if (p1 > p2) { return  1; }

   return 0;
}

hypre_MemoryTrackerEvent
hypre_MemoryTrackerGetNext(hypre_MemoryTracker *tracker)
{
   hypre_MemoryTrackerEvent i, k = HYPRE_MEMORY_NUM_EVENTS;
   hypre_MemoryTrackerQueue *q = tracker->queue;

   for (i = HYPRE_MEMORY_EVENT_ALLOC; i < HYPRE_MEMORY_NUM_EVENTS; i++)
   {
      if (q[i].head >= q[i].actual_size)
      {
         continue;
      }

      if (k == HYPRE_MEMORY_NUM_EVENTS || q[i].data[q[i].head].time_step < q[k].data[q[k].head].time_step)
      {
         k = i;
      }
   }

   return k;
}

HYPRE_Int
hypre_MemoryTrackerSortQueue(hypre_MemoryTrackerQueue *q)
{
   size_t i = 0;

   if (!q) { return hypre_error_flag; }

   free(q->sorted_data);
   free(q->sorted_data_compressed_offset);
   free(q->sorted_data_compressed);

   q->sorted_data = (hypre_MemoryTrackerEntry *) malloc(q->actual_size * sizeof(
                                                           hypre_MemoryTrackerEntry));
   memcpy(q->sorted_data, q->data, q->actual_size * sizeof(hypre_MemoryTrackerEntry));
   qsort(q->sorted_data, q->actual_size, sizeof(hypre_MemoryTrackerEntry),
         hypre_MemoryTrackerQueueCompSort);

   q->sorted_data_compressed_len = 0;
   q->sorted_data_compressed_offset = (size_t *) malloc(q->actual_size * sizeof(size_t));
   q->sorted_data_compressed = (hypre_MemoryTrackerEntry **) malloc((q->actual_size + 1) * sizeof(
                                                                       hypre_MemoryTrackerEntry *));

   for (i = 0; i < q->actual_size; i++)
   {
      if (i == 0 || q->sorted_data[i].ptr != q->sorted_data[i - 1].ptr)
      {
         q->sorted_data_compressed_offset[q->sorted_data_compressed_len] = i;
         q->sorted_data_compressed[q->sorted_data_compressed_len] = &q->sorted_data[i];
         q->sorted_data_compressed_len ++;
      }
   }
   q->sorted_data_compressed[q->sorted_data_compressed_len] = q->sorted_data + q->actual_size;

   q->sorted_data_compressed_offset = (size_t *)
                                      realloc(q->sorted_data_compressed_offset, q->sorted_data_compressed_len * sizeof(size_t));

   q->sorted_data_compressed = (hypre_MemoryTrackerEntry **)
                               realloc(q->sorted_data_compressed,
                                       (q->sorted_data_compressed_len + 1) * sizeof(hypre_MemoryTrackerEntry *));

   return hypre_error_flag;
}

hypre_MemoryTracker *
hypre_MemoryTrackerCreate()
{
   hypre_MemoryTracker *ptr = (hypre_MemoryTracker *) calloc(1, sizeof(hypre_MemoryTracker));
   return ptr;
}

void
hypre_MemoryTrackerDestroy(hypre_MemoryTracker *tracker)
{
   if (tracker)
   {
      HYPRE_Int i;

      for (i = 0; i < HYPRE_MEMORY_NUM_EVENTS; i++)
      {
         free(tracker->queue[i].data);
         free(tracker->queue[i].sorted_data);
         free(tracker->queue[i].sorted_data_compressed_offset);
         free(tracker->queue[i].sorted_data_compressed);
      }

      free(tracker);
   }
}

HYPRE_Int
hypre_MemoryTrackerSetPrint(HYPRE_Int do_print)
{
   hypre_memory_tracker_print = do_print;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MemoryTrackerSetFileName(const char *file_name)
{
   snprintf(hypre_memory_tracker_filename, HYPRE_MAX_FILE_NAME_LEN, "%s", file_name);

   return hypre_error_flag;
}

void
hypre_MemoryTrackerInsert1(const char           *action,
                           void                 *ptr,
                           size_t                nbytes,
                           hypre_MemoryLocation  memory_location,
                           const char           *filename,
                           const char           *function,
                           HYPRE_Int             line)
{
   hypre_MemoryTrackerInsert2(action, ptr, NULL, nbytes, memory_location, hypre_MEMORY_UNDEFINED,
                              filename, function, line);
}

void
hypre_MemoryTrackerInsert2(const char           *action,
                           void                 *ptr,
                           void                 *ptr2,
                           size_t                nbytes,
                           hypre_MemoryLocation  memory_location,
                           hypre_MemoryLocation  memory_location2,
                           const char           *filename,
                           const char           *function,
                           HYPRE_Int             line)
{
   if (ptr == NULL)
   {
      return;
   }

   hypre_MemoryTracker *tracker = hypre_memory_tracker();

   hypre_MemoryTrackerEvent q;

   /* Get the proper queue based on the action */

   if (strstr(action, "alloc") != NULL)
   {
      /* including malloc, alloc and the malloc in realloc */
      q = HYPRE_MEMORY_EVENT_ALLOC;
   }
   else if (strstr(action, "free") != NULL)
   {
      /* including free and the free in realloc */
      q = HYPRE_MEMORY_EVENT_FREE;
   }
   else if (strstr(action, "memcpy") != NULL)
   {
      /* including memcpy */
      q = HYPRE_MEMORY_EVENT_COPY;
   }
   else
   {
      return;
   }

   hypre_MemoryTrackerQueue *queue = &tracker->queue[q];

#ifdef HYPRE_USING_OPENMP
   #pragma omp critical
#endif
   {
      /* resize if not enough space */

      if (queue->alloced_size <= queue->actual_size)
      {
         queue->alloced_size = 2 * queue->alloced_size + 1;
         queue->data = (hypre_MemoryTrackerEntry *) realloc(queue->data,
                                                            queue->alloced_size * sizeof(hypre_MemoryTrackerEntry));
      }

      hypre_assert(queue->actual_size < queue->alloced_size);

      /* insert an entry */
      hypre_MemoryTrackerEntry *entry = queue->data + queue->actual_size;

      entry->index = queue->actual_size;
      entry->time_step = tracker->curr_time_step;
      sprintf(entry->action, "%s", action);
      entry->ptr = ptr;
      entry->ptr2 = ptr2;
      entry->nbytes = nbytes;
      entry->memory_location = memory_location;
      entry->memory_location2 = memory_location2;
      sprintf(entry->filename, "%s", filename);
      sprintf(entry->function, "%s", function);
      entry->line = line;
      entry->pair = (size_t) -1;

#if 0
      HYPRE_Int myid;
      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
      if (myid == 0 && entry->time_step == 28111) {assert(0);}
#endif

      /* increase the time step */
      tracker->curr_time_step ++;

      /* increase the queue length by 1 */
      queue->actual_size ++;
   }
}

HYPRE_Int
hypre_PrintMemoryTracker( size_t     *totl_bytes_o,
                          size_t     *peak_bytes_o,
                          size_t     *curr_bytes_o,
                          HYPRE_Int   do_print,
                          const char *fname )
{
   char   filename[HYPRE_MAX_FILE_NAME_LEN + 16];
   FILE  *file = NULL;
   size_t totl_bytes[hypre_NUM_MEMORY_LOCATION] = {0};
   size_t peak_bytes[hypre_NUM_MEMORY_LOCATION] = {0};
   size_t curr_bytes[hypre_NUM_MEMORY_LOCATION] = {0};
   size_t copy_bytes[hypre_MEMCPY_NUM_TYPES] = {0};
   size_t j;
   hypre_MemoryTrackerEvent i;
   //HYPRE_Real t0 = hypre_MPI_Wtime();

   HYPRE_Int leakcheck = 1;

   hypre_MemoryTracker *tracker = hypre_memory_tracker();
   hypre_MemoryTrackerQueue *qq = tracker->queue;
   hypre_MemoryTrackerQueue *qa = &qq[HYPRE_MEMORY_EVENT_ALLOC];
   hypre_MemoryTrackerQueue *qf = &qq[HYPRE_MEMORY_EVENT_FREE];

   if (do_print)
   {
      HYPRE_Int myid;
      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

      if (fname)
      {
         hypre_sprintf(filename, "%s.%05d.csv", fname, myid);
      }
      else
      {
         hypre_sprintf(filename, "HypreMemoryTrack.log.%05d.csv", myid);
      }

      if ((file = fopen(filename, "w")) == NULL)
      {
         fprintf(stderr, "Error: can't open output file %s\n", filename);
         return hypre_error_flag;
      }

      fprintf(file, "\"==== Operations:\"\n");
      fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
              "ID", "EVENT", "ADDRESS1", "ADDRESS2", "BYTE", "LOCATION1", "LOCATION2",
              "FILE", "LINE", "FUNCTION", "HOST", "PINNED", "DEVICE", "UNIFIED");
   }

   if (leakcheck)
   {
      //HYPRE_Real t0 = hypre_MPI_Wtime();
      hypre_MemoryTrackerSortQueue(qf);
      //HYPRE_Real t1 = hypre_MPI_Wtime() - t0;
      //printf("Sort Time %.2f\n", t1);
   }

   size_t total_num_events = 0;
   size_t total_num_events_2 = 0;
   for (i = HYPRE_MEMORY_EVENT_ALLOC; i < HYPRE_MEMORY_NUM_EVENTS; i++)
   {
      total_num_events_2 += qq[i].actual_size;
   }

   for (i = hypre_MemoryTrackerGetNext(tracker); i < HYPRE_MEMORY_NUM_EVENTS;
        i = hypre_MemoryTrackerGetNext(tracker))
   {
      total_num_events ++;

      hypre_MemoryTrackerEntry *entry = &qq[i].data[qq[i].head++];

      if (strstr(entry->action, "alloc") != NULL)
      {
         totl_bytes[entry->memory_location] += entry->nbytes;

         if (leakcheck)
         {
            curr_bytes[entry->memory_location] += entry->nbytes;
            peak_bytes[entry->memory_location] = hypre_max( curr_bytes[entry->memory_location],
                                                            peak_bytes[entry->memory_location] );
         }

         if (leakcheck && entry->pair == (size_t) -1)
         {
            hypre_MemoryTrackerEntry key = { .ptr = entry->ptr };
            hypre_MemoryTrackerEntry *key_ptr = &key;

            hypre_MemoryTrackerEntry **result = bsearch(&key_ptr,
                                                        qf->sorted_data_compressed,
                                                        qf->sorted_data_compressed_len,
                                                        sizeof(hypre_MemoryTrackerEntry *),
                                                        hypre_MemoryTrackerQueueCompSearch);
            if (result)
            {
               j = result - qf->sorted_data_compressed;
               hypre_MemoryTrackerEntry *p = qf->sorted_data + qf->sorted_data_compressed_offset[j];

               if (p < qf->sorted_data_compressed[j + 1])
               {
                  hypre_assert(p->ptr == entry->ptr);
                  entry->pair = p->index;
                  hypre_assert(qf->data[p->index].pair == -1);
                  hypre_assert(qq[i].head - 1 == entry->index);
                  qf->data[p->index].pair = entry->index;
                  qf->data[p->index].nbytes = entry->nbytes;

                  qf->sorted_data_compressed_offset[j] ++;
               }
            }
         }
      }
      else if (leakcheck && strstr(entry->action, "free") != NULL)
      {
         if (entry->pair < qa->actual_size)
         {
            curr_bytes[entry->memory_location] -= qa->data[entry->pair].nbytes;
         }
      }
      else if (strstr(entry->action, "memcpy") != NULL)
      {
         copy_bytes[hypre_GetMemcpyType(entry->memory_location, entry->memory_location2)] += entry->nbytes;
      }

      if (do_print)
      {
         char memory_location[256];
         char memory_location2[256];
         char nbytes[32];

         hypre_GetMemoryLocationName(entry->memory_location, memory_location);
         hypre_GetMemoryLocationName(entry->memory_location2, memory_location2);

         if (entry->nbytes != (size_t) -1)
         {
            sprintf(nbytes, "%zu", entry->nbytes);
         }
         else
         {
            sprintf(nbytes, "%s", "--");
         }

         fprintf(file,
                 " %6zu, %9s, %16p, %16p, %10s, %10s, %10s, %28s, %8d, %54s, %11zu, %11zu, %11zu, %11zu\n",
                 entry->time_step,
                 entry->action,
                 entry->ptr,
                 entry->ptr2,
                 nbytes,
                 memory_location,
                 memory_location2,
                 hypre_basename(entry->filename),
                 entry->line,
                 entry->function,
                 curr_bytes[hypre_MEMORY_HOST],
                 curr_bytes[hypre_MEMORY_HOST_PINNED],
                 curr_bytes[hypre_MEMORY_DEVICE],
                 curr_bytes[hypre_MEMORY_UNIFIED]
                );
      }
   }

   hypre_assert(total_num_events == total_num_events_2);

   if (do_print)
   {
      fprintf(file, "\n\"==== Total Allocation (byte):\"\n");
      fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
              "", "", "", "", "", "", "", "", "", "", "HOST", "PINNED", "DEVICE", "UNIFIED");
      fprintf(file,
              " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11zu, %11zu, %11zu, %11zu\n",
              "", "", "", "", "", "", "", "", "", "",
              totl_bytes[hypre_MEMORY_HOST],
              totl_bytes[hypre_MEMORY_HOST_PINNED],
              totl_bytes[hypre_MEMORY_DEVICE],
              totl_bytes[hypre_MEMORY_UNIFIED]);

      fprintf(file, "\n\"==== Peak Allocation (byte):\"\n");
      /*fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
            "", "", "", "", "", "", "", "", "", "", "HOST", "PINNED", "DEVICE", "UNIFIED"); */
      fprintf(file,
              " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11zu, %11zu, %11zu, %11zu\n",
              "", "", "", "", "", "", "", "", "", "",
              peak_bytes[hypre_MEMORY_HOST],
              peak_bytes[hypre_MEMORY_HOST_PINNED],
              peak_bytes[hypre_MEMORY_DEVICE],
              peak_bytes[hypre_MEMORY_UNIFIED]);

      fprintf(file, "\n\"==== Reachable Allocation (byte):\"\n");
      /* fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
            "", "", "", "", "", "", "", "", "", "", "HOST", "PINNED", "DEVICE", "UNIFIED"); */
      fprintf(file,
              " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11zu, %11zu, %11zu, %11zu\n",
              "", "", "", "", "", "", "", "", "", "",
              curr_bytes[hypre_MEMORY_HOST],
              curr_bytes[hypre_MEMORY_HOST_PINNED],
              curr_bytes[hypre_MEMORY_DEVICE],
              curr_bytes[hypre_MEMORY_UNIFIED]);

      fprintf(file, "\n\"==== Memory Copy (byte):\"\n");
      fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
              "", "", "", "", "", "", "", "", "", "", "H2H", "D2H", "H2D", "D2D");
      fprintf(file,
              " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11zu, %11zu, %11zu, %11zu\n",
              "", "", "", "", "", "", "", "", "", "",
              copy_bytes[hypre_MEMCPY_H2H],
              copy_bytes[hypre_MEMCPY_D2H],
              copy_bytes[hypre_MEMCPY_H2D],
              copy_bytes[hypre_MEMCPY_D2D]);

   }

   if (totl_bytes_o)
   {
      totl_bytes_o[hypre_MEMORY_HOST] = totl_bytes[hypre_MEMORY_HOST];
      totl_bytes_o[hypre_MEMORY_HOST_PINNED] = totl_bytes[hypre_MEMORY_HOST_PINNED];
      totl_bytes_o[hypre_MEMORY_DEVICE] = totl_bytes[hypre_MEMORY_DEVICE];
      totl_bytes_o[hypre_MEMORY_UNIFIED] = totl_bytes[hypre_MEMORY_UNIFIED];
   }

   if (peak_bytes_o)
   {
      peak_bytes_o[hypre_MEMORY_HOST] = peak_bytes[hypre_MEMORY_HOST];
      peak_bytes_o[hypre_MEMORY_HOST_PINNED] = peak_bytes[hypre_MEMORY_HOST_PINNED];
      peak_bytes_o[hypre_MEMORY_DEVICE] = peak_bytes[hypre_MEMORY_DEVICE];
      peak_bytes_o[hypre_MEMORY_UNIFIED] = peak_bytes[hypre_MEMORY_UNIFIED];
   }

   if (curr_bytes_o)
   {
      curr_bytes_o[hypre_MEMORY_HOST] = curr_bytes[hypre_MEMORY_HOST];
      curr_bytes_o[hypre_MEMORY_HOST_PINNED] = curr_bytes[hypre_MEMORY_HOST_PINNED];
      curr_bytes_o[hypre_MEMORY_DEVICE] = curr_bytes[hypre_MEMORY_DEVICE];
      curr_bytes_o[hypre_MEMORY_UNIFIED] = curr_bytes[hypre_MEMORY_UNIFIED];
   }

#if defined(HYPRE_DEBUG)
   for (i = HYPRE_MEMORY_EVENT_ALLOC; i < HYPRE_MEMORY_NUM_EVENTS; i++)
   {
      hypre_assert(qq[i].head == qq[i].actual_size);
   }
#endif

   if (leakcheck && do_print)
   {
      fprintf(file, "\n\"==== Warnings:\"\n");

      for (j = 0; j < qa->actual_size; j++)
      {
         hypre_MemoryTrackerEntry *entry = &qa->data[j];
         if (entry->pair == (size_t) -1)
         {
            fprintf(file, " %6zu, %9s, %16p, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
                    entry->time_step, entry->action, entry->ptr, "", "", "", "", "", "", "Not freed", "", "", "", "");
         }
         else
         {
            hypre_assert(entry->pair < qf->actual_size);
            hypre_assert(qf->data[entry->pair].ptr == entry->ptr);
            hypre_assert(qf->data[entry->pair].nbytes == entry->nbytes);
            hypre_assert(qf->data[entry->pair].memory_location == entry->memory_location);
            hypre_assert(qf->data[entry->pair].pair == j);
         }
      }

      for (j = 0; j < qf->actual_size; j++)
      {
         hypre_MemoryTrackerEntry *entry = &qf->data[j];
         if (entry->pair == (size_t) -1)
         {
            fprintf(file, " %6zu, %9s, %16p, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
                    entry->time_step, entry->action, entry->ptr, "", "", "", "", "", "", "Unpaired free", "", "", "",
                    "");
         }
         else
         {
            hypre_assert(entry->pair < qa->actual_size);
            hypre_assert(qa->data[entry->pair].ptr == entry->ptr);
            hypre_assert(qa->data[entry->pair].nbytes == entry->nbytes);
            hypre_assert(qa->data[entry->pair].memory_location == entry->memory_location);
            hypre_assert(qa->data[entry->pair].pair == j);
         }
      }
   }

   if (file)
   {
      fclose(file);
   }

   if (leakcheck)
   {
      hypre_MemoryLocation t;

      for (t = hypre_MEMORY_HOST; t <= hypre_MEMORY_UNIFIED; t++)
      {
         if (curr_bytes[t])
         {
            char memory_location[256];
            hypre_GetMemoryLocationName(t, memory_location);
            fprintf(stderr, "%zu bytes of %s memory may not be freed\n", curr_bytes[t], memory_location);
         }

      }

      for (t = hypre_MEMORY_HOST; t <= hypre_MEMORY_UNIFIED; t++)
      {
         hypre_assert(curr_bytes[t] == 0);
      }
   }

   //HYPRE_Real t1 = hypre_MPI_Wtime() - t0;
   //printf("Tracker Print Time %.2f\n", t1);

   return hypre_error_flag;
}

#endif


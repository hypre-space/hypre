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

size_t hypre_total_bytes[hypre_MEMORY_UNIFIED + 1];
size_t hypre_peak_bytes[hypre_MEMORY_UNIFIED + 1];
size_t hypre_current_bytes[hypre_MEMORY_UNIFIED + 1];
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

hypre_int
hypre_MemoryTrackerQueueComp(const void *e1,
                             const void *e2)
{
   void *p1 = ((hypre_MemoryTrackerEntry *) e1) -> ptr;
   void *p2 = ((hypre_MemoryTrackerEntry *) e2) -> ptr;

    if (p1 < p2) { return -1; }
    if (p1 > p2) { return  1; }

    return 0;
}


HYPRE_Int
hypre_GetMemoryLocationName(hypre_MemoryLocation  memory_location,
                            char                 *memory_location_name)
{
   if (memory_location == hypre_MEMORY_HOST)
   {
      sprintf(memory_location_name, "%s", "HOST");
   }
   else if (memory_location == hypre_MEMORY_HOST_PINNED)
   {
      sprintf(memory_location_name, "%s", "HOST_PINNED");
   }
   else if (memory_location == hypre_MEMORY_DEVICE)
   {
      sprintf(memory_location_name, "%s", "DEVICE");
   }
   else if (memory_location == hypre_MEMORY_UNIFIED)
   {
      sprintf(memory_location_name, "%s", "UNIFIED");
   }
   else
   {
      sprintf(memory_location_name, "%s", "--");
   }

   return hypre_error_flag;
}

hypre_MemoryTrackerEvent
hypre_MemoryTrackerGetNext(hypre_MemoryTracker *tracker)
{
   hypre_MemoryTrackerEvent i, k = HYPRE_MEMORY_EVENT_NUM;
   hypre_MemoryTrackerQueue *q = tracker->queue;

   for (i = HYPRE_MEMORY_EVENT_ALLOC; i < HYPRE_MEMORY_EVENT_NUM; i++)
   {
      if (q[i].head >= q[i].actual_size)
      {
         continue;
      }

      if (k == HYPRE_MEMORY_EVENT_NUM || q[i].data[q[i].head].time_step < q[k].data[q[k].head].time_step)
      {
         k = i;
      }
   }

   return k;
}

HYPRE_Int
hypre_MemoryTrackerSortQueue(hypre_MemoryTrackerQueue *q)
{
   if (!q) { return hypre_error_flag; }

   free(q->sorted_data);
   q->sorted_data = (hypre_MemoryTrackerEntry *) malloc(q->actual_size * sizeof(hypre_MemoryTrackerEntry));
   memcpy(q->sorted_data, q->data, q->actual_size * sizeof(hypre_MemoryTrackerEntry));
   qsort(q->sorted_data, q->actual_size, sizeof(hypre_MemoryTrackerEntry), hypre_MemoryTrackerQueueComp);

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

      for (i = 0; i < HYPRE_MEMORY_EVENT_NUM; i++)
      {
         free(tracker->queue[i].data);
         free(tracker->queue[i].sorted_data);
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
   hypre_MemoryTrackerInsert2(action, ptr, NULL, nbytes, memory_location, hypre_MEMORY_UNDEFINED, filename, function, line);
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

   //if (entry->time_step == 1643183) {assert(0);}

   /* increase the time step */
   tracker->curr_time_step ++;

   /* increase the queue length by 1 */
   queue->actual_size ++;
}

#define HYPRE_MEMORY_TRACKER_BINARY_SEARCH 1

HYPRE_Int
hypre_PrintMemoryTracker( size_t     *totl_bytes_o,
                          size_t     *peak_bytes_o,
                          size_t     *curr_bytes_o,
                          HYPRE_Int   do_print,
                          const char *fname )
{
   char   filename[HYPRE_MAX_FILE_NAME_LEN + 16];
   FILE  *file = NULL;
   size_t totl_bytes[hypre_MEMORY_UNIFIED + 1] = {0};
   size_t peak_bytes[hypre_MEMORY_UNIFIED + 1] = {0};
   size_t curr_bytes[hypre_MEMORY_UNIFIED + 1] = {0};
   size_t j;
   hypre_MemoryTrackerEvent i;

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

#if HYPRE_MEMORY_TRACKER_BINARY_SEARCH
   if (leakcheck)
   {
      hypre_MemoryTrackerSortQueue(qf);
   }
#endif

   for (i = hypre_MemoryTrackerGetNext(tracker); i < HYPRE_MEMORY_EVENT_NUM; i = hypre_MemoryTrackerGetNext(tracker))
   {
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
#if HYPRE_MEMORY_TRACKER_BINARY_SEARCH
            hypre_MemoryTrackerEntry key = { .ptr = entry->ptr };
            hypre_MemoryTrackerEntry *result = bsearch(&key,
                                                       qf->sorted_data,
                                                       qf->actual_size,
                                                       sizeof(hypre_MemoryTrackerEntry),
                                                       hypre_MemoryTrackerQueueComp);

            if (result)
            {
               hypre_MemoryTrackerEntry *p;

               for (; result >= qf->sorted_data && result->ptr == entry->ptr; result --);

               for (p = NULL, result ++;
                    result < qf->sorted_data + qf->actual_size && result->ptr == entry->ptr;
                    result ++)
               {
                  if (qf->data[result->index].pair == (size_t) -1 &&
                      qf->data[result->index].memory_location == entry->memory_location)
                  {
                     if (!p || result->time_step < p->time_step)
                     {
                        p = result;
                     }
                  }
               }

               //hypre_assert(p);

               if (p)
               {
                  entry->pair = p->index;
                  qf->data[entry->pair].pair = qq[i].head - 1;
                  qf->data[entry->pair].nbytes = entry->nbytes;
               }
            }
#else
            for (j = qf->head; j < qf->actual_size; j++)
            {
               if ( qf->data[j].pair == (size_t) -1 &&
                    qf->data[j].ptr == entry->ptr &&
                    qf->data[j].memory_location == entry->memory_location )
               {
                  entry->pair = j;
                  qf->data[j].pair = qq[i].head - 1;
                  qf->data[j].nbytes = entry->nbytes;
                  break;
               }
            }
#endif
         }
      }
      else if (leakcheck && strstr(entry->action, "free") != NULL)
      {
         if (entry->pair < qa->actual_size)
         {
            curr_bytes[entry->memory_location] -= qa->data[entry->pair].nbytes;
         }
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

         fprintf(file, " %6zu, %9s, %16p, %16p, %10s, %10s, %10s, %28s, %8d, %54s, %11zu, %11zu, %11zu, %11zu\n",
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

   if (do_print)
   {
      fprintf(file, "\n\"==== Total (byte):\"\n");
      fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
            "", "", "", "", "", "", "", "", "", "", "HOST", "PINNED", "DEVICE", "UNIFIED");
      fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11zu, %11zu, %11zu, %11zu\n",
            "", "", "", "", "", "", "", "", "", "",
            totl_bytes[hypre_MEMORY_HOST],
            totl_bytes[hypre_MEMORY_HOST_PINNED],
            totl_bytes[hypre_MEMORY_DEVICE],
            totl_bytes[hypre_MEMORY_UNIFIED]);

      fprintf(file, "\n\"==== Peak (byte):\"\n");
      /*fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
            "", "", "", "", "", "", "", "", "", "", "HOST", "PINNED", "DEVICE", "UNIFIED"); */
      fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11zu, %11zu, %11zu, %11zu\n",
            "", "", "", "", "", "", "", "", "", "",
            peak_bytes[hypre_MEMORY_HOST],
            peak_bytes[hypre_MEMORY_HOST_PINNED],
            peak_bytes[hypre_MEMORY_DEVICE],
            peak_bytes[hypre_MEMORY_UNIFIED]);

      fprintf(file, "\n\"==== Reachable (byte):\"\n");
      /* fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
            "", "", "", "", "", "", "", "", "", "", "HOST", "PINNED", "DEVICE", "UNIFIED"); */
      fprintf(file, " %6s, %9s, %16s, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11zu, %11zu, %11zu, %11zu\n",
            "", "", "", "", "", "", "", "", "", "",
            curr_bytes[hypre_MEMORY_HOST],
            curr_bytes[hypre_MEMORY_HOST_PINNED],
            curr_bytes[hypre_MEMORY_DEVICE],
            curr_bytes[hypre_MEMORY_UNIFIED]);

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
   for (i = HYPRE_MEMORY_EVENT_ALLOC; i < HYPRE_MEMORY_EVENT_NUM; i++)
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
            hypre_assert(qf->data[entry->pair].pair == j);
         }
      }

      for (j = 0; j < qf->actual_size; j++)
      {
         hypre_MemoryTrackerEntry *entry = &qf->data[j];
         if (entry->pair == (size_t) -1)
         {
            fprintf(file, " %6zu, %9s, %16p, %16s, %10s, %10s, %10s, %28s, %8s, %54s, %11s, %11s, %11s, %11s\n",
                  entry->time_step, entry->action, entry->ptr, "", "", "", "", "", "", "Unpaired free", "", "", "", "");
         }
         else
         {
            hypre_assert(entry->pair < qa->actual_size);
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
      hypre_assert(curr_bytes[hypre_MEMORY_HOST] == 0);
      hypre_assert(curr_bytes[hypre_MEMORY_HOST_PINNED] == 0);
      hypre_assert(curr_bytes[hypre_MEMORY_DEVICE] == 0);
      hypre_assert(curr_bytes[hypre_MEMORY_UNIFIED] == 0);
   }

   return hypre_error_flag;
}

#endif


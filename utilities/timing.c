/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Routines for doing timing.
 *
 *****************************************************************************/

#define HYPRE_TIMING_GLOBALS
#include "utilities.h"
#include "timing.h"


/*-------------------------------------------------------
 * Timing macros
 *-------------------------------------------------------*/

#define hypre_StartTiming() \
hypre_TimingWallCount -= time_getWallclockSeconds();\
hypre_TimingCPUCount -= time_getCPUSeconds()

#define hypre_StopTiming() \
hypre_TimingWallCount += time_getWallclockSeconds();\
hypre_TimingCPUCount += time_getCPUSeconds()

#ifndef HYPRE_USE_PTHREADS
#define hypre_global_timing_ref(index,field) hypre_global_timing->field
#else
#define hypre_global_timing_ref(index,field) \
                                     hypre_global_timing[index].field
#endif

/*--------------------------------------------------------------------------
 * hypre_InitializeTiming
 *--------------------------------------------------------------------------*/

int
hypre_InitializeTiming( char *name )
{
   int      time_index;

   double  *old_wall_time;
   double  *old_cpu_time;
   double  *old_flops;
   char   **old_name;
   int     *old_state;
   int     *old_num_regs;

   int      new_name;
   int      i;
#ifdef HYPRE_USE_PTHREADS
   int      threadid = hypre_GetThreadID();
#endif

   /*-------------------------------------------------------
    * Allocate global TimingType structure if needed
    *-------------------------------------------------------*/

   if (hypre_global_timing == NULL)
   {
#ifndef HYPRE_USE_PTHREADS
      hypre_global_timing = hypre_CTAlloc(hypre_TimingType, 1);
#else
      hypre_global_timing = hypre_CTAlloc(hypre_TimingType,
                                          hypre_NumThreads + 1);
#endif
   }

   /*-------------------------------------------------------
    * Check to see if name has already been registered
    *-------------------------------------------------------*/

   new_name = 1;
   for (i = 0; i < (hypre_global_timing_ref(threadid, size)); i++)
   {
      if (hypre_TimingNumRegs(i) > 0)
      {
         if (strcmp(name, hypre_TimingName(i)) == 0)
         {
            new_name = 0;
            time_index = i;
            hypre_TimingNumRegs(time_index) ++;
            break;
         }
      }
   }

   if (new_name)
   {
      for (i = 0; i < hypre_global_timing_ref(threadid ,size); i++)
      {
         if (hypre_TimingNumRegs(i) == 0)
         {
            break;
         }
      }
      time_index = i;
   }

   /*-------------------------------------------------------
    * Register the new timing name
    *-------------------------------------------------------*/

   if (new_name)
   {
      if (time_index == (hypre_global_timing_ref(threadid, size)))
      {
         old_wall_time = (hypre_global_timing_ref(threadid, wall_time));
         old_cpu_time  = (hypre_global_timing_ref(threadid, cpu_time));
         old_flops     = (hypre_global_timing_ref(threadid, flops));
         old_name      = (hypre_global_timing_ref(threadid, name));
         old_state     = (hypre_global_timing_ref(threadid, state));
         old_num_regs  = (hypre_global_timing_ref(threadid, num_regs));
    
         (hypre_global_timing_ref(threadid, wall_time)) =
            hypre_CTAlloc(double, (time_index+1));
         (hypre_global_timing_ref(threadid, cpu_time))  =
            hypre_CTAlloc(double, (time_index+1));
         (hypre_global_timing_ref(threadid, flops))     =
            hypre_CTAlloc(double, (time_index+1));
         (hypre_global_timing_ref(threadid, name))      =
            hypre_CTAlloc(char *, (time_index+1));
         (hypre_global_timing_ref(threadid, state))     =
            hypre_CTAlloc(int,    (time_index+1));
         (hypre_global_timing_ref(threadid, num_regs))  =
            hypre_CTAlloc(int,    (time_index+1));
         (hypre_global_timing_ref(threadid, size)) ++;

         for (i = 0; i < time_index; i++)
         {
            hypre_TimingWallTime(i) = old_wall_time[i];
            hypre_TimingCPUTime(i)  = old_cpu_time[i];
            hypre_TimingFLOPS(i)    = old_flops[i];
            hypre_TimingName(i)     = old_name[i];
            hypre_TimingState(i)    = old_state[i];
            hypre_TimingNumRegs(i)  = old_num_regs[i];
         }

         hypre_TFree(old_wall_time);
         hypre_TFree(old_cpu_time);
         hypre_TFree(old_flops);
         hypre_TFree(old_name);
         hypre_TFree(old_state);
         hypre_TFree(old_num_regs);
      }

      hypre_TimingName(time_index) = hypre_CTAlloc(char, 80);
      strncpy(hypre_TimingName(time_index), name, 79);
      hypre_TimingState(time_index)   = 0;
      hypre_TimingNumRegs(time_index) = 1;
      (hypre_global_timing_ref(threadid, num_names)) ++;
   }

   return time_index;
}

/*--------------------------------------------------------------------------
 * hypre_FinalizeTiming
 *--------------------------------------------------------------------------*/

int
hypre_FinalizeTiming( int time_index )
{
   int  ierr = 0;
   int  i;
#ifdef HYPRE_USE_PTHREADS
   int  threadid = hypre_GetThreadID();
   int  free_global_timing;
#endif

   if (hypre_global_timing == NULL)
      return;

   if (time_index < (hypre_global_timing_ref(threadid, size)))
   {
      if (hypre_TimingNumRegs(time_index) > 0)
      {
         hypre_TimingNumRegs(time_index) --;
      }

      if (hypre_TimingNumRegs(time_index) == 0)
      {
         hypre_TFree(hypre_TimingName(time_index));
         (hypre_global_timing_ref(threadid, num_names)) --;
      }
   }

#ifdef HYPRE_USE_PTHREADS

   free_global_timing = 1;
   for (i = 0; i <= hypre_NumThreads; i++)
   {  
      if (hypre_global_timing_ref(i, num_names))
      {
         free_global_timing = 0;
         break;
      }  
   }

   if (free_global_timing)
   {   
      pthread_mutex_lock(&time_mtx);
      if(hypre_global_timing)
      {
         for (i = 0; i <= hypre_NumThreads; i++)  

         {  
            hypre_TFree(hypre_global_timing_ref(i, wall_time));
            hypre_TFree(hypre_global_timing_ref(i, cpu_time));
            hypre_TFree(hypre_global_timing_ref(i, flops));
            hypre_TFree(hypre_global_timing_ref(i, name));
            hypre_TFree(hypre_global_timing_ref(i, state));
            hypre_TFree(hypre_global_timing_ref(i, num_regs));
         }
      
         hypre_TFree(hypre_global_timing);
         hypre_global_timing = NULL;
      }
      pthread_mutex_unlock(&time_mtx);
   }

#else

   if ((hypre_global_timing -> num_names) == 0)
   {
      for (i = 0; i < (hypre_global_timing -> size); i++)
      {  
         hypre_TFree(hypre_global_timing_ref(i, wall_time));
         hypre_TFree(hypre_global_timing_ref(i, cpu_time));
         hypre_TFree(hypre_global_timing_ref(i, flops));
         hypre_TFree(hypre_global_timing_ref(i, name));
         hypre_TFree(hypre_global_timing_ref(i, state));
         hypre_TFree(hypre_global_timing_ref(i, num_regs));
      }
      
      hypre_TFree(hypre_global_timing);
      hypre_global_timing = NULL;
   }

#endif

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_IncFLOPCount
 *--------------------------------------------------------------------------*/

int
hypre_IncFLOPCount( int inc )
{
   int  ierr = 0;
#ifdef HYPRE_USE_PTHREADS
   int threadid = hypre_GetThreadID();
#endif

   if (hypre_global_timing == NULL)
      return;

   hypre_TimingFLOPCount += (double) (inc);

#ifdef HYPRE_USE_PTHREADS
   if (threadid != hypre_NumThreads)
   {
      pthread_mutex_lock(&time_mtx);
      hypre_TimingAllFLOPS += (double) (inc);
      pthread_mutex_unlock(&time_mtx);
   }
#endif

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BeginTiming
 *--------------------------------------------------------------------------*/

int
hypre_BeginTiming( int time_index )
{
   int  ierr = 0;
#ifdef HYPRE_USE_PTHREADS
   int threadid = hypre_GetThreadID();
#endif

   if (hypre_global_timing == NULL)
      return;

   if (hypre_TimingState(time_index) == 0)
   {
      hypre_StopTiming();
      hypre_TimingWallTime(time_index) -= hypre_TimingWallCount;
      hypre_TimingCPUTime(time_index)  -= hypre_TimingCPUCount;
#ifdef HYPRE_USE_PTHREADS
      if (threadid != hypre_NumThreads)
         hypre_TimingFLOPS(time_index)    -= hypre_TimingFLOPCount;
      else
         hypre_TimingFLOPS(time_index)    -= hypre_TimingAllFLOPS;
#else
      hypre_TimingFLOPS(time_index)    -= hypre_TimingFLOPCount;
#endif

      hypre_StartTiming();
   }
   hypre_TimingState(time_index) ++;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_EndTiming
 *--------------------------------------------------------------------------*/

int
hypre_EndTiming( int time_index )
{
   int  ierr = 0;
#ifdef HYPRE_USE_PTHREADS
   int  threadid = hypre_GetThreadID();
#endif

   if (hypre_global_timing == NULL)
      return;

   hypre_TimingState(time_index) --;
   if (hypre_TimingState(time_index) == 0)
   {
      hypre_StopTiming();
      hypre_TimingWallTime(time_index) += hypre_TimingWallCount;
      hypre_TimingCPUTime(time_index)  += hypre_TimingCPUCount;
#ifdef HYPRE_USE_PTHREADS
      if (threadid != hypre_NumThreads)
         hypre_TimingFLOPS(time_index)    += hypre_TimingFLOPCount;
      else
         hypre_TimingFLOPS(time_index)    += hypre_TimingAllFLOPS;
#else
      hypre_TimingFLOPS(time_index)    += hypre_TimingFLOPCount;
#endif
      hypre_StartTiming();
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ClearTiming
 *--------------------------------------------------------------------------*/

int
hypre_ClearTiming( )
{
   int  ierr = 0;
   int  i;
#ifdef HYPRE_USE_PTHREADS
   int  threadid = hypre_GetThreadID();
#endif

   if (hypre_global_timing == NULL)
      return;

   for (i = 0; i < (hypre_global_timing_ref(threadid,size)); i++)
   {
      hypre_TimingWallTime(i) = 0.0;
      hypre_TimingCPUTime(i)  = 0.0;
      hypre_TimingFLOPS(i)    = 0.0;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PrintTiming
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_USE_PTHREADS  /* non-threaded version of hypre_PrintTiming */

int
hypre_PrintTiming( char     *heading,
                   MPI_Comm  comm  )
{
   int  ierr = 0;

   double  local_wall_time;
   double  local_cpu_time;
   double  wall_time;
   double  cpu_time;
   double  wall_mflops;
   double  cpu_mflops;

   int     i;
   int     myrank;

   if (hypre_global_timing == NULL)
      return;

   MPI_Comm_rank(comm, &myrank );

   /* print heading */
   if (myrank == 0)
   {
      printf("=============================================\n");
      printf("%s:\n", heading);
      printf("=============================================\n");
   }

   for (i = 0; i < (hypre_global_timing -> size); i++)
   {
      if (hypre_TimingNumRegs(i) > 0)
      {
         local_wall_time = hypre_TimingWallTime(i);
         local_cpu_time  = hypre_TimingCPUTime(i);
         MPI_Allreduce(&local_wall_time, &wall_time, 1,
                       MPI_DOUBLE, MPI_MAX, comm);
         MPI_Allreduce(&local_cpu_time, &cpu_time, 1,
                       MPI_DOUBLE, MPI_MAX, comm);

         if (myrank == 0)
         {
            printf("%s:\n", hypre_TimingName(i));

            /* print wall clock info */
            printf("  wall clock time = %f seconds\n", wall_time);
            if (wall_time)
               wall_mflops = hypre_TimingFLOPS(i) / wall_time / 1.0E6;
            else
               wall_mflops = 0.0;
            printf("  wall MFLOPS     = %f\n", wall_mflops);

            /* print CPU clock info */
            printf("  cpu clock time  = %f seconds\n", cpu_time);
            if (cpu_time)
               cpu_mflops = hypre_TimingFLOPS(i) / cpu_time / 1.0E6;
            else
               cpu_mflops = 0.0;
            printf("  cpu MFLOPS      = %f\n\n", cpu_mflops);
         }
      }
   }

   return ierr;
}

#else /* threaded version of hypre_PrintTiming */

#ifdef MPI_Comm_rank
#undef MPI_Comm_rank
#endif
#ifdef MPI_Allreduce
#undef MPI_Allreduce
#endif

int
hypre_PrintTiming( char     *heading,
                   MPI_Comm  comm  )
{
   int  ierr = 0;

   double  local_wall_time;
   double  local_cpu_time;
   double  wall_time;
   double  cpu_time;
   double  wall_mflops;
   double  cpu_mflops;

   int     i, j, index;
   int     myrank;
   int     my_thread = hypre_GetThreadID();
   int     threadid;
   int     max_size;
   int     num_regs;

   char    target_name[32];

   if (my_thread == hypre_NumThreads)
   {
      if (hypre_global_timing == NULL)
         return;

      MPI_Comm_rank(comm, &myrank );

      /* print heading */
      if (myrank == 0)
      {
         printf("=============================================\n");
         printf("%s:\n", heading);
         printf("=============================================\n");
      }

      for (i = 0; i < 7; i++)
      {
         switch (i)
         {
            case 0:  
               threadid = my_thread;
               strcpy(target_name, hypre_TimingName(i));
               break;
            case 1:
               strcpy(target_name, "SMG");
               break;
            case 2:
               strcpy(target_name, "SMGRelax");
               break;
            case 3:
               strcpy(target_name, "SMGResidual");
               break;
            case 4:
               strcpy(target_name, "CyclicReduction");
               break;
            case 5:
               strcpy(target_name, "SMGIntAdd");
               break;
            case 6:
               strcpy(target_name, "SMGRestrict");
               break;
         }

         threadid = 0;
         for (j = 0; j < hypre_global_timing[threadid].size; j++)
         {
            if (strcmp(target_name, hypre_TimingName(j)) == 0)
            {
               index = j;
               break;
            }
            else
               index = -1;
         }

         if (i < hypre_global_timing[my_thread].size)
         {
            threadid = my_thread;
            num_regs = hypre_TimingNumRegs(i);
         }
         else
            num_regs = hypre_TimingNumRegs(index);

         if (num_regs > 0)
         {
            local_wall_time = 0.0;
            local_cpu_time  = 0.0;
            if (index >= 0)
            {
               for (threadid = 0; threadid < hypre_NumThreads; threadid++)
               {
                  local_wall_time = 
                     hypre_max(local_wall_time, hypre_TimingWallTime(index));
                  local_cpu_time  = 
                     hypre_max(local_cpu_time, hypre_TimingCPUTime(index));
               }
            }

            if (i < hypre_global_timing[my_thread].size)
            {
               threadid         = my_thread;
               local_wall_time += hypre_TimingWallTime(i);
               local_cpu_time  += hypre_TimingCPUTime(i);
            }

            MPI_Allreduce(&local_wall_time, &wall_time, 1,
                          MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(&local_cpu_time, &cpu_time, 1,
                          MPI_DOUBLE, MPI_MAX, comm);

            if (myrank == 0)
            {
               printf("%s:\n", target_name);

               /* print wall clock info */
               printf("  wall clock time = %f seconds\n", wall_time);
               wall_mflops = 0.0;
               if (wall_time)
               {
                  if (index >= 0)
                  {
                     for (threadid = 0; threadid < hypre_NumThreads; threadid++)
                     {
                        wall_mflops += 
                           hypre_TimingFLOPS(index) / wall_time / 1.0E6;
                     }
                  }
                  if (i < hypre_global_timing[my_thread].size)
                  {
                     threadid = my_thread;
                     wall_mflops += hypre_TimingFLOPS(i) / wall_time / 1.0E6;
                  }
               }

               printf("  wall MFLOPS     = %f\n", wall_mflops);

               /* print CPU clock info */
               printf("  cpu clock time  = %f seconds\n", cpu_time);
               cpu_mflops = 0.0;
               if (cpu_time)
               {
                  if (index >= 0)
                  {
                     for (threadid = 0; threadid < hypre_NumThreads; threadid++)
                     {
                        cpu_mflops += 
                           hypre_TimingFLOPS(index) / cpu_time / 1.0E6;
                     }
                  }
                  if (i < hypre_global_timing[my_thread].size)
                  {
                     threadid = my_thread;
                     cpu_mflops += hypre_TimingFLOPS(i) / cpu_time / 1.0E6;
                  }
               }

               printf("  cpu MFLOPS      = %f\n\n", cpu_mflops);
            }
         }
      }
   }

   return ierr;
}

#endif

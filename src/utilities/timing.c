/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * Routines for doing timing.
 *
 *****************************************************************************/

#define HYPRE_TIMING_GLOBALS
#include "_hypre_utilities.h"
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

HYPRE_Int
hypre_InitializeTiming( const char *name )
{
   HYPRE_Int      time_index;

   double  *old_wall_time;
   double  *old_cpu_time;
   double  *old_flops;
   char   **old_name;
   HYPRE_Int     *old_state;
   HYPRE_Int     *old_num_regs;

   HYPRE_Int      new_name;
   HYPRE_Int      i;
#ifdef HYPRE_USE_PTHREADS
   HYPRE_Int      threadid = hypre_GetThreadID();
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
            hypre_CTAlloc(HYPRE_Int,    (time_index+1));
         (hypre_global_timing_ref(threadid, num_regs))  =
            hypre_CTAlloc(HYPRE_Int,    (time_index+1));
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

HYPRE_Int
hypre_FinalizeTiming( HYPRE_Int time_index )
{
   HYPRE_Int  ierr = 0;
   HYPRE_Int  i;
#ifdef HYPRE_USE_PTHREADS
   HYPRE_Int  threadid = hypre_GetThreadID();
   HYPRE_Int  free_global_timing;
#endif

   if (hypre_global_timing == NULL)
      return ierr;

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

HYPRE_Int
hypre_IncFLOPCount( HYPRE_Int inc )
{
   HYPRE_Int  ierr = 0;
#ifdef HYPRE_USE_PTHREADS
   HYPRE_Int threadid = hypre_GetThreadID();
#endif

   if (hypre_global_timing == NULL)
      return ierr;

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

HYPRE_Int
hypre_BeginTiming( HYPRE_Int time_index )
{
   HYPRE_Int  ierr = 0;
#ifdef HYPRE_USE_PTHREADS
   HYPRE_Int threadid = hypre_GetThreadID();
#endif

   if (hypre_global_timing == NULL)
      return ierr;

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

HYPRE_Int
hypre_EndTiming( HYPRE_Int time_index )
{
   HYPRE_Int  ierr = 0;
#ifdef HYPRE_USE_PTHREADS
   HYPRE_Int  threadid = hypre_GetThreadID();
#endif

   if (hypre_global_timing == NULL)
      return ierr;

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

HYPRE_Int
hypre_ClearTiming( )
{
   HYPRE_Int  ierr = 0;
   HYPRE_Int  i;
#ifdef HYPRE_USE_PTHREADS
   HYPRE_Int  threadid = hypre_GetThreadID();
#endif

   if (hypre_global_timing == NULL)
      return ierr;

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

HYPRE_Int
hypre_PrintTiming( const char     *heading,
                   MPI_Comm        comm  )
{
   HYPRE_Int  ierr = 0;

   double  local_wall_time;
   double  local_cpu_time;
   double  wall_time;
   double  cpu_time;
   double  wall_mflops;
   double  cpu_mflops;

   HYPRE_Int     i;
   HYPRE_Int     myrank;

   if (hypre_global_timing == NULL)
      return ierr;

   hypre_MPI_Comm_rank(comm, &myrank );

   /* print heading */
   if (myrank == 0)
   {
      hypre_printf("=============================================\n");
      hypre_printf("%s:\n", heading);
      hypre_printf("=============================================\n");
   }

   for (i = 0; i < (hypre_global_timing -> size); i++)
   {
      if (hypre_TimingNumRegs(i) > 0)
      {
         local_wall_time = hypre_TimingWallTime(i);
         local_cpu_time  = hypre_TimingCPUTime(i);
         hypre_MPI_Allreduce(&local_wall_time, &wall_time, 1,
                       hypre_MPI_DOUBLE, hypre_MPI_MAX, comm);
         hypre_MPI_Allreduce(&local_cpu_time, &cpu_time, 1,
                       hypre_MPI_DOUBLE, hypre_MPI_MAX, comm);

         if (myrank == 0)
         {
            hypre_printf("%s:\n", hypre_TimingName(i));

            /* print wall clock info */
            hypre_printf("  wall clock time = %f seconds\n", wall_time);
            if (wall_time)
               wall_mflops = hypre_TimingFLOPS(i) / wall_time / 1.0E6;
            else
               wall_mflops = 0.0;
            hypre_printf("  wall MFLOPS     = %f\n", wall_mflops);

            /* print CPU clock info */
            hypre_printf("  cpu clock time  = %f seconds\n", cpu_time);
            if (cpu_time)
               cpu_mflops = hypre_TimingFLOPS(i) / cpu_time / 1.0E6;
            else
               cpu_mflops = 0.0;
            hypre_printf("  cpu MFLOPS      = %f\n\n", cpu_mflops);
         }
      }
   }

   return ierr;
}

#else /* threaded version of hypre_PrintTiming */

#ifdef hypre_MPI_Comm_rank
#undef hypre_MPI_Comm_rank
#endif
#ifdef hypre_MPI_Allreduce
#undef hypre_MPI_Allreduce
#endif

HYPRE_Int
hypre_PrintTiming( const char     *heading,
                   MPI_Comm        comm  )
{
   HYPRE_Int  ierr = 0;

   double  local_wall_time;
   double  local_cpu_time;
   double  wall_time;
   double  cpu_time;
   double  wall_mflops;
   double  cpu_mflops;

   HYPRE_Int     i, j, index;
   HYPRE_Int     myrank;
   HYPRE_Int     my_thread = hypre_GetThreadID();
   HYPRE_Int     threadid;
   HYPRE_Int     max_size;
   HYPRE_Int     num_regs;

   char    target_name[32];

   if (my_thread == hypre_NumThreads)
   {
      if (hypre_global_timing == NULL)
         return ierr;

      hypre_MPI_Comm_rank(comm, &myrank );

      /* print heading */
      if (myrank == 0)
      {
         hypre_printf("=============================================\n");
         hypre_printf("%s:\n", heading);
         hypre_printf("=============================================\n");
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

            hypre_MPI_Allreduce(&local_wall_time, &wall_time, 1,
                          hypre_MPI_DOUBLE, hypre_MPI_MAX, comm);
            hypre_MPI_Allreduce(&local_cpu_time, &cpu_time, 1,
                          hypre_MPI_DOUBLE, hypre_MPI_MAX, comm);

            if (myrank == 0)
            {
               hypre_printf("%s:\n", target_name);

               /* print wall clock info */
               hypre_printf("  wall clock time = %f seconds\n", wall_time);
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

               hypre_printf("  wall MFLOPS     = %f\n", wall_mflops);

               /* print CPU clock info */
               hypre_printf("  cpu clock time  = %f seconds\n", cpu_time);
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

               hypre_printf("  cpu MFLOPS      = %f\n\n", cpu_mflops);
            }
         }
      }
   }

   return ierr;
}

#endif

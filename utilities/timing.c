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
#include "memory.h"
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

   int      i;

   /*-------------------------------------------------------
    * Allocate global TimingType structure if needed
    *-------------------------------------------------------*/

   if (hypre_global_timing == NULL)
   {
      hypre_global_timing = hypre_CTAlloc(hypre_TimingType, 1);
   }

   /*-------------------------------------------------------
    * Check to see if name has already been registered
    *-------------------------------------------------------*/

   for (i = 0; i < (hypre_global_timing -> size); i++)
   {
      if (strcmp(name, hypre_TimingName(i)) == 0)
      {
         hypre_TimingNumRegs(i) ++;
         break;
      }
   }
   time_index = i;

   /*-------------------------------------------------------
    * Register the new timing name
    *-------------------------------------------------------*/

   if (time_index == (hypre_global_timing -> size))
   {
      old_wall_time = (hypre_global_timing -> wall_time);
      old_cpu_time  = (hypre_global_timing -> cpu_time);
      old_flops     = (hypre_global_timing -> flops);
      old_name      = (hypre_global_timing -> name);
      old_state     = (hypre_global_timing -> state);
      old_num_regs  = (hypre_global_timing -> num_regs);

      (hypre_global_timing -> wall_time) = hypre_CTAlloc(double, (time_index+1));
      (hypre_global_timing -> cpu_time)  = hypre_CTAlloc(double, (time_index+1));
      (hypre_global_timing -> flops)     = hypre_CTAlloc(double, (time_index+1));
      (hypre_global_timing -> name)      = hypre_CTAlloc(char *, (time_index+1));
      (hypre_global_timing -> state)     = hypre_CTAlloc(int,    (time_index+1));
      (hypre_global_timing -> num_regs)  = hypre_CTAlloc(int,    (time_index+1));
      (hypre_global_timing -> size) ++;

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

      hypre_TimingName(time_index) = hypre_CTAlloc(char, 80);
      strncpy(hypre_TimingName(time_index), name, 79);
      hypre_TimingState(time_index)   = 0;
      hypre_TimingNumRegs(time_index) = 1;
   }

   return time_index;
}

/*--------------------------------------------------------------------------
 * hypre_IncFLOPCount
 *--------------------------------------------------------------------------*/

void
hypre_IncFLOPCount( int inc )
{
   if (hypre_global_timing == NULL)
      return;

   hypre_TimingFLOPCount += (double) (inc);
}

/*--------------------------------------------------------------------------
 * hypre_BeginTiming
 *--------------------------------------------------------------------------*/

void
hypre_BeginTiming( int time_index )
{
   if (hypre_global_timing == NULL)
      return;

   if (hypre_TimingState(time_index) == 0)
   {
      hypre_StopTiming();
      hypre_TimingWallTime(time_index) -= hypre_TimingWallCount;
      hypre_TimingCPUTime(time_index)  -= hypre_TimingCPUCount;
      hypre_TimingFLOPS(time_index)    -= hypre_TimingFLOPCount;
      hypre_StartTiming();
   }
   hypre_TimingState(time_index) ++;
}

/*--------------------------------------------------------------------------
 * hypre_EndTiming
 *--------------------------------------------------------------------------*/

void
hypre_EndTiming( int time_index )
{
   if (hypre_global_timing == NULL)
      return;

   hypre_TimingState(time_index) --;
   if (hypre_TimingState(time_index) == 0)
   {
      hypre_StopTiming();
      hypre_TimingWallTime(time_index) += hypre_TimingWallCount;
      hypre_TimingCPUTime(time_index)  += hypre_TimingCPUCount;
      hypre_TimingFLOPS(time_index)    += hypre_TimingFLOPCount;
      hypre_StartTiming();
   }
}

/*--------------------------------------------------------------------------
 * hypre_PrintTiming
 *--------------------------------------------------------------------------*/

void
hypre_PrintTiming( MPI_Comm *comm )
{
   double local_wall_time;
   double local_cpu_time;
   double wall_time;
   double cpu_time;
   double wall_mflops;
   double cpu_mflops;

   int     i, myrank;

   if (hypre_global_timing == NULL)
      return;

   MPI_Comm_rank(*comm, &myrank );

   for (i = 0; i < (hypre_global_timing -> size); i++)
   {
      local_wall_time = hypre_TimingWallTime(i);
      local_cpu_time  = hypre_TimingCPUTime(i);
      MPI_Allreduce(&local_wall_time, &wall_time, 1,
                    MPI_DOUBLE, MPI_MAX, *comm);
      MPI_Allreduce(&local_cpu_time, &cpu_time, 1,
                    MPI_DOUBLE, MPI_MAX, *comm);

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
	 printf("  cpu MFLOPS      = %f\n", cpu_mflops);
      }
   }
}

/*--------------------------------------------------------------------------
 * hypre_FinalizeTiming
 *--------------------------------------------------------------------------*/

void
hypre_FinalizeTiming( int time_index )
{
   int  i;

   if (hypre_global_timing == NULL)
      return;

   if (time_index < (hypre_global_timing -> size))
   {
      if (hypre_TimingNumRegs(time_index) == 1)
      {
         hypre_TFree(hypre_TimingName(time_index));
         (hypre_global_timing -> size) --;
         for (i = time_index; i < (hypre_global_timing -> size); i++)
         {
            hypre_TimingWallTime(i) = hypre_TimingWallTime(i+1);
            hypre_TimingCPUTime(i)  = hypre_TimingCPUTime(i+1);
            hypre_TimingFLOPS(i)    = hypre_TimingFLOPS(i+1);
            hypre_TimingName(i)     = hypre_TimingName(i+1);
            hypre_TimingState(i)    = hypre_TimingState(i+1);
            hypre_TimingNumRegs(i)  = hypre_TimingNumRegs(i+1);
         }

         if ((hypre_global_timing -> size) == 0)
         {
            hypre_TFree(hypre_global_timing -> wall_time);
            hypre_TFree(hypre_global_timing -> cpu_time);
            hypre_TFree(hypre_global_timing -> flops);
            hypre_TFree(hypre_global_timing -> name);
            hypre_TFree(hypre_global_timing -> state);
            hypre_TFree(hypre_global_timing);
            hypre_global_timing = NULL;
         }
      }
      else
      {
         hypre_TimingNumRegs(time_index) --;
      }
   }
}


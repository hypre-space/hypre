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

#define ZZZ_TIMING_GLOBALS
#include "memory.h"
#include "timing.h"

/*-------------------------------------------------------
 * Timing macros
 *-------------------------------------------------------*/

#define zzz_StartTiming() \
zzz_TimingWallCount -= time_getWallclockSeconds();\
zzz_TimingCPUCount -= time_getCPUSeconds()

#define zzz_StopTiming() \
zzz_TimingWallCount += time_getWallclockSeconds();\
zzz_TimingCPUCount += time_getCPUSeconds()

/*--------------------------------------------------------------------------
 * zzz_InitializeTiming
 *--------------------------------------------------------------------------*/

int
zzz_InitializeTiming( char *name )
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

   if (zzz_global_timing == NULL)
   {
      zzz_global_timing = zzz_CTAlloc(zzz_TimingType, 1);
   }

   /*-------------------------------------------------------
    * Check to see if name has already been registered
    *-------------------------------------------------------*/

   for (i = 0; i < (zzz_global_timing -> size); i++)
   {
      if (strcmp(name, zzz_TimingName(i)) == 0)
      {
         zzz_TimingNumRegs(i) ++;
         break;
      }
   }
   time_index = i;

   /*-------------------------------------------------------
    * Register the new timing name
    *-------------------------------------------------------*/

   if (time_index == (zzz_global_timing -> size))
   {
      old_wall_time = (zzz_global_timing -> wall_time);
      old_cpu_time  = (zzz_global_timing -> cpu_time);
      old_flops     = (zzz_global_timing -> flops);
      old_name      = (zzz_global_timing -> name);
      old_state     = (zzz_global_timing -> state);
      old_num_regs  = (zzz_global_timing -> num_regs);

      (zzz_global_timing -> wall_time) = zzz_CTAlloc(double, (time_index+1));
      (zzz_global_timing -> cpu_time)  = zzz_CTAlloc(double, (time_index+1));
      (zzz_global_timing -> flops)     = zzz_CTAlloc(double, (time_index+1));
      (zzz_global_timing -> name)      = zzz_CTAlloc(char *, (time_index+1));
      (zzz_global_timing -> state)     = zzz_CTAlloc(int,    (time_index+1));
      (zzz_global_timing -> num_regs)  = zzz_CTAlloc(int,    (time_index+1));
      (zzz_global_timing -> size) ++;

      for (i = 0; i < time_index; i++)
      {
         zzz_TimingWallTime(i) = old_wall_time[i];
         zzz_TimingCPUTime(i)  = old_cpu_time[i];
         zzz_TimingFLOPS(i)    = old_flops[i];
         zzz_TimingName(i)     = old_name[i];
         zzz_TimingState(i)    = old_state[i];
         zzz_TimingNumRegs(i)  = old_num_regs[i];
      }

      zzz_TFree(old_wall_time);
      zzz_TFree(old_cpu_time);
      zzz_TFree(old_flops);
      zzz_TFree(old_name);
      zzz_TFree(old_state);
      zzz_TFree(old_num_regs);

      zzz_TimingName(time_index) = zzz_CTAlloc(char, 80);
      strncpy(zzz_TimingName(time_index), name, 79);
      zzz_TimingState(time_index)   = 0;
      zzz_TimingNumRegs(time_index) = 1;
   }

   return time_index;
}

/*--------------------------------------------------------------------------
 * zzz_IncFLOPCount
 *--------------------------------------------------------------------------*/

void
zzz_IncFLOPCount( int inc )
{
   if (zzz_global_timing == NULL)
      return;

   zzz_TimingFLOPCount += (double) (inc);
}

/*--------------------------------------------------------------------------
 * zzz_BeginTiming
 *--------------------------------------------------------------------------*/

void
zzz_BeginTiming( int time_index )
{
   if (zzz_global_timing == NULL)
      return;

   if (zzz_TimingState(time_index) == 0)
   {
      zzz_StopTiming();
      zzz_TimingWallTime(time_index) -= zzz_TimingWallCount;
      zzz_TimingCPUTime(time_index)  -= zzz_TimingCPUCount;
      zzz_TimingFLOPS(time_index)    -= zzz_TimingFLOPCount;
      zzz_StartTiming();
   }
   zzz_TimingState(time_index) ++;
}

/*--------------------------------------------------------------------------
 * zzz_EndTiming
 *--------------------------------------------------------------------------*/

void
zzz_EndTiming( int time_index )
{
   if (zzz_global_timing == NULL)
      return;

   zzz_TimingState(time_index) --;
   if (zzz_TimingState(time_index) == 0)
   {
      zzz_StopTiming();
      zzz_TimingWallTime(time_index) += zzz_TimingWallCount;
      zzz_TimingCPUTime(time_index)  += zzz_TimingCPUCount;
      zzz_TimingFLOPS(time_index)    += zzz_TimingFLOPCount;
      zzz_StartTiming();
   }
}

/*--------------------------------------------------------------------------
 * zzz_PrintTiming
 *--------------------------------------------------------------------------*/

void
zzz_PrintTiming( MPI_Comm *comm )
{
   double local_wall_time;
   double local_cpu_time;
   double wall_time;
   double cpu_time;
   double wall_mflops;
   double cpu_mflops;

   int     i, myrank;

   if (zzz_global_timing == NULL)
      return;

   MPI_Comm_rank(*comm, &myrank );

   for (i = 0; i < (zzz_global_timing -> size); i++)
   {
      local_wall_time = zzz_TimingWallTime(i);
      local_cpu_time  = zzz_TimingCPUTime(i);
      MPI_Allreduce(&local_wall_time, &wall_time, 1,
                    MPI_DOUBLE, MPI_MAX, *comm);
      MPI_Allreduce(&local_cpu_time, &cpu_time, 1,
                    MPI_DOUBLE, MPI_MAX, *comm);

      if (myrank == 0)
      {
	 printf("%s:\n", zzz_TimingName(i));

         /* print wall clock info */
	 printf("  wall clock time = %f seconds\n", wall_time);
         if (wall_time)
            wall_mflops = zzz_TimingFLOPS(i) / wall_time / 1.0E6;
         else
            wall_mflops = 0.0;
	 printf("  wall MFLOPS     = %f\n", wall_mflops);

         /* print CPU clock info */
	 printf("  cpu clock time  = %f seconds\n", cpu_time);
         if (cpu_time)
            cpu_mflops = zzz_TimingFLOPS(i) / cpu_time / 1.0E6;
         else
            cpu_mflops = 0.0;
	 printf("  cpu MFLOPS      = %f\n", cpu_mflops);
      }
   }
}

/*--------------------------------------------------------------------------
 * zzz_FinalizeTiming
 *--------------------------------------------------------------------------*/

void
zzz_FinalizeTiming( int time_index )
{
   int  i;

   if (zzz_global_timing == NULL)
      return;

   if (time_index < (zzz_global_timing -> size))
   {
      if (zzz_TimingNumRegs(time_index) == 1)
      {
         zzz_TFree(zzz_TimingName(time_index));
         (zzz_global_timing -> size) --;
         for (i = time_index; i < (zzz_global_timing -> size); i++)
         {
            zzz_TimingWallTime(i) = zzz_TimingWallTime(i+1);
            zzz_TimingCPUTime(i)  = zzz_TimingCPUTime(i+1);
            zzz_TimingFLOPS(i)    = zzz_TimingFLOPS(i+1);
            zzz_TimingName(i)     = zzz_TimingName(i+1);
            zzz_TimingState(i)    = zzz_TimingState(i+1);
            zzz_TimingNumRegs(i)  = zzz_TimingNumRegs(i+1);
         }

         if ((zzz_global_timing -> size) == 0)
         {
            zzz_TFree(zzz_global_timing -> wall_time);
            zzz_TFree(zzz_global_timing -> cpu_time);
            zzz_TFree(zzz_global_timing -> flops);
            zzz_TFree(zzz_global_timing -> name);
            zzz_TFree(zzz_global_timing -> state);
            zzz_TFree(zzz_global_timing);
            zzz_global_timing = NULL;
         }
      }
      else
      {
         zzz_TimingNumRegs(time_index) --;
      }
   }
}


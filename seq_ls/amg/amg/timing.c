#include "timing.h"

amg_Clock_t amg_start_clock=0;
   
long AMG_CPU_TICKS_PER_SEC;
 
void HYPRE_AMGClock_init()
{
   struct timeval r_time;
 
   /* get the current time */
   gettimeofday(&r_time, 0);
 
   amg_start_clock = r_time.tv_sec;

   AMG_CPU_TICKS_PER_SEC = sysconf(_SC_CLK_TCK);

}

amg_Clock_t HYPRE_AMGClock()
{
   struct timeval r_time;
   amg_Clock_t micro_sec;
 
   /* get the current time */
   gettimeofday(&r_time, 0);
 
   /* get the seconds part */
   micro_sec = (r_time.tv_sec - amg_start_clock);
   micro_sec = micro_sec*10000;
 
   /* get the lower order part */
   micro_sec += r_time.tv_usec/100;
 
   return(micro_sec);
}
 

 
 
 
amg_CPUClock_t HYPRE_AMGCPUClock()
{
   struct tms cpu_tms;
 
   times(&cpu_tms);
 
   return(cpu_tms.tms_utime);
}
 

/*--------------------------------------------------------------------------
 * PrintTiming
 *--------------------------------------------------------------------------*/
 
void  HYPRE_AMGPrintTiming(double time_ticks, double cpu_ticks)
{

  
   printf(" wall clock time = %f seconds\n", time_ticks/AMG_TICKS_PER_SEC);
   printf(" CPU clock time  = %f seconds\n", cpu_ticks/AMG_CPU_TICKS_PER_SEC);

}


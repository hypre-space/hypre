/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
#include "headers.h"

/*--------------------------------------------------------------------------
 * amg_Clock_init
 *--------------------------------------------------------------------------*/
 
void     hypre_NAME_C_FOR_FORTRAN(amg_clock_init)()
{
 
   HYPRE_AMGClock_init();
}
 
/*--------------------------------------------------------------------------
 * amg_Clock
 *--------------------------------------------------------------------------*/
 
void hypre_NAME_C_FOR_FORTRAN(amg_clock)(time_ticks)
long *time_ticks;
{
 
 *time_ticks = HYPRE_AMGClock();
 
}
 
/*--------------------------------------------------------------------------
 * amg_CPUClock__
 *--------------------------------------------------------------------------*/
 
void hypre_NAME_C_FOR_FORTRAN(amg_cpuclock)(cpu_ticks)
long *cpu_ticks;
{
 
   *cpu_ticks = HYPRE_AMGCPUClock();
}
 
 
/*--------------------------------------------------------------------------
 * PrintTiming__
 *--------------------------------------------------------------------------*/

void  hypre_NAME_C_FOR_FORTRAN(amg_printtiming)(time_ticks, cpu_ticks)
double *time_ticks;
double *cpu_ticks;
{
 
   HYPRE_AMGPrintTiming(*time_ticks, *cpu_ticks);
 
}
 
 
 


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
 
void     NAME_C_FOR_FORTRAN(amg_clock_init)()
{
 
   amg_Clock_init();
}
 
/*--------------------------------------------------------------------------
 * amg_Clock
 *--------------------------------------------------------------------------*/
 
void NAME_C_FOR_FORTRAN(amg_clock)(time_ticks)
long *time_ticks;
{
 
 *time_ticks = amg_Clock();
 
}
 
/*--------------------------------------------------------------------------
 * amg_CPUClock__
 *--------------------------------------------------------------------------*/
 
void NAME_C_FOR_FORTRAN(amg_cpuclock)(cpu_ticks)
long *cpu_ticks;
{
 
   *cpu_ticks = amg_CPUClock();
}
 
 
/*--------------------------------------------------------------------------
 * PrintTiming__
 *--------------------------------------------------------------------------*/

void  NAME_C_FOR_FORTRAN(amg_printtiming)(time_ticks, cpu_ticks)
double *time_ticks;
double *cpu_ticks;
{
 
   PrintTiming(*time_ticks, *cpu_ticks);
 
}
 
 
 


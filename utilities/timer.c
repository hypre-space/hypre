/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/*
 * File:	timer.c
 * Copyright:	(c) 1997 The Regents of the University of California
 * Author:	Scott Kohn (skohn@llnl.gov)
 * Description:	somewhat portable timing routines for C++, C, and Fortran
 *
 * If TIMER_USE_MPI is defined, then the MPI timers are used to get
 * wallclock seconds, since we assume that the MPI timers have better
 * resolution than the system timers.
 */

#include "timer.h"
#include <time.h>
#include <sys/times.h>
#ifdef TIMER_USE_MPI
#include "mpi.h"
#endif

double time_getWallclockSeconds(void)
{
#ifdef TIMER_USE_MPI
   return(MPI_Wtime());
#else

#ifdef TIMER_NO_SYS
   return(0.0);
#else
   struct tms usage;
   long wallclock = times(&usage);
   return(((double) wallclock)/((double) CLK_TCK));
#endif

#endif
}

double time_getCPUSeconds(void)
{
#ifdef TIMER_NO_SYS
   return(0.0);
#else
   struct tms usage;
   (void) times(&usage);
   return(((double) (usage.tms_utime+usage.tms_stime))/((double) CLK_TCK));
#endif
}

double time_get_wallclock_seconds_(void)
{
   return(time_getWallclockSeconds());
}

double time_get_cpu_seconds_(void)
{
   return(time_getCPUSeconds());
}

/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.15 $
 ***********************************************************************EHEADER*/

/*
 * File:	timer.c
 * Author:	Scott Kohn (skohn@llnl.gov)
 * Description:	somewhat portable timing routines for C++, C, and Fortran
 *
 * If TIMER_USE_MPI is defined, then the MPI timers are used to get
 * wallclock seconds, since we assume that the MPI timers have better
 * resolution than the system timers.
 */

#include "_hypre_utilities.h"

#include <time.h>
#ifndef WIN32
#include <unistd.h>
#include <sys/times.h>
#endif
#ifdef TIMER_USE_MPI
#include "mpi.h"
#endif

double time_getWallclockSeconds(void)
{
#ifdef TIMER_USE_MPI
   return(hypre_MPI_Wtime());
#else
#ifdef WIN32
   clock_t cl=clock();
   return(((double) cl)/((double) CLOCKS_PER_SEC));
#else
   struct tms usage;
   hypre_longint wallclock = times(&usage);
   return(((double) wallclock)/((double) sysconf(_SC_CLK_TCK)));
#endif
#endif
}

double time_getCPUSeconds(void)
{
#ifndef TIMER_NO_SYS
   clock_t cpuclock = clock();
   return(((double) (cpuclock))/((double) CLOCKS_PER_SEC));
#else
   return(0.0);
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

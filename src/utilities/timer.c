/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

HYPRE_Real time_getWallclockSeconds(void)
{
#ifdef TIMER_USE_MPI
   return(hypre_MPI_Wtime());
#else
#ifdef WIN32
   clock_t cl=clock();
   return(((HYPRE_Real) cl)/((HYPRE_Real) CLOCKS_PER_SEC));
#else
   struct tms usage;
   hypre_longint wallclock = times(&usage);
   return(((HYPRE_Real) wallclock)/((HYPRE_Real) sysconf(_SC_CLK_TCK)));
#endif
#endif
}

HYPRE_Real time_getCPUSeconds(void)
{
#ifndef TIMER_NO_SYS
   clock_t cpuclock = clock();
   return(((HYPRE_Real) (cpuclock))/((HYPRE_Real) CLOCKS_PER_SEC));
#else
   return(0.0);
#endif
}

HYPRE_Real time_get_wallclock_seconds_(void)
{
   return(time_getWallclockSeconds());
}

HYPRE_Real time_get_cpu_seconds_(void)
{
   return(time_getCPUSeconds());
}

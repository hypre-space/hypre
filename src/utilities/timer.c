/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
 * File: timer.c
 * Author:  Scott Kohn (skohn@llnl.gov)
 * Description:   somewhat portable timing routines for C++, C, and Fortran
 *
 * This has been modified many times since the original author's version.
 */

#include "_hypre_utilities.h"

#include <time.h>
#ifndef WIN32
#include <unistd.h>
#include <sys/times.h>
#endif

HYPRE_Real time_getWallclockSeconds(void)
{
#ifndef HYPRE_SEQUENTIAL
   return (hypre_MPI_Wtime());
#else
#ifdef WIN32
   clock_t cl = clock();
   return (((HYPRE_Real) cl) / ((HYPRE_Real) CLOCKS_PER_SEC));
#else
   struct tms usage;
   hypre_longint wallclock = times(&usage);
   return (((HYPRE_Real) wallclock) / ((HYPRE_Real) sysconf(_SC_CLK_TCK)));
#endif
#endif
}

HYPRE_Real time_getCPUSeconds(void)
{
#ifndef TIMER_NO_SYS
   clock_t cpuclock = clock();
   return (((HYPRE_Real) (cpuclock)) / ((HYPRE_Real) CLOCKS_PER_SEC));
#else
   return (0.0);
#endif
}

HYPRE_Real time_get_wallclock_seconds_(void)
{
   return (time_getWallclockSeconds());
}

HYPRE_Real time_get_cpu_seconds_(void)
{
   return (time_getCPUSeconds());
}

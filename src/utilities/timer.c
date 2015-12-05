/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/


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

#include <time.h>
#include <unistd.h>
#ifndef WIN32
#include <sys/times.h>
#endif
#ifdef TIMER_USE_MPI
#include "mpi.h"
#endif

double time_getWallclockSeconds(void)
{
#ifdef TIMER_USE_MPI
   return(MPI_Wtime());
#else
#ifdef WIN32
   clock_t cl=clock();
   return(((double) cl)/((double) CLOCKS_PER_SEC));
#else
   struct tms usage;
   long wallclock = times(&usage);
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

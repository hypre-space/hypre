/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/*
 * File:	timer.h
 * Copyright:	(c) 1997 The Regents of the University of California
 * Author:	Scott Kohn (skohn@llnl.gov)
 * Description:	somewhat portable timing routines for C++, C, and Fortran
 */

#ifndef _included_timer_
#define _included_timer_

#ifdef  __cplusplus
extern "C" {
#endif

/**
 * @name Simple Timing Routines
 *
 * These timing routines provide a relatively portable method to obtain
 * both CPU time and wallclock time.  They should run under MPI and any
 * POSIX system.
 *
 * The Fortran routines assume that the Fortran compiler follows the name
 * mangling convention that subroutine names are converted to lower case
 * with a trailing underscore ``_''.
 *
 * All of these routines use the package prefix time_.
 */
/*@{*/
/**
 * Return wall clock seconds measured from some unknown reference point.
 */
extern double time_getWallclockSeconds(void);

/**
 * Return CPU seconds measured from some unknown reference point.
 */
extern double time_getCPUSeconds(void);

/**
 * Fortran-callable routine to get wall clock seconds.
 */
extern double time_get_wallclock_seconds_(void);

/**
 * Fortran-callable routine to get CPU seconds.
 */
extern double time_get_cpu_seconds_(void);
/*@}*/

#ifdef  __cplusplus
}
#endif
#endif

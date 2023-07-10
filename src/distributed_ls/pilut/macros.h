/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef MACROS_H
#define MACROS_H

/*
 * macros.h
 *
 * This file contains utility and debugging macros
 *
 * 7/1
 *    Ownership changed from Karypis to MGates
 * 7/8
 *  - moved all constants, timer, and MIS macros to here
 *  - redid timer in MPI and verified
 *  - made RandomSeed() macro and verified
 * 7/11
 *  - removed some unused macros
 * 7/29
 *  - removed more unused things (randoms, CommInterface)
 *
 * 8/30: ownership changed from MGates to ACleary
 *
 * 12/10
 *  - added MACHINE_IS macro
 *  - added HYPRE_Real precision prototypes to Fortran BLAS
 *
 * 1/13
 *  - added two macros to deal with 0 vs 1-based indexing
 *
 * 1/14
 *  - Modified COLIND macro to translate from local to global indices
 */

/*********************************************************************
* Constants
**********************************************************************/
/* Define what machine we are on. -AC */
#define MACHINE_IS_CRAY
#undef MACHINE_IS_CRAY
#define MACHINE_IS_SOLARIS

/* Define precision. -AC */
/* Cray T3D uses 32bit shorts and 64bit IEEE doubles */
#define  USE_SHORT
#undef USE_SHORT
#define USE_DOUBLE

/* Maximum number of levels */
#define MAXNLEVEL  500

/* communication limits */
#define MAX_NPES   256   /* Maximum # of supported processors */

/* Macros for names of Fortran BLAS routines */
/* AJC: added HYPRE_Real precision prototypes using MACHINE_IS_ */
/* DOK: We can include _hypre_blas.h and use hypre_<dnrm2, ddot, dcopy> directly instead */
/*
#ifdef MACHINE_IS_CRAY
#ifdef USE_SHORT
#define SNRM2 SNRM2
#define SDOT SDOT
#define SCOPY SCOPY
#else
#define SNRM2 DNRM2
#define SDOT DDOT
#define SCOPY DCOPY
#endif
#else
#ifdef MACHINE_IS_SOLARIS
#include "_hypre_blas.h"
#ifdef USE_SHORT
#define SNRM2 hypre_snrm2
#define SDOT hypre_sdot
#define SCOPY hypre_scopy
#else
#define SNRM2 hypre_dnrm2
#define SDOT hypre_ddot
#define SCOPY hypre_dcopy
#endif
#else
#ifdef USE_SHORT
#define SNRM2 SNRM2
#define SDOT SDOT
#define SCOPY SCOPY
#else
#define SNRM2 DNRM2
#define SDOT DDOT
#define SCOPY DCOPY
#endif
#endif
#endif
*/
/*********************************************************************
* Utility Macros
**********************************************************************/
/* MPI and Cray native timers. Note MPI uses doubles while Cray uses longs */
#if defined(MACHINE_IS_CRAY) && MACHINE_IS_CRAY
# define cleartimer(tmr) (tmr = 0)
# define starttimer(tmr) (tmr -= rtclock())
# define stoptimer(tmr)  (tmr += rtclock())
# define gettimer(tmr)   ((HYPRE_Real) tmr*_secpertick)
  typedef hypre_longint timer ;
#else
# define cleartimer(tmr) (tmr = 0.0)
# define starttimer(tmr) (tmr -= hypre_MPI_Wtime())
# define stoptimer(tmr)  (tmr += hypre_MPI_Wtime())
# define gettimer(tmr)   (tmr)
  typedef HYPRE_Real timer ;
#endif

/* This random seed maybe should be dynamic? That produces
 * different results each run though--is that what we want?? */
#define RandomSeed()     (srand(mype+1111))

#define SWAP(a, b, tmp) do {(tmp) = (a); (a) = (b); (b) = (tmp);} while(0)

#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#define MIN(a, b) ((a) >= (b) ? (b) : (a))

/* used in parilut.c. The LSB is MIS flag, all higher bits are the row index. */
#define IsInMIS(a)    (((a)&1) == 1)
#define StripMIS(a)   (((a)>>1))

#define IsLocal(a)    (((a)&1) == 0)
#define StripLocal(a) (((a)>>1))

#endif

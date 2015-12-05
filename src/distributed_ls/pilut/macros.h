/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




#ifndef MACROS_H
#define MACROS_H

#include "../../utilities/general.h"
#include "../../utilities/fortran.h"

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
 *  - added double precision prototypes to Fortran BLAS
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
/* AJC: added double precision prototypes using MACHINE_IS_ */
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
#ifdef USE_SHORT
#define SNRM2 hypre_F90_NAME_BLAS(snrm2, SNRM2)
#define SDOT hypre_F90_NAME_BLAS(sdot, SDOT)
#define SCOPY hypre_F90_NAME_BLAS(scopy, SCOPY)
#else
#define SNRM2 hypre_F90_NAME_BLAS(dnrm2, DNRM2)
#define SDOT hypre_F90_NAME_BLAS(ddot, DDOT)
#define SCOPY hypre_F90_NAME_BLAS(dcopy, DCOPY)
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

/*********************************************************************
* Utility Macros 
**********************************************************************/
/* MPI and Cray native timers. Note MPI uses doubles while Cray uses longs */
#if MACHINE_IS_CRAY
# define cleartimer(tmr) (tmr = 0)
# define starttimer(tmr) (tmr -= rtclock())
# define stoptimer(tmr)  (tmr += rtclock())
# define gettimer(tmr)   ((double) tmr*_secpertick)
  typedef hypre_longint timer ;
#else
# define cleartimer(tmr) (tmr = 0.0)
# define starttimer(tmr) (tmr -= hypre_MPI_Wtime())
# define stoptimer(tmr)  (tmr += hypre_MPI_Wtime())
# define gettimer(tmr)   (tmr)
  typedef double timer ;
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

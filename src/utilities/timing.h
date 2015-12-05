/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * Header file for doing timing
 *
 *****************************************************************************/

#ifndef HYPRE_TIMING_HEADER
#define HYPRE_TIMING_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Prototypes for low-level timing routines
 *--------------------------------------------------------------------------*/

/* timer.c */
double time_getWallclockSeconds( void );
double time_getCPUSeconds( void );
double time_get_wallclock_seconds_( void );
double time_get_cpu_seconds_( void );

/*--------------------------------------------------------------------------
 * With timing off
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_TIMING

#define hypre_InitializeTiming(name) 0
#define hypre_IncFLOPCount(inc)
#define hypre_BeginTiming(i)
#define hypre_EndTiming(i)
#define hypre_PrintTiming(heading, comm)
#define hypre_FinalizeTiming(index)

/*--------------------------------------------------------------------------
 * With timing on
 *--------------------------------------------------------------------------*/

#else

/*-------------------------------------------------------
 * Global timing structure
 *-------------------------------------------------------*/

typedef struct
{
   double  *wall_time;
   double  *cpu_time;
   double  *flops;
   char   **name;
   HYPRE_Int     *state;     /* boolean flag to allow for recursive timing */
   HYPRE_Int     *num_regs;  /* count of how many times a name is registered */

   HYPRE_Int      num_names;
   HYPRE_Int      size;

   double   wall_count;
   double   CPU_count;
   double   FLOP_count;

} hypre_TimingType;

#ifdef HYPRE_TIMING_GLOBALS
hypre_TimingType *hypre_global_timing = NULL;
#else
extern hypre_TimingType *hypre_global_timing;
#endif

/*-------------------------------------------------------
 * Accessor functions
 *-------------------------------------------------------*/

#ifndef HYPRE_USE_PTHREADS
#define hypre_TimingWallTime(i) (hypre_global_timing -> wall_time[(i)])
#define hypre_TimingCPUTime(i)  (hypre_global_timing -> cpu_time[(i)])
#define hypre_TimingFLOPS(i)    (hypre_global_timing -> flops[(i)])
#define hypre_TimingName(i)     (hypre_global_timing -> name[(i)])
#define hypre_TimingState(i)    (hypre_global_timing -> state[(i)])
#define hypre_TimingNumRegs(i)  (hypre_global_timing -> num_regs[(i)])
#define hypre_TimingWallCount   (hypre_global_timing -> wall_count)
#define hypre_TimingCPUCount    (hypre_global_timing -> CPU_count)
#define hypre_TimingFLOPCount   (hypre_global_timing -> FLOP_count)
#else
#define hypre_TimingWallTime(i) (hypre_global_timing[threadid].wall_time[(i)])
#define hypre_TimingCPUTime(i)  (hypre_global_timing[threadid].cpu_time[(i)])
#define hypre_TimingFLOPS(i)    (hypre_global_timing[threadid].flops[(i)])
#define hypre_TimingName(i)     (hypre_global_timing[threadid].name[(i)])
#define hypre_TimingState(i)    (hypre_global_timing[threadid].state[(i)])
#define hypre_TimingNumRegs(i)  (hypre_global_timing[threadid].num_regs[(i)])
#define hypre_TimingWallCount   (hypre_global_timing[threadid].wall_count)
#define hypre_TimingCPUCount    (hypre_global_timing[threadid].CPU_count)
#define hypre_TimingFLOPCount   (hypre_global_timing[threadid].FLOP_count)
#define hypre_TimingAllFLOPS    (hypre_global_timing[hypre_NumThreads].FLOP_count)
#endif

/*-------------------------------------------------------
 * Prototypes
 *-------------------------------------------------------*/

/* timing.c */
HYPRE_Int hypre_InitializeTiming( const char *name );
HYPRE_Int hypre_FinalizeTiming( HYPRE_Int time_index );
HYPRE_Int hypre_IncFLOPCount( HYPRE_Int inc );
HYPRE_Int hypre_BeginTiming( HYPRE_Int time_index );
HYPRE_Int hypre_EndTiming( HYPRE_Int time_index );
HYPRE_Int hypre_ClearTiming( void );
HYPRE_Int hypre_PrintTiming( const char *heading , MPI_Comm comm );

#endif

#ifdef __cplusplus
}
#endif

#endif

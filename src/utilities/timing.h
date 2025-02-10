/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
HYPRE_Real time_getWallclockSeconds( void );
HYPRE_Real time_getCPUSeconds( void );
HYPRE_Real time_get_wallclock_seconds_( void );
HYPRE_Real time_get_cpu_seconds_( void );

/*--------------------------------------------------------------------------
 * With timing off
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_TIMING

#define hypre_InitializeTiming(name) 0
#define hypre_FinalizeTiming(index)
#define hypre_FinalizeAllTimings()
#define hypre_IncFLOPCount(inc)
#define hypre_BeginTiming(i)
#define hypre_EndTiming(i)
#define hypre_PrintTiming(heading, comm)
#define hypre_ClearTiming()
#define hypre_GetTiming(heading, comm, time)

/*--------------------------------------------------------------------------
 * With timing on
 *--------------------------------------------------------------------------*/

#else

/*-------------------------------------------------------
 * Global timing structure
 *-------------------------------------------------------*/

typedef struct
{
   HYPRE_Real  *wall_time;
   HYPRE_Real  *cpu_time;
   HYPRE_Real  *flops;
   char   **name;
   HYPRE_Int     *state;     /* boolean flag to allow for recursive timing */
   HYPRE_Int     *num_regs;  /* count of how many times a name is registered */

   HYPRE_Int      num_names;
   HYPRE_Int      size;

   HYPRE_Real   wall_count;
   HYPRE_Real   CPU_count;
   HYPRE_Real   FLOP_count;

} hypre_TimingType;

#ifdef HYPRE_TIMING_GLOBALS
hypre_TimingType *hypre_global_timing = NULL;
#else
extern hypre_TimingType *hypre_global_timing;
#endif

/*-------------------------------------------------------
 * Accessor functions
 *-------------------------------------------------------*/

#define hypre_TimingWallTime(i) (hypre_global_timing -> wall_time[(i)])
#define hypre_TimingCPUTime(i)  (hypre_global_timing -> cpu_time[(i)])
#define hypre_TimingFLOPS(i)    (hypre_global_timing -> flops[(i)])
#define hypre_TimingName(i)     (hypre_global_timing -> name[(i)])
#define hypre_TimingState(i)    (hypre_global_timing -> state[(i)])
#define hypre_TimingNumRegs(i)  (hypre_global_timing -> num_regs[(i)])
#define hypre_TimingWallCount   (hypre_global_timing -> wall_count)
#define hypre_TimingCPUCount    (hypre_global_timing -> CPU_count)
#define hypre_TimingFLOPCount   (hypre_global_timing -> FLOP_count)

/*-------------------------------------------------------
 * Prototypes
 *-------------------------------------------------------*/

/* timing.c */
HYPRE_Int hypre_InitializeTiming( const char *name );
HYPRE_Int hypre_FinalizeTiming( HYPRE_Int time_index );
HYPRE_Int hypre_FinalizeAllTimings( void );
HYPRE_Int hypre_IncFLOPCount( HYPRE_BigInt inc );
HYPRE_Int hypre_BeginTiming( HYPRE_Int time_index );
HYPRE_Int hypre_EndTiming( HYPRE_Int time_index );
HYPRE_Int hypre_ClearTiming( void );
HYPRE_Int hypre_PrintTiming( const char *heading, MPI_Comm comm );
HYPRE_Int hypre_GetTiming( const char *heading, HYPRE_Real *wall_time_ptr, MPI_Comm comm );

#endif

#ifdef __cplusplus
}
#endif

#endif


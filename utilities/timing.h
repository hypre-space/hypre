/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header file for doing timing
 *
 *****************************************************************************/

#ifndef ZZZ_TIMING_HEADER
#define ZZZ_TIMING_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"
#include "timer.h"

/*--------------------------------------------------------------------------
 * With timing off
 *--------------------------------------------------------------------------*/

#ifndef ZZZ_TIMING

#define zzz_InitializeTiming(name) (int)(name)
#define zzz_IncFLOPCount(inc)
#define zzz_BeginTiming(i)
#define zzz_EndTiming(i)
#define zzz_PrintTiming(comm)
#define zzz_FinalizeTiming(index)

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
   int     *state;     /* boolean flag to allow for recursive timing */
   int     *num_regs;  /* count of how many times a name is registered */

   int      size;

   double   wall_count;
   double   CPU_count;
   double   FLOP_count;

} zzz_TimingType;

#ifdef ZZZ_TIMING_GLOBALS
zzz_TimingType *zzz_global_timing = NULL;
#else
extern zzz_TimingType *zzz_global_timing;
#endif

/*-------------------------------------------------------
 * Accessor functions
 *-------------------------------------------------------*/

#define zzz_TimingWallTime(i) (zzz_global_timing -> wall_time[(i)])
#define zzz_TimingCPUTime(i)  (zzz_global_timing -> cpu_time[(i)])
#define zzz_TimingFLOPS(i)    (zzz_global_timing -> flops[(i)])
#define zzz_TimingName(i)     (zzz_global_timing -> name[(i)])
#define zzz_TimingState(i)    (zzz_global_timing -> state[(i)])
#define zzz_TimingNumRegs(i)  (zzz_global_timing -> num_regs[(i)])
#define zzz_TimingWallCount   (zzz_global_timing -> wall_count)
#define zzz_TimingCPUCount    (zzz_global_timing -> CPU_count)
#define zzz_TimingFLOPCount   (zzz_global_timing -> FLOP_count)

/*-------------------------------------------------------
 * Prototypes
 *-------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif
 
 
/* timing.c */
int zzz_InitializeTiming P((char *name ));
void zzz_IncFLOPCount P((int inc ));
void zzz_BeginTiming P((int time_index ));
void zzz_EndTiming P((int time_index ));
void zzz_PrintTiming P((MPI_Comm *comm ));
void zzz_FinalizeTiming P((int time_index ));
 
#undef P

#endif

#endif

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_Euclid.h"
/* #include "Timer_dh.h" */
/* #include "Mem_dh.h" */

#undef __FUNC__
#define __FUNC__ "Timer_dhCreate"
void Timer_dhCreate(Timer_dh *t)
{
  START_FUNC_DH
  struct _timer_dh* tmp = (struct _timer_dh*)MALLOC_DH(sizeof(struct _timer_dh)); CHECK_V_ERROR;
  *t = tmp;

  tmp->isRunning = false;
  tmp->begin_wall = 0.0;
  tmp->end_wall = 0.0;
#ifdef EUCLID_TIMING
  tmp->sc_clk_tck = sysconf(_SC_CLK_TCK);
#else
  tmp->sc_clk_tck = CLOCKS_PER_SEC;
#endif

#if defined(EUCLID_TIMING)
  hypre_sprintf(msgBuf_dh, "using EUCLID_TIMING; _SC_CLK_TCK = %i", (HYPRE_Int)tmp->sc_clk_tck);
  SET_INFO(msgBuf_dh);
#elif defined(hypre_MPI_TIMING)
  SET_INFO("using MPI timing")
#else
  SET_INFO("using JUNK timing")
#endif
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Timer_dhDestroy"
void Timer_dhDestroy(Timer_dh t)
{
  START_FUNC_DH
  FREE_DH(t);
  END_FUNC_DH
}

/*-------------------------------------------------------------------------------
 * EUCLID_TIMING timing methods; these use times() to record
 * both wall and cpu time.
 *-------------------------------------------------------------------------------*/

#ifdef EUCLID_TIMING

#undef __FUNC__
#define __FUNC__ "Timer_dhStart"
void Timer_dhStart(Timer_dh t)
{
  START_FUNC_DH
  t->begin_wall = times(&(t->begin_cpu));
  t->isRunning = true;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Timer_dhStop"
void Timer_dhStop(Timer_dh t)
{
  START_FUNC_DH
  t->end_wall = times(&(t->end_cpu));
  t->isRunning = false;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Timer_dhReadWall"
HYPRE_Real Timer_dhReadWall(Timer_dh t)
{
  START_FUNC_DH
  HYPRE_Real retval = 0.0;
  hypre_longint sc_clk_tck = t->sc_clk_tck;
  if (t->isRunning) t->end_wall = times(&(t->end_cpu));
  retval = (HYPRE_Real)(t->end_wall - t->begin_wall) / (HYPRE_Real)sc_clk_tck;
  END_FUNC_VAL(retval)
}

#undef __FUNC__
#define __FUNC__ "Timer_dhReadCPU"
HYPRE_Real Timer_dhReadCPU(Timer_dh t)
{
  START_FUNC_DH
  HYPRE_Real retval;
  hypre_longint sc_clk_tck = t->sc_clk_tck;
  if (t->isRunning) t->end_wall = times(&(t->end_cpu));
  retval = (HYPRE_Real)(t->end_cpu.tms_utime - t->begin_cpu.tms_utime
          + t->end_cpu.tms_stime -  t->begin_cpu.tms_stime
          + t->end_cpu.tms_cutime - t->begin_cpu.tms_cutime
          + t->end_cpu.tms_cstime -  t->begin_cpu.tms_cstime)
                      /(HYPRE_Real)sc_clk_tck;
  END_FUNC_VAL(retval)
}

#undef __FUNC__
#define __FUNC__ "Timer_dhReadUsage"
HYPRE_Real Timer_dhReadUsage(Timer_dh t)
{
  START_FUNC_DH
  HYPRE_Real cpu = Timer_dhReadCPU(t);
  HYPRE_Real wall = Timer_dhReadWall(t);
  HYPRE_Real retval = 100.0*cpu/wall;
  END_FUNC_VAL(retval);
}

/*-------------------------------------------------------------------------------
 * Parallel timing functions; these use hypre_MPI_Wtime() to record
 * wall-clock time only.
 *-------------------------------------------------------------------------------*/

#elif defined(hypre_MPI_TIMING)

#undef __FUNC__
#define __FUNC__ "Timer_dhStart"
void Timer_dhStart(Timer_dh t)
{
  START_FUNC_DH
  t->begin_wall = hypre_MPI_Wtime();
  t->isRunning = true;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Timer_dhStop"
void Timer_dhStop(Timer_dh t)
{
  START_FUNC_DH
  t->end_wall = hypre_MPI_Wtime();
  t->isRunning = false;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Timer_dhReadWall"
HYPRE_Real Timer_dhReadWall(Timer_dh t)
{
  START_FUNC_DH
  HYPRE_Real retval;
  if (t->isRunning) t->end_wall = hypre_MPI_Wtime();
  retval = t->end_wall - t->begin_wall;
  END_FUNC_VAL(retval)
}

#undef __FUNC__
#define __FUNC__ "Timer_dhReadCPU"
HYPRE_Real Timer_dhReadCPU(Timer_dh t)
{
  START_FUNC_DH
  END_FUNC_VAL(-1.0)
}

#undef __FUNC__
#define __FUNC__ "Timer_dhReadUsage"
HYPRE_Real Timer_dhReadUsage(Timer_dh t)
{
  START_FUNC_DH
  END_FUNC_VAL(-1.0);
}


/*-------------------------------------------------------------------------------
 * junk timing methods -- these do nothing!
 *-------------------------------------------------------------------------------*/

#else

#undef __FUNC__
#define __FUNC__ "Timer_dhStart"
void Timer_dhStart(Timer_dh t)
{
  HYPRE_UNUSED_VAR(t);

  START_FUNC_DH
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Timer_dhStop"
void Timer_dhStop(Timer_dh t)
{
  HYPRE_UNUSED_VAR(t);

  START_FUNC_DH
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "Timer_dhReadWall"
HYPRE_Real Timer_dhReadWall(Timer_dh t)
{
  HYPRE_UNUSED_VAR(t);

  START_FUNC_DH
  END_FUNC_VAL(-1.0)
}

#undef __FUNC__
#define __FUNC__ "Timer_dhReadCPU"
HYPRE_Real Timer_dhReadCPU(Timer_dh t)
{
  HYPRE_UNUSED_VAR(t);

  START_FUNC_DH
  END_FUNC_VAL(-1.0)
}

#undef __FUNC__
#define __FUNC__ "Timer_dhReadUsage"
HYPRE_Real Timer_dhReadUsage(Timer_dh t)
{
  HYPRE_UNUSED_VAR(t);

  START_FUNC_DH
  END_FUNC_VAL(-1.0);
}

#endif

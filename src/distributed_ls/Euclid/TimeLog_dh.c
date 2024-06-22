/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_Euclid.h"
/* #include "TimeLog_dh.h" */
/* #include "Timer_dh.h" */
/* #include "Mem_dh.h" */

#define MAX_TIME_MARKS  100
#define MAX_DESC_LENGTH 60

struct _timeLog_dh {
  HYPRE_Int first;
  HYPRE_Int last; 
  HYPRE_Real time[MAX_TIME_MARKS];
  char   desc[MAX_TIME_MARKS][MAX_DESC_LENGTH];
  Timer_dh timer; 
};

#undef __FUNC__
#define __FUNC__ "TimeLog_dhCreate"
void TimeLog_dhCreate(TimeLog_dh *t)
{
  START_FUNC_DH
  HYPRE_Int i;
  struct _timeLog_dh* tmp = (struct _timeLog_dh*)MALLOC_DH(sizeof(struct _timeLog_dh)); CHECK_V_ERROR;
  *t = tmp;
  tmp->first = tmp->last = 0;
  Timer_dhCreate(&tmp->timer);
  for (i=0; i<MAX_TIME_MARKS; ++i) strcpy(tmp->desc[i], "X");
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "TimeLog_dhDestroy"
void TimeLog_dhDestroy(TimeLog_dh t)
{
  START_FUNC_DH
  Timer_dhDestroy(t->timer); 
  FREE_DH(t);
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "TimeLog_dhStart"
void TimeLog_dhStart(TimeLog_dh t)
{
  START_FUNC_DH
  Timer_dhStart(t->timer);
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "TimeLog_dhStop"
void TimeLog_dhStop(TimeLog_dh t)
{
  START_FUNC_DH
  Timer_dhStop(t->timer);
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "TimeLog_dhMark"
void TimeLog_dhMark(TimeLog_dh t, const char *desc)
{
  START_FUNC_DH
  if (t->last < MAX_TIME_MARKS - 3) {
/*     SET_V_ERROR("overflow; please increase MAX_TIME_MARKS and recompile"); */
    Timer_dhStop(t->timer);
    t->time[t->last] = Timer_dhReadWall(t->timer);
    Timer_dhStart(t->timer);
    hypre_sprintf(t->desc[t->last], "%s", desc);
    t->last += 1;
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "TimeLog_dhReset"
void TimeLog_dhReset(TimeLog_dh t)
{
  START_FUNC_DH
  if (t->last < MAX_TIME_MARKS - 2) {
    HYPRE_Real total = 0.0;
    HYPRE_Int i, first = t->first, last = t->last;
    for (i=first; i<last; ++i) total += t->time[i];
    t->time[last] = total;
    hypre_sprintf(t->desc[last], "========== totals, and reset ==========\n");
    t->last += 1;
    t->first = t->last;
    Timer_dhStart(t->timer);
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "TimeLog_dhPrint"
void TimeLog_dhPrint(TimeLog_dh t, FILE *fp, bool allPrint)
{
  START_FUNC_DH
  HYPRE_Int i;
  HYPRE_Real total = 0.0;
  HYPRE_Real timeMax[MAX_TIME_MARKS]; HYPRE_Real timeMin[MAX_TIME_MARKS];
  static bool wasSummed = false;


  if (! wasSummed) {
    for (i=t->first; i<t->last; ++i) total += t->time[i];
    t->time[t->last] = total;
    hypre_sprintf(t->desc[t->last], "========== totals, and reset ==========\n");
    t->last += 1;

    hypre_MPI_Allreduce(t->time, timeMax, t->last, hypre_MPI_REAL, hypre_MPI_MAX, comm_dh);
    hypre_MPI_Allreduce(t->time, timeMin, t->last, hypre_MPI_REAL, hypre_MPI_MIN, comm_dh);
    wasSummed = true;
  }

  if (fp != NULL) {
    if (myid_dh == 0 || allPrint) {
      hypre_fprintf(fp,"\n----------------------------------------- timing report\n");
      hypre_fprintf(fp, "\n   self     max     min\n");
      for (i=0; i<t->last; ++i) {
        hypre_fprintf(fp, "%7.3f %7.3f %7.3f   #%s\n", t->time[i],
                                timeMax[i], timeMin[i], 
                                t->desc[i]);
      }
      fflush(fp);
    } 
  } /* if (fp != NULL) */
  END_FUNC_DH
}

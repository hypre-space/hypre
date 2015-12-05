/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/




#include "TimeLog_dh.h"
#include "Timer_dh.h"
#include "Mem_dh.h"

#define MAX_TIME_MARKS  100
#define MAX_DESC_LENGTH 60

struct _timeLog_dh {
  HYPRE_Int first;
  HYPRE_Int last; 
  double time[MAX_TIME_MARKS];
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
    double total = 0.0;
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
  double total = 0.0;
  double timeMax[MAX_TIME_MARKS]; double timeMin[MAX_TIME_MARKS];
  static bool wasSummed = false;


  if (! wasSummed) {
    for (i=t->first; i<t->last; ++i) total += t->time[i];
    t->time[t->last] = total;
    hypre_sprintf(t->desc[t->last], "========== totals, and reset ==========\n");
    t->last += 1;

    hypre_MPI_Allreduce(t->time, timeMax, t->last, hypre_MPI_DOUBLE, hypre_MPI_MAX, comm_dh);
    hypre_MPI_Allreduce(t->time, timeMin, t->last, hypre_MPI_DOUBLE, hypre_MPI_MIN, comm_dh);
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

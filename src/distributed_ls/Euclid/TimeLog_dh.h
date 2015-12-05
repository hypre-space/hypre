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




/*
    Euclid employs a global object: 

        TimeLog_dh timlog_dh;

    for recording timing information.  
*/



#ifndef TIMELOG_DH_DH
#define TIMELOG_DH_DH

#include "euclid_common.h"

extern void TimeLog_dhCreate(TimeLog_dh *t);
extern void TimeLog_dhDestroy(TimeLog_dh t);
extern void TimeLog_dhStart(TimeLog_dh t);
extern void TimeLog_dhStop(TimeLog_dh t);
extern void TimeLog_dhReset(TimeLog_dh t);
extern void TimeLog_dhMark(TimeLog_dh t, const char *description);
extern void TimeLog_dhPrint(TimeLog_dh t, FILE *fp, bool allPrint);

#endif

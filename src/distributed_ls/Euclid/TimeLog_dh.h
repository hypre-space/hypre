/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.3 $
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
extern void TimeLog_dhMark(TimeLog_dh t, char *description);
extern void TimeLog_dhPrint(TimeLog_dh t, FILE *fp, bool allPrint);

#endif

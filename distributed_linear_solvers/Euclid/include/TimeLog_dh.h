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

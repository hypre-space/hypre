/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * RowPatt.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Hash.h"

#ifndef _ROWPATT_H
#define _ROWPATT_H

typedef struct
{
    int   maxlen;
    int   len;
    int   prev_len;
    int  *ind;
    int  *back;
    Hash *hash;
}
RowPatt;

RowPatt *RowPattCreate(int maxlen);
void RowPattDestroy(RowPatt *p);
void RowPattReset(RowPatt *p);
void RowPattMerge(RowPatt *p, int len, int *ind);
void RowPattMergeExt(RowPatt *p, int len, int *ind, int beg, int end);
void RowPattGet(RowPatt *p, int *lenp, int **indp);
void RowPattPrevLevel(RowPatt *p, int *lenp, int **indp);

#endif /* _ROWPATT_H */

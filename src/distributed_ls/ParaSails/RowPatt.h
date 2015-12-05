/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * RowPatt.h header file.
 *
 *****************************************************************************/

#include <stdio.h>

#ifndef _ROWPATT_H
#define _ROWPATT_H

typedef struct
{
    int  maxlen;
    int  len;
    int  prev_len;
    int *ind;
    int *mark;
    int *buffer; /* buffer used for outputting indices */
    int  buflen; /* length of this buffer */
}
RowPatt;

RowPatt *RowPattCreate(int maxlen);
void RowPattDestroy(RowPatt *p);
void RowPattReset(RowPatt *p);
void RowPattMerge(RowPatt *p, int len, int *ind);
void RowPattMergeExt(RowPatt *p, int len, int *ind, int num_loc);
void RowPattGet(RowPatt *p, int *lenp, int **indp);
void RowPattPrevLevel(RowPatt *p, int *lenp, int **indp);

#endif /* _ROWPATT_H */

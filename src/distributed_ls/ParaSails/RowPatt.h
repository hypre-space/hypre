/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
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
    HYPRE_Int  maxlen;
    HYPRE_Int  len;
    HYPRE_Int  prev_len;
    HYPRE_Int *ind;
    HYPRE_Int *mark;
    HYPRE_Int *buffer; /* buffer used for outputting indices */
    HYPRE_Int  buflen; /* length of this buffer */
}
RowPatt;

RowPatt *RowPattCreate(HYPRE_Int maxlen);
void RowPattDestroy(RowPatt *p);
void RowPattReset(RowPatt *p);
void RowPattMerge(RowPatt *p, HYPRE_Int len, HYPRE_Int *ind);
void RowPattMergeExt(RowPatt *p, HYPRE_Int len, HYPRE_Int *ind, HYPRE_Int num_loc);
void RowPattGet(RowPatt *p, HYPRE_Int *lenp, HYPRE_Int **indp);
void RowPattPrevLevel(RowPatt *p, HYPRE_Int *lenp, HYPRE_Int **indp);

#endif /* _ROWPATT_H */

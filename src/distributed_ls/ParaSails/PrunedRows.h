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
 * PrunedRows.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Mem.h"
#include "DiagScale.h"

#ifndef _PRUNEDROWS_H
#define _PRUNEDROWS_H

typedef struct
{
    Mem      *mem;   /* storage for arrays, indices, and values */
    int      size;

    int     *len;
    int    **ind;
}
PrunedRows;

PrunedRows *PrunedRowsCreate(Matrix *mat, int size, DiagScale *diag_scale,
  double thresh);
void PrunedRowsDestroy(PrunedRows *p);
int *PrunedRowsAlloc(PrunedRows *p, int len);
void PrunedRowsPut(PrunedRows *p, int index, int len, int *ind);
void PrunedRowsGet(PrunedRows *p, int index, int *lenp, int **indp);

#endif /* _PRUNEDROWS_H */

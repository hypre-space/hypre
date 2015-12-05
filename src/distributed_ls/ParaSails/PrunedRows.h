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
    HYPRE_Int      size;

    HYPRE_Int     *len;
    HYPRE_Int    **ind;
}
PrunedRows;

PrunedRows *PrunedRowsCreate(Matrix *mat, HYPRE_Int size, DiagScale *diag_scale,
  double thresh);
void PrunedRowsDestroy(PrunedRows *p);
HYPRE_Int *PrunedRowsAlloc(PrunedRows *p, HYPRE_Int len);
void PrunedRowsPut(PrunedRows *p, HYPRE_Int index, HYPRE_Int len, HYPRE_Int *ind);
void PrunedRowsGet(PrunedRows *p, HYPRE_Int index, HYPRE_Int *lenp, HYPRE_Int **indp);

#endif /* _PRUNEDROWS_H */

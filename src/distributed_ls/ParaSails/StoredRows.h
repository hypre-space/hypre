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
 * StoredRows.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Mem.h"
#include "Matrix.h"

#ifndef _STOREDROWS_H
#define _STOREDROWS_H

typedef struct
{
    Matrix   *mat;   /* the matrix corresponding to the rows stored here */
    Mem      *mem;   /* storage for arrays, indices, and values */

    HYPRE_Int      size;
    HYPRE_Int      num_loc;

    HYPRE_Int     *len;
    HYPRE_Int    **ind;
    double **val;

    HYPRE_Int      count;
}
StoredRows;

StoredRows *StoredRowsCreate(Matrix *mat, HYPRE_Int size);
void    StoredRowsDestroy(StoredRows *p);
HYPRE_Int    *StoredRowsAllocInd(StoredRows *p, HYPRE_Int len);
double *StoredRowsAllocVal(StoredRows *p, HYPRE_Int len);
void    StoredRowsPut(StoredRows *p, HYPRE_Int index, HYPRE_Int len, HYPRE_Int *ind, double *val);
void    StoredRowsGet(StoredRows *p, HYPRE_Int index, HYPRE_Int *lenp, HYPRE_Int **indp, 
          double **valp);

#endif /* _STOREDROWS_H */

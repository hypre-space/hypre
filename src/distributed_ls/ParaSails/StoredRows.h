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

    int      size;
    int      num_loc;

    int     *len;
    int    **ind;
    double **val;

    int      count;
}
StoredRows;

StoredRows *StoredRowsCreate(Matrix *mat, int size);
void    StoredRowsDestroy(StoredRows *p);
int    *StoredRowsAllocInd(StoredRows *p, int len);
double *StoredRowsAllocVal(StoredRows *p, int len);
void    StoredRowsPut(StoredRows *p, int index, int len, int *ind, double *val);
void    StoredRowsGet(StoredRows *p, int index, int *lenp, int **indp, 
          double **valp);

#endif /* _STOREDROWS_H */

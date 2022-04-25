/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
    HYPRE_Real **val;

    HYPRE_Int      count;
}
StoredRows;

StoredRows *StoredRowsCreate(Matrix *mat, HYPRE_Int size);
void    StoredRowsDestroy(StoredRows *p);
HYPRE_Int    *StoredRowsAllocInd(StoredRows *p, HYPRE_Int len);
HYPRE_Real *StoredRowsAllocVal(StoredRows *p, HYPRE_Int len);
void    StoredRowsPut(StoredRows *p, HYPRE_Int index, HYPRE_Int len, HYPRE_Int *ind, HYPRE_Real *val);
void    StoredRowsGet(StoredRows *p, HYPRE_Int index, HYPRE_Int *lenp, HYPRE_Int **indp, 
          HYPRE_Real **valp);

#endif /* _STOREDROWS_H */

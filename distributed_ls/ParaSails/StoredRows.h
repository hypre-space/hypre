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
 * StoredRows.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Mem.h"
#include "Hash.h"
#include "Matrix.h"

#ifndef _STOREDROWS_H
#define _STOREDROWS_H

typedef struct
{
    Matrix   *mat;   /* the matrix corresponding to the rows stored here */
    Hash     *hash;  /* hash table for accessing */
    Mem      *mem;   /* storage for arrays, indices, and values */

    int     *len;
    int    **ind;
    double **val;
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

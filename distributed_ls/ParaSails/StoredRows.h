/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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

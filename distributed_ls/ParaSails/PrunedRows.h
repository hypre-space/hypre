/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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

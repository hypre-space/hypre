/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

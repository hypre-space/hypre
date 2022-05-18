/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Numbering.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Common.h"
#include "Matrix.h"
#include "Hash.h"

#ifndef _NUMBERING_H
#define _NUMBERING_H

struct numbering
{
    HYPRE_Int   size;    /* max number of indices that can be stored */
    HYPRE_Int   beg_row;
    HYPRE_Int   end_row;
    HYPRE_Int   num_loc; /* number of local indices */
    HYPRE_Int   num_ind; /* number of indices */

    HYPRE_Int  *local_to_global;
    Hash *hash;
};

typedef struct numbering Numbering;

Numbering *NumberingCreate(Matrix *m, HYPRE_Int size);
Numbering *NumberingCreateCopy(Numbering *orig);
void NumberingDestroy(Numbering *numb);
void NumberingLocalToGlobal(Numbering *numb, HYPRE_Int len, HYPRE_Int *local, HYPRE_Int *global);
void NumberingGlobalToLocal(Numbering *numb, HYPRE_Int len, HYPRE_Int *global, HYPRE_Int *local);

#endif /* _NUMBERING_H */

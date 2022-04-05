/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * DiagScale.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Hash.h"
#include "Matrix.h"
#include "Numbering.h"

#ifndef _DIAGSCALE_H
#define _DIAGSCALE_H

typedef struct
{
    HYPRE_Int     offset;      /* number of on-processor entries */
    HYPRE_Real *local_diags; /* on-processor entries */
    HYPRE_Real *ext_diags;   /* off-processor entries */
}
DiagScale;

DiagScale *DiagScaleCreate(Matrix *A, Numbering *numb);
void DiagScaleDestroy(DiagScale *p);
HYPRE_Real DiagScaleGet(DiagScale *p, HYPRE_Int index);

#endif /* _DIAGSCALE_H */

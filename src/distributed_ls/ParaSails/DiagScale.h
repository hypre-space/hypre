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
    double *local_diags; /* on-processor entries */
    double *ext_diags;   /* off-processor entries */
}
DiagScale;

DiagScale *DiagScaleCreate(Matrix *A, Numbering *numb);
void DiagScaleDestroy(DiagScale *p);
double DiagScaleGet(DiagScale *p, HYPRE_Int index);

#endif /* _DIAGSCALE_H */

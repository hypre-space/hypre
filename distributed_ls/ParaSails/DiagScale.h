/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
    int     offset;      /* number of on-processor entries */
    double *local_diags; /* on-processor entries */
    double *ext_diags;   /* off-processor entries */
}
DiagScale;

DiagScale *DiagScaleCreate(Matrix *A, Numbering *numb);
void DiagScaleDestroy(DiagScale *p);
double DiagScaleGet(DiagScale *p, int index);

#endif /* _DIAGSCALE_H */

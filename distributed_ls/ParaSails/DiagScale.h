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
    Matrix    *mat;
    double    *local_diags;
    double    *ext_diags;
}
DiagScale;

DiagScale *DiagScaleCreate(Matrix *mat);
void DiagScaleDestroy(DiagScale *p);
double DiagScaleGet(DiagScale *p, int global_index);

#endif /* _DIAGSCALE_H */

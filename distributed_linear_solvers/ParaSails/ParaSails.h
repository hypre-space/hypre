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
 * ParaSails.h header file.
 *
 *****************************************************************************/

#include "Matrix.h"
#include "Numbering.h"
#include "PrunedRows.h"
#include "StoredRows.h"
#include "RowPatt.h"
#include "LoadBal.h"

#ifndef _PARASAILS_H
#define _PARASAILS_H

typedef struct
{
    Matrix     *A;  /* original matrix     */
    Matrix     *M;  /* approximate inverse */
    Numbering  *numb;

    int         symmetric;
    int         num_levels;
    double      thresh;

    DiagScale  *diag_scale;
    PrunedRows *pruned_rows;
    StoredRows *stored_rows;
    LoadBal    *load_bal;
}
ParaSails;

ParaSails *ParaSailsCreate(Matrix *A);
void ParaSailsDestroy(ParaSails *ps);
void ParaSailsSetSym(ParaSails *ps, int sym);
double ParaSailsSelectThresh(ParaSails *ps, double param);
void ParaSailsSetupPattern(ParaSails *ps, double thresh, int num_levels);
void ParaSailsSetupValues(ParaSails *ps, Matrix *A);
double ParaSailsSelectFilter(ParaSails *ps, double param);
void ParaSailsFilterValues(ParaSails *ps, double filter);
void ParaSailsComplete(ParaSails *ps);
void ParaSailsApply(ParaSails *ps, double *u, double *v);

#endif /* _PARASAILS_H */

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
 * HYPRE_ParaSails
 *
 *****************************************************************************/

#include "Common.h"
#include "HYPRE_distributed_matrix_types.h"
#include "HYPRE_distributed_matrix_protos.h"
#include "HYPRE_ParaSails.h"
#include "Matrix.h"
#include "ParaSails.h"

typedef struct
{
    ParaSails *ps;
    Matrix    *A;
}
hypre_ParaSails;

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsCreate - Return a ParaSails preconditioner object "obj"
 * for the matrix "distmat".
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSailsCreate(MPI_Comm comm, HYPRE_DistributedMatrix *distmat, 
  HYPRE_ParaSails *obj, int symmetric)
{
    int beg_row, end_row, row, dummy;
    int len, *ind;
    double *val;
    hypre_ParaSails *internal;

    internal = (hypre_ParaSails *) hypre_CTAlloc(hypre_ParaSails, 1);

    /* Convert distributed matrix to native data structure of ParaSails */

    HYPRE_DistributedMatrixGetLocalRange(distmat, &beg_row, &end_row,
        &dummy, &dummy);
    internal->A = MatrixCreate(comm, beg_row, end_row);

    for (row=beg_row; row<=end_row; row++)
    {
	HYPRE_DistributedMatrixGetRow(distmat, row, &len, &ind, &val);
	MatrixSetRow(internal->A, row-beg_row, len, ind, val);
	HYPRE_DistributedMatrixRestoreRow(distmat, row, &len, &ind, &val);
    }

    MatrixComplete(internal->A);

    /* Call the native code */

    internal->ps = ParaSailsCreate(comm, beg_row, end_row, symmetric);

    *obj = (HYPRE_ParaSails) internal;

    return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsDestroy - Destroy a ParaSails object "ps".
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSailsDestroy(HYPRE_ParaSails ps)
{
    hypre_ParaSails *internal = (hypre_ParaSails *) ps;

    MatrixDestroy(internal->A);
    ParaSailsDestroy(internal->ps);

    hypre_TFree(internal);

    return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetup - Set up a ParaSails preconditioner, using the
 * input parameters "thresh" and "nlevels".
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSailsSetup(HYPRE_ParaSails ps, int sym, double thresh, 
  int nlevels, double filter)
{
    hypre_ParaSails *internal = (hypre_ParaSails *) ps;

    ParaSailsSetupPattern(internal->ps, internal->A, thresh, nlevels);

    ParaSailsSetupValues(internal->ps, internal->A, filter);

    return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsApply - Apply the ParaSails preconditioner to an array 
 * "u", and return the result in the array "v".
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSailsApply(HYPRE_ParaSails ps, double *u, double *v)
{
    hypre_ParaSails *internal = (hypre_ParaSails *) ps;

    ParaSailsApply(internal->ps, u, v);

    return 0;
}

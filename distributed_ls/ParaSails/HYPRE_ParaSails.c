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
    MPI_Comm   comm;
    ParaSails *ps;
}
hypre_ParaSails;

/*--------------------------------------------------------------------------
 * convert_matrix -  Create and convert distributed matrix to native 
 * data structure of ParaSails
 *--------------------------------------------------------------------------*/

static Matrix *convert_matrix(MPI_Comm comm, HYPRE_DistributedMatrix *distmat)
{
    int beg_row, end_row, row, dummy;
    int len, *ind;
    double *val;
    Matrix *mat;

    HYPRE_DistributedMatrixGetLocalRange(distmat, &beg_row, &end_row,
        &dummy, &dummy);

    mat = MatrixCreate(comm, beg_row, end_row);

    for (row=beg_row; row<=end_row; row++)
    {
	HYPRE_DistributedMatrixGetRow(distmat, row, &len, &ind, &val);
	MatrixSetRow(mat, row, len, ind, val);
	HYPRE_DistributedMatrixRestoreRow(distmat, row, &len, &ind, &val);
    }

    MatrixComplete(mat);

    return mat;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsCreate - Return a ParaSails preconditioner object "obj"
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSailsCreate(MPI_Comm comm, HYPRE_ParaSails *obj)
{
    hypre_ParaSails *internal;

    internal = (hypre_ParaSails *) hypre_CTAlloc(hypre_ParaSails, 1);

    internal->comm = comm;
    internal->ps   = NULL;

    *obj = (HYPRE_ParaSails) internal;

    return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsDestroy - Destroy a ParaSails object "ps".
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSailsDestroy(HYPRE_ParaSails obj)
{
    hypre_ParaSails *internal = (hypre_ParaSails *) obj;

    ParaSailsDestroy(internal->ps);

    hypre_TFree(internal);

    return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetup - This function should be used if the preconditioner
 * pattern and values are set up with the same distributed matrix.
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSailsSetup(HYPRE_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, int sym, double thresh, int nlevels,
  double filter, double loadbal, int logging)
{
    double cost;
    Matrix *mat;
    hypre_ParaSails *internal = (hypre_ParaSails *) obj;

    mat = convert_matrix(internal->comm, distmat);

    ParaSailsDestroy(internal->ps);

    internal->ps = ParaSailsCreate(internal->comm, 
        mat->beg_row, mat->end_row, sym);

    ParaSailsSetupPattern(internal->ps, mat, thresh, nlevels);

    if (logging)
        cost = ParaSailsStatsPattern(internal->ps, mat);

    internal->ps->loadbal_beta = loadbal;

    ParaSailsSetupValues(internal->ps, mat, filter);

    if (logging)
        ParaSailsStatsValues(internal->ps, mat);

    MatrixDestroy(mat);

    return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetupPattern - Set up pattern using a distributed matrix.
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSailsSetupPattern(HYPRE_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, int sym, double thresh, int nlevels,
  int logging)
{
    double cost;
    Matrix *mat;
    hypre_ParaSails *internal = (hypre_ParaSails *) obj;

    mat = convert_matrix(internal->comm, distmat);

    ParaSailsDestroy(internal->ps);

    internal->ps = ParaSailsCreate(internal->comm, 
        mat->beg_row, mat->end_row, sym);

    ParaSailsSetupPattern(internal->ps, mat, thresh, nlevels);

    if (logging)
        cost = ParaSailsStatsPattern(internal->ps, mat);

    MatrixDestroy(mat);

    return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetupValues - Set up values using a distributed matrix.
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSailsSetupValues(HYPRE_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, double filter, double loadbal,
  int logging)
{
    Matrix *mat;
    hypre_ParaSails *internal = (hypre_ParaSails *) obj;

    mat = convert_matrix(internal->comm, distmat);

    internal->ps->loadbal_beta = loadbal;
    internal->ps->setup_pattern_time = 0.0;

    ParaSailsSetupValues(internal->ps, mat, filter);

    if (logging)
        ParaSailsStatsValues(internal->ps, mat);

    MatrixDestroy(mat);

    return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsApply - Apply the ParaSails preconditioner to an array 
 * "u", and return the result in the array "v".
 *--------------------------------------------------------------------------*/

int HYPRE_ParaSailsApply(HYPRE_ParaSails obj, double *u, double *v)
{
    hypre_ParaSails *internal = (hypre_ParaSails *) obj;

    ParaSailsApply(internal->ps, u, v);

    return 0;
}

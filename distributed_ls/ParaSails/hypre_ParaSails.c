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
 * hypre_ParaSails
 *
 *****************************************************************************/

#include "Common.h"
#include "HYPRE_distributed_matrix_types.h"
#include "HYPRE_distributed_matrix_protos.h"
#include "hypre_ParaSails.h"
#include "Matrix.h"
#include "ParaSails.h"

typedef struct
{
    MPI_Comm   comm;
    ParaSails *ps;
}
hypre_ParaSails_struct;

/*--------------------------------------------------------------------------
 * balance_info - Dump out information about the partitioning of the 
 * matrix, which affects load balance
 *--------------------------------------------------------------------------*/

#ifdef BALANCE_INFO
static void balance_info(MPI_Comm comm, Matrix *mat)
{
    int mype, num_local, i, total;

    MPI_Comm_rank(comm, &mype);
    num_local = mat->end_row - mat->beg_row + 1;

    /* compute number of nonzeros on local matrix */
    total = 0;
    for (i=0; i<num_local; i++)
        total += mat->lens[i];

    /* each processor prints out its own info */
    printf("%4d: nrows %d, nnz %d, send %d (%d), recv %d (%d)\n", mype, 
    num_local, total, mat->num_send, mat->sendlen, mat->num_recv, mat->recvlen);
}

static void matvec_timing(MPI_Comm comm, Matrix *mat)
{
    double time0, time1;
    double trial1, trial2, trial3, trial4, trial5, trial6;
    double *temp1, *temp2;
    int i, mype;
    int n = mat->end_row - mat->beg_row + 1;

    temp1 = (double *) calloc(n, sizeof(double));
    temp2 = (double *) calloc(n, sizeof(double));

    /* warm-up */
    MPI_Barrier(comm);
    for (i=0; i<100; i++)
       MatrixMatvec(mat, temp1, temp2);

    MPI_Barrier(comm);
    time0 = MPI_Wtime();
    for (i=0; i<100; i++)
       MatrixMatvec(mat, temp1, temp2);
    MPI_Barrier(comm);
    time1 = MPI_Wtime();
    trial1 = time1-time0;

    MPI_Barrier(comm);
    time0 = MPI_Wtime();
    for (i=0; i<100; i++)
       MatrixMatvec(mat, temp1, temp2);
    MPI_Barrier(comm);
    time1 = MPI_Wtime();
    trial2 = time1-time0;

    MPI_Barrier(comm);
    time0 = MPI_Wtime();
    for (i=0; i<100; i++)
       MatrixMatvec(mat, temp1, temp2);
    MPI_Barrier(comm);
    time1 = MPI_Wtime();
    trial3 = time1-time0;

    MPI_Barrier(comm);
    time0 = MPI_Wtime();
    for (i=0; i<100; i++)
       MatrixMatvecSerial(mat, temp1, temp2);
    MPI_Barrier(comm);
    time1 = MPI_Wtime();
    trial4 = time1-time0;

    MPI_Barrier(comm);
    time0 = MPI_Wtime();
    for (i=0; i<100; i++)
       MatrixMatvecSerial(mat, temp1, temp2);
    MPI_Barrier(comm);
    time1 = MPI_Wtime();
    trial5 = time1-time0;

    MPI_Barrier(comm);
    time0 = MPI_Wtime();
    for (i=0; i<100; i++)
       MatrixMatvecSerial(mat, temp1, temp2);
    MPI_Barrier(comm);
    time1 = MPI_Wtime();
    trial6 = time1-time0;

    MPI_Comm_rank(comm, &mype);
    if (mype == 0)
        printf("Timings: %f %f %f Serial: %f %f %f\n", 
	   trial1, trial2, trial3, trial4, trial5, trial6);

    fflush(stdout);

    /* this is all we wanted, so don't waste any more cycles */
    exit(0);
}
#endif

/*--------------------------------------------------------------------------
 * convert_matrix - Create and convert distributed matrix to native 
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

#ifdef BALANCE_INFO
    matvec_timing(comm, mat);
    balance_info(comm, mat);
#endif

    return mat;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsCreate - Return a ParaSails preconditioner object "obj"
 *--------------------------------------------------------------------------*/

int hypre_ParaSailsCreate(MPI_Comm comm, hypre_ParaSails *obj)
{
    hypre_ParaSails_struct *internal;

    internal = (hypre_ParaSails_struct *)
                   hypre_CTAlloc(hypre_ParaSails_struct, 1);

    internal->comm = comm;
    internal->ps   = NULL;

    *obj = (hypre_ParaSails) internal;

    return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsDestroy - Destroy a ParaSails object "ps".
 *--------------------------------------------------------------------------*/

int hypre_ParaSailsDestroy(hypre_ParaSails obj)
{
    hypre_ParaSails_struct *internal = (hypre_ParaSails_struct *) obj;

    ParaSailsDestroy(internal->ps);

    hypre_TFree(internal);

    return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsSetup - This function should be used if the preconditioner
 * pattern and values are set up with the same distributed matrix.
 *--------------------------------------------------------------------------*/

int hypre_ParaSailsSetup(hypre_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, int sym, double thresh, int nlevels,
  double filter, double loadbal, int logging)
{
    /* double cost; */
    Matrix *mat;
    hypre_ParaSails_struct *internal = (hypre_ParaSails_struct *) obj;

    mat = convert_matrix(internal->comm, distmat);

    ParaSailsDestroy(internal->ps);

    internal->ps = ParaSailsCreate(internal->comm, 
        mat->beg_row, mat->end_row, sym);

    ParaSailsSetupPattern(internal->ps, mat, thresh, nlevels);

    if (logging)
        /* cost = */ ParaSailsStatsPattern(internal->ps, mat);

    internal->ps->loadbal_beta = loadbal;

    ParaSailsSetupValues(internal->ps, mat, filter);

    if (logging)
        ParaSailsStatsValues(internal->ps, mat);

    MatrixDestroy(mat);

    return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsSetupPattern - Set up pattern using a distributed matrix.
 *--------------------------------------------------------------------------*/

int hypre_ParaSailsSetupPattern(hypre_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, int sym, double thresh, int nlevels,
  int logging)
{
    /* double cost; */
    Matrix *mat;
    hypre_ParaSails_struct *internal = (hypre_ParaSails_struct *) obj;

    mat = convert_matrix(internal->comm, distmat);

    ParaSailsDestroy(internal->ps);

    internal->ps = ParaSailsCreate(internal->comm, 
        mat->beg_row, mat->end_row, sym);

    ParaSailsSetupPattern(internal->ps, mat, thresh, nlevels);

    if (logging)
        /* cost = */ ParaSailsStatsPattern(internal->ps, mat);

    MatrixDestroy(mat);

    return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsSetupValues - Set up values using a distributed matrix.
 *--------------------------------------------------------------------------*/

int hypre_ParaSailsSetupValues(hypre_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, double filter, double loadbal,
  int logging)
{
    Matrix *mat;
    hypre_ParaSails_struct *internal = (hypre_ParaSails_struct *) obj;

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
 * hypre_ParaSailsApply - Apply the ParaSails preconditioner to an array 
 * "u", and return the result in the array "v".
 *--------------------------------------------------------------------------*/

int hypre_ParaSailsApply(hypre_ParaSails obj, double *u, double *v)
{
    hypre_ParaSails_struct *internal = (hypre_ParaSails_struct *) obj;

    ParaSailsApply(internal->ps, u, v);

    return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParaSailsApplyTrans - Apply the ParaSails preconditioner, transposed
 * to an array "u", and return the result in the array "v".
 *--------------------------------------------------------------------------*/

int hypre_ParaSailsApplyTrans(hypre_ParaSails obj, double *u, double *v)
{
    hypre_ParaSails_struct *internal = (hypre_ParaSails_struct *) obj;

    ParaSailsApplyTrans(internal->ps, u, v);

    return 0;
}


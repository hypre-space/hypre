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
 * Matrix - Matrix stored and accessible by rows.  Indices and values for
 * the matrix nonzeros are copied into the matrix a row at a time, in any
 * order using the MatrixGetRow function.  The MatrixPutRow function returns
 * a pointer to the indices and values of a row.  The matrix has a set of
 * row and column indices such that these indices begin at "beg" and end 
 * at "end", where 0 <= "beg" <= "end".  In other words, the matrix indices
 * have any nonnegative base value, and the base values of the row and column 
 * indices must agree.
 *
 *****************************************************************************/

#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include "mpi.h"
#include "Matrix.h"

/* prototypes for some static functions used in this file */
static void MatrixMatvecSetup(Matrix *mat);


/*--------------------------------------------------------------------------
 * MATRIX_EXIT - Print message, flush all output streams, return -1 to 
 * operating system, and exit to operating system.  Used internally only.
 *--------------------------------------------------------------------------*/

#define MATRIX_EXIT \
{  printf("Exiting...\n"); \
   fflush(NULL); \
   MPI_Abort(MPI_COMM_WORLD, -1); \
}

/*--------------------------------------------------------------------------
 * MatrixCreate - Return (a pointer to) a matrix object.
 *--------------------------------------------------------------------------*/

Matrix *MatrixCreate(MPI_Comm comm, int beg_row, int end_row)
{
    int num_rows, mype, npes;

    Matrix *mat = (Matrix *) malloc(sizeof(Matrix));

    mat->comm = comm;

    mat->beg_row = beg_row;
    mat->end_row = end_row;

    mat->mem = (Mem *) MemCreate();

    num_rows = mat->end_row - mat->beg_row + 1;

    mat->lens = (int *)     MemAlloc(mat->mem, num_rows * sizeof(int));
    mat->inds = (int **)    MemAlloc(mat->mem, num_rows * sizeof(int *));
    mat->vals = (double **) MemAlloc(mat->mem, num_rows * sizeof(double *));

    /* Send beg_row and end_row to all processors */
    /* This is needed in order to map row numbers to processors */

    MPI_Comm_rank(comm, &mype);
    MPI_Comm_size(comm, &npes);

    mat->beg_rows = (int *) MemAlloc(mat->mem, npes * sizeof(int));
    mat->end_rows = (int *) MemAlloc(mat->mem, npes * sizeof(int));

    MPI_Allgather(&beg_row, 1, MPI_INT, mat->beg_rows, 1, MPI_INT, comm);
    MPI_Allgather(&end_row, 1, MPI_INT, mat->end_rows, 1, MPI_INT, comm);

    mat->matvec_setup = 0;

    return mat;
}

/*--------------------------------------------------------------------------
 * MatrixDestroy - Destroy a matrix object "mat".
 *--------------------------------------------------------------------------*/

void MatrixDestroy(Matrix *mat)
{
    if (mat->matvec_setup)
        MatrixMatvecComplete(mat);

    MemDestroy(mat->mem);
    free(mat);
}

/*--------------------------------------------------------------------------
 * MatrixSetRow - Set a row in a matrix.  Only local rows can be set.
 * Once a row has been set, it should not be set again, or else the 
 * memory used by the existing row will not be recovered until 
 * the matrix is destroyed.
 *--------------------------------------------------------------------------*/

void MatrixSetRow(Matrix *mat, int row, int len, int *ind, double *val)
{
    int local_row = row - mat->beg_row;

    mat->lens[local_row] = len;
    mat->inds[local_row] = (int *) MemAlloc(mat->mem, len*sizeof(int));
    mat->vals[local_row] = (double *) MemAlloc(mat->mem, len*sizeof(double));

    if (ind != NULL)
        memcpy(mat->inds[local_row], ind, len*sizeof(int));

    if (val != NULL)
        memcpy(mat->vals[local_row], val, len*sizeof(double));
}

/*--------------------------------------------------------------------------
 * MatrixSetRow - Get a *local* row in a matrix.  
 *--------------------------------------------------------------------------*/

void MatrixGetRow(Matrix *mat, int row, int *lenp, int **indp, double **valp)
{
    int local_row = row - mat->beg_row;

    *lenp = mat->lens[local_row];
    *indp = mat->inds[local_row];
    *valp = mat->vals[local_row];
}

/*--------------------------------------------------------------------------
 * MatrixRowPe - Map "row" to a processor number.
 *--------------------------------------------------------------------------*/

int MatrixRowPe(Matrix *mat, int row)
{
    int npes, pe;

    int *beg = mat->beg_rows;
    int *end = mat->end_rows;

    MPI_Comm_size(mat->comm, &npes);

    for (pe=0; pe<npes; pe++)
    {
        if (row >= beg[pe] && row <= end[pe])
            return pe;
    }

    printf("MatrixRowPe: could not map row %d.\n", row);
    MATRIX_EXIT;

    return -1; /* for picky compilers */
}

/*--------------------------------------------------------------------------
 * MatrixNnz - Return total number of nonzeros in preconditioner.
 *--------------------------------------------------------------------------*/

int MatrixNnz(Matrix *mat)
{
    int num_local, i, total, alltotal;

    num_local = mat->end_row - mat->beg_row + 1;

    total = 0;
    for (i=0; i<num_local; i++)
	total += mat->lens[i];

    MPI_Allreduce(&total, &alltotal, 1, MPI_INT, MPI_SUM, mat->comm);

    return alltotal;
}

/*--------------------------------------------------------------------------
 * MatrixPrint - Print a matrix to a file "filename".  Each processor 
 * appends to the file in order, but the file is overwritten if it exists.
 *--------------------------------------------------------------------------*/

void MatrixPrint(Matrix *mat, char *filename)
{
    int mype, npes, pe;
    int row, i, len, *ind;
    double *val;

    MPI_Comm_rank(mat->comm, &mype);
    MPI_Comm_size(mat->comm, &npes);

    for (pe=0; pe<npes; pe++)
    {
        FILE *file = fopen(filename, (pe==0 ? "w" : "a"));
        assert(file != NULL);

	MPI_Barrier(mat->comm);

	if (mype == pe)
	{

            for (row=mat->beg_row; row<=mat->end_row; row++)
            {
                MatrixGetRow(mat, row, &len, &ind, &val);

                for (i=0; i<len; i++)
                    fprintf(file, "%d %d %.14e\n", row, ind[i], val[i]);
            }
	}

        fclose(file);
    }
}

/*--------------------------------------------------------------------------
 * MatrixReadMaster - MatrixRead routine for processor 0.  Internal use.
 *--------------------------------------------------------------------------*/

static void MatrixReadMaster(Matrix *mat, char *filename)
{
    MPI_Comm comm = mat->comm;
    int mype, npes;
    FILE *file;
    int ret;
    int num_rows, curr_proc;
    int row, col;
    double value;
    long offset;
    long outbuf;

    int curr_row;
    int len;
    int ind[10000];
    double val[10000];
    int dummy;

    MPI_Request request;
    MPI_Status  status;

    MPI_Comm_size(mat->comm, &npes);
    MPI_Comm_rank(mat->comm, &mype);

    file = fopen(filename, "r");
    assert(file != NULL);

    ret = fscanf(file, "%d %d %d\n", &num_rows, &dummy, &dummy);
    assert(ret == 3);

    offset = ftell(file);
    fscanf(file, "%d %d %lf\n", &row, &col, &value);

    request = MPI_REQUEST_NULL;
    curr_proc = 1; /* proc for which we are looking for the beginning */
    while (curr_proc < npes)
    {
	if (row == mat->beg_rows[curr_proc])
	{
            MPI_Wait(&request, &status);
	    outbuf = offset;
	    MPI_Isend(&outbuf, 1, MPI_LONG, curr_proc, 0, comm, &request);
	    curr_proc++;
	}
        offset = ftell(file);
        fscanf(file, "%d %d %lf\n", &row, &col, &value);
    }

    /* Now read our own part */
    rewind(file);
    fscanf(file, "%d %d %d\n", &num_rows, &dummy, &dummy);

    ret = fscanf(file, "%d %d %lf\n", &row, &col, &value);
    curr_row = row;
    len = 0;

    while (ret != EOF && row <= mat->end_row)
    {
	if (row != curr_row)
	{
	    /* store this row */
	    MatrixSetRow(mat, curr_row, len, ind, val);

	    curr_row = row;

	    /* reset row pointer */
	    len = 0;
	}

	ind[len] = col;
	val[len] = value;
	len++;

        ret = fscanf(file, "%d %d %lf\n", &row, &col, &value);
    }

    /* Store the final row */
    if (ret == EOF || row > mat->end_row)
	MatrixSetRow(mat, mat->end_row, len, ind, val);

    fclose(file);

    MPI_Wait(&request, &status);
}

/*--------------------------------------------------------------------------
 * MatrixReadSlave - MatrixRead routine for other processors.  Internal use.
 *--------------------------------------------------------------------------*/

static void MatrixReadSlave(Matrix *mat, char *filename)
{
    MPI_Comm comm = mat->comm;
    MPI_Status status;
    int mype;
    FILE *file;
    int ret;
    int row, col;
    double value;
    long offset;

    int curr_row;
    int len;
    int ind[10000];
    double val[10000];

    double time0, time1;

    file = fopen(filename, "r");
    assert(file != NULL);

    MPI_Comm_rank(mat->comm, &mype);

    MPI_Recv(&offset, 1, MPI_LONG, 0, 0, comm, &status);
    time0 = MPI_Wtime();

    ret = fseek(file, offset, SEEK_SET);
    assert(ret == 0);

    ret = fscanf(file, "%d %d %lf\n", &row, &col, &value);
    curr_row = row;
    len = 0;

    while (ret != EOF && row <= mat->end_row)
    {
	if (row != curr_row)
	{
	    /* store this row */
	    MatrixSetRow(mat, curr_row, len, ind, val);

	    curr_row = row;

	    /* reset row pointer */
	    len = 0;
	}

	ind[len] = col;
	val[len] = value;
	len++;

        ret = fscanf(file, "%d %d %lf\n", &row, &col, &value);
    }

    /* Store the final row */
    if (ret == EOF || row > mat->end_row)
	MatrixSetRow(mat, mat->end_row, len, ind, val);

    fclose(file);
    time1 = MPI_Wtime();
    printf("%d: Time for slave read: %f\n", mype, time1-time0);
}

/*--------------------------------------------------------------------------
 * MatrixRead - Read a matrix file "filename" from disk and store in the 
 * matrix "mat" which has already been created using MatrixCreate.  The format
 * assumes no nonzero rows, the rows are in order, and there will be at least
 * one row per processor.
 *--------------------------------------------------------------------------*/

void MatrixRead(Matrix *mat, char *filename)
{
    int mype;
    double time0, time1;

    MPI_Comm_rank(mat->comm, &mype);

    time0 = MPI_Wtime();
    if (mype == 0)
	MatrixReadMaster(mat, filename);
    else
	MatrixReadSlave(mat, filename);
    time1 = MPI_Wtime();
    printf("%d: Time for reading matrix: %f\n", mype, time1-time0);
}

/*--------------------------------------------------------------------------
 * RhsRead - Read a right-hand side file "filename" from disk and store in the 
 * location pointed to by "rhs".  "mat" is needed to provide the partitioning
 * information.  The expected format is: a header line (n, nrhs) followed
 * by n values.
 *--------------------------------------------------------------------------*/

void RhsRead(double *rhs, Matrix *mat, char *filename)
{
    FILE *file;
    MPI_Status status;
    int mype, npes;
    int num_rows, dummy, num_local, pe, i;
    double *buffer = NULL;
    int buflen = 0;

    MPI_Comm_size(mat->comm, &npes);
    MPI_Comm_rank(mat->comm, &mype);

    num_local = mat->end_row - mat->beg_row + 1;

    if (mype != 0)
    {
	MPI_Recv(rhs, num_local, MPI_DOUBLE, 0, 0, mat->comm, &status);
	return;
    }

    file = fopen(filename, "r");
    assert(file != NULL);

    i = fscanf(file, "%d %d\n", &num_rows, &dummy);
    assert(i == 2);
    assert(num_rows == mat->end_rows[npes-1]);

    /* Read own rows first */
    for (i=0; i<num_local; i++)
        fscanf(file, "%lf\n", &rhs[i]);

    for (pe=1; pe<npes; pe++)
    {
	num_local = mat->end_rows[pe] - mat->beg_rows[pe]+ 1;

	if (buflen < num_local)
	{
	    free(buffer);
	    buflen = num_local;
            buffer = (double *) malloc(buflen * sizeof(double));
	}

        for (i=0; i<num_local; i++)
            fscanf(file, "%lf\n", &buffer[i]);

	MPI_Send(buffer, num_local, MPI_DOUBLE, pe, 0, mat->comm);
    }

    free(buffer);
}

/*
local_to_global = one simple array = ind (using base 0 for external)
global_to_local = indexed by hash table
*/


/* input: hash table that was created, and buffer to put indices */
/* output: external indices */

static void GetExternalIndices(Matrix *mat, Hash *hash, int *lenp)
{
    int row, i, len, *ind, index, inserted;
    double *val;
    int num_external = 0;
    int num_local = mat->end_row - mat->beg_row + 1;

    for (row=mat->beg_row; row<=mat->end_row; row++)
    {
        MatrixGetRow(mat, row, &len, &ind, &val);

	for (i=0; i<len; i++)
	{
	    /* only interested in external indices */
	    if (ind[i] < mat->beg_row || ind[i] > mat->end_row)
	    {
                index = HashInsert(hash, ind[i], &inserted);

                if (inserted)
		{
                    mat->local_to_global[num_external++] = ind[i];
		}
	    }
	}
    }

    /* sort the indices */
    shell_sort(num_external, mat->local_to_global);

    /* Redo the hash table for the sorted indices */
    HashReset(hash, num_external, mat->local_to_global);
    for (i=0; i<num_external; i++)
    {
        index = HashInsert(hash, mat->local_to_global[i], &inserted);
        mat->global_to_local[index] = i + num_local;
    }

    *lenp = num_external;
}

/* hash table and global_to_local maps global to local */

/* map local indices to global indices */
/* also return map vector for external part */

/* converts in place */

static void ConvertToLocalIndices(Matrix *mat, Hash *hash)
{
    int row, len, *ind, i, index;
    double *val;

    for (row=mat->beg_row; row<=mat->end_row; row++)
    {
        MatrixGetRow(mat, row, &len, &ind, &val);

        for (i=0; i<len; i++)
        {
	    if (ind[i] < mat->beg_row || ind[i] > mat->end_row)
	    {
		index = HashLookup(hash, ind[i]);
		ind[i] = mat->global_to_local[index];
	    }
	    else
	    {
		ind[i] -= mat->beg_row;
	    }
	}
    }
}

static void ConvertToGlobalIndices(Matrix *mat, Hash *hash, int *local_to_global)
{
    int row, i, len, *ind;
    double *val;

    int num_local = mat->end_row - mat->beg_row + 1;

    for (row=mat->beg_row; row<=mat->end_row; row++)
    {
        MatrixGetRow(mat, row, &len, &ind, &val);

        for (i=0; i<len; i++)
        {
	    if (ind[i] >= num_local)
	    {
		ind[i] = mat->local_to_global[ind[i] - num_local];
	    }
	    else
	    {
		ind[i] += mat->beg_row;
	    }
	}
    }
}

static void SetupReceives(Matrix *mat, int reqlen, int *reqind, int *outlist)
{
    int i, j, this_pe, mype;
    MPI_Request request;
    MPI_Comm comm = mat->comm;
    int num_local = mat->end_row - mat->beg_row + 1;

    MPI_Comm_rank(comm, &mype);

    mat->num_recv = 0;

    /* Allocate recvbuf */
    /* recvbuf has numlocal entires saved for local part of x, used in matvec */
    mat->recvlen = reqlen; /* used for the transpose multiply */
    mat->recvbuf = (double *) malloc((reqlen+num_local) * sizeof(double));

    for (i=0; i<reqlen; i=j) /* j is set below */
    {
        /* The processor that owns the row with index reqind[i] */
        this_pe = MatrixRowPe(mat, reqind[i]);

        /* Figure out other rows we need from this_pe */
        for (j=i+1; j<reqlen; j++)
        {
            /* if row is on different pe */
            if (reqind[j] < mat->beg_rows[this_pe] ||
                reqind[j] > mat->end_rows[this_pe])
                   break;
        }

        /* Request rows in reqind[i..j-1] */
        MPI_Isend(&reqind[i], j-i, MPI_INT, this_pe, 444, comm, &request);
        MPI_Request_free(&request);

	/* Count of number of number of indices needed from this_pe */
        outlist[this_pe] = j-i;

        MPI_Recv_init(&mat->recvbuf[i+num_local], j-i, MPI_DOUBLE, this_pe, 555,
	    comm, &mat->recv_req[mat->num_recv]);

        MPI_Send_init(&mat->recvbuf[i+num_local], j-i, MPI_DOUBLE, this_pe, 666,
	    comm, &mat->send_req2[mat->num_recv]);

#ifdef DEBUG
printf("%d: recv_init(%d) to %d at %d of len %d\n", mype, mat->num_recv,
this_pe, i, j-i);
#endif

        mat->num_recv++;
    }
}

/* this func waits for all receives to complete */
static void SetupSends(Matrix *mat, int *inlist)
{
    int i, j, k, mype, npes;
    MPI_Request request;
    MPI_Request *requests;
    MPI_Status  *statuses;
    MPI_Comm comm = mat->comm;

    MPI_Comm_rank(comm, &mype);
    MPI_Comm_size(comm, &npes);

    requests = (MPI_Request *) malloc(npes * sizeof(MPI_Request));
    statuses = (MPI_Status *)  malloc(npes * sizeof(MPI_Status));

    /* Determine size of and allocate sendbuf and sendind */
    mat->sendlen = 0;
    for (i=0; i<npes; i++)
        mat->sendlen += inlist[i];
    mat->sendbuf = (double *) malloc(mat->sendlen * sizeof(double));
    mat->sendind = (int *) malloc(mat->sendlen * sizeof(int));

    j = 0;
    mat->num_send = 0;
    for (i=0; i<npes; i++)
    {
	if (inlist[i] != 0)
	{
	    /* Post receive for the actual indices */
	    MPI_Irecv(&mat->sendind[j], inlist[i], MPI_INT, i, 444, comm, 
                &requests[mat->num_send]);

	    /* Set up the send */
	    MPI_Send_init(&mat->sendbuf[j], inlist[i], MPI_DOUBLE, i, 555, comm,
		&mat->send_req[mat->num_send]);

	    /* Set up the receive for the transpose  */
	    MPI_Recv_init(&mat->sendbuf[j], inlist[i], MPI_DOUBLE, i, 666, comm,
		&mat->recv_req2[mat->num_send]);

#ifdef DEBUG
printf("%d: send_init(%d) to %d at %d of len %d\n", mype, mat->num_send,
i, j, inlist[i]);
#endif

	    mat->num_send++;
	    j += inlist[i];
	}

    }

    MPI_Waitall(mat->num_send, requests, statuses);
    free(requests);
    free(statuses);

    /* convert global indices to local indices */
    /* these are all indices on this processor */
    for (i=0; i<mat->sendlen; i++)
        mat->sendind[i] -= mat->beg_row;
}


static void MatrixMatvecSetup(Matrix *mat)
{
    int mype, npes, len;
    int *outlist, *inlist;
    Hash *hash;

    mat->matvec_setup = 1;

    MPI_Comm_rank(mat->comm, &mype);
    MPI_Comm_size(mat->comm, &npes);

    mat->recv_req = (MPI_Request *) malloc(npes * sizeof(MPI_Request));
    mat->send_req = (MPI_Request *) malloc(npes * sizeof(MPI_Request));
    mat->recv_req2 = (MPI_Request *) malloc(npes * sizeof(MPI_Request));
    mat->send_req2 = (MPI_Request *) malloc(npes * sizeof(MPI_Request));
    mat->statuses = (MPI_Status *)  malloc(npes * sizeof(MPI_Status));

    outlist = (int *) calloc(npes, sizeof(int));
    inlist  = (int *) calloc(npes, sizeof(int));

    mat->global_to_local = (int *) malloc(50021 * sizeof(int));
    mat->local_to_global = (int *) malloc(10000 * sizeof(int));

    mat->hash_numbering = HashCreate(50021);

    GetExternalIndices(mat, mat->hash_numbering, &len);

    ConvertToLocalIndices(mat, mat->hash_numbering);

    SetupReceives(mat, len, mat->local_to_global, outlist);

    MPI_Alltoall(outlist, 1, MPI_INT, inlist, 1, MPI_INT, mat->comm);

    SetupSends(mat, inlist);

    free(outlist);
    free(inlist);
}


void MatrixMatvecComplete(Matrix *mat)
{
    int i;

    mat->matvec_setup = 0;

    /* Change back to local numbering */
    ConvertToGlobalIndices(mat, mat->hash_numbering, mat->local_to_global);
    HashDestroy(mat->hash_numbering);

    for (i=0; i<mat->num_recv; i++)
        MPI_Request_free(&mat->recv_req[i]);

    for (i=0; i<mat->num_send; i++)
        MPI_Request_free(&mat->send_req[i]);

    free(mat->recv_req);
    free(mat->send_req);
    free(mat->statuses);

    free(mat->global_to_local);
    free(mat->local_to_global);

    free(mat->sendind);
    free(mat->sendbuf);
    free(mat->recvbuf);
}



/* could be done in place */

void MatrixMatvec(Matrix *mat, double *x, double *y)
{
    int row, i, len, *ind;
    double *val, temp;
    int num_local = mat->end_row - mat->beg_row + 1;

    /* Change to local numbering and set up persistent communications */
    if (!mat->matvec_setup)
        MatrixMatvecSetup(mat);

    /* Put components of x into the right outgoing buffers */
    for (i=0; i<mat->sendlen; i++)
        mat->sendbuf[i] = x[mat->sendind[i]];

    MPI_Startall(mat->num_recv, mat->recv_req);
    MPI_Startall(mat->num_send, mat->send_req);

    /* Copy local part of x into top part of recvbuf */
    for (i=0; i<num_local; i++)
	mat->recvbuf[i] = x[i];

    MPI_Waitall(mat->num_recv, mat->recv_req, mat->statuses);

    /* do the multiply */
    for (row=mat->beg_row; row<=mat->end_row; row++)
    {
        MatrixGetRow(mat, row, &len, &ind, &val);

	temp = 0.0;
	for (i=0; i<len; i++)
	{
	    temp = temp + val[i] * mat->recvbuf[ind[i]];
	}
	y[row-mat->beg_row] = temp;
    } 

    MPI_Waitall(mat->num_send, mat->send_req, mat->statuses);
}

/* can be done in place */

void MatrixMatvecTrans(Matrix *mat, double *x, double *y)
{
    int row, i, len, *ind;
    double *val, temp;
    int num_local = mat->end_row - mat->beg_row + 1;

    /* Change to local numbering and set up persistent communications */
    if (!mat->matvec_setup)
        MatrixMatvecSetup(mat);

    /* Post receives for local parts of the solution y */
    MPI_Startall(mat->num_send, mat->recv_req2);

    /* initialize accumulator buffer to zero */
    for (i=0; i<mat->recvlen+num_local; i++)
        mat->recvbuf[i] = 0.0;

    /* do the multiply */
    for (row=mat->beg_row; row<=mat->end_row; row++)
    {
        MatrixGetRow(mat, row, &len, &ind, &val);

        for (i=0; i<len; i++)
        {
            mat->recvbuf[ind[i]] += val[i] * x[row-mat->beg_row];
        }
    }

    /* Now can send nonlocal parts of solution to other procs */
    MPI_Startall(mat->num_recv, mat->send_req2);

    /* copy local part of solution into y */
    for (i=0; i<num_local; i++)
        y[i] = mat->recvbuf[i];

    /* alternatively, loop over a wait any */
    MPI_Waitall(mat->num_send, mat->recv_req2, mat->statuses);

    /* add all the incoming partial sums to y */
    for (i=0; i<mat->sendlen; i++)
        y[mat->sendind[i]] += mat->sendbuf[i];

    MPI_Waitall(mat->num_recv, mat->send_req2, mat->statuses);
}

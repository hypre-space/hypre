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
 * ParaSails - Parallel sparse approximate inverse least squares.
 *
 *****************************************************************************/

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "Common.h"
#include "Matrix.h"
#include "Numbering.h"
#include "RowPatt.h"
#include "StoredRows.h"
#include "PrunedRows.h"
#include "OrderStat.h"
#include "LoadBal.h"
#include "ParaSails.h"

#ifdef ESSL
#include <essl.h>
#else
void hypre_F90_NAME(dpotrf)(char *, int *, double *, int *, int *);
void hypre_F90_NAME(dpotrs)(char *, int *, int *, double *, int *, double *, 
  int *, int *);
#endif

double parasails_loadbal_beta = 0.0; /* load balance factor */

/******************************************************************************
 *
 * ParaSails private functions
 *
 *****************************************************************************/

int FindNumReplies(MPI_Comm comm, int *replies_list)
{
    int num_replies;
    int npes, mype;
    int *replies_list2;

    MPI_Comm_rank(comm, &mype);
    MPI_Comm_size(comm, &npes);

    replies_list2 = (int *) malloc(npes * sizeof(int));

    MPI_Allreduce(replies_list, replies_list2, npes, MPI_INT, MPI_SUM, comm);
    num_replies = replies_list2[mype];

    free(replies_list2);

    return num_replies;
}


/*--------------------------------------------------------------------------
 * SendRequests - Given a list of indices "reqind" of length "reqlen",
 * send a sublist to the appropriate processors, thereby requesting 
 * the rows (for example) corresponding to these indices.  The number
 * of requests made is returned in "num_requests".
 *
 * comm   - MPI communicator (input)
 * mat    - matrix used to map row and column numbers to processors (input)
 * reqlen - length of request list (input)
 * reqind - list of indices (input)
 * num_requests - number of requests made (output)
 * replies_list - if non-null, on input this should be a buffer initialized
 *          to zero of size the number of nonzero entries.  On output this
 *          buffer contains a 1 in position i if a request was made to 
 *          processor i.  This array can be used to count (using 
 *          MPI_AllReduce) the number of requests made to the current
 *          processor when the communication pattern is nonsymmetric.
 *
 *--------------------------------------------------------------------------*/

static void SendRequests(MPI_Comm comm, Matrix *mat, int reqlen, int *reqind, 
  int *num_requests, int *replies_list)
{
    MPI_Request request;
    int i, j, this_pe;

    shell_sort(reqlen, reqind);

    *num_requests = 0;

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
        MPI_Isend(&reqind[i], j-i, MPI_INT, this_pe, ROW_REQ_TAG,
            comm, &request);
        MPI_Request_free(&request);
        (*num_requests)++;

        if (replies_list != NULL)
            replies_list[this_pe] = 1;

#ifdef DEBUG
        {
        int mype;
        MPI_Comm_rank(comm, &mype);
        printf("%d: sent request for %d indices to %d\n", mype, j-i, this_pe);
        fflush(NULL);
        }
#endif
    }
}

/*--------------------------------------------------------------------------
 * ReceiveRequest - Receive a request sent with SendRequests by another 
 * processor.  This function should be placed inside a loop which is 
 * executed once for every request that this processor expects to receive.
 * This is the number of requests this processor made in SendRequests
 * in the symmetric case.
 *
 * comm   - MPI communicator (input)
 * source - number of the processor that sent the message (output)
 * buffer - buffer provided by the user.  On output, it contains a 
 *          list of indices.  Buffer will be reallocated if too small
 *          (input/output)
 * buflen - size of the buffer (input).  Size will be updated if buffer 
 *          is too small (input/output)
 * count  - number of indices in the output buffer (output)
 *--------------------------------------------------------------------------*/

static void ReceiveRequest(MPI_Comm comm, int *source, int **buffer, 
  int *buflen, int *count)
{
    MPI_Status status;

    MPI_Probe(MPI_ANY_SOURCE, ROW_REQ_TAG, comm, &status);
    *source = status.MPI_SOURCE;
    MPI_Get_count(&status, MPI_INT, count);

    if (*count > *buflen)
    {
	free(*buffer);
	*buflen = *count;
	*buffer = (int *) malloc(*buflen * sizeof(int));
    }

    MPI_Recv(*buffer, *count, MPI_INT, *source, ROW_REQ_TAG, comm, &status);

#ifdef DEBUG
    {
    int mype;
    MPI_Comm_rank(comm, &mype);
    printf("%d: received req for %d indices from %d\n", mype, *count, *source);
    fflush(NULL);
    }
#endif
}

/*--------------------------------------------------------------------------
 * SendReplyPrunedRows - Send a reply of pruned rows for each request 
 * received by this processor using ReceiveRequest.
 *
 * comm    - MPI communicator (input)
 * dest    - pe to send to (input)
 * buffer  - list of indices (input)
 * count   - number of indices in buffer (input)
 * pruned_rows - the pruned_rows object where the pruned rows reside (input)
 * mem     - pointer to memory object used for reply buffers (input)
 * request - request handle of send (output)
 * 
 * The function allocates space for each send buffer using "mem", and the
 * caller must free this space when all the sends have completed..
 *
 * The reply has the following structure for the integer data in indbuf:
 * num_rows, index_1, ..., index_n, len_1, row_1_indices, len_2, indices, ...
 *--------------------------------------------------------------------------*/

static void SendReplyPrunedRows(MPI_Comm comm, Numbering *numb,
  int dest, int *buffer, int count,
  PrunedRows *pruned_rows, Mem *mem, MPI_Request *request)
{
    MPI_Status status;
    int sendbacksize, j;
    int len, *ind, *indbuf, *indbufp;
    int temp;

    /* Determine the size of the integer message we need to send back */
    sendbacksize = count+1; /* length of header part */
    for (j=0; j<count; j++)
    {
	NumberingGlobalToLocal(numb, 1, &buffer[j], &temp);
        PrunedRowsGet(pruned_rows, temp, &len, &ind);
        sendbacksize += (len+1);  /* add one for the row length */
    }

    /* Reply buffer - will be freed by caller */
    indbuf = (int *) MemAlloc(mem, sendbacksize * sizeof(int));

    /* Pointer used to construct reply message */
    indbufp = indbuf;

    /* Construct integer reply message in local buffer, with this format:
       number of rows to send, row numbers, indices of each row */

    *indbufp++ = count; /* number of rows to send */

    for (j=0; j<count; j++)
        *indbufp++ = buffer[j]; /* row numbers */

    for (j=0; j<count; j++)
    {
	NumberingGlobalToLocal(numb, 1, &buffer[j], &temp);
        PrunedRowsGet(pruned_rows, temp, &len, &ind);

        *indbufp++ = len;
        /* memcpy(indbufp, ind, sizeof(int)*len); */
 	NumberingLocalToGlobal(numb, len, ind, indbufp);
        indbufp += len;
    }

    MPI_Isend(indbuf, indbufp-indbuf, MPI_INT, dest, ROW_REPI_TAG,
        comm, request);
}

/*--------------------------------------------------------------------------
 * ReceiveReplyPrunedRows - Receive a reply sent by SendReplyPrunedRows
 *
 * comm    - MPI communicator (input)
 * pruned_rows - the pruned_rows object where the rows should be stored
 * patt    - each pruned row is merged into patt before returning (input).
 *           Only the external indices of the pattern is merged
 * mat     - Matrix argument used for determining the external indices
 *--------------------------------------------------------------------------*/

static void ReceiveReplyPrunedRows(MPI_Comm comm, Numbering *numb,
  PrunedRows *pruned_rows, 
  RowPatt *patt, Matrix *mat)
{
    MPI_Status status;
    int source, count;
    int len, *ind, num_rows, *row_nums, j;

    /* Don't know the size of reply, so use probe and get count */
    MPI_Probe(MPI_ANY_SOURCE, ROW_REPI_TAG, comm, &status);
    source = status.MPI_SOURCE;
    MPI_Get_count(&status, MPI_INT, &count);

    /* Allocate space in stored rows data structure */
    ind = PrunedRowsAlloc(pruned_rows, count);
    MPI_Recv(ind, count, MPI_INT, source, ROW_REPI_TAG, comm, &status);

    /* Parse the message */
    num_rows = *ind++; /* number of rows */
    row_nums = ind;    /* row numbers */
    ind += num_rows;

    /* Convert global row numbers to local row numbers */
    NumberingGlobalToLocal(numb, num_rows, row_nums, row_nums);

    /* Set the pointers to the individual rows */
    for (j=0; j<num_rows; j++)
    {
        len = *ind++;
	NumberingGlobalToLocal(numb, len, ind, ind);
        PrunedRowsPut(pruned_rows, row_nums[j], len, ind);
        RowPattMergeExt(patt, len, ind, numb->num_loc);
        ind += len;
    }
}

/*--------------------------------------------------------------------------
 * SendReplyStoredRows - Send a reply of stored rows for each request 
 * received by this processor using ReceiveRequest.
 *
 * comm    - MPI communicator (input)
 * dest    - pe to send to (input)
 * buffer  - list of indices (input)
 * count   - number of indices in buffer (input)
 * stored_rows - the stored_rows object where the rows reside (input)
 * mem     - pointer to memory object used for reply buffers (input)
 * request - request handle of send (output)
 * 
 * The function allocates space for each send buffer using "mem", and the
 * caller must free this space when all the sends have completed..
 *
 * The reply has the following structure for the integer data in indbuf:
 * num_rows, index_1, ..., index_n, len_1, row_1_indices, len_2, indices, ...
 *
 * The reply has the following structure for the value data:
 * row_1_values, row_2_values, ...
 *--------------------------------------------------------------------------*/

static void SendReplyStoredRows(MPI_Comm comm, Numbering *numb,
  int dest, int *buffer, int count,
  StoredRows *stored_rows, Mem *mem, MPI_Request *request)
{
    MPI_Status status;
    int sendbacksize, j;
    int len, *ind, *indbuf, *indbufp;
    double *val, *valbuf, *valbufp;
    int temp;

    /* Determine the size of the integer message we need to send back */
    sendbacksize = count+1; /* length of header part */
    for (j=0; j<count; j++)
    {
	NumberingGlobalToLocal(numb, 1, &buffer[j], &temp);
        StoredRowsGet(stored_rows, temp, &len, &ind, &val);
        sendbacksize += (len+1);  /* add one for the row length */
    }

    /* Reply buffers - will be freed by caller */
    indbuf = (int *)    MemAlloc(mem, sendbacksize * sizeof(int));
    valbuf = (double *) MemAlloc(mem, sendbacksize * sizeof(double));

    /* Pointers used to construct reply messages */
    indbufp = indbuf;
    valbufp = valbuf;

    /* Construct integer reply message in local buffer, with this format:
       number of rows to send, row numbers, len of row, indices each row,
       len of next row, indices of row, etc. */

    *indbufp++ = count; /* number of rows to send */

    for (j=0; j<count; j++)
        *indbufp++ = buffer[j]; /* row numbers */

    for (j=0; j<count; j++)
    {
	NumberingGlobalToLocal(numb, 1, &buffer[j], &temp);
        StoredRowsGet(stored_rows, temp, &len, &ind, &val);

        *indbufp++ = len;
        /* memcpy(indbufp, ind, sizeof(int)*len); */
	NumberingLocalToGlobal(numb, len, ind, indbufp);
        memcpy(valbufp, val, sizeof(double)*len);
        indbufp += len;
        valbufp += len;
    }

    MPI_Isend(indbuf, indbufp-indbuf, MPI_INT, dest, ROW_REPI_TAG,
        comm, request);

    MPI_Request_free(request);

    MPI_Isend(valbuf, valbufp-valbuf, MPI_DOUBLE, dest, ROW_REPV_TAG,
        comm, request);
}

/*--------------------------------------------------------------------------
 * ReceiveReplyStoredRows - Receive a reply sent by SendReplyStoredRows
 *
 * comm    - MPI communicator (input)
 * stored_rows - the stored_rows object where the rows should be stored
 *--------------------------------------------------------------------------*/

static void ReceiveReplyStoredRows(MPI_Comm comm, Numbering *numb,
  StoredRows *stored_rows)
{
    MPI_Status status;
    int source, count;
    int len, *ind, num_rows, *row_nums, j;
    double *val;

    /* Don't know the size of reply, so use probe and get count */
    MPI_Probe(MPI_ANY_SOURCE, ROW_REPI_TAG, comm, &status);
    source = status.MPI_SOURCE;
    MPI_Get_count(&status, MPI_INT, &count);

    /* Allocate space in stored rows data structure */
    ind = StoredRowsAllocInd(stored_rows, count);
    MPI_Recv(ind, count, MPI_INT, source, ROW_REPI_TAG, comm, &status);
    val = StoredRowsAllocVal(stored_rows, count);
    MPI_Recv(val, count, MPI_DOUBLE, source, ROW_REPV_TAG, comm, &status);

    /* Parse the message */
    num_rows = *ind++; /* number of rows */
    row_nums = ind;    /* row numbers */
    ind += num_rows;

    /* Convert global row numbers to local row numbers */
    NumberingGlobalToLocal(numb, num_rows, row_nums, row_nums);

    /* Set the pointers to the individual rows */
    for (j=0; j<num_rows; j++)
    {
        len = *ind++;
	NumberingGlobalToLocal(numb, len, ind, ind);
        StoredRowsPut(stored_rows, row_nums[j], len, ind, val);
        ind += len;
        val += len;
    }
}

/*--------------------------------------------------------------------------
 * ExchangePrunedRows
 *--------------------------------------------------------------------------*/

static void ExchangePrunedRows(MPI_Comm comm, Matrix *M, Numbering *numb,
  PrunedRows *pruned_rows, int num_levels)
{
    RowPatt *patt;
    int *ind_global;
    int row, len, *ind;

    int num_requests;
    int source;

    int bufferlen;
    int *buffer;

    int level;

    int i;
    int count;
    MPI_Request *requests;
    MPI_Status *statuses;
    int npes;
    int num_replies, *replies_list;

    Mem *mem;

    MPI_Comm_size(comm, &npes);
    requests = (MPI_Request *) malloc(npes * sizeof(MPI_Request));
    statuses = (MPI_Status *) malloc(npes * sizeof(MPI_Status));

    /* Merged pattern of pruned rows on this processor */

    patt = RowPattCreate(ROWPATT_MAXLEN);
    ind_global = (int *) malloc(ROWPATT_MAXLEN * sizeof(int));

    for (row=0; row<=M->end_row - M->beg_row; row++)
    {
        PrunedRowsGet(pruned_rows, row, &len, &ind);
        RowPattMergeExt(patt, len, ind, numb->num_loc);
    }

    /* Loop to construct pattern of pruned rows on this processor */

    bufferlen = 10; /* size will grow if get a long msg */
    buffer = (int *) malloc(bufferlen * sizeof(int));

    for (level=1; level<=num_levels; level++)
    {
        mem = (Mem *) MemCreate();

        /* Get list of indices that were just merged */
        RowPattPrevLevel(patt, &len, &ind);

	/* Convert local row numbers to global row numbers */
        NumberingLocalToGlobal(numb, len, ind, ind_global);

        replies_list = (int *) calloc(npes, sizeof(int));

        SendRequests(comm, M, len, ind_global, &num_requests, replies_list);

        num_replies = FindNumReplies(comm, replies_list);
        free(replies_list);

        for (i=0; i<num_replies; i++)
        {
            /* Receive count indices stored in buffer */
            ReceiveRequest(comm, &source, &buffer, &bufferlen, &count);

            SendReplyPrunedRows(comm, numb, source, buffer, count,
                pruned_rows, mem, &requests[i]);
        }

        for (i=0; i<num_requests; i++)
        {
	    /* Will also merge the pattern of received rows into "patt" */
            ReceiveReplyPrunedRows(comm, numb, pruned_rows, patt, M);
        }

        MPI_Waitall(num_replies, requests, statuses);
        MemDestroy(mem);
    }

    RowPattDestroy(patt);
    free(ind_global);
    free(buffer);
    free(requests);
    free(statuses);
}

/*--------------------------------------------------------------------------
 * ExchangeStoredRows
 *--------------------------------------------------------------------------*/

static void ExchangeStoredRows(MPI_Comm comm, Matrix *A, Matrix *M, 
  Numbering *numb, StoredRows *stored_rows, LoadBal *load_bal)
{
    RowPatt *patt;
    int *ind_global;
    int row, len, *ind;
    double *val;

    int num_requests;
    int source;

    int bufferlen;
    int *buffer;

    int i;
    int count;
    MPI_Request *requests;
    MPI_Status *statuses;
    int npes;
    int num_replies, *replies_list;

    Mem *mem = (Mem *) MemCreate();

    MPI_Comm_size(comm, &npes);

    /* Merge the patterns of all the rows of M on this processor */
    /* The merged pattern is not already known, since M is triangular */

    patt = RowPattCreate(ROWPATT_MAXLEN);
    ind_global = (int *) malloc(ROWPATT_MAXLEN * sizeof(int));

    for (row=load_bal->beg_row; row<=M->end_row; row++)
    {
        MatrixGetRow(M, row - M->beg_row, &len, &ind, &val);
        RowPattMergeExt(patt, len, ind, numb->num_loc);
    }

    /* Merge patterns for load balancing recipient rows */

    for (i=0; i<load_bal->num_taken; i++)
    {
      for (row=0; row <= load_bal->recip_data[i].mat->end_row - 
                         load_bal->recip_data[i].mat->beg_row; row++)
      {
        MatrixGetRow(load_bal->recip_data[i].mat, row, &len, &ind, &val);
        RowPattMergeExt(patt, len, ind, numb->num_loc);
      }
    }

    RowPattGet(patt, &len, &ind);

    /* Convert local row numbers to global row numbers */
    NumberingLocalToGlobal(numb, len, ind, ind_global);

    replies_list = (int *) calloc(npes, sizeof(int));

    SendRequests(comm, A, len, ind_global, &num_requests, replies_list);

    num_replies = FindNumReplies(comm, replies_list);
    free(replies_list);

    requests = (MPI_Request *) malloc(num_replies * sizeof(MPI_Request));
    statuses = (MPI_Status *) malloc(num_replies * sizeof(MPI_Status));

    bufferlen = 10; /* size will grow if get a long msg */
    buffer = (int *) malloc(bufferlen * sizeof(int));

    for (i=0; i<num_replies; i++)
    {
	/* Receive count indices stored in buffer */
        ReceiveRequest(comm, &source, &buffer, &bufferlen, &count);

	SendReplyStoredRows(comm, numb, source, buffer, count, 
            stored_rows, mem, &requests[i]);
    }

    for (i=0; i<num_requests; i++)
    {
	ReceiveReplyStoredRows(comm, numb, stored_rows);
    }

    MPI_Waitall(num_replies, requests, statuses);

    /* Free all send buffers */
    MemDestroy(mem);

    RowPattDestroy(patt);
    free(buffer);
    free(requests);
    free(statuses);
}

/*--------------------------------------------------------------------------
 * ConstructPatternForEachRow
 *
 * pruned_rows - pruned rows, used for constructing row patterns (input) 
 * num_levels  - number of levels in pattern (input)
 * M           - matrix where the row patterns will be stored (input/output).
 *               This is the approximate inverse with lower triangular pattern
 *--------------------------------------------------------------------------*/

static void ConstructPatternForEachRow(int symmetric, PrunedRows *pruned_rows,
  int num_levels, Numbering *numb, Matrix *M, double *costp)
{
    int row, len, *ind, level, lenprev, *indprev;
    int i, j;
    RowPatt *row_patt;
    int nnz = 0;
    int npes, pe;

    MPI_Comm_size(M->comm, &npes);
    *costp = 0.0;

    row_patt = RowPattCreate(ROWPATT_MAXLEN);

    for (row=0; row<=M->end_row - M->beg_row; row++)
    {
        /* Get initial pattern for row */
        PrunedRowsGet(pruned_rows, row, &len, &ind);
        RowPattMerge(row_patt, len, ind);

        /* Loop */
        for (level=1; level<=num_levels; level++)
        {
            /* Get the indices that were just added */
            RowPattPrevLevel(row_patt, &lenprev, &indprev);

            for (i=0; i<lenprev; i++)
            {
                PrunedRowsGet(pruned_rows, indprev[i], &len, &ind);
                RowPattMerge(row_patt, len, ind);
            }
        }

        RowPattGet(row_patt, &len, &ind);

	/* do reset here, because now we mess with ind array */
        RowPattReset(row_patt);

	if (symmetric)
	{
            /* Store the lower triangular part of row pattern into the matrix */
            j = 0;
            for (i=0; i<len; i++)
            {
                if (numb->local_to_global[ind[i]] <= numb->local_to_global[row])
                    ind[j++] = ind[i];
            }
            len = j;
	}

        /* Store structure of row in matrix M */
        /* Following statement allocates space but does not store values */
        MatrixSetRow(M, row, len, ind, NULL);

        nnz += len;
        (*costp) += (double) len*len*len;
    }

#ifdef PARASAILS_TIME
    {
    int mype;
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    printf("%d: nnz: %10d  ********* cost %7.1e\n", mype, nnz, *costp);
    fflush(NULL);
    }
#endif

    RowPattDestroy(row_patt);
}

/*--------------------------------------------------------------------------
 * ComputeValues
 *--------------------------------------------------------------------------*/

static void ComputeValuesSym(StoredRows *stored_rows, Matrix *mat,
  int local_beg_row, Numbering *numb)
{
    int *marker;
    int row, maxlen, len, *ind;
    double *val;

    double *ahat, *ahatp;
    int i, j, len2, *ind2, loc;
    double *val2, temp;
    double time0, time1, timet = 0.0, timea = 0.0;

    double ahatcost = 0.0;

#ifndef ESSL
    char uplo = 'L';
    int one = 1;
    int info;
#endif

    /* Allocate and initialize full length marker array */
    marker = (int *) malloc(numb->num_ind * sizeof(int));
    for (i=0; i<numb->num_ind; i++)
	marker[i] = -1;

    /* Determine the length of the longest row of M on this processor */
    /* This determines the maximum storage required for the ahat matrix */
    maxlen = 0;
    for (row=local_beg_row; row<=mat->end_row; row++)
    {
        MatrixGetRow(mat, row - mat->beg_row, &len, &ind, &val);
        maxlen = (len > maxlen ? len : maxlen);
    }

#ifdef ESSL
    ahat = (double *) malloc(maxlen*(maxlen+1)/2 * sizeof(double));
#else
    ahat = (double *) malloc(maxlen*maxlen * sizeof(double));
#endif

    /* Compute values for row "row" of approximate inverse */
    for (row=local_beg_row; row<=mat->end_row; row++)
    {
	/* Retrieve local indices */
        MatrixGetRow(mat, row - mat->beg_row, &len, &ind, &val);

	/* Fill marker array in locations of local indices */
	for (i=0; i<len; i++)
	    marker[ind[i]] = i;

        /* Initialize ahat to zero */
#ifdef ESSL
        bzero((char *) ahat, len*(len+1)/2 * sizeof(double));
#else
        bzero((char *) ahat, len*len * sizeof(double));
#endif

        time0 = MPI_Wtime();

        /* Form ahat matrix, entries correspond to indices in "ind" only */
        ahatp = ahat;
        for (i=0; i<len; i++)
        {
            StoredRowsGet(stored_rows, ind[i], &len2, &ind2, &val2);
	    assert(len2 > 0);

#ifdef ESSL
            for (j=0; j<len2; j++)
            {
                loc = marker[ind2[j]];

                if (loc != -1) /* redundant */
                    if (loc >= i)
                        ahatp[loc - i] = val2[j];
            }

            ahatp += (len-i);
#else
            for (j=0; j<len2; j++)
            {
                loc = marker[ind2[j]];

                if (loc != -1)
                    ahatp[loc] = val2[j];
            }

            ahatp += len;
#endif
        }

        time1 = MPI_Wtime();
        timea += (time1-time0);
	ahatcost += (double) (len*len2);

        /* Set the right-hand side */
        bzero((char *) val, len*sizeof(double));
	NumberingGlobalToLocal(numb, 1, &row, &loc);
	loc = marker[loc];
	assert(loc != -1);
	val[loc] = 1.0;

	/* Reset marker array */
	for (i=0; i<len; i++)
	    marker[ind[i]] = -1;

        time0 = MPI_Wtime();

#ifdef ESSL
        dppf(ahat, len, 1);
        dpps(ahat, len, val, 1);
#else
        /* Solve local linear system - factor phase */
        hypre_F90_NAME(dpotrf)(&uplo, &len, ahat, &len, &info);
        if (info != 0)
        {
            printf("ParaSails: row %d, dpotrf returned %d.\n", row, info);
            printf("ParaSails: len %d, ahat: %f %f %f %f\n", row,
                ahat[0], ahat[1], ahat[2], ahat[3]);
            fflush(NULL);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        /* Solve local linear system - solve phase */
        hypre_F90_NAME(dpotrs)(&uplo, &len, &one, ahat, &len, val, &len, &info);
        if (info != 0)
        {
            printf("ParaSails: row %d, dpotrs returned %d.\n", row, info);
            printf("ParaSails: len %d, ahat: %f %f %f %f\n", row,
                ahat[0], ahat[1], ahat[2], ahat[3]);
            fflush(NULL);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
#endif
        time1 = MPI_Wtime();
        timet += (time1-time0);

        /* Scale the result */
        temp = 1.0 / sqrt(ABS(val[loc]));
        for (i=0; i<len; i++)
            val[i] = val[i] * temp;
    }

    free(marker);
    free(ahat);

#ifdef PARASAILS_TIME
    {
    int mype;
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    printf("%d: Time for ahat: %f, for local solves: %f\n", mype, timea, timet);
    printf("%d: ahatcost: %7.1e, numrows: %d, maxlen: %d\n", 
        mype, ahatcost, mat->end_row-local_beg_row+1, maxlen);
    fflush(NULL);
    }
#endif
}

/* Numbering may need additional indices .... */
/* all stored rows have been converted to local indexing by the time get here */
/* num_loc is the highest local index number */

static void ComputeValuesNonsym(StoredRows *stored_rows, Matrix *mat,
  int local_beg_row, Numbering *numb)
{
    int *marker;
    double *ahat, *ahatp, *bhat;
    double *work;
    int ahat_size = 10000, bhat_size = 1000, work_size = 400*64;

    int row, maxlen, len, *ind;
    double *val;

    int i, j, len2, *ind2, loc;
    double *val2, temp;
    double time0, time1, timet = 0.0, timea = 0.0;

    int npat;
    /* int *patt; */
    int patt[100000];

    int info;

#ifndef ESSL
    char trans = 'N';
    int one = 1;
#endif

    /* Allocate and initialize marker array */
    marker = (int *) malloc(numb->num_ind * sizeof(int));
    for (i=0; i<numb->num_ind; i++)
	marker[i] = -1;

    bhat = (double *) malloc(bhat_size * sizeof(double));
    ahat = (double *) malloc(ahat_size * sizeof(double));
    work = (double *) calloc(work_size,  sizeof(double));

    /* Compute values for row "row" of approximate inverse */
    for (row=local_beg_row; row<=mat->end_row; row++)
    {
        time0 = MPI_Wtime();

	/* Retrieve local indices */
        MatrixGetRow(mat, row - mat->beg_row, &len, &ind, &val);

	npat = 0;

	/* Fill marker array */
        for (i=0; i<len; i++)
        {
            StoredRowsGet(stored_rows, ind[i], &len2, &ind2, &val2);
	    assert(len2 > 0);

            for (j=0; j<len2; j++)
            {
                loc = marker[ind2[j]];

                if (loc == -1)
		{
		    marker[ind2[j]] = npat;
		    patt[npat++] = ind2[j];
		}
            }
        }

	if (len*npat > ahat_size)
	{
	    free(ahat);
	    ahat_size = len*npat;
            ahat = (double *) malloc(ahat_size * sizeof(double));
	}

        /* Initialize ahat to zero */
        bzero((char *) ahat, len*npat * sizeof(double));

        /* Form ahat matrix, entries correspond to indices in "ind" only */
        ahatp = ahat;
        for (i=0; i<len; i++)
        {
            StoredRowsGet(stored_rows, ind[i], &len2, &ind2, &val2);

            for (j=0; j<len2; j++)
            {
                loc = marker[ind2[j]];
                ahatp[loc] = val2[j];
            }
            ahatp += npat;
        }

        time1 = MPI_Wtime();
        timea += (time1-time0);

	/* Reallocate bhat if necessary */
	if (npat > bhat_size)
	{
	    free(bhat);
	    bhat_size = npat;
            bhat = (double *) malloc(bhat_size * sizeof(double));
	}

        /* Set the right-hand side, bhat */
        bzero((char *) bhat, npat*sizeof(double));
	NumberingGlobalToLocal(numb, 1, &row, &loc);
        loc = marker[loc];
        assert(loc != -1);
        bhat[loc] = 1.0;

	/* Reset marker array */
	for (i=0; i<npat; i++)
	    marker[patt[i]] = -1;

        time0 = MPI_Wtime();

#ifdef ESSL
	/* rhs in bhat, and put solution in val */
        dgells(0, ahat, npat, bhat, npat, val, len, NULL, 1.e-12, npat, len, 1,
            &info, work, work_size);
#else
	/* rhs in bhat, and put solution in bhat */
        hypre_F90_NAME(dgels)(&trans, &npat, &len, &one, ahat, &npat,
            bhat, &npat, work, &work_size, &info);

        if (info != 0)
        {
            printf("ParaSails: row %d, dgels returned %d.\n", row, info);
            printf("ParaSails: len %d, ahat: %f %f %f %f\n", row,
                ahat[0], ahat[1], ahat[2], ahat[3]);
            fflush(NULL);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

	/* Copy result into row */
        for (j=0; j<len; j++)
            val[j] = bhat[j];
#endif
        time1 = MPI_Wtime();
        timet += (time1-time0);
    }

    free(marker);
    free(bhat);
    free(ahat);
    free(work);

#ifdef PARASAILS_TIME
    {
    int mype;
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    printf("%d: Time for ahat: %f, for local solves: %f\n", mype, timea, timet);
    fflush(NULL);
    }
#endif
}

static void Rescale(Numbering *numb, StoredRows *stored_rows, Matrix *M)
{
    int len, *ind, len2, *ind2;
    double *val, *val2, *w;
    int row, j, i;
    double accum, prod;

    /* Allocate full-length workspace */
    w = (double *) calloc(numb->num_ind, sizeof(double));

    /* Loop over rows */
    for (row=0; row<=M->end_row - M->beg_row; row++)
    {
        MatrixGetRow(M, row, &len, &ind, &val);

	accum = 0.0;

	/* Loop over nonzeros in row */
        for (j=0; j<len; j++)
        {
	    /* Get the row of A corresponding to current nonzero */
            StoredRowsGet(stored_rows, ind[j], &len2, &ind2, &val2);

	    /* Scatter nonzeros of A */
	    for (i=0; i<len2; i++)
	    {
		assert(ind2[i] < numb->num_ind);
		w[ind2[i]] = val2[i];
	    }

	    /* Form inner product of current row with this row */
	    prod = 0.0;
	    for (i=0; i<len; i++)
	    {
		assert(ind[i] < numb->num_ind);
		prod += val[i] * w[ind[i]];
	    }

	    accum += val[j] * prod;

	    /* Reset workspace */
	    for (i=0; i<len2; i++)
		w[ind2[i]] = 0.0;
        }

	/* Scale the row */
	accum = 1./sqrt(accum);
        for (j=0; j<len; j++)
	    val[j] *= accum;
    }
}

/******************************************************************************
 *
 * ParaSails public functions
 *
 * After creating a ParaSails object, the preconditioner requires two set up
 * steps:  one for the pattern, and one for the numerical values.  Once the
 * pattern has been set up, the numerical values can be set up for different
 * matrices, i.e., ParaSailsSetupValues can be called again with a different
 * matrix, and used in another iterative solve.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * ParaSailsCreate - Return (a pointer to) a ParaSails preconditioner object.
 * The input matrix "A" is the matrix that will be used for setting up the 
 * sparsity pattern of the preconditioner.
 *--------------------------------------------------------------------------*/

ParaSails *ParaSailsCreate(Matrix *A)
{
    ParaSails *ps = (ParaSails *) malloc(sizeof(ParaSails));

    ps->symmetric = 0;

    ps->A = A;

    ps->M = NULL;

    ps->numb = A->numb;

    ps->pruned_rows = NULL;

    ps->stored_rows = NULL;

    ps->diag_scale = DiagScaleCreate(A);

    return ps;
}

/*--------------------------------------------------------------------------
 * ParaSailsDestroy - Destroy a ParaSails object "ps".
 *--------------------------------------------------------------------------*/

void ParaSailsDestroy(ParaSails *ps)
{
    if (ps->pruned_rows)
        PrunedRowsDestroy(ps->pruned_rows);

    if (ps->stored_rows)
        StoredRowsDestroy(ps->stored_rows);

    DiagScaleDestroy(ps->diag_scale);

    MatrixDestroy(ps->M);

    free(ps);
}

/*--------------------------------------------------------------------------
 * ParaSailsSetSym -
 * Must be the first call after ParaSailsCreate
 * Nonzero indicates symmetric.
 *--------------------------------------------------------------------------*/

void ParaSailsSetSym(ParaSails *ps, int sym)
{
    ps->symmetric = sym;
}

/*--------------------------------------------------------------------------
 * ParaSailsSetupPattern - Set up a pattern for the ParaSails preconditioner.
 * The pattern is defined by the matrix used in the Create call, and by the
 * input parameters "thresh" and "num_levels".
 *--------------------------------------------------------------------------*/

void ParaSailsSetupPattern(ParaSails *ps, double thresh, int num_levels)
{
    int mype;
    double cost;

    MPI_Comm_rank(ps->A->comm, &mype);

    ps->thresh     = thresh;
    ps->num_levels = num_levels;

#ifdef PARASAILS_TIME
    if (mype == 0)
       printf("ParaSails: thresh %e, level %d\n", thresh, num_levels);
#endif

    if (ps->M)
        MatrixDestroy(ps->M);

    ps->M = MatrixCreate(ps->A->comm, ps->A->beg_row, ps->A->end_row);

    if (ps->pruned_rows)
        PrunedRowsDestroy(ps->pruned_rows);

    ps->pruned_rows = PrunedRowsCreate(ps->A, 100000,
        ps->diag_scale, ps->thresh);

    ExchangePrunedRows(ps->A->comm, ps->A, ps->numb, ps->pruned_rows, 
        ps->num_levels);

    /* set structure in approx inverse */
    ConstructPatternForEachRow(ps->symmetric, ps->pruned_rows, ps->num_levels, 
        ps->numb, ps->M, &cost);

    ps->load_bal = LoadBalDonate(ps->A->comm, ps->M, ps->numb, cost, 
        parasails_loadbal_beta);
}

/*--------------------------------------------------------------------------
 * ParaSailsSetupValues - Compute the numerical values of the ParaSails
 * preconditioner, for the pattern set up using ParaSailsSetupPattern.
 * This function may be called repeatedly with different input matrices
 * "A", for which a preconditioner is constructed.  (A call with the same
 * matrix as a previous call is not treated any differently.)
 *--------------------------------------------------------------------------*/

void ParaSailsSetupValues(ParaSails *ps, Matrix *A)
{
    int mype;
    int i;
    double time0, time1;

    MPI_Comm_rank(A->comm, &mype);

    if (ps->stored_rows)
        StoredRowsDestroy(ps->stored_rows);

    ps->stored_rows = StoredRowsCreate(A, 100000);

    ExchangeStoredRows(ps->A->comm, A, ps->M, ps->numb,
        ps->stored_rows, ps->load_bal);

    time0 = MPI_Wtime();

    if (ps->symmetric)
    {
        ComputeValuesSym(ps->stored_rows, ps->M, ps->load_bal->beg_row,
            ps->numb);

        for (i=0; i<ps->load_bal->num_taken; i++)
        {
            ComputeValuesSym(ps->stored_rows, ps->load_bal->recip_data[i].mat,
	        ps->load_bal->recip_data[i].mat->beg_row, ps->numb);
        }
    }
    else
    {
        ComputeValuesNonsym(ps->stored_rows, ps->M, ps->load_bal->beg_row,
            ps->numb);

        for (i=0; i<ps->load_bal->num_taken; i++)
        {
            ComputeValuesNonsym(ps->stored_rows, 
		ps->load_bal->recip_data[i].mat,
	        ps->load_bal->recip_data[i].mat->beg_row, ps->numb);
        }
    }

    time1 = MPI_Wtime();

#ifdef PARASAILS_TIME
    printf("%d: Total Time for computing values: %f\n", mype, time1-time0);
#endif

    /* load bal affects code in ExchangeStoredRows and ComputeValues */
    /* note that this will effectively synchronize the processors */

    LoadBalReturn(ps->load_bal, ps->A->comm, ps->M);
}

/*--------------------------------------------------------------------------
 * ParaSailsSelectThresh - select a threshold for the preconditioner pattern.  
 * The threshold attempts to be chosen such that approximately (1-param) of 
 * all the matrix elements is larger than this threshold.  This is accomplished
 * by finding the element in each row that is smaller than (1-param) of the 
 * elements in that row, and averaging these elements over all rows.  The 
 * threshold is selected on the diagonally scaled matrix.
 *--------------------------------------------------------------------------*/

double ParaSailsSelectThresh(ParaSails *ps, double param)
{
    int row, len, *ind, i, npes;
    double *val;
    double localsum = 0.0, sum;
    double temp;

    MPI_Comm comm = ps->A->comm;

    /* Buffer for storing the values in each row when computing the 
       i-th smallest element - buffer will grow if necessary */
    double *buffer;
    int buflen = 10;
    buffer = (double *) malloc(buflen * sizeof(double));

    for (row=0; row<=ps->A->end_row - ps->A->beg_row; row++)
    {
        MatrixGetRow(ps->A, row, &len, &ind, &val);

	if (len > buflen)
	{
	    free(buffer);
	    buflen = len;
            buffer = (double *) malloc(buflen * sizeof(double));
	}

	/* Copy the scaled absolute values into a work buffer */
        temp = DiagScaleGet(ps->diag_scale, row);
	for (i=0; i<len; i++)
	{
	    buffer[i] = temp*ABS(val[i])*DiagScaleGet(ps->diag_scale, ind[i]);
	    if (ind[i] == row)
		buffer[i] = 0.0; /* diagonal is not same scale as off-diag */
	}

        /* Compute which element to select */
	i = (int) (len * param) + 1;

	/* Select the i-th smallest element */
        localsum += randomized_select(buffer, 0, len-1, i);
    }

    /* Find the average across all processors */
    MPI_Allreduce(&localsum, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Comm_size(comm, &npes);

    free(buffer);
    return sum / (ps->A->end_rows[npes-1] - ps->A->beg_rows[0] + 1);
}

/*--------------------------------------------------------------------------
 * ParaSailsSelectFilter - select a threshold for the preconditioner pattern.  
 * The threshold attempts to be chosen such that approximately (1-param) of 
 * all the matrix elements is larger than this threshold.  This is accomplished
 * by finding the element in each row that is smaller than (1-param) of the 
 * elements in that row, and averaging these elements over all rows.  The 
 * threshold is selected on the diagonally scaled matrix.
 *
 * Assumes matrix is in local indexing.
 *--------------------------------------------------------------------------*/

double ParaSailsSelectFilter(ParaSails *ps, double param)
{
    int row, len, *ind, i, npes;
    double *val;
    double localsum = 0.0, sum;

    MPI_Comm comm = ps->A->comm;

    /* Buffer for storing the values in each row when computing the 
       i-th smallest element - buffer will grow if necessary */
    double *buffer;
    int buflen = 10;
    buffer = (double *) malloc(buflen * sizeof(double));

    for (row=0; row<=ps->M->end_row - ps->M->beg_row; row++)
    {
        MatrixGetRow(ps->M, row, &len, &ind, &val);

	if (len > buflen)
	{
	    free(buffer);
	    buflen = len;
            buffer = (double *) malloc(buflen * sizeof(double));
	}

	/* Copy the absolute values into a work buffer */
	for (i=0; i<len; i++)
	    if (ind[i] == row)
	        buffer[i] = 0.0;
	    else
	        buffer[i] = ABS(val[i]);

        /* Compute which element to select */
	i = (int) (len * param) + 1;

	/* Select the i-th smallest element */
        localsum += randomized_select(buffer, 0, len-1, i);
    }

    /* Find the average across all processors */
    MPI_Allreduce(&localsum, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Comm_size(comm, &npes);

    free(buffer);
    return sum / (ps->A->end_rows[npes-1] - ps->A->beg_rows[0] + 1);
}

/*--------------------------------------------------------------------------
 * ParaSailsFilterValues - constructs another matrix, free existing matrix
 * Assumes M is in local indexing.
 *--------------------------------------------------------------------------*/

void ParaSailsFilterValues(ParaSails *ps, double filter)
{
    int i, j;
    int row, len, *ind;
    double *val;
    Matrix *F;

    F = MatrixCreate(ps->M->comm, ps->M->beg_row, ps->M->end_row);

    for (row=0; row<=ps->M->end_row - ps->M->beg_row; row++)
    {
        MatrixGetRow(ps->M, row, &len, &ind, &val);

        j = 0;
        for (i=0; i<len; i++)
        {
            if (ABS(val[i]) >= filter || row == ind[i])
	    {
                val[j] = val[i];
                ind[j] = ind[i];
		j++;
	    }
        }

        MatrixSetRow(F, row, j, ind, val);
    }

    Rescale(ps->numb, ps->stored_rows, F);

    MatrixDestroy(ps->M);
    ps->M = F;
}

/*--------------------------------------------------------------------------
 * ParaSailsComplete -
 *--------------------------------------------------------------------------*/

void ParaSailsComplete(ParaSails *ps)
{
    int row, len, *ind;
    double *val;

    /* convert to global indexing before calling MatrixComplete */
    for (row=0; row<=ps->M->end_row - ps->M->beg_row; row++)
    {
	MatrixGetRow(ps->M, row, &len, &ind, &val);
	NumberingLocalToGlobal(ps->A->numb, len, ind, ind);
    }

    MatrixComplete(ps->M);
}

/*--------------------------------------------------------------------------
 * ParaSailsApply - Apply the ParaSails preconditioner
 *
 * ps - input ParaSails object
 * u  - input array of doubles
 * v  - output array of doubles
 *
 * Although this computation can be done in place, it typically will not
 * be used this way, since the caller usually needs to preserve the input
 * vector.
 *--------------------------------------------------------------------------*/

void ParaSailsApply(ParaSails *ps, double *u, double *v)
{
    if (ps->symmetric)
    {
        MatrixMatvec(ps->M, u, v);      /* need to preserve u */
        MatrixMatvecTrans(ps->M, v, v); /* do the second mult in place */
    }
    else
    {
        MatrixMatvec(ps->M, u, v);
    }
}


/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParaSails - Parallel sparse approximate inverse least squares.
 *
 *****************************************************************************/
#include "HYPRE_config.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "Common.h"
#include "Matrix.h"
#include "Numbering.h"
#include "RowPatt.h"
#include "StoredRows.h"
#include "PrunedRows.h"
#include "OrderStat.h"
#include "LoadBal.h"
#include "ParaSails.h"

#include "_hypre_lapack.h"

#define ROW_PRUNED_REQ_TAG        221
#define ROW_STORED_REQ_TAG        222
#define ROW_REPI_TAG              223
#define ROW_REPV_TAG              224

#ifdef ESSL
#include <essl.h>
#endif

#if 0 /* no longer need this since using 'memset' now */
#ifdef WIN32
static void bzero(char *a, HYPRE_Int n) {HYPRE_Int i; for (i=0; i<n; i++) {a[i]=0;}}
#endif
#endif


/******************************************************************************
 *
 * ParaSails private functions
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * FindNumReplies - Find the number of replies that this processor should
 * expect.  The input "replies_list" is an array that indicates what
 * processors were sent a message from the local processor.  An Allreduce
 * operation determines the total number of messages sent to the local
 * processor.
 *--------------------------------------------------------------------------*/

HYPRE_Int FindNumReplies(MPI_Comm comm, HYPRE_Int *replies_list)
{
    HYPRE_Int num_replies;
    HYPRE_Int npes, mype;
    HYPRE_Int *replies_list2;

    hypre_MPI_Comm_rank(comm, &mype);
    hypre_MPI_Comm_size(comm, &npes);

    replies_list2 = hypre_TAlloc(HYPRE_Int, npes , HYPRE_MEMORY_HOST);

    hypre_MPI_Allreduce(replies_list, replies_list2, npes, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
    num_replies = replies_list2[mype];

    hypre_TFree(replies_list2,HYPRE_MEMORY_HOST);

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
 *          hypre_MPI_AllReduce) the number of requests made to the current
 *          processor when the communication pattern is nonsymmetric.
 *--------------------------------------------------------------------------*/

static void SendRequests(MPI_Comm comm, HYPRE_Int tag, Matrix *mat, HYPRE_Int reqlen, HYPRE_Int *reqind,
  HYPRE_Int *num_requests, HYPRE_Int *replies_list)
{
    hypre_MPI_Request request;
    HYPRE_Int i, j, this_pe;

    hypre_shell_sort(reqlen, reqind);

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
        hypre_MPI_Isend(&reqind[i], j-i, HYPRE_MPI_INT, this_pe, tag,
            comm, &request);
        hypre_MPI_Request_free(&request);
        (*num_requests)++;

        if (replies_list != NULL)
            replies_list[this_pe] = 1;
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

static void ReceiveRequest(MPI_Comm comm, HYPRE_Int *source, HYPRE_Int tag, HYPRE_Int **buffer,
  HYPRE_Int *buflen, HYPRE_Int *count)
{
    hypre_MPI_Status status;

    hypre_MPI_Probe(hypre_MPI_ANY_SOURCE, tag, comm, &status);
    *source = status.hypre_MPI_SOURCE;
    hypre_MPI_Get_count(&status, HYPRE_MPI_INT, count);

    if (*count > *buflen)
    {
        hypre_TFree(*buffer,HYPRE_MEMORY_HOST);
        *buflen = *count;
        *buffer = hypre_TAlloc(HYPRE_Int, *buflen , HYPRE_MEMORY_HOST);
    }

    hypre_MPI_Recv(*buffer, *count, HYPRE_MPI_INT, *source, tag, comm, &status);
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
  HYPRE_Int dest, HYPRE_Int *buffer, HYPRE_Int count,
  PrunedRows *pruned_rows, Mem *mem, hypre_MPI_Request *request)
{
    HYPRE_Int sendbacksize, j;
    HYPRE_Int len, *ind, *indbuf, *indbufp;
    HYPRE_Int temp;

    /* Determine the size of the integer message we need to send back */
    sendbacksize = count+1; /* length of header part */
    for (j=0; j<count; j++)
    {
        NumberingGlobalToLocal(numb, 1, &buffer[j], &temp);
        PrunedRowsGet(pruned_rows, temp, &len, &ind);
        sendbacksize += (len+1);  /* add one for the row length */
    }

    /* Reply buffer - will be freed by caller */
    indbuf = (HYPRE_Int *) MemAlloc(mem, sendbacksize * sizeof(HYPRE_Int));

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
        /* memcpy(indbufp, ind, sizeof(HYPRE_Int)*len); */
        NumberingLocalToGlobal(numb, len, ind, indbufp);
        indbufp += len;
    }

    hypre_MPI_Isend(indbuf, indbufp-indbuf, HYPRE_MPI_INT, dest, ROW_REPI_TAG,
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
  PrunedRows *pruned_rows, RowPatt *patt)
{
    hypre_MPI_Status status;
    HYPRE_Int source, count;
    HYPRE_Int len, *ind, num_rows, *row_nums, j;

    /* Don't know the size of reply, so use probe and get count */
    hypre_MPI_Probe(hypre_MPI_ANY_SOURCE, ROW_REPI_TAG, comm, &status);
    source = status.hypre_MPI_SOURCE;
    hypre_MPI_Get_count(&status, HYPRE_MPI_INT, &count);

    /* Allocate space in stored rows data structure */
    ind = PrunedRowsAlloc(pruned_rows, count);
    hypre_MPI_Recv(ind, count, HYPRE_MPI_INT, source, ROW_REPI_TAG, comm, &status);

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
  HYPRE_Int dest, HYPRE_Int *buffer, HYPRE_Int count,
  StoredRows *stored_rows, Mem *mem, hypre_MPI_Request *request)
{
    HYPRE_Int sendbacksize, j;
    HYPRE_Int len, *ind, *indbuf, *indbufp;
    HYPRE_Real *val, *valbuf, *valbufp;
    HYPRE_Int temp;

    /* Determine the size of the integer message we need to send back */
    sendbacksize = count+1; /* length of header part */
    for (j=0; j<count; j++)
    {
        NumberingGlobalToLocal(numb, 1, &buffer[j], &temp);
        StoredRowsGet(stored_rows, temp, &len, &ind, &val);
        sendbacksize += (len+1);  /* add one for the row length */
    }

    /* Reply buffers - will be freed by caller */
    indbuf = (HYPRE_Int *)    MemAlloc(mem, sendbacksize * sizeof(HYPRE_Int));
    valbuf = (HYPRE_Real *) MemAlloc(mem, sendbacksize * sizeof(HYPRE_Real));

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
        /* memcpy(indbufp, ind, sizeof(HYPRE_Int)*len); */
        NumberingLocalToGlobal(numb, len, ind, indbufp);
        hypre_TMemcpy(valbufp,  val, HYPRE_Real, len, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
        indbufp += len;
        valbufp += len;
    }

    hypre_MPI_Isend(indbuf, indbufp-indbuf, HYPRE_MPI_INT, dest, ROW_REPI_TAG,
        comm, request);

    hypre_MPI_Request_free(request);

    hypre_MPI_Isend(valbuf, valbufp-valbuf, hypre_MPI_REAL, dest, ROW_REPV_TAG,
        comm, request);
}

/*--------------------------------------------------------------------------
 * ReceiveReplyStoredRows - Receive a reply sent by SendReplyStoredRows
 *
 * comm    - MPI communicator (input)
 * numb    - Numbering object (input)
 * stored_rows - the stored_rows object where the rows should be stored
 *--------------------------------------------------------------------------*/

static void ReceiveReplyStoredRows(MPI_Comm comm, Numbering *numb,
  StoredRows *stored_rows)
{
    hypre_MPI_Status status;
    HYPRE_Int source, count;
    HYPRE_Int len, *ind, num_rows, *row_nums, j;
    HYPRE_Real *val;

    /* Don't know the size of reply, so use probe and get count */
    hypre_MPI_Probe(hypre_MPI_ANY_SOURCE, ROW_REPI_TAG, comm, &status);
    source = status.hypre_MPI_SOURCE;
    hypre_MPI_Get_count(&status, HYPRE_MPI_INT, &count);

    /* Allocate space in stored rows data structure */
    ind = StoredRowsAllocInd(stored_rows, count);
    hypre_MPI_Recv(ind, count, HYPRE_MPI_INT, source, ROW_REPI_TAG, comm, &status);
    val = StoredRowsAllocVal(stored_rows, count);
    hypre_MPI_Recv(val, count, hypre_MPI_REAL, source, ROW_REPV_TAG, comm, &status);

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
  PrunedRows *pruned_rows, HYPRE_Int num_levels)
{
    RowPatt *patt;
    HYPRE_Int row, len, *ind;

    HYPRE_Int num_requests;
    HYPRE_Int source;

    HYPRE_Int bufferlen;
    HYPRE_Int *buffer;

    HYPRE_Int level;

    HYPRE_Int i;
    HYPRE_Int count;
    hypre_MPI_Request *requests;
    hypre_MPI_Status *statuses;
    HYPRE_Int npes;
    HYPRE_Int num_replies, *replies_list;

    Mem *mem;

    hypre_MPI_Comm_size(comm, &npes);
    requests = hypre_TAlloc(hypre_MPI_Request, npes , HYPRE_MEMORY_HOST);
    statuses = hypre_TAlloc(hypre_MPI_Status, npes , HYPRE_MEMORY_HOST);

    /* Merged pattern of pruned rows on this processor */

    patt = RowPattCreate(PARASAILS_MAXLEN);

    for (row=0; row<=M->end_row - M->beg_row; row++)
    {
        PrunedRowsGet(pruned_rows, row, &len, &ind);
        RowPattMergeExt(patt, len, ind, numb->num_loc);
    }

    /* Loop to construct pattern of pruned rows on this processor */

    bufferlen = 10; /* size will grow if get a long msg */
    buffer = hypre_TAlloc(HYPRE_Int, bufferlen , HYPRE_MEMORY_HOST);

    for (level=1; level<=num_levels; level++)
    {
        mem = (Mem *) MemCreate();

        /* Get list of indices that were just merged */
        RowPattPrevLevel(patt, &len, &ind);

        /* Convert local row numbers to global row numbers */
        NumberingLocalToGlobal(numb, len, ind, ind);

        replies_list = hypre_CTAlloc(HYPRE_Int, npes, HYPRE_MEMORY_HOST);

        SendRequests(comm, ROW_PRUNED_REQ_TAG, M, len, ind, &num_requests, replies_list);

        num_replies = FindNumReplies(comm, replies_list);
        hypre_TFree(replies_list,HYPRE_MEMORY_HOST);

        for (i=0; i<num_replies; i++)
        {
            /* Receive count indices stored in buffer */
            ReceiveRequest(comm, &source, ROW_PRUNED_REQ_TAG, &buffer, &bufferlen, &count);

            SendReplyPrunedRows(comm, numb, source, buffer, count,
                pruned_rows, mem, &requests[i]);
        }

        for (i=0; i<num_requests; i++)
        {
            /* Will also merge the pattern of received rows into "patt" */
            ReceiveReplyPrunedRows(comm, numb, pruned_rows, patt);
        }

        hypre_MPI_Waitall(num_replies, requests, statuses);
        MemDestroy(mem);
    }

    RowPattDestroy(patt);
    hypre_TFree(buffer,HYPRE_MEMORY_HOST);
    hypre_TFree(requests,HYPRE_MEMORY_HOST);
    hypre_TFree(statuses,HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------------
 * ExchangePrunedRowsExt
 *--------------------------------------------------------------------------*/

static void ExchangePrunedRowsExt(MPI_Comm comm, Matrix *M, Numbering *numb,
  PrunedRows *pruned_rows_global, PrunedRows *pruned_rows_local, HYPRE_Int num_levels)
{
    RowPatt *patt;
    HYPRE_Int row, len, *ind;

    HYPRE_Int num_requests;
    HYPRE_Int source;

    HYPRE_Int bufferlen;
    HYPRE_Int *buffer;

    HYPRE_Int level;

    HYPRE_Int i;
    HYPRE_Int count;
    hypre_MPI_Request *requests;
    hypre_MPI_Status *statuses;
    HYPRE_Int npes;
    HYPRE_Int num_replies, *replies_list;

    Mem *mem;

    hypre_MPI_Comm_size(comm, &npes);
    requests = hypre_TAlloc(hypre_MPI_Request, npes , HYPRE_MEMORY_HOST);
    statuses = hypre_TAlloc(hypre_MPI_Status, npes , HYPRE_MEMORY_HOST);

    /* Merged pattern of pruned rows on this processor */

    patt = RowPattCreate(PARASAILS_MAXLEN);

    for (row=0; row<=M->end_row - M->beg_row; row++)
    {
        PrunedRowsGet(pruned_rows_global, row, &len, &ind);
        RowPattMergeExt(patt, len, ind, numb->num_loc);
    }

    /* Loop to construct pattern of pruned rows on this processor */

    bufferlen = 10; /* size will grow if get a long msg */
    buffer = hypre_TAlloc(HYPRE_Int, bufferlen , HYPRE_MEMORY_HOST);

    for (level=0; level<=num_levels; level++)  /* MUST DO THIS AT LEAST ONCE */
    {
        mem = (Mem *) MemCreate();

        /* Get list of indices that were just merged */
        RowPattPrevLevel(patt, &len, &ind);

        /* Convert local row numbers to global row numbers */
        NumberingLocalToGlobal(numb, len, ind, ind);

        replies_list = hypre_CTAlloc(HYPRE_Int, npes, HYPRE_MEMORY_HOST);

        SendRequests(comm, ROW_PRUNED_REQ_TAG, M, len, ind, &num_requests, replies_list);

        num_replies = FindNumReplies(comm, replies_list);
        hypre_TFree(replies_list,HYPRE_MEMORY_HOST);

        for (i=0; i<num_replies; i++)
        {
            /* Receive count indices stored in buffer */
	    ReceiveRequest(comm, &source, ROW_PRUNED_REQ_TAG, &buffer, &bufferlen, &count);

            SendReplyPrunedRows(comm, numb, source, buffer, count,
                pruned_rows_local, mem, &requests[i]);
        }

        for (i=0; i<num_requests; i++)
        {
            /* Will also merge the pattern of received rows into "patt" */
            ReceiveReplyPrunedRows(comm, numb, pruned_rows_local, patt);
        }

        hypre_MPI_Waitall(num_replies, requests, statuses);
        MemDestroy(mem);
    }

    RowPattDestroy(patt);
    hypre_TFree(buffer,HYPRE_MEMORY_HOST);
    hypre_TFree(requests,HYPRE_MEMORY_HOST);
    hypre_TFree(statuses,HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------------
 * ExchangePrunedRowsExt2 - part 2 of the algorithm
 *--------------------------------------------------------------------------*/

static void ExchangePrunedRowsExt2(MPI_Comm comm, Matrix *M, Numbering *numb,
  PrunedRows *pruned_rows_global, PrunedRows *pruned_rows_local, HYPRE_Int num_levels)
{
    RowPatt *patt;
    HYPRE_Int row, len, *ind;

    HYPRE_Int num_requests;
    HYPRE_Int source;

    HYPRE_Int bufferlen;
    HYPRE_Int *buffer;

    HYPRE_Int level;

    HYPRE_Int i;
    HYPRE_Int count;
    hypre_MPI_Request *requests;
    hypre_MPI_Status *statuses;
    HYPRE_Int npes;
    HYPRE_Int num_replies, *replies_list;

    Mem *mem;

    hypre_MPI_Comm_size(comm, &npes);
    requests = hypre_TAlloc(hypre_MPI_Request, npes , HYPRE_MEMORY_HOST);
    statuses = hypre_TAlloc(hypre_MPI_Status, npes , HYPRE_MEMORY_HOST);

    /* Merged pattern of pruned rows on this processor */

    patt = RowPattCreate(PARASAILS_MAXLEN);

    for (row=0; row<=M->end_row - M->beg_row; row++)
    {
        PrunedRowsGet(pruned_rows_local, row, &len, &ind);
        RowPattMergeExt(patt, len, ind, numb->num_loc);
    }

    /* Compute powers with local matrix - no communication is needed */

    for (level=1; level<=num_levels; level++)
    {
        HYPRE_Int lenprev, *indprev;

        /* Get the indices that were just added */
        RowPattPrevLevel(patt, &lenprev, &indprev);

        for (i=0; i<lenprev; i++)
        {
            PrunedRowsGet(pruned_rows_local, indprev[i], &len, &ind);
            RowPattMergeExt(patt, len, ind, numb->num_loc);
        }
    }

    /* Now get rows from pruned_rows_global */

    bufferlen = 10; /* size will grow if get a long msg */
    buffer = hypre_TAlloc(HYPRE_Int, bufferlen , HYPRE_MEMORY_HOST);

    /* DO THIS ONCE */
    {
        mem = (Mem *) MemCreate();

	/* Get list of indices - these are all nonlocal indices */
        RowPattGet(patt, &len, &ind);

        /* Convert local row numbers to global row numbers */
        NumberingLocalToGlobal(numb, len, ind, ind);

        replies_list = hypre_CTAlloc(HYPRE_Int, npes, HYPRE_MEMORY_HOST);

        SendRequests(comm, ROW_PRUNED_REQ_TAG, M, len, ind, &num_requests, replies_list);

        num_replies = FindNumReplies(comm, replies_list);
        hypre_TFree(replies_list,HYPRE_MEMORY_HOST);

        for (i=0; i<num_replies; i++)
        {
            /* Receive count indices stored in buffer */
            ReceiveRequest(comm, &source, ROW_PRUNED_REQ_TAG, &buffer, &bufferlen, &count);

            SendReplyPrunedRows(comm, numb, source, buffer, count,
                pruned_rows_global, mem, &requests[i]);
        }

        for (i=0; i<num_requests; i++)
        {
            /* Will also merge the pattern of received rows into "patt" */
            ReceiveReplyPrunedRows(comm, numb, pruned_rows_global, patt);
        }

        hypre_MPI_Waitall(num_replies, requests, statuses);
        MemDestroy(mem);
    }

    RowPattDestroy(patt);
    hypre_TFree(buffer,HYPRE_MEMORY_HOST);
    hypre_TFree(requests,HYPRE_MEMORY_HOST);
    hypre_TFree(statuses,HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------------
 * ExchangeStoredRows
 *--------------------------------------------------------------------------*/

static void ExchangeStoredRows(MPI_Comm comm, Matrix *A, Matrix *M,
  Numbering *numb, StoredRows *stored_rows, LoadBal *load_bal)
{
    RowPatt *patt;
    HYPRE_Int row, len, *ind;
    HYPRE_Real *val;

    HYPRE_Int num_requests;
    HYPRE_Int source;

    HYPRE_Int bufferlen;
    HYPRE_Int *buffer;

    HYPRE_Int i;
    HYPRE_Int count;
    hypre_MPI_Request *requests = NULL;
    hypre_MPI_Status *statuses = NULL;
    HYPRE_Int npes;
    HYPRE_Int num_replies, *replies_list;

    Mem *mem = (Mem *) MemCreate();

    hypre_MPI_Comm_size(comm, &npes);

    /* Merge the patterns of all the rows of M on this processor */
    /* The merged pattern is not already known, since M is triangular */

    patt = RowPattCreate(PARASAILS_MAXLEN);

    /* for (row=load_bal->beg_row; row<=M->end_row; row++) */
    /* We need the additional rows if we need to Rescale */
    /* i.e., if filter is nonzero and we are in symmetric case */

    for (row=M->beg_row; row<=M->end_row; row++)
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
    NumberingLocalToGlobal(numb, len, ind, ind);

    replies_list = hypre_CTAlloc(HYPRE_Int, npes, HYPRE_MEMORY_HOST);

    SendRequests(comm, ROW_STORED_REQ_TAG, A, len, ind, &num_requests, replies_list);

    num_replies = FindNumReplies(comm, replies_list);
    hypre_TFree(replies_list,HYPRE_MEMORY_HOST);

    if (num_replies)
    {
        requests = hypre_TAlloc(hypre_MPI_Request, num_replies , HYPRE_MEMORY_HOST);
        statuses = hypre_TAlloc(hypre_MPI_Status, num_replies , HYPRE_MEMORY_HOST);
    }

    bufferlen = 10; /* size will grow if get a long msg */
    buffer = hypre_TAlloc(HYPRE_Int, bufferlen , HYPRE_MEMORY_HOST);

    for (i=0; i<num_replies; i++)
    {
        /* Receive count indices stored in buffer */
        ReceiveRequest(comm, &source, ROW_STORED_REQ_TAG, &buffer, &bufferlen, &count);

        SendReplyStoredRows(comm, numb, source, buffer, count,
            stored_rows, mem, &requests[i]);
    }

    for (i=0; i<num_requests; i++)
    {
        ReceiveReplyStoredRows(comm, numb, stored_rows);
    }

    hypre_MPI_Waitall(num_replies, requests, statuses);

    /* Free all send buffers */
    MemDestroy(mem);

    RowPattDestroy(patt);
    hypre_TFree(buffer,HYPRE_MEMORY_HOST);
    hypre_TFree(requests,HYPRE_MEMORY_HOST);
    hypre_TFree(statuses,HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------------
 * ConstructPatternForEachRow
 *
 * pruned_rows - pruned rows, used for constructing row patterns (input)
 * num_levels  - number of levels in pattern (input)
 * M           - matrix where the row patterns will be stored (input/output).
 *               This is the approximate inverse with lower triangular pattern
 *--------------------------------------------------------------------------*/

static void ConstructPatternForEachRow(HYPRE_Int symmetric, PrunedRows *pruned_rows,
  HYPRE_Int num_levels, Numbering *numb, Matrix *M, HYPRE_Real *costp)
{
    HYPRE_Int row, len, *ind, level, lenprev, *indprev;
    HYPRE_Int i, j;
    RowPatt *row_patt;
    HYPRE_Int nnz = 0;
    HYPRE_Int npes;

    hypre_MPI_Comm_size(M->comm, &npes);
    *costp = 0.0;

    row_patt = RowPattCreate(PARASAILS_MAXLEN);

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
        MatrixSetRow(M, row+M->beg_row, len, ind, NULL);

        nnz += len;
        (*costp) += (HYPRE_Real) len*len*len;
    }

#if 0
    {
    HYPRE_Int mype;
    hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &mype);
    hypre_printf("%d: nnz: %10d  ********* cost %7.1e\n", mype, nnz, *costp);
    fflush(stdout);
    }
#endif

    RowPattDestroy(row_patt);
}

/*--------------------------------------------------------------------------
 * ConstructPatternForEachRowExt - extended version
 *
 * pruned_rows - pruned rows, used for constructing row patterns (input)
 * num_levels  - number of levels in pattern (input)
 * M           - matrix where the row patterns will be stored (input/output).
 *               This is the approximate inverse with lower triangular pattern
 *--------------------------------------------------------------------------*/

static void ConstructPatternForEachRowExt(HYPRE_Int symmetric, 
  PrunedRows *pruned_rows_global, PrunedRows *pruned_rows_local, 
  HYPRE_Int num_levels, Numbering *numb, Matrix *M, HYPRE_Real *costp)
{
    HYPRE_Int row, len, *ind, level, lenprev, *indprev;
    HYPRE_Int i, j;
    RowPatt *row_patt;
    RowPatt *row_patt2;
    HYPRE_Int nnz = 0;
    HYPRE_Int npes;

    hypre_MPI_Comm_size(M->comm, &npes);
    *costp = 0.0;

    row_patt = RowPattCreate(PARASAILS_MAXLEN);
    row_patt2 = RowPattCreate(PARASAILS_MAXLEN);

    for (row=0; row<=M->end_row - M->beg_row; row++)
    {
        /* Get initial pattern for row */
        PrunedRowsGet(pruned_rows_global, row, &len, &ind);
        RowPattMerge(row_patt, len, ind);

        /* Loop */
        for (level=0; level<=num_levels; level++) /* at least once */
        {
            /* Get the indices that were just added */
            RowPattPrevLevel(row_patt, &lenprev, &indprev);

            for (i=0; i<lenprev; i++)
            {
                PrunedRowsGet(pruned_rows_local, indprev[i], &len, &ind);
                RowPattMerge(row_patt, len, ind);
            }
        }

        /***********************
	 * Now do the transpose 
	 ***********************/

        /* Get initial pattern for row */
        PrunedRowsGet(pruned_rows_local, row, &len, &ind);
        RowPattMerge(row_patt2, len, ind);

        /* Loop */
        for (level=1; level<=num_levels; level++)
        {
            /* Get the indices that were just added */
            RowPattPrevLevel(row_patt2, &lenprev, &indprev);

            for (i=0; i<lenprev; i++)
            {
                PrunedRowsGet(pruned_rows_local, indprev[i], &len, &ind);
                RowPattMerge(row_patt2, len, ind);
            }
        }

	/* One more merge, with pruned_rows_global */
        RowPattGet(row_patt2, &lenprev, &indprev);
        for (i=0; i<lenprev; i++)
        {
            PrunedRowsGet(pruned_rows_global, indprev[i], &len, &ind);
            RowPattMerge(row_patt2, len, ind);
        }


        /****************************
	 * Merge the two row patterns
	 ****************************/

        RowPattGet(row_patt2, &len, &ind);
        RowPattMerge(row_patt, len, ind);

        /****************************
	 * Done computing pattern!
	 ****************************/

	/* get the indices in the pattern */
        RowPattGet(row_patt, &len, &ind);

        /* do reset here, because now we mess with ind array */
        RowPattReset(row_patt);
        RowPattReset(row_patt2);

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
        MatrixSetRow(M, row+M->beg_row, len, ind, NULL);

        nnz += len;
        (*costp) += (HYPRE_Real) len*len*len;
    }

#if 0
    {
    HYPRE_Int mype;
    hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &mype);
    hypre_printf("%d: nnz: %10d  ********* cost %7.1e\n", mype, nnz, *costp);
    fflush(stdout);
    }
#endif

    RowPattDestroy(row_patt);
    RowPattDestroy(row_patt2);
}

/*--------------------------------------------------------------------------
 * ComputeValuesSym
 *--------------------------------------------------------------------------*/

static HYPRE_Int ComputeValuesSym(StoredRows *stored_rows, Matrix *mat,
  HYPRE_Int local_beg_row, Numbering *numb, HYPRE_Int symmetric)
{
    HYPRE_Int *marker;
    HYPRE_Int row, maxlen, len, *ind;
    HYPRE_Real *val;

    HYPRE_Real *ahat, *ahatp;
    HYPRE_Int i, j, len2, *ind2, loc;
    HYPRE_Real *val2, temp;
    HYPRE_Real time0, time1, timet = 0.0, timea = 0.0;

    HYPRE_Real ahatcost = 0.0;

    HYPRE_Real error = 0;

#ifndef ESSL
    char uplo = 'L';
    HYPRE_Int one = 1;
    HYPRE_Int info;
#endif

    /* Allocate and initialize full length marker array */
    marker = hypre_TAlloc(HYPRE_Int, numb->num_ind , HYPRE_MEMORY_HOST);
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
    ahat = hypre_TAlloc(HYPRE_Real, maxlen*(maxlen+1)/2 , HYPRE_MEMORY_HOST);
#else
    ahat = hypre_TAlloc(HYPRE_Real, maxlen*maxlen , HYPRE_MEMORY_HOST);
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
/*        bzero((char *) ahat, len*(len+1)/2 * sizeof(HYPRE_Real));*/
        memset(ahat, 0, len*(len+1)/2 * sizeof(HYPRE_Real));
#else
/*        bzero((char *) ahat, len*len * sizeof(HYPRE_Real));*/
        memset(ahat, 0, len*len * sizeof(HYPRE_Real));
#endif

        time0 = hypre_MPI_Wtime();

        /* Form ahat matrix, entries correspond to indices in "ind" only */
        ahatp = ahat;
        for (i=0; i<len; i++)
        {
            StoredRowsGet(stored_rows, ind[i], &len2, &ind2, &val2);
            hypre_assert(len2 > 0);

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

        if (symmetric == 2)
        {
#ifdef ESSL
            hypre_printf("Symmetric precon for nonsym problem not yet available\n");
            hypre_printf("for ESSL version.  Please contact the author.\n");
            PARASAILS_EXIT;
#else
            HYPRE_Int k, kk;
            k = 0;
            for (i=0; i<len; i++)
            {
                for (j=0; j<len; j++)
                {
                    kk = j*len + i;
                    ahat[k] = (ahat[k] + ahat[kk]) / 2.0;
                    k++;
                }
            }
#endif
        }

        time1 = hypre_MPI_Wtime();
        timea += (time1-time0);
        ahatcost += (HYPRE_Real) (len*len2);

        /* Set the right-hand side */
/*        bzero((char *) val, len*sizeof(HYPRE_Real));*/
        memset(val, 0, len*sizeof(HYPRE_Real));
        NumberingGlobalToLocal(numb, 1, &row, &loc);
        loc = marker[loc];
        hypre_assert(loc != -1);
        val[loc] = 1.0;

        /* Reset marker array */
        for (i=0; i<len; i++)
            marker[ind[i]] = -1;

        time0 = hypre_MPI_Wtime();

#ifdef ESSL
        dppf(ahat, len, 1);
        dpps(ahat, len, val, 1);
#else
        /* Solve local linear system - factor phase */
        hypre_dpotrf(&uplo, &len, ahat, &len, &info);
        if (info != 0)
        {
#if 0
            hypre_printf("Matrix may not be symmetric positive definite.\n");
            hypre_printf("ParaSails: row %d, dpotrf returned %d.\n", row, info);
            hypre_printf("ParaSails: len %d, ahat: %f %f %f %f\n", len,
                ahat[0], ahat[1], ahat[2], ahat[3]);
            PARASAILS_EXIT;
#endif
            error = 1;
        }

        /* Solve local linear system - solve phase */
        hypre_dpotrs(&uplo, &len, &one, ahat, &len, val, &len, &info);
        if (info != 0)
        {
#if 0
            hypre_printf("ParaSails: row %d, dpotrs returned %d.\n", row, info);
            hypre_printf("ParaSails: len %d, ahat: %f %f %f %f\n", len,
                ahat[0], ahat[1], ahat[2], ahat[3]);
            PARASAILS_EXIT;
#endif
            error = 1;
        }
#endif
        time1 = hypre_MPI_Wtime();
        timet += (time1-time0);

        /* Scale the result */
        temp = 1.0 / sqrt(ABS(val[loc]));
        for (i=0; i<len; i++)
            val[i] = val[i] * temp;
    }

    hypre_TFree(marker,HYPRE_MEMORY_HOST);
    hypre_TFree(ahat,HYPRE_MEMORY_HOST);

#if 0
    {
    HYPRE_Int mype;
    hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &mype);
    hypre_printf("%d: Time for ahat: %f, for local solves: %f\n", mype, timea, timet);
    hypre_printf("%d: ahatcost: %7.1e, numrows: %d, maxlen: %d\n",
        mype, ahatcost, mat->end_row-local_beg_row+1, maxlen);
    fflush(stdout);
    }
#endif

    return error;
}

/*--------------------------------------------------------------------------
 * ComputeValuesNonsym
 *--------------------------------------------------------------------------*/

static HYPRE_Int ComputeValuesNonsym(StoredRows *stored_rows, Matrix *mat,
  HYPRE_Int local_beg_row, Numbering *numb)
{
    HYPRE_Int *marker;
    HYPRE_Real *ahat, *ahatp, *bhat;
    HYPRE_Real *work;
    HYPRE_Int ahat_size = 10000, bhat_size = 1000, work_size = 2000*64;

    HYPRE_Int row, len, *ind;
    HYPRE_Real *val;

    HYPRE_Int i, j, len2, *ind2, loc;
    HYPRE_Real *val2;
    HYPRE_Real time0, time1, timet = 0.0, timea = 0.0;

    HYPRE_Int npat;
    HYPRE_Int pattsize = 1000;
    HYPRE_Int *patt = hypre_TAlloc(HYPRE_Int, pattsize, HYPRE_MEMORY_HOST);

    HYPRE_Int info;

    HYPRE_Int error = 0;

#ifndef ESSL
    char trans = 'N';
    HYPRE_Int one = 1;
#endif

    /* Allocate and initialize marker array */
    /* Since numb already knows about the indices of the external rows that
       will be needed, numb_ind is the maximum size of the marker array */
    marker = hypre_TAlloc(HYPRE_Int, numb->num_ind , HYPRE_MEMORY_HOST);
    for (i=0; i<numb->num_ind; i++)
        marker[i] = -1;

    bhat = hypre_TAlloc(HYPRE_Real, bhat_size , HYPRE_MEMORY_HOST);
    ahat = hypre_TAlloc(HYPRE_Real, ahat_size , HYPRE_MEMORY_HOST);
    work = hypre_CTAlloc(HYPRE_Real, work_size, HYPRE_MEMORY_HOST);

    /* Compute values for row "row" of approximate inverse */
    for (row=local_beg_row; row<=mat->end_row; row++)
    {
        time0 = hypre_MPI_Wtime();

        /* Retrieve local indices */
        MatrixGetRow(mat, row - mat->beg_row, &len, &ind, &val);

        npat = 0;

        /* Put the diagonal entry into the marker array */
        NumberingGlobalToLocal(numb, 1, &row, &loc);
        marker[loc] = npat;
        patt[npat++] = loc;

        /* Fill marker array */
        for (i=0; i<len; i++)
        {
            StoredRowsGet(stored_rows, ind[i], &len2, &ind2, &val2);
            hypre_assert(len2 > 0);

            for (j=0; j<len2; j++)
            {
                loc = marker[ind2[j]];

                if (loc == -1)
                {
                    marker[ind2[j]] = npat;
                    if (npat >= pattsize)
                    {
                        pattsize = npat*2;
                        patt = hypre_TReAlloc(patt,HYPRE_Int,  pattsize, HYPRE_MEMORY_HOST);
                    }
                    patt[npat++] = ind2[j];
                }
            }
        }

        if (len*npat > ahat_size)
        {
            hypre_TFree(ahat,HYPRE_MEMORY_HOST);
            ahat_size = len*npat;
            ahat = hypre_TAlloc(HYPRE_Real, ahat_size , HYPRE_MEMORY_HOST);
        }

        /* Initialize ahat to zero */
/*        bzero((char *) ahat, len*npat * sizeof(HYPRE_Real));*/
        memset(ahat, 0, len*npat * sizeof(HYPRE_Real));

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

        time1 = hypre_MPI_Wtime();
        timea += (time1-time0);

        /* Reallocate bhat if necessary */
        if (npat > bhat_size)
        {
            hypre_TFree(bhat,HYPRE_MEMORY_HOST);
            bhat_size = npat;
            bhat = hypre_TAlloc(HYPRE_Real, bhat_size , HYPRE_MEMORY_HOST);
        }

        /* Set the right-hand side, bhat */
/*        bzero((char *) bhat, npat*sizeof(HYPRE_Real));*/
        memset(bhat, 0, npat*sizeof(HYPRE_Real));
        NumberingGlobalToLocal(numb, 1, &row, &loc);
        loc = marker[loc];
        hypre_assert(loc != -1);
        bhat[loc] = 1.0;

        /* Reset marker array */
        for (i=0; i<npat; i++)
            marker[patt[i]] = -1;

        time0 = hypre_MPI_Wtime();

#ifdef ESSL
        /* rhs in bhat, and put solution in val */
        dgells(0, ahat, npat, bhat, npat, val, len, NULL, 1.e-12, npat, len, 1,
            &info, work, work_size);
#else
        /* rhs in bhat, and put solution in bhat */
        hypre_dgels(&trans, &npat, &len, &one, ahat, &npat,
            bhat, &npat, work, &work_size, &info);

        if (info != 0)
        {
#if 0
            hypre_printf("ParaSails: row %d, dgels returned %d.\n", row, info);
            hypre_printf("ParaSails: len %d, ahat: %f %f %f %f\n", len,
                ahat[0], ahat[1], ahat[2], ahat[3]);
            PARASAILS_EXIT;
#endif
            error = 1;
        }

        /* Copy result into row */
        for (j=0; j<len; j++)
            val[j] = bhat[j];
#endif
        time1 = hypre_MPI_Wtime();
        timet += (time1-time0);
    }

    hypre_TFree(patt,HYPRE_MEMORY_HOST);
    hypre_TFree(marker,HYPRE_MEMORY_HOST);
    hypre_TFree(bhat,HYPRE_MEMORY_HOST);
    hypre_TFree(ahat,HYPRE_MEMORY_HOST);
    hypre_TFree(work,HYPRE_MEMORY_HOST);

#if 0
    {
    HYPRE_Int mype;
    hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &mype);
    hypre_printf("%d: Time for ahat: %f, for local solves: %f\n", mype, timea, timet);
    fflush(stdout);
    }
#endif

    return error;
}

/*--------------------------------------------------------------------------
 * SelectThresh - select a threshold for the preconditioner pattern.
 * The threshold attempts to be chosen such that approximately (1-param) of
 * all the matrix elements is larger than this threshold.  This is accomplished
 * by finding the element in each row that is smaller than (1-param) of the
 * elements in that row, and averaging these elements over all rows.  The
 * threshold is selected on the diagonally scaled matrix.
 *--------------------------------------------------------------------------*/

static HYPRE_Real SelectThresh(MPI_Comm comm, Matrix *A, DiagScale *diag_scale,
  HYPRE_Real param)
{
    HYPRE_Int row, len, *ind, i, npes;
    HYPRE_Real *val;
    HYPRE_Real localsum = 0.0, sum;
    HYPRE_Real temp;

    /* Buffer for storing the values in each row when computing the
       i-th smallest element - buffer will grow if necessary */
    HYPRE_Real *buffer;
    HYPRE_Int buflen = 10;
    buffer = hypre_TAlloc(HYPRE_Real, buflen , HYPRE_MEMORY_HOST);

    for (row=0; row<=A->end_row - A->beg_row; row++)
    {
        MatrixGetRow(A, row, &len, &ind, &val);

        if (len > buflen)
        {
            hypre_TFree(buffer,HYPRE_MEMORY_HOST);
            buflen = len;
            buffer = hypre_TAlloc(HYPRE_Real, buflen , HYPRE_MEMORY_HOST);
        }

        /* Copy the scaled absolute values into a work buffer */
        temp = DiagScaleGet(diag_scale, row);
        for (i=0; i<len; i++)
        {
            buffer[i] = temp*ABS(val[i])*DiagScaleGet(diag_scale, ind[i]);
            if (ind[i] == row)
                buffer[i] = 0.0; /* diagonal is not same scale as off-diag */
        }

        /* Compute which element to select */
        i = (HYPRE_Int) (len * param) + 1;

        /* Select the i-th smallest element */
        localsum += randomized_select(buffer, 0, len-1, i);
    }

    /* Find the average across all processors */
    hypre_MPI_Allreduce(&localsum, &sum, 1, hypre_MPI_REAL, hypre_MPI_SUM, comm);
    hypre_MPI_Comm_size(comm, &npes);

    hypre_TFree(buffer,HYPRE_MEMORY_HOST);
    return sum / (A->end_rows[npes-1] - A->beg_rows[0] + 1);
}

/*--------------------------------------------------------------------------
 * SelectFilter - Similar to SelectThresh, but on the preconditioner.
 * Assumes matrix is in local indexing.
 *--------------------------------------------------------------------------*/

static HYPRE_Real SelectFilter(MPI_Comm comm, Matrix *M, DiagScale *diag_scale,
  HYPRE_Real param, HYPRE_Int symmetric)
{
    HYPRE_Int row, len, *ind, i, npes;
    HYPRE_Real *val;
    HYPRE_Real localsum = 0.0, sum;
    HYPRE_Real temp = 1.0;

    /* Buffer for storing the values in each row when computing the
       i-th smallest element - buffer will grow if necessary */
    HYPRE_Real *buffer;
    HYPRE_Int buflen = 10;
    buffer = hypre_TAlloc(HYPRE_Real, buflen , HYPRE_MEMORY_HOST);

    for (row=0; row<=M->end_row - M->beg_row; row++)
    {
        MatrixGetRow(M, row, &len, &ind, &val);

        if (len > buflen)
        {
            hypre_TFree(buffer,HYPRE_MEMORY_HOST);
            buflen = len;
            buffer = hypre_TAlloc(HYPRE_Real, buflen , HYPRE_MEMORY_HOST);
        }

        if (symmetric == 0)
            temp = 1. / DiagScaleGet(diag_scale, row);

        /* Copy the scaled absolute values into a work buffer */
        for (i=0; i<len; i++)
        {
            buffer[i] = temp * ABS(val[i]) / DiagScaleGet(diag_scale, ind[i]);
            if (ind[i] == row)
                buffer[i] = 0.0;
        }

        /* Compute which element to select */
        i = (HYPRE_Int) (len * param) + 1;

        /* Select the i-th smallest element */
        localsum += randomized_select(buffer, 0, len-1, i);
    }

    /* Find the average across all processors */
    hypre_MPI_Allreduce(&localsum, &sum, 1, hypre_MPI_REAL, hypre_MPI_SUM, comm);
    hypre_MPI_Comm_size(comm, &npes);

    hypre_TFree(buffer,HYPRE_MEMORY_HOST);
    return sum / (M->end_rows[npes-1] - M->beg_rows[0] + 1);
}

/*--------------------------------------------------------------------------
 * FilterValues - Filter the values in a preconditioner matrix.
 * M - original matrix, in local ordering
 * F - new matrix, that has been created already
 * Also, return the cost estimate, in case SetupValues is called again
 * with load balancing - the old cost estimate would be incorrect.
 *--------------------------------------------------------------------------*/

static void FilterValues(Matrix *M, Matrix *F, DiagScale *diag_scale,
  HYPRE_Real filter, HYPRE_Int symmetric, HYPRE_Real *newcostp)
{
    HYPRE_Int i, j;
    HYPRE_Int row, len, *ind;
    HYPRE_Real *val, temp = 1.0;
    HYPRE_Real cost = 0.0;

    for (row=0; row<=M->end_row - M->beg_row; row++)
    {
        MatrixGetRow(M, row, &len, &ind, &val);

        j = 0;
        for (i=0; i<len; i++)
        {
            if (symmetric == 0)
                temp = 1. / DiagScaleGet(diag_scale, row);

            if (temp * ABS(val[i]) / DiagScaleGet(diag_scale, ind[i]) >= filter
              || row == ind[i])
            {
                val[j] = val[i];
                ind[j] = ind[i];
                j++;
            }
        }

        MatrixSetRow(F, row+F->beg_row, j, ind, val);

        cost += (HYPRE_Real) j*j*j;
    }

    *newcostp = cost;
}

/*--------------------------------------------------------------------------
 * Rescale - Rescaling to be used after filtering, in symmetric case.
 *--------------------------------------------------------------------------*/

static void Rescale(Matrix *M, StoredRows *stored_rows, HYPRE_Int num_ind)
{
    HYPRE_Int len, *ind, len2, *ind2;
    HYPRE_Real *val, *val2, *w;
    HYPRE_Int row, j, i;
    HYPRE_Real accum, prod;

    /* Allocate full-length workspace */
    w = hypre_CTAlloc(HYPRE_Real, num_ind, HYPRE_MEMORY_HOST);

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
                hypre_assert(ind2[i] < num_ind);
                w[ind2[i]] = val2[i];
            }

            /* Form inner product of current row with this row */
            prod = 0.0;
            for (i=0; i<len; i++)
            {
                hypre_assert(ind[i] < num_ind);
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

    hypre_TFree(w,HYPRE_MEMORY_HOST);
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
 * ParaSailsCreate - Allocate, initialize, and return a pointer to a
 * ParaSails preconditioner data structure.
 *--------------------------------------------------------------------------*/

ParaSails *ParaSailsCreate(MPI_Comm comm, HYPRE_Int beg_row, HYPRE_Int end_row, HYPRE_Int sym)
{
    ParaSails *ps = hypre_TAlloc(ParaSails, 1, HYPRE_MEMORY_HOST);
    HYPRE_Int npes;

    ps->symmetric          = sym;
    ps->thresh             = 0.1;
    ps->num_levels         = 1;
    ps->filter             = 0.0;
    ps->loadbal_beta       = 0.0;
    ps->cost               = 0.0;
    ps->setup_pattern_time = 0.0;
    ps->setup_values_time  = 0.0;
    ps->numb               = NULL;
    ps->M                  = NULL;
    ps->comm               = comm;
    ps->beg_row            = beg_row;
    ps->end_row            = end_row;

    hypre_MPI_Comm_size(comm, &npes);

    ps->beg_rows = hypre_TAlloc(HYPRE_Int, npes , HYPRE_MEMORY_HOST);
    ps->end_rows = hypre_TAlloc(HYPRE_Int, npes , HYPRE_MEMORY_HOST);

    hypre_MPI_Allgather(&beg_row, 1, HYPRE_MPI_INT, ps->beg_rows, 1, HYPRE_MPI_INT, comm);
    hypre_MPI_Allgather(&end_row, 1, HYPRE_MPI_INT, ps->end_rows, 1, HYPRE_MPI_INT, comm);

    return ps;
}

/*--------------------------------------------------------------------------
 * ParaSailsDestroy - Deallocate a ParaSails data structure.
 *--------------------------------------------------------------------------*/

void ParaSailsDestroy(ParaSails *ps)
{
    if (ps == NULL)
        return;

    if (ps->numb)
        NumberingDestroy(ps->numb);

    if (ps->M)
        MatrixDestroy(ps->M);

    hypre_TFree(ps->beg_rows,HYPRE_MEMORY_HOST);
    hypre_TFree(ps->end_rows,HYPRE_MEMORY_HOST);

    hypre_TFree(ps,HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------------
 * ParaSailsSetupPattern - Set up a pattern for the ParaSails preconditioner.
 *--------------------------------------------------------------------------*/

void ParaSailsSetupPattern(ParaSails *ps, Matrix *A,
  HYPRE_Real thresh, HYPRE_Int num_levels)
{
    DiagScale  *diag_scale;
    PrunedRows *pruned_rows;
    HYPRE_Real time0, time1;

    time0 = hypre_MPI_Wtime();

    ps->thresh     = thresh;
    ps->num_levels = num_levels;

    if (ps->numb) NumberingDestroy(ps->numb);
    ps->numb = NumberingCreateCopy(A->numb);

    if (ps->M) MatrixDestroy(ps->M);
    ps->M = MatrixCreate(ps->comm, ps->beg_row, ps->end_row);

    diag_scale = DiagScaleCreate(A, A->numb);

    if (ps->thresh < 0.0)
        ps->thresh = SelectThresh(ps->comm, A, diag_scale, -ps->thresh);

    pruned_rows = PrunedRowsCreate(A, PARASAILS_NROWS, diag_scale, ps->thresh);

    ExchangePrunedRows(ps->comm, A, ps->numb, pruned_rows, ps->num_levels);

    ConstructPatternForEachRow(ps->symmetric, pruned_rows, ps->num_levels,
        ps->numb, ps->M, &ps->cost);

    DiagScaleDestroy(diag_scale);
    PrunedRowsDestroy(pruned_rows);

    time1 = hypre_MPI_Wtime();
    ps->setup_pattern_time = time1 - time0;
}

/*--------------------------------------------------------------------------
 * ParaSailsSetupPatternExt - Set up a pattern for the ParaSails preconditioner.
 * Extended version.
 *--------------------------------------------------------------------------*/

void ParaSailsSetupPatternExt(ParaSails *ps, Matrix *A,
  HYPRE_Real thresh_global, HYPRE_Real thresh_local, HYPRE_Int num_levels)
{
    DiagScale  *diag_scale;
    PrunedRows *pruned_rows_global;
    PrunedRows *pruned_rows_local;
    HYPRE_Real time0, time1;

    time0 = hypre_MPI_Wtime();

    ps->thresh     = thresh_global*1000000.+thresh_local; /* dummy */
    ps->num_levels = num_levels;

    if (ps->numb) NumberingDestroy(ps->numb);
    ps->numb = NumberingCreateCopy(A->numb);

    if (ps->M) MatrixDestroy(ps->M);
    ps->M = MatrixCreate(ps->comm, ps->beg_row, ps->end_row);

    diag_scale = DiagScaleCreate(A, A->numb);

    if (ps->thresh < 0.0)
        ps->thresh = SelectThresh(ps->comm, A, diag_scale, -ps->thresh);

    pruned_rows_global = PrunedRowsCreate(A, PARASAILS_NROWS, diag_scale, 
         thresh_global);
    pruned_rows_local = PrunedRowsCreate(A, PARASAILS_NROWS, diag_scale, 
         thresh_local);

    ExchangePrunedRowsExt(ps->comm, A, ps->numb, 
        pruned_rows_global, pruned_rows_local, ps->num_levels);

    ExchangePrunedRowsExt2(ps->comm, A, ps->numb, 
        pruned_rows_global, pruned_rows_local, ps->num_levels);

    ConstructPatternForEachRowExt(ps->symmetric, pruned_rows_global,
	pruned_rows_local, ps->num_levels, ps->numb, ps->M, &ps->cost);

    DiagScaleDestroy(diag_scale);
    PrunedRowsDestroy(pruned_rows_global);
    PrunedRowsDestroy(pruned_rows_local);

    time1 = hypre_MPI_Wtime();
    ps->setup_pattern_time = time1 - time0;
}

/*--------------------------------------------------------------------------
 * ParaSailsSetupValues - Compute the numerical values of the ParaSails
 * preconditioner, for the pattern set up using ParaSailsSetupPattern.
 * This function may be called repeatedly with different input matrices
 * "A", for which a preconditioner is constructed.
 *--------------------------------------------------------------------------*/

HYPRE_Int ParaSailsSetupValues(ParaSails *ps, Matrix *A, HYPRE_Real filter)
{
    LoadBal    *load_bal;
    StoredRows *stored_rows;
    HYPRE_Int row, len, *ind;
    HYPRE_Real *val;
    HYPRE_Int i;
    HYPRE_Real time0, time1;
    MPI_Comm comm = ps->comm;
    HYPRE_Int error = 0, error_sum;

    time0 = hypre_MPI_Wtime();

    /*
     * If the preconditioner matrix has its own numbering object, then we
     * assume it is in its own local numbering, and we change the numbering
     * in the matrix to the ParaSails numbering.
     */

    if (ps->M->numb != NULL)
    {
        /* Make a new numbering object in case pattern of A has changed */
        if (ps->numb) NumberingDestroy(ps->numb);
        ps->numb = NumberingCreateCopy(A->numb);

        for (row=0; row<=ps->M->end_row - ps->M->beg_row; row++)
        {
           MatrixGetRow(ps->M, row, &len, &ind, &val);
           NumberingLocalToGlobal(ps->M->numb, len, ind, ind);
           NumberingGlobalToLocal(ps->numb,    len, ind, ind);
        }
    }

    load_bal = LoadBalDonate(ps->comm, ps->M, ps->numb, ps->cost,
        ps->loadbal_beta);

    stored_rows = StoredRowsCreate(A, PARASAILS_NROWS);

    ExchangeStoredRows(ps->comm, A, ps->M, ps->numb, stored_rows, load_bal);

    if (ps->symmetric)
    {
        error += 
          ComputeValuesSym(stored_rows, ps->M, load_bal->beg_row, ps->numb,
            ps->symmetric);

        for (i=0; i<load_bal->num_taken; i++)
        {
            error += ComputeValuesSym(stored_rows,
                load_bal->recip_data[i].mat,
                load_bal->recip_data[i].mat->beg_row, ps->numb,
                ps->symmetric);
        }
    }
    else
    {
        error += 
          ComputeValuesNonsym(stored_rows, ps->M, load_bal->beg_row, ps->numb);

        for (i=0; i<load_bal->num_taken; i++)
        {
            error += ComputeValuesNonsym(stored_rows,
                load_bal->recip_data[i].mat,
                load_bal->recip_data[i].mat->beg_row, ps->numb);
        }
    }

    time1 = hypre_MPI_Wtime();
    ps->setup_values_time = time1 - time0;

    LoadBalReturn(load_bal, ps->comm, ps->M);

    /* check if there was an error in computing the approximate inverse */
    hypre_MPI_Allreduce(&error, &error_sum, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
    if (error_sum != 0)
    {
        hypre_printf("Hypre-ParaSails detected a problem.  The input matrix\n");
        hypre_printf("may not be full-rank, or if you are using the SPD version,\n");
        hypre_printf("the input matrix may not be positive definite.\n");
        hypre_printf("This error is being returned to the calling function.\n");
        return error_sum;
    }

    /* Filtering */

    ps->filter = filter;

    if (ps->filter != 0.0)
    {
        DiagScale *diag_scale = DiagScaleCreate(A, ps->numb);
        Matrix    *filtered_matrix = MatrixCreate(ps->comm,
                                         ps->beg_row, ps->end_row);

        if (ps->filter < 0.0)
            ps->filter = SelectFilter(ps->comm, ps->M, diag_scale, -ps->filter,
                ps->symmetric);

        FilterValues(ps->M, filtered_matrix, diag_scale, ps->filter,
            ps->symmetric, &ps->cost);

        DiagScaleDestroy(diag_scale);
        MatrixDestroy(ps->M);
        ps->M = filtered_matrix;

        /* Rescale if factored preconditioner */
        if (ps->symmetric != 0)
            Rescale(ps->M, stored_rows, ps->numb->num_ind);
    }

    /*
     * If the preconditioner matrix has its own numbering object, then we
     * change the numbering in the matrix to this numbering.  If not, then
     * we put the preconditioner matrix in global numbering, and call
     * MatrixComplete (to create numbering object, convert the indices,
     * and create the matvec info).
     */

    if (ps->M->numb != NULL)
    {
        /* Convert to own numbering system */
        for (row=0; row<=ps->M->end_row - ps->M->beg_row; row++)
        {
            MatrixGetRow(ps->M, row, &len, &ind, &val);
            NumberingLocalToGlobal(ps->numb,    len, ind, ind);
            NumberingGlobalToLocal(ps->M->numb, len, ind, ind);
        }
    }
    else
    {
        /* Convert to global numbering system and call MatrixComplete */
        for (row=0; row<=ps->M->end_row - ps->M->beg_row; row++)
        {
            MatrixGetRow(ps->M, row, &len, &ind, &val);
            NumberingLocalToGlobal(ps->numb, len, ind, ind);
        }

        MatrixComplete(ps->M);
    }

    StoredRowsDestroy(stored_rows);

    return 0;
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

void ParaSailsApply(ParaSails *ps, HYPRE_Real *u, HYPRE_Real *v)
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

/*--------------------------------------------------------------------------
 * ParaSailsApplyTrans - Apply the ParaSails preconditioner, transposed
 *
 * ps - input ParaSails object
 * u  - input array of doubles
 * v  - output array of doubles
 *
 * Although this computation can be done in place, it typically will not
 * be used this way, since the caller usually needs to preserve the input
 * vector.
 *--------------------------------------------------------------------------*/

void ParaSailsApplyTrans(ParaSails *ps, HYPRE_Real *u, HYPRE_Real *v)
{
    if (ps->symmetric)
    {
        MatrixMatvec(ps->M, u, v);      /* need to preserve u */
        MatrixMatvecTrans(ps->M, v, v); /* do the second mult in place */
    }
    else
    {
        MatrixMatvecTrans(ps->M, u, v);
    }
}

/*--------------------------------------------------------------------------
 * ParaSailsStatsPattern - Print some statistics about ParaSailsSetupPattern.
 * Returns a cost, which can be used to preempt ParaSailsSetupValues if the
 * cost is too high.
 *--------------------------------------------------------------------------*/

HYPRE_Real ParaSailsStatsPattern(ParaSails *ps, Matrix *A)
{
    HYPRE_Int mype, npes;
    HYPRE_Int n, nnzm, nnza;
    MPI_Comm comm = ps->comm;
    HYPRE_Real max_pattern_time, max_cost, ave_cost;

    hypre_MPI_Comm_rank(comm, &mype);
    hypre_MPI_Comm_size(comm, &npes);

    nnzm = MatrixNnz(ps->M);
    nnza = MatrixNnz(A);
    if (ps->symmetric)
    {
        n = ps->end_rows[npes-1] - ps->beg_rows[0] + 1;
	nnza = (nnza - n) / 2 + n;
    }

    hypre_MPI_Allreduce(&ps->setup_pattern_time, &max_pattern_time, 
	1, hypre_MPI_REAL, hypre_MPI_MAX, comm);
    hypre_MPI_Allreduce(&ps->cost, &max_cost, 1, hypre_MPI_REAL, hypre_MPI_MAX, comm);
    hypre_MPI_Allreduce(&ps->cost, &ave_cost, 1, hypre_MPI_REAL, hypre_MPI_SUM, comm);
    ave_cost = ave_cost / (HYPRE_Real) npes;

    if (mype)
	return ave_cost;

    if (ps->symmetric == 0)
        max_cost *= 8.0;  /* nonsymmetric method is harder */

    hypre_printf("** ParaSails Setup Pattern Statistics ***********\n");
    hypre_printf("symmetric             : %d\n", ps->symmetric);
    hypre_printf("thresh                : %f\n", ps->thresh);
    hypre_printf("num_levels            : %d\n", ps->num_levels);
    hypre_printf("Max cost (average)    : %7.1e (%7.1e)\n", max_cost, ave_cost);
    hypre_printf("Nnz (ratio)           : %d (%5.2f)\n", nnzm, nnzm/(HYPRE_Real)nnza);
    hypre_printf("Max setup pattern time: %8.1f\n", max_pattern_time);
    hypre_printf("*************************************************\n");
    fflush(stdout);

    return ave_cost;
}

/*--------------------------------------------------------------------------
 * ParaSailsStatsValues - Print some statistics about ParaSailsSetupValues.
 *--------------------------------------------------------------------------*/

void ParaSailsStatsValues(ParaSails *ps, Matrix *A)
{
    HYPRE_Int mype, npes;
    HYPRE_Int n, nnzm, nnza;
    MPI_Comm comm = ps->comm;
    HYPRE_Real max_values_time;
    HYPRE_Real temp, *setup_times = NULL;
    HYPRE_Int i;

    hypre_MPI_Comm_rank(comm, &mype);
    hypre_MPI_Comm_size(comm, &npes);

    nnzm = MatrixNnz(ps->M);
    nnza = MatrixNnz(A);
    if (ps->symmetric)
    {
        n = ps->end_rows[npes-1] - ps->beg_rows[0] + 1;
        nnza = (nnza - n) / 2 + n;
    }

    hypre_MPI_Allreduce(&ps->setup_values_time, &max_values_time, 
	1, hypre_MPI_REAL, hypre_MPI_MAX, comm);

    if (!mype)
        setup_times = hypre_TAlloc(HYPRE_Real, npes , HYPRE_MEMORY_HOST);

    temp = ps->setup_pattern_time + ps->setup_values_time;
    hypre_MPI_Gather(&temp, 1, hypre_MPI_REAL, setup_times, 1, hypre_MPI_REAL, 0, comm);

    if (mype)
        return;

    hypre_printf("** ParaSails Setup Values Statistics ************\n");
    hypre_printf("filter                : %f\n", ps->filter);
    hypre_printf("loadbal               : %f\n", ps->loadbal_beta);
    hypre_printf("Final Nnz (ratio)     : %d (%5.2f)\n", nnzm, nnzm/(HYPRE_Real)nnza);
    hypre_printf("Max setup values time : %8.1f\n", max_values_time);
    hypre_printf("*************************************************\n");
    hypre_printf("Setup (pattern and values) times:\n");

    temp = 0.0;
    for (i=0; i<npes; i++)
    {
        hypre_printf("%3d: %8.1f\n", i, setup_times[i]);
        temp += setup_times[i];
    }
    hypre_printf("ave: %8.1f\n", temp / (HYPRE_Real) npes);
    hypre_printf("*************************************************\n");

    hypre_TFree(setup_times,HYPRE_MEMORY_HOST);

    fflush(stdout);
}

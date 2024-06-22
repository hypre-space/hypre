/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * LoadBal - Load balancing module for ParaSails.
 *
 *****************************************************************************/

#include <stdlib.h>
#include "Common.h"
#include "Matrix.h"
#include "Numbering.h"
#include "LoadBal.h"

/*--------------------------------------------------------------------------
 * LoadBalInit - determine the amount of work to be donated and received by
 * each processor, given the amount of work that each processor has
 * ("local_cost").  The number of processors that this processor will donate
 * to is "num_given" and the number of processors from which this processor
 * will receive is "num_taken".  Additional donor information is stored in
 * "donor_data_pe" and "donor_data_cost".
 *
 * local_cost - amount of work that this processor has
 * beta - target load balance factor
 *--------------------------------------------------------------------------*/

void LoadBalInit(MPI_Comm comm, HYPRE_Real local_cost, HYPRE_Real beta,
  HYPRE_Int *num_given, HYPRE_Int *donor_data_pe, HYPRE_Real *donor_data_cost,
  HYPRE_Int *num_taken)
{
    HYPRE_Int mype, npes;
    HYPRE_Real *cost, average, upper, move, accept;
    HYPRE_Int i, jj, j;

    *num_given = 0;
    *num_taken = 0;

    if (beta == 0.0)
	return;

    hypre_MPI_Comm_rank(comm, &mype);
    hypre_MPI_Comm_size(comm, &npes);

    cost = hypre_TAlloc(HYPRE_Real, npes , HYPRE_MEMORY_HOST);

    hypre_MPI_Allgather(&local_cost, 1, hypre_MPI_REAL, cost, 1, hypre_MPI_REAL, comm);

    /* Compute the average cost */
    average = 0.0;
    for (i=0; i<npes; i++)
        average += cost[i];
    average = average / npes;

    /* Maximum cost allowed by load balancer */
    upper = average / beta;

    for (i=0; i<npes; i++)
    {
        if (cost[i] > upper)
        {
            move = cost[i] - upper;

            /* for j=[i+1:n 1:i-1] */
            for (jj=i+1; jj<=i+npes; jj++)
            {
		j = jj % npes;
		if (j == i)
		    continue;

                if (cost[j] < average)
                {
                    accept = upper - cost[j];

                    /* If we are sender, record it */
                    if (mype == i)
                    {
                        donor_data_pe[*num_given] = j;
                        donor_data_cost[*num_given] = MIN(move, accept);
                        (*num_given)++;
                    }

                    /* If we are receiver, record it */
                    if (mype == j)
                    {
                        (*num_taken)++;
                    }

                    if (move <= accept)
                    {
                        cost[i] = cost[i] - move;
                        cost[j] = cost[j] + move;
#ifdef PARASAILS_DEBUG
			if (mype == 0)
                            hypre_printf("moved from %d to %d (%7.1e)\n", i,j,move);
#endif
                        /*nummoves = nummoves + 1;*/
                        break;
                    }
                    else
                    {
                        cost[i] = cost[i] - accept;
                        cost[j] = cost[j] + accept;
#ifdef PARASAILS_DEBUG
			if (mype == 0)
                            hypre_printf("moved from %d to %d (%7.1e)\n", i,j,accept);
#endif
                        /*nummoves = nummoves + 1;*/
                        move = cost[i] - upper;
                    }
                }
            }
        }
    }

    hypre_TFree(cost,HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------------
 * LoadBalDonorSend - send the indices of the donated rows.
 * The message structure is: beg_row, end_row, len1, indices1, len2, ....
 * Caller must free the allocated buffers.
 *--------------------------------------------------------------------------*/

void LoadBalDonorSend(MPI_Comm comm, Matrix *mat, Numbering *numb,
  HYPRE_Int num_given, const HYPRE_Int *donor_data_pe, const HYPRE_Real *donor_data_cost,
  DonorData *donor_data, HYPRE_Int *local_beg_row, hypre_MPI_Request *request)
{
    HYPRE_Int send_beg_row, send_end_row;
    HYPRE_Int i, row;
    HYPRE_Real accum;
    HYPRE_Int buflen;
    HYPRE_Int *bufferp;
    HYPRE_Int len, *ind;
    HYPRE_Real *val;

    send_end_row = mat->beg_row - 1; /* imaginary end of previous block */

    for (i=0; i<num_given; i++)
    {
	send_beg_row = send_end_row + 1;
        send_end_row = send_beg_row - 1;

        /* Portion out rows that add up to the workload to be sent out */
	/* and determine the size of the buffer needed */

        accum = 0.0; /* amount of work portioned out so far */
        buflen = 2;  /* front of buffer will contain beg_row, end_row */

        do
        {
            send_end_row++;
            hypre_assert(send_end_row <= mat->end_row);
            MatrixGetRow(mat, send_end_row - mat->beg_row, &len, &ind, &val);
            accum += (HYPRE_Real) len*len*len;
            buflen += (len+1); /* additional one for row length */
        }
        while (accum < donor_data_cost[i]);

        /* Create entry in donor_data structure */

        donor_data[i].pe      = donor_data_pe[i];
        donor_data[i].beg_row = send_beg_row;
        donor_data[i].end_row = send_end_row;
        donor_data[i].buffer  = hypre_TAlloc(HYPRE_Int, (buflen) , HYPRE_MEMORY_HOST);

	/* Construct send buffer */

         bufferp   = donor_data[i].buffer;
        *bufferp++ = send_beg_row;
        *bufferp++ = send_end_row;

        for (row=send_beg_row; row<=send_end_row; row++)
        {
            MatrixGetRow(mat, row - mat->beg_row, &len, &ind, &val);
            *bufferp++ = len;
            /* memcpy(bufferp, ind, len*sizeof(HYPRE_Int)); */ /* copy into buffer */
	    NumberingLocalToGlobal(numb, len, ind, bufferp);
            bufferp += len;
        }

        hypre_MPI_Isend(donor_data[i].buffer, buflen, HYPRE_MPI_INT, donor_data[i].pe,
            LOADBAL_REQ_TAG, comm, &request[i]);
    }

    *local_beg_row = send_end_row + 1;
}

/*--------------------------------------------------------------------------
 * LoadBalRecipRecv - receive the indices of the donated rows.
 * The message structure is: beg_row, end_row, len1, indices1, len2, ....
 *--------------------------------------------------------------------------*/

void LoadBalRecipRecv(MPI_Comm comm, Numbering *numb,
  HYPRE_Int num_taken, RecipData *recip_data)
{
    HYPRE_Int i, row;
    HYPRE_Int count;
    hypre_MPI_Status status;
    HYPRE_Int *buffer, *bufferp;
    HYPRE_Int beg_row, end_row;
    HYPRE_Int len;

    for (i=0; i<num_taken; i++)
    {
        hypre_MPI_Probe(hypre_MPI_ANY_SOURCE, LOADBAL_REQ_TAG, comm, &status);
        recip_data[i].pe = status.hypre_MPI_SOURCE;
        hypre_MPI_Get_count(&status, HYPRE_MPI_INT, &count);

        buffer = hypre_TAlloc(HYPRE_Int, count , HYPRE_MEMORY_HOST);
        hypre_MPI_Recv(buffer, count, HYPRE_MPI_INT, recip_data[i].pe, LOADBAL_REQ_TAG,
           comm, &status);

	bufferp =  buffer;
        beg_row = *bufferp++;
        end_row = *bufferp++;

        recip_data[i].mat = MatrixCreateLocal(beg_row, end_row);

	/* Set the indices of the local matrix containing donated rows */

        for (row=beg_row; row<=end_row; row++)
        {
            len = *bufferp++;
	    NumberingGlobalToLocal(numb, len, bufferp, bufferp);
            MatrixSetRow(recip_data[i].mat, row, len, bufferp, NULL);
            bufferp += len;
        }

	hypre_TFree(buffer,HYPRE_MEMORY_HOST);
    }
}

/*--------------------------------------------------------------------------
 * LoadBalRecipSend - send back the computed values of the donated rows.
 * Traverse all the donated local matrices.
 * Assume indices are in the same order.
 * Caller must free the allocated buffers.
 *--------------------------------------------------------------------------*/

void LoadBalRecipSend(MPI_Comm comm, HYPRE_Int num_taken,
  RecipData *recip_data, hypre_MPI_Request *request)
{
    HYPRE_Int i, row, buflen;
    HYPRE_Real *bufferp;
    Matrix *mat;
    HYPRE_Int len, *ind;
    HYPRE_Real *val;

    for (i=0; i<num_taken; i++)
    {
        mat = recip_data[i].mat;

        /* Find size of output buffer */
	buflen = 0;
        for (row=0; row<=mat->end_row - mat->beg_row; row++)
        {
            MatrixGetRow(mat, row, &len, &ind, &val);
	    buflen += len;
	}

	recip_data[i].buffer = hypre_TAlloc(HYPRE_Real, buflen , HYPRE_MEMORY_HOST);

	/* Construct send buffer */

	bufferp = recip_data[i].buffer;
        for (row=0; row<=mat->end_row - mat->beg_row; row++)
        {
            MatrixGetRow(mat, row, &len, &ind, &val);
            hypre_TMemcpy(bufferp,  val, HYPRE_Real, len, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST); /* copy into buffer */
            bufferp += len;
        }

        hypre_MPI_Isend(recip_data[i].buffer, buflen, hypre_MPI_REAL, recip_data[i].pe,
            LOADBAL_REP_TAG, comm, &request[i]);

        MatrixDestroy(mat);
    }
}

/*--------------------------------------------------------------------------
 * LoadBalDonorRecv - receive the computed values of the donated rows.
 * Traverse all the donated local matrices.
 * Assume indices are in the same order.
 *--------------------------------------------------------------------------*/

void LoadBalDonorRecv(MPI_Comm comm, Matrix *mat,
  HYPRE_Int num_given, DonorData *donor_data)
{
    HYPRE_Int i, j, row;
    HYPRE_Int source, count;
    hypre_MPI_Status status;
    HYPRE_Real *buffer, *bufferp;
    HYPRE_Int len, *ind;
    HYPRE_Real *val;

    for (i=0; i<num_given; i++)
    {
        hypre_MPI_Probe(hypre_MPI_ANY_SOURCE, LOADBAL_REP_TAG, comm, &status);
        source = status.hypre_MPI_SOURCE;
        hypre_MPI_Get_count(&status, hypre_MPI_REAL, &count);

        buffer = hypre_TAlloc(HYPRE_Real, count , HYPRE_MEMORY_HOST);
        hypre_MPI_Recv(buffer, count, hypre_MPI_REAL, source, LOADBAL_REP_TAG,
           comm, &status);

	/* search for which entry in donor_data this message corresponds to */
	for (j=0; j<num_given; j++)
	{
	    if (donor_data[j].pe == source)
		break;
	}
	hypre_assert(j < num_given);

        /* Parse the message and put row values into local matrix */
	bufferp = buffer;
        for (row=donor_data[j].beg_row; row<=donor_data[j].end_row; row++)
        {
            MatrixGetRow(mat, row - mat->beg_row, &len, &ind, &val);
			hypre_TMemcpy(val,  bufferp, HYPRE_Real, len, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST); /* copy into matrix */
            bufferp += len;
        }

	hypre_TFree(buffer,HYPRE_MEMORY_HOST);
    }
}

/*--------------------------------------------------------------------------
 * LoadBalDonate
 *--------------------------------------------------------------------------*/

LoadBal *LoadBalDonate(MPI_Comm comm, Matrix *mat, Numbering *numb,
  HYPRE_Real local_cost, HYPRE_Real beta)
{
    LoadBal *p;
    HYPRE_Int i, npes;
    HYPRE_Int    *donor_data_pe;
    HYPRE_Real *donor_data_cost;
    hypre_MPI_Request *requests = NULL;
    hypre_MPI_Status  *statuses = NULL;

    p = hypre_TAlloc(LoadBal, 1, HYPRE_MEMORY_HOST);

    hypre_MPI_Comm_size(comm, &npes);

    donor_data_pe   = hypre_TAlloc(HYPRE_Int, npes , HYPRE_MEMORY_HOST);
    donor_data_cost = hypre_TAlloc(HYPRE_Real, npes , HYPRE_MEMORY_HOST);

    LoadBalInit(comm, local_cost, beta, &p->num_given,
        donor_data_pe, donor_data_cost, &p->num_taken);

    p->recip_data = NULL;
    p->donor_data = NULL;

    if (p->num_taken)
        p->recip_data = hypre_TAlloc(RecipData, p->num_taken , HYPRE_MEMORY_HOST);

    if (p->num_given)
    {
        p->donor_data = hypre_TAlloc(DonorData, p->num_given , HYPRE_MEMORY_HOST);
        requests = hypre_TAlloc(hypre_MPI_Request, p->num_given , HYPRE_MEMORY_HOST);
        statuses = hypre_TAlloc(hypre_MPI_Status, p->num_given , HYPRE_MEMORY_HOST);
    }

    LoadBalDonorSend(comm, mat, numb, p->num_given,
        donor_data_pe, donor_data_cost, p->donor_data, &p->beg_row, requests);

    hypre_TFree(donor_data_pe,HYPRE_MEMORY_HOST);
    hypre_TFree(donor_data_cost,HYPRE_MEMORY_HOST);

    LoadBalRecipRecv(comm, numb, p->num_taken, p->recip_data);

    hypre_MPI_Waitall(p->num_given, requests, statuses);

    hypre_TFree(requests,HYPRE_MEMORY_HOST);
    hypre_TFree(statuses,HYPRE_MEMORY_HOST);

    /* Free the send buffers which were allocated by LoadBalDonorSend */
    for (i=0; i<p->num_given; i++)
	hypre_TFree(p->donor_data[i].buffer,HYPRE_MEMORY_HOST);

    return p;
}

/*--------------------------------------------------------------------------
 * LoadBalReturn
 *--------------------------------------------------------------------------*/

void LoadBalReturn(LoadBal *p, MPI_Comm comm, Matrix *mat)
{
    HYPRE_Int i;

    hypre_MPI_Request *requests = NULL;
    hypre_MPI_Status  *statuses = NULL;

    if (p->num_taken)
    {
        requests = hypre_TAlloc(hypre_MPI_Request, p->num_taken , HYPRE_MEMORY_HOST);
        statuses = hypre_TAlloc(hypre_MPI_Status, p->num_taken , HYPRE_MEMORY_HOST);
    }

    LoadBalRecipSend(comm, p->num_taken, p->recip_data, requests);

    LoadBalDonorRecv(comm, mat, p->num_given, p->donor_data);

    hypre_MPI_Waitall(p->num_taken, requests, statuses);

    hypre_TFree(requests,HYPRE_MEMORY_HOST);
    hypre_TFree(statuses,HYPRE_MEMORY_HOST);

    /* Free the send buffers which were allocated by LoadBalRecipSend */
    for (i=0; i<p->num_taken; i++)
	hypre_TFree(p->recip_data[i].buffer,HYPRE_MEMORY_HOST);

    hypre_TFree(p->donor_data,HYPRE_MEMORY_HOST);
    hypre_TFree(p->recip_data,HYPRE_MEMORY_HOST);

    hypre_TFree(p,HYPRE_MEMORY_HOST);
}


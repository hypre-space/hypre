/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

// ParaSAILS - Parallel Sparse Approximate Inverse, Least Squares
// Version 2 (multithreaded)
//
// Center for Applied Scientific Computing
// Lawrence Livermore National Laboratory
// December 1998
//
// Hypre version: August 1999

#include <iostream.h>
#include <iomanip.h>
#include <fstream.h>
#include <string.h>
#include <stdio.h>

#include "mpi.h"

#include "HYPRE_distributed_matrix_types.h"
#include "HYPRE_distributed_matrix_protos.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE.h" // needed for HYPRE_PARCSR

//#include "BlasLapack.h"
#include "HYPRE_ParaSAILS.h"
#include "ParaSAILS.h"

int num_threads_workers = 4;
int num_threads_servers = 1;

extern "C" void esvdgells(int, void *, int, void *, int, void *, int, double *,
                 double, int, int, int, int *, double *, int *);


#if 0
#if 1
#include <essl.h>
#else
extern "C" int dgels(char* ch, int*, int*, int*,
    double*, int*, double*, int*, double*, int*, int*);
#endif
#endif

#include "util.h" // new


#define MAX(a,b) ((a)>(b)?(a):(b))
#define ABS(a) ((a)>=0?(a):-(a))

#define SERVER_TAG              1000

typedef struct
{
    int index;  // index into list_request containing row numbers
    int size;   // number of rows requested
} RequestData;


ParaSAILS::ParaSAILS(const HYPRE_DistributedMatrix& mat)
{
    int ierr;

    A = mat; // store matrix in object

    ierr = HYPRE_GetDistributedMatrixDims(A, &n, &n); // assumes square matrix
    assert(!ierr);
    comm = HYPRE_GetDistributedMatrixContext(A);
    ierr = HYPRE_GetDistributedMatrixLocalRange(A, &my_start_row, &my_end_row);
    assert(!ierr);
    my_start_row++; // convert to 1-based indexing
    my_end_row++;

    assert(0); // UNDONE
    //start_rows = 1;
    //end_rows = n;

    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &myid);

    ierr = HYPRE_NewIJMatrix(comm, &M, n, n); // store matrix in object
    assert(!ierr);
    ierr = HYPRE_SetIJMatrixLocalStorageType(M, HYPRE_PARCSR);
    assert(!ierr);
    ierr = HYPRE_SetIJMatrixLocalSize(M, my_end_row-my_start_row+1, n);
    assert(!ierr);
    ierr = HYPRE_InitializeIJMatrix(M);
    assert(!ierr);

    // default values
    nlevels = 0;
    thresh = 0.0;
    lfil = n;
    prune_alg = PRUNE_NO;
    dump = 0;
}


// only processor 0 returns the correct result
int ParaSAILS::nnz()
{
// UNDONE: should get the underlying parcsr matrix to find out row lengths
#if 0
    int i, nnz, total;

    nnz = 0;
    for (i=my_start_row; i<=my_end_row; i++)
        nnz += M->rowLength(i);

    MPI_Reduce(&nnz, &total, 1, MPI_INT, MPI_SUM, 0, comm);

    if (myid == 0)
        return total;
#endif
    return -1;
}

// sorts x[0:n-1] in place, ascending order

void shell_sort(const int n, int x[])
{
    int m, max, j, k, itemp;

    m = n/2;

    while (m > 0) {
        max = n - m;
        for (j=0; j<max; j++)
        {
            for (k=j; k>=0; k-=m)
            {
                if (x[k+m] >= x[k])
                    break;
                itemp = x[k+m];
                x[k+m] = x[k];
                x[k] = itemp;
            }
        }
        m = m/2;
    }
}

void reverse_shell_sort(const int n, int x[])
{
    int m, max, j, k, itemp;

    m = n/2;

    while (m > 0) {
        max = n - m;
        for (j=0; j<max; j++)
        {
            for (k=j; k>=0; k-=m)
            {
                if (x[k+m] <= x[k])  // only line different
                    break;
                itemp = x[k+m];
                x[k+m] = x[k];
                x[k] = itemp;
            }
        }
        m = m/2;
    }
}

// randomize a list
// modified from shell sort

void sort_rand(const int n, double x[], int p[])
{
    int m, max, j, k, ptemp;
    double xtemp;

    m = n/2;

    while (m > 0) {
        max = n - m;
        for (j=0; j<max; j++)
        {
            for (k=j; k>=0; k-=m)
            {
                if (x[k+m] >= x[k])
                    break;
                xtemp = x[k+m];
                x[k+m] = x[k];
                x[k] = xtemp;
                ptemp = p[k+m];
                p[k+m] = p[k];
                p[k] = ptemp;
            }
        }
        m = m/2;
    }
}

// n is number of elements
// ncut is number desired

void qsplit(int rownum, double *a, int *ind, int n, int ncut)
{
        double tmp, abskey;
        int itmp, first, last, mid;

        // modification to not drop diagonal element
        for (int i=0; i<n; i++)
        {
            if (ind[i] == rownum)
            {
                ind[i] = ind[0];
                ind[0] = rownum;
                tmp = a[i];
                a[i] = a[0];
                a[0] = tmp;
                break;
            }
        }
        if (*ind == rownum) // make sure there was a diagonal
        {
            ncut--;
            n--;
            a++;
            ind++;
        }
        // else there was not a diagonal in the first place... ignore for now
        // end of modification

        ncut--;
        first = 0;
        last = n-1;
        if (ncut < first || ncut > last) return;

//    outer loop -- while mid .ne. ncut do

      while (1)
      {
        mid = first;
        abskey = ABS(a[mid]);
        for (int j=first+1; j<=last; j++)
        {
           if (ABS(a[j]) > abskey)
           {
              mid = mid+1;
              // interchange
              tmp = a[mid];
              itmp = ind[mid];
              a[mid] = a[j];
              ind[mid] = ind[j];
              a[j]  = tmp;
              ind[j] = itmp;
           }
        }

//      interchange

        tmp = a[mid];
        a[mid] = a[first];
        a[first]  = tmp;

        itmp = ind[mid];
        ind[mid] = ind[first];
        ind[first] = itmp;
//      test for while loop

        if (mid == ncut) return;
        if (mid > ncut)
           last = mid-1;
        else
           first = mid+1;

     } // endwhile
}

void ParaSAILS::dump_pattern()
{
    // UNDONE needs to get underlying parcsr matrix
#if 0

    const double *val;
    const int *ind;
    int len, row, i;
    char filename[40];

    sprintf(filename, "M_matrix.%d", myid);
    ofstream outfile(filename);
    outfile.precision(14);
    outfile.setf(ios::scientific,ios::floatfield);

    for (row=my_start_row; row<=my_end_row; row++)
    {
        val = M->getPointerToCoef(len, row);
        ind = M->getPointerToColIndex(len, row);

        for (i=0; i<len; i++)
            outfile << setw(6) << row << setw(6) << ind[i]
                << setw(24) << val[i] << endl;
    }
    outfile.close();

#endif
}

//////////////////////////////////////////
int row_to_pe(SharedData *shared, int row)
{
    int pe;
    int *start = shared->start_rows;
    int *end   = shared->end_rows;

    for (pe=0; pe<shared->npes; pe++)
    {
        if (row >= start[pe] && row <= end[pe])
            return pe;
    }

    printf("row to pe failed %d \n", row);
    assert(0);
    return -1;
}

void init_get_row_rand(SharedData *shared)
{
    int num_rows = shared->my_end_row - shared->my_start_row + 1;
    double *temp = new double[num_rows];
    int i;

    for (i=0; i<num_rows; i++)
    {
	shared->rownums[i] = shared->my_start_row + i;
	temp[i] = (double) rand() / (double) RAND_MAX;
    }

    sort_rand(num_rows, temp, shared->rownums);

    delete temp;
}

// returns -1 if no more rows
int get_row_rand(SharedData *shared)
{
    int job;

    if (shared->current_row > shared->my_end_row)
        return -1;

    pthread_mutex_lock(&shared->job_mutex);
    job = shared->current_row;
    shared->current_row++;
    pthread_mutex_unlock(&shared->job_mutex);

    if (job > shared->my_end_row)
        return -1;

    return shared->rownums[job-shared->my_start_row]; // only line different
}

// returns -1 if no more rows
int get_row(SharedData *shared)
{
    int job;

    if (shared->current_row > shared->my_end_row)
        return -1;

    pthread_mutex_lock(&shared->job_mutex);
    job = shared->current_row;
    shared->current_row++;
    pthread_mutex_unlock(&shared->job_mutex);

    if (job > shared->my_end_row)
        return -1;

    return job;
}

//////////////////////////////////////////

void *server(void *);
void *worker(void *);

typedef struct
{
    int        threadid;
    SharedData *shared;
} LocalData;

void ParaSAILS::calculate()
{
    SharedData shared(this);

    pthread_t wthreads[NUM_THREADS_WORKERS];
    pthread_t sthreads[NUM_THREADS_SERVERS];

    LocalData wlocal[NUM_THREADS_WORKERS];
    LocalData slocal[NUM_THREADS_SERVERS];

    MPI_Request request;
    int i, rc;

    pthread_attr_t attr;
    pthread_attr_init(&attr);

    // need this?
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

//#define IBM_AIX
#ifdef  IBM_AIX
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_UNDETACHED);
#endif

    printf("workers servers %d %d\n", num_threads_workers, num_threads_servers);

    double time0 = MPI_Wtime();

    // create threads for workers
    for (i=0; i<num_threads_workers; i++)
    {
        wlocal[i].threadid = i;
        wlocal[i].shared = &shared;
        rc = pthread_create(&wthreads[i], &attr, worker, (void *)(wlocal+i));
        assert(rc==0);
    }

    // create threads for servicing requests for rows
    for (i=0; i<num_threads_servers; i++)
    {
        slocal[i].threadid = i;
        slocal[i].shared = &shared;
        rc = pthread_create(&sthreads[i], &attr, server, (void *)(slocal+i));
        assert(rc==0);
    }

    // wait for worker threads to complete
    for (i=0; i<num_threads_workers; i++)
    {
        rc = pthread_join(wthreads[i], NULL);
        assert(rc==0);
    }

    time0 = MPI_Wtime() - time0;
    printf("Time for worker threads: %#10.4f\n", time0);

    // wait for all nodes to complete work before shutting down servers
    MPI_Barrier(comm);

    // send message to all servers to tell them to return
    int minusone = -1;
    for (i=0; i<num_threads_servers; i++)
    {
        MPI_Isend(&minusone, 1, MPI_INT, shared.myid, SERVER_TAG, 
            comm, &request);
        MPI_Request_free(&request);
    }

    // wait for server threads to complete
    for (i=0; i<num_threads_servers; i++)
    {
        rc = pthread_join(sthreads[i], NULL);
        assert(rc==0);
    }

    time0 = MPI_Wtime();
    time0 = MPI_Wtime() - time0;
    printf("assemble time: %#10.4f\n", time0);

    if (dump)
        dump_pattern();
}


// DESIGN CONSIDERATION:
// question: do we retrieve the actual rows or do we retrieve
// the pruned rows???  Since we need the actual rows eventually
// retrieve the actual rows....
// THUS we prune every time we load a row, we prune it, and also
// store the pruned row

// needs: prune_alg, threshes, buffer pointers and base pointers
// use a prune_table data structure???

// similarly, use a rows data structure??? that would need access to mpi

inline void prune_row(int row, RowRecord *rec, SharedData *shared,
  int *const prn_space, int *&prn_space_p)
{
    int i;
    const int *ind;
    const double *val;
    double temp_buffer[MAX_PATTERN];

    switch (shared->prune_alg)
    {
        case PRUNE_NO:

            rec->pruned_len = rec->len;
            rec->pruned_ind = (int *) rec->ind;  // breaks const-ness
            break;

        case PRUNE_THRESH:

            assert(prn_space_p - prn_space + rec->len
                < MAX_PRUNE_SPACE);

            rec->pruned_ind = prn_space_p;

            for (i=0, ind=rec->ind, val=rec->val; i<rec->len; i++, ind++, val++)
                if (ABS(*val) >= shared->thresh || *ind == row)
                    *prn_space_p++ = *ind;

            rec->pruned_len = prn_space_p - rec->pruned_ind;
            break;

        case PRUNE_LFIL:

            assert(prn_space_p - prn_space + rec->len
                < MAX_PRUNE_SPACE);

            assert(rec->len <= MAX_PATTERN);

            memcpy(prn_space_p, rec->ind, sizeof(int) * rec->len);
            memcpy(temp_buffer, rec->val, sizeof(double) * rec->len);

            rec->pruned_len = rec->len;
            if (rec->pruned_len > shared->lfil)
            {
                qsplit(row, temp_buffer, prn_space_p, 
                  rec->pruned_len, shared->lfil);
                rec->pruned_len = shared->lfil;
            }
            rec->pruned_ind = prn_space_p;
            prn_space_p += rec->pruned_len;
            break;
    }
}

//////////////////////////////

// server needs the local row records, they are stored contiguously

void *server(void *local)
{
    int threadid = ((LocalData *)local)->threadid;
    SharedData *shared = ((LocalData *)local)->shared;

    MPI_Status  status;
    MPI_Status  statuses[2];
    MPI_Request requests[2];
    int tag, source;

    int len;
    const int *ind;
    const double *val;
    int rows[MAX_ROWS_PER_REQUEST+1];

    int count;
    int i;

    requests[0] = MPI_REQUEST_NULL;
    requests[1] = MPI_REQUEST_NULL;

    int *indbufp, *const indbuf = new int[MAX_SERVER_BUF];
    double *valbufp, *const valbuf = new double[MAX_SERVER_BUF];

    assert(valbuf != NULL);

    // cannot start servicing rows until workers have set things up
    // is this still true???

    barrier_wait(shared->barrier, 0);

    while (1)
    {
        // only allow one server to probe at a time
        // have to receive it before the probe goes away?
        pthread_mutex_lock(&shared->probe_mutex);
        MPI_Probe(MPI_ANY_SOURCE, SERVER_TAG, shared->comm, &status);

        source = status.MPI_SOURCE;

        // printf("server got request from %d\n", source);fflush(stdout);

        // get the number of rows desired
        MPI_Get_count(&status, MPI_INT, &count);

        assert (count-1 < MAX_ROWS_PER_REQUEST);

        // get the indices of the desired rows
        MPI_Recv(rows, count, MPI_INT, source, SERVER_TAG, 
            shared->comm, &status);

        pthread_mutex_unlock(&shared->probe_mutex);

        // last entry in message is the threadid of originating thread
        tag = rows[count-1];
        count--;

        if (tag == -1)
            break;

        // wait until buffer is clear
        MPI_Waitall(2, requests, statuses);

        //
        // construct message in local buffer
        //

        // set the beginnings of these pointers
        indbufp = indbuf + count;
        valbufp = valbuf;

        for (i=0; i<count; i++)
        {
#if 0
            ind = shared->sails->A.getPointerToColIndex(len, rows[i]);
            val = shared->sails->A.getPointerToCoef(len, rows[i]);
#endif
            int ierr;
            ierr = HYPRE_GetDistributedMatrixRow(shared->sails->A, 
              rows[i], &len, (int **) &ind, (double **) &val); // kludge
            assert(!ierr);

            indbuf[i] = len;

            assert(indbufp - indbuf + len < MAX_SERVER_BUF);
            // assert(valbufp - valbuf + len < MAX_SERVER_BUF);

            memcpy(indbufp, ind, sizeof(int)*len);    // row indices
            memcpy(valbufp, val, sizeof(double)*len); // row values
            indbufp += len;
            valbufp += len;
        }

        // tag was set above

        MPI_Isend(indbuf, indbufp - indbuf, MPI_INT,
            source, tag, shared->comm, &requests[0]);
        MPI_Isend(valbuf, valbufp - valbuf, MPI_DOUBLE, 
            source, tag, shared->comm, &requests[1]);

        // printf("sent reply to node %d thread %d\n",
        //   source, tag);fflush(stdout);
    }

    delete indbuf;
    delete valbuf;

    return NULL;
}

void prune_local_rows(int threadid, SharedData *shared, 
  int *prn_space, int *&prn_space_p)
{
    double ave;
    int start, end;
    int row, len;

    // indices are 1-based 
    // printf("%d %d\n", shared->my_start_row, shared->my_end_row);

    // compute range of rows that this thread will prune
    ave = (shared->my_end_row - shared->my_start_row + 1) 
      / (double) num_threads_workers;
    start = (int) (ave*threadid) + shared->my_start_row;
    end   = (int) (ave*(threadid+1)) + shared->my_start_row - 1;
    if (threadid == num_threads_workers-1)
        end = shared->my_end_row;

    // printf("Worker thread %d, start %d end %d\n", threadid, start, end);

    RowRecord *rec;
    for (row=start; row<=end; row++)
    {
        rec = &shared->local_recs[row-shared->my_start_row];
#if 0
        rec->ind = shared->sails->A.getPointerToColIndex(len, row);
        rec->val = shared->sails->A.getPointerToCoef(len, row);
#else
        int ierr;
        ierr = HYPRE_GetDistributedMatrixRow(shared->sails->A, row, &len, 
          (int **) &rec->ind, (double **) &rec->val); //kludge
        assert(!ierr);
#endif
        rec->len = len;
        prune_row(row, rec, shared, 
          prn_space, prn_space_p); // fills prune fields
    }
}

void send_requests(int threadid, SharedData *shared, 
  RowPattern &pattern, RequestData *req_data, int &num_requests,
  RowRecord **list_avail, int *list_request, int merge)
{
    int sendbuf[MAX_ROWS_PER_REQUEST+1];

    int list_avail_size = 0;
    int list_request_size = 0;

    RowRecord *rec;
    int prev_len;
    const int *prev_ind;
    int i;

    pattern.prev_level(prev_len, prev_ind);

    // traverse the new entries in the pattern (entries in previous level)
    // to determine which rows we already have and which we need to request

    for (i=0; i<prev_len; i++, prev_ind++)
    {
        // search for row records
        // this is the most delayed place we can search for these records

        rec = shared->stored_rows->search(*prev_ind);

        if (rec != NULL) // row is already on this node
        {
            assert(list_avail_size < MAX_LIST);
            list_avail[list_avail_size] = rec;
            list_avail_size++;
            // printf("on this node: %d\n", *prev_ind);
        }
        else // row is not on this node
        {
            assert(list_request_size < MAX_LIST);
            list_request[list_request_size] = *prev_ind;
            list_request_size++;
            // printf("not on this node: %d\n", *prev_ind);
        }
    }

    //
    // send requests for rows
    //

    int j;
    int this_pe;
    MPI_Request request;

    int k;

    // if ((double) rand() / (double) RAND_MAX < 0.5)
    shell_sort(list_request_size, list_request);
    // else
    // reverse_shell_sort(list_request_size, list_request);

    num_requests = 0;

    for (i=0; i<list_request_size; i=j) // j is set below
    {
        this_pe = row_to_pe(shared, list_request[i]);

        for (j=i+1; j<list_request_size; j++)
        {
            // if row is on different pe
            if (list_request[j] < shared->start_rows[this_pe] ||
                list_request[j] > shared->end_rows[this_pe])
                   break;
        }

        // send rows in list_request[i..j-1]

        req_data[this_pe].index = i;
        req_data[this_pe].size  = j-i;

        num_requests++;

        // new version that has threadid in the message, and SERVER_TAG
        assert(j-i <= MAX_ROWS_PER_REQUEST);
        for (k=i; k<j; k++)
            sendbuf[k-i] = list_request[k];
        sendbuf[j-i] = threadid;

        MPI_Isend(sendbuf, j-i+1, MPI_INT, this_pe,
            SERVER_TAG, shared->comm, &request);
        MPI_Request_free(&request);
/*
        MPI_Isend(&list_request[i], j-i, MPI_INT, this_pe,
            threadid, shared->comm, &request);
*/

        // printf("sent request to node %d, for %d rows: %d\n",
        //   this_pe, j-i, list_request[i]);fflush(stdout);
    }

    //
    // in the meantime, merge patterns for rows in list_avail
    //

    if (merge)
    {
        for (i=0; i<list_avail_size; i++)
        {
            rec = list_avail[i];
            pattern.merge(rec->pruned_len, rec->pruned_ind);
        }
    }
}


void receive_replies(
  int threadid, SharedData *shared,
  RowPattern &pattern,
  int &num_requests,
  RequestData *req_data, int *list_request,
  int *&ind_space_p, double *&val_space_p, int *const ind_space,
  RowRecord *nonlocal_recs, int &num_nonlocal_recs, 
  int *const prn_space, int *&prn_space_p,
  int merge)
{
    int i, j;
    MPI_Status status;
    int source;
    RequestData *rd;
    int count;
    int *lengths;
    RowRecord *rec;
    int tag;

    for (i=0; i<num_requests; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, threadid, shared->comm, &status);
        tag = status.MPI_TAG;
        source = status.MPI_SOURCE;

        // printf("received reply from node %d\n",
        //   source);fflush(stdout);

        MPI_Get_count(&status, MPI_INT, &count);

        // check we have enough space
        assert(ind_space_p - ind_space + count <= MAX_STORED_ROWS_SPACE);

        // receive indices (with row lengths as first rd->size elements)
        MPI_Recv(ind_space_p, count, 
            MPI_INT, source, tag, shared->comm, &status);

        // first size entries in message are row lengths
        lengths = ind_space_p;
        rd = &req_data[source];
        ind_space_p += rd->size;

        // assume message with values will follow immediately after
        MPI_Recv(val_space_p, count-rd->size,
            MPI_DOUBLE, source, tag, shared->comm, &status);

        // set pointers to the data in ind_space_p and val_space_p
        assert(num_nonlocal_recs+rd->size < MAX_NONLOCAL_RECS);
        for (j=0; j<rd->size; j++)
        {
            nonlocal_recs[num_nonlocal_recs+j].ind = ind_space_p;
            nonlocal_recs[num_nonlocal_recs+j].val = val_space_p;
            nonlocal_recs[num_nonlocal_recs+j].len = lengths[j];
            ind_space_p += lengths[j];
            val_space_p += lengths[j];
            prune_row(list_request[rd->index+j], 
              &nonlocal_recs[num_nonlocal_recs+j], shared, 
              prn_space, prn_space_p);
        }

        // add these rows to stored rows table

        // printf("adding %d rows to shared table\n",
          // rd->size);fflush(stdout);
          // can size be zero??

        pthread_mutex_lock(&shared->table_mutex);
        for (j=0; j<rd->size; j++)
        {
            shared->stored_rows->insert(list_request[rd->index+j], 
              &nonlocal_recs[num_nonlocal_recs+j]);
        }
        pthread_mutex_unlock(&shared->table_mutex);

        // merge the pattern of these rows
        if (merge)
        {
            for (j=0; j<rd->size; j++)
            {
                rec = &nonlocal_recs[num_nonlocal_recs+j];
                pattern.merge(rec->pruned_len, rec->pruned_ind);
            }
        }

        // finally update number of nonlocal records stored by this thread
        num_nonlocal_recs += rd->size;
    }
}


// input to this function, we have a pattern
// we can also run the above one more time to get the neighbor pattern,
// actually that case is different because we want to get orig rows,
// not pruned rows

// need to give it static workspace - since this is now
// called repeatedly

// initialize size of buffers is 0, and not init.

// initialize large buffers to minimize realloc

//// note:
// this function will re-search for row records
// it is possible to remove this cost by storing the row records
// with the patterns when when are searched for the first time

void dump_ahat(int row, double *ahat, const int *indrow, int dimrow, int dimcol)
{
    char filename[40];
    int i, j;
    sprintf(filename, "row.%d", row);
    ofstream outfile(filename);

    for (i=0; i<dimcol; i++)
    {
      outfile << "----" << endl;
      for (j=0; j<dimrow; j++)
        outfile << indrow[j] << "  " << *ahat++ << endl;
    }
    outfile.close();
}

void compute_values(
  SharedData *shared,
  int row, 
  const RowPattern &rowpat, // pattern of row
  ColPattern &colpat,
  double *&ahat,
  int &ahat_size,
  double *&bvec,            // output solution
  int &bvec_size,
  int    *&local_ind,
  int &local_ind_size,
  double *&workspace,
  int &workspace_size,
  double &timet,
  int &maxnrow)
{
    double values[MAX_PATTERN];

    int i, j;
    const double *val;

    double *ahatp;

    int dimcol;
    int dimrow;
    const int *indcol;
    const int *indrow;

    RowRecord *recs[MAX_PATTERN];
    RowRecord *rec;

    int num_entries;
    int *local_ind_p;

    colpat.init();
    rowpat.get_pattern(dimcol, indcol);

    // search for all the row records in the pattern
    // and count number of entries
    num_entries = 0;
    for (i=0; i<dimcol; i++)
    {
        recs[i] = shared->stored_rows->search(*indcol++);
        num_entries += recs[i]->len;
    }
        
    // reallocate space for local indices if necessary
    if (num_entries > local_ind_size)
    {
        delete local_ind;
        local_ind_size = num_entries;
        local_ind = new int[local_ind_size];
    }

    // merge row patterns while saving local indices
    local_ind_p = local_ind;
    for (i=0; i<dimcol; i++)
    {
        rec = recs[i];
        colpat.merge(rec->len, rec->ind, local_ind_p);
    }

    colpat.get_pattern(dimrow, indrow);

    // form b (0-based)

    if (dimrow > bvec_size)
    {
        delete bvec;
        bvec_size = dimrow;
        bvec = new double[bvec_size];
    }

    for (i=0; i<dimrow; i++)
        bvec[i] = 0.0;

    for (i=0; i<dimrow; i++)
    {
        if (indrow[i] == row)
        {
            bvec[i] = 1.0;
            break;
        }
    }

    // check that ahat is large enough

    if (dimcol*dimrow > ahat_size)
    {
        delete ahat;
        ahat_size = dimcol*dimrow;
        ahat = new double[ahat_size];
    }

    for (i=0; i<dimcol*dimrow; i++)
        ahat[i] = 0.0;

    // form ahat using local indices

    ahatp = ahat;
    local_ind_p = local_ind;
    for (i=0; i<dimcol; i++)
    {
        rec = recs[i];
        val = rec->val;
        for (j=0; j<rec->len; j++)
        {
            ahatp[*local_ind_p++] = *val++;
        }
        ahatp += dimrow;
    }

    // solve least squares system
#if 0
    char TRANS = 'N';
    int NROW = dimrow;
    int NCOL = dimcol;

    // printf("nrow ncol %d %d\n", NROW, NCOL); fflush(stdout);
    // assert (NROW >= NCOL);
    int ONE = 1;
#endif
    int INFO;

    // maxnrow = maxnrow + (NROW*NCOL*NCOL)/1000;

#if 0
    // kludge - is essl thread safe?
    pthread_mutex_lock(&shared->dgels_mutex);
    dgels(&TRANS, &NROW, &NCOL, &ONE, ahat, &NROW,
        bvec, &NROW, workspace, &workspace_size, &INFO);
    pthread_mutex_unlock(&shared->dgels_mutex);
    if (INFO != 0)
    {
        cout << "dgels returned: " << INFO << endl;
    }

#else
// dgells (iopt, a, lda, b, ldb, x, ldx, rn, tau, m, n, 
//         nb, k, aux, naux); 

    assert(dimcol <= MAX_PATTERN);

    // dump_ahat(row, ahat, indrow, dimrow, dimcol);

 // double time0 = MPI_Wtime();
 esvdgells(0, ahat, dimrow, bvec, dimrow, values, dimcol, NULL, 1.e-12, dimrow, 
      dimcol, 1, &INFO, workspace, &workspace_size);
 // timet += (MPI_Wtime() - time0);

    for (i=0; i<dimcol; i++)
    {
       bvec[i] = values[i];
       // if (ABS(bvec[i]) < 0.)
          // printf("BAD: row %d col %d val %f\n", row, indcol[i], values[i]);
    }
#endif
}


// no values are ever used after the initial thresholding
// however, values ARE used when computing numerical values, so....
// and in fact, values for these rows are needed... even after the 
// pattern has been computed

// level=0 sparsified pattern

// maybe servers should allloc this space

void *worker(void *local)
{
    int threadid = ((LocalData *)local)->threadid;
    SharedData *shared = ((LocalData *)local)->shared;

    int    *const ind_space = new int   [MAX_STORED_ROWS_SPACE];
    double *const val_space = new double[MAX_STORED_ROWS_SPACE];
    int    *const prn_space = new int   [MAX_PRUNE_SPACE];

    assert(prn_space != NULL);

    int    *ind_space_p = ind_space;
    double *val_space_p = val_space;
    int    *prn_space_p = prn_space;

    RowRecord *nonlocal_recs = new RowRecord[MAX_NONLOCAL_RECS];
    assert(nonlocal_recs != NULL);
    int num_nonlocal_recs = 0;

    RowPattern rowpat(MAX_PATTERN);
    ColPattern colpat(MAX_PATTERN);

    RowRecord **list_avail   = new RowRecordP[MAX_LIST];
    int *list_request = new int[MAX_LIST];

    RequestData *req_data = new RequestData[shared->npes];

    int ahat_size      =  10000;
    int bvec_size      =   1000;
    int local_ind_size =   5000;
    int workspace_size = 400*64;
    double *ahat      = new double[ahat_size];
    double *bvec      = new double[bvec_size];
    int    *local_ind = new int[local_ind_size];
    double *workspace = new double[workspace_size];

    assert(workspace != NULL);

    int row;
    RowRecord *rec;
    int level;

    int len;
    const int *ind;

    int num_requests;

    //
    // real work begins here
    //

    prune_local_rows(threadid, shared, prn_space, prn_space_p);

    barrier_wait(shared->barrier, 0);
    // printf("%d got past barrier\n", threadid);fflush(stdout);

    //row = get_row_rand(shared);
    row = get_row(shared);

    // double time0 = MPI_Wtime();
    double time_calc = 0.;
    double time_ls = 0.;
    double time1;
    int maxnrow = 0;

    while (row != -1)
    {
        rec = shared->stored_rows->search(row);  // must be a local row

        // printf("working on row %d which has length %d\n", 
        //    row, rec->len);fflush(stdout);

        rowpat.init();
        rowpat.merge(rec->pruned_len, rec->pruned_ind);

        for (level=0; level<shared->nlevels; level++)
        {
            send_requests(threadid, shared, rowpat, req_data, 
              num_requests, list_avail, list_request, 1);

            receive_replies(threadid, shared, rowpat, num_requests, req_data,
              list_request, ind_space_p, val_space_p, ind_space,
              nonlocal_recs, num_nonlocal_recs, prn_space, prn_space_p, 1);
        }

        // send and receive for rows in the pattern
        // but without merging their patterns

        send_requests(threadid, shared, rowpat, req_data, 
          num_requests, list_avail, list_request, 0);

        receive_replies(threadid, shared, rowpat, num_requests, req_data,
          list_request, ind_space_p, val_space_p, ind_space,
          nonlocal_recs, num_nonlocal_recs, prn_space, prn_space_p, 0);

        // time1 = MPI_Wtime();
        compute_values(shared, row, rowpat, colpat, ahat, ahat_size, bvec, 
          bvec_size, local_ind, local_ind_size, workspace, workspace_size,
          time_ls, maxnrow);
        // time_calc += (MPI_Wtime() - time1);

        rowpat.get_pattern(len, ind);
#if 0
        shared->sails->M->setRowLength(len, row);

        int    *ind2 = shared->sails->M->getPointerToColIndex(len, row);
        double *val2 = shared->sails->M->getPointerToCoef(len, row);

	int i;
        for (i=0; i<len; i++)
        {
            *ind2++ = *ind++;
            *val2++ = bvec[i];
        }
#else
        int ierr;
        ierr = HYPRE_InsertIJMatrixRow(shared->sails->M, len, row,
	  (int *) ind, bvec); // kludge
	assert(!ierr);
#endif

        // get next row to compute
        //row = get_row_rand(shared);
        row = get_row(shared);
    }

    // time0 = MPI_Wtime() - time0;
    // printf("Time for thread %d: %#10.4f\n", threadid, time0);

    // printf("Compute time for thread %d: %#10.4f %#10.4f\n", 
    //     threadid, time_calc, time_ls);

    // printf("Num indices cached on %d thread: %d, and maxnrow %d\n", threadid,
        // ind_space_p-ind_space, maxnrow);

/*
struct timespec delay;
delay.tv_sec = 10;
delay.tv_nsec = 0;
pthread_delay_np(&delay);
*/

    delete ind_space;
    delete val_space;
    delete prn_space;

    delete [] nonlocal_recs;

    delete list_avail;
    delete list_request;

    delete [] req_data;

    delete ahat;
    delete bvec;
    delete local_ind;
    delete workspace;

    return NULL;
}


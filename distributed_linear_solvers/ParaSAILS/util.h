/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "ParaSAILS.h"
#include <pthread.h>
#include <assert.h>
#include "barrier.h"
#include "mpi.h"

#ifndef _UTIL_H
#define _UTIL_H

#define NUM_THREADS_WORKERS 16 // these are maximums
#define NUM_THREADS_SERVERS 16

// this is shared data
#define MAX_STORED_ROWS       (110*110*110)

// number of indices buffered per thread
#define MAX_STORED_ROWS_SPACE (60*60*6*5*7) 

// assumes 100^3 problem and 3 nnz per row, per thread
#define MAX_PRUNE_SPACE   (60*60*60*3) 

#define MAX_ROWS_PER_REQUEST    1000 // max number of rows of each request
#define MAX_SERVER_BUF         10000 // max number of indices in each reply
#define MAX_NONLOCAL_RECS  (60*60*6*5) // max number of nonlocal rows on node
#define MAX_PATTERN             2000
#define MAX_LIST                1000


// prototype
class SharedData;
void init_get_row_rand(SharedData *shared);

// record for row, stored locally by each thread

typedef struct
{
    const int    *ind;         // indices
    const double *val;         // values
    int           len;         // length
    int          *pruned_ind;  // indices of pruned row
    int           pruned_len;  // length of pruned row
} RowRecord;


////////////////////////////////////////
// shared table of rows (local and those cached on this node)

class StoredRows
{
public:
    inline  StoredRows(const int max_size_, RowRecord *local_recs_,
      int my_start_row_, int my_end_row_);
    inline ~StoredRows();

    inline RowRecord *search(int row_num);
    inline void insert(int row, RowRecord *rec);

private:
    int  size;       // number of stored records for nonlocal cached rows
    int  max_size;   // max size of arrays below
    int *rows;       // array of row numbers associated with row records
    RowRecord **recs;// array of pointers to records for nonlocal cached rows

    RowRecord *local_recs; // array of records for local rows
    int my_start_row;
    int my_end_row;
};

typedef RowRecord *RowRecordP;

inline StoredRows::StoredRows(const int max_size_, RowRecord *local_recs_,
  int my_start_row_, int my_end_row_)
{
    size     = 0;
    max_size = max_size_;
    rows     = new int[max_size];
    //recs     = new (RowRecord *)[max_size];
    recs     = new RowRecordP[max_size];

    assert(rows != NULL);
    assert(recs != NULL);

    local_recs = local_recs_;
    my_start_row = my_start_row_;
    my_end_row = my_end_row_;
}

inline StoredRows::~StoredRows()
{
    delete [] rows;
    delete [] recs;
}

inline RowRecord *StoredRows::search(int row_num)
{
    int i;

    if (row_num >= my_start_row && row_num <= my_end_row)
    {
	return &local_recs[row_num - my_start_row];
    }

    for (i=0; i<size; i++)
    {
	if (rows[i] == row_num)
	    return recs[i];
    }

    return NULL;
}

inline void StoredRows::insert(int row, RowRecord *rec)
{
    assert(size < max_size);
    rows[size] = row;
    recs[size] = rec;
    // increment size after adding record
    size++;
}

// for above insert procedure:
// it is the responsibility of the caller to lock and unlock the mutex
// around calling this procedure, i.e., caller can insert more than
// one row during one lock
// also, row record must be full, since other threads may try to use
// this row immediately after this insertion

/////////////////////////

// RowPattern
// also store the pointer to the row record, so we never have to search again

// need a RowPattern class that does not store row records

class RowPattern
{
public:
    inline  RowPattern(int);
    inline ~RowPattern();

    // return the pattern
    inline void init();

    // initialize the pattern to zero length
    inline void get_pattern(int &len, const int *&ind) const;

    // returns the previous level (since the last call to prev_level)
    inline void prev_level(int &len, const int *&ind);

    // input elements are distinct from each other
    inline void merge(int len, const int *ind);

private:
    int  max_size;
    int *pattern;    // list of elements in pattern
    int  curr;       // pointer to next free space to hold elements
    int  prev;       // the curr pointer last time prev_level was called
};

inline RowPattern::RowPattern(int max_size_)
{
    max_size = max_size_;
    pattern  = new int[max_size];
    assert(pattern != NULL);
    curr     = 0;
    prev     = 0;
}

inline RowPattern::~RowPattern()
{
    delete [] pattern;
}

inline void RowPattern::init()
{
    curr = 0;
    prev = 0;
}

inline void RowPattern::get_pattern(int &len, const int *&ind) const
{
    ind = pattern;
    len = curr;
}

inline void RowPattern::prev_level(int &len, const int *&ind)
{
    ind = &pattern[prev];
    len = curr - prev;
    prev = curr;
}

// algorithm assumes that entries in "elements" are unique

inline void RowPattern::merge(int len, const int *ind)
{
    int i;
    int *stop, *p;

    // indicates where to stop searching
    stop = &pattern[curr];

    // loop on all elements to possibly merge
    for (i=0; i<len; i++, ind++)
    {
        p = pattern;
        while (p != stop)
        {
	    if (*p++ == *ind)
		goto found;
        }

	// not found, so add this element
        assert(curr < max_size);
	pattern[curr++] = *ind;

        found:
            ;
    }
}

/////////////////////////

// ColPattern
// differences: does not store row records, no levels, but returns
// local indices

class ColPattern
{
public:
    inline  ColPattern(int);
    inline ~ColPattern();

    // return the pattern
    inline void init();

    // initialize the pattern to zero length
    inline void get_pattern(int &len, const int *&ind);

    // input elements are distinct from each other
    inline void merge(int len, const int *ind, int *&localind);

private:
    int  max_size;
    int *pattern;    // list of elements in pattern
    int *curr;       // pointer to next free space to hold elements
};

inline ColPattern::ColPattern(int max_size_)
{
    max_size = max_size_;
    pattern  = new int[max_size];
    assert(pattern != NULL);
    curr     = pattern;
}

inline ColPattern::~ColPattern()
{
    delete [] pattern;
}

inline void ColPattern::init()
{
    curr = pattern;
}

inline void ColPattern::get_pattern(int &len, const int *&ind)
{
    ind = pattern;
    len = (int)(curr - pattern);
}

// same as merge, but insert the values
// into the vector a
// do not need row records
// .. local indices are 0-based
// do we increment localind pointer??? yes

inline void ColPattern::merge(int len, const int *ind, int *&localind)
{
    int i;
    int *stop, *p;

    // indicates where to stop searching
    stop = curr;

    // loop on all elements to possibly merge
    for (i=0; i<len; i++, ind++)
    {
        p = pattern;
        while (p != stop)
        {
	    if (*p++ == *ind)
		goto found;
        }

	// not found, so add this element
        assert(curr-pattern < max_size);
	*curr++ = *ind;
        p = curr; // so that localind will be set correctly below

	found:

	*localind++ = (int)(p-pattern-1); // 0-based
    }
}

// shared data
// there is only one copy of this data
// this is created by the main thread

class SharedData
{
public:
    inline  SharedData(ParaSAILS *sails);
    inline ~SharedData();

    // read-write data

    int             current_row;
    pthread_mutex_t job_mutex;
    pthread_mutex_t table_mutex;
    pthread_mutex_t probe_mutex;
    pthread_mutex_t dgels_mutex;
    barrier_t       barrier;
    barrier_t       barrier_end;

    StoredRows     *stored_rows;

    // read-only data

    ParaSAILS *sails;

    int myid;
    int npes;
    int nlevels;
    double thresh;
    int lfil;
    int prune_alg;
    int dump;

    int    my_start_row;
    int    my_end_row;

    int    *start_rows;
    int    *end_rows;

    RowRecord *local_recs;

    int *rownums; // row numbers to use in assigning jobs randomly

    MPI_Comm comm;
};


inline SharedData::SharedData(ParaSAILS *sailspc)
{
    pthread_mutex_init(&job_mutex, NULL);
    pthread_mutex_init(&table_mutex, NULL);
    pthread_mutex_init(&probe_mutex, NULL);
    pthread_mutex_init(&dgels_mutex, NULL);
    barrier = barrier_init(num_threads_workers+num_threads_servers);
    barrier_end = barrier_init(num_threads_workers);

    sails = sailspc;

    myid = sails->myid;
    npes = sails->npes;
    nlevels = sails->nlevels;
    thresh = sails->thresh;
    lfil = sails->lfil;
    prune_alg = sails->prune_alg;
    dump = sails->dump;

    my_start_row = sails->my_start_row;
    my_end_row   = sails->my_end_row;

    start_rows = sails->start_rows;
    end_rows   = sails->end_rows;

#if 0
    my_start_row = sails->A.getMap().startRow();
    my_end_row   = sails->A.getMap().endRow();

    start_rows = sails->A.getMap().globalStartRow();
    end_rows   = sails->A.getMap().globalEndRow();
#endif

    local_recs = new RowRecord[my_end_row-my_start_row+1];

    stored_rows = new StoredRows(MAX_STORED_ROWS, local_recs,
        my_start_row, my_end_row);
    assert(stored_rows != NULL);

    current_row = my_start_row;
    rownums = new int[my_end_row-my_start_row+1];
    init_get_row_rand(this);

    comm = sails->comm;
}

inline SharedData::~SharedData()
{
    pthread_mutex_destroy(&job_mutex);
    pthread_mutex_destroy(&table_mutex);
    pthread_mutex_destroy(&probe_mutex);
    pthread_mutex_destroy(&dgels_mutex);
    barrier_destroy(barrier);
    barrier_destroy(barrier_end);

    delete [] local_recs;
    delete rownums;
}

#endif /* _UTIL_H */

/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/* Reference:  Kleiman et al., Programming with Threads, p. 289. */

#include <pthread.h>

#ifndef _BARRIER_H
#define _BARRIER_H

typedef struct barrier_struct
{
    pthread_mutex_t lock; /* Mutex lock for the entire structure */
    int n_clients;        /* Number of threads to wait for at barrier */
    int n_waiting;        /* Number of threads have called barrier_wait */
    int phase;            /* Flag to separate waiters from fast workers */
    int sum;              /* Sum of arguments passed to barrier_wait */
    int result;           /* Answer to be returned by barrier_wait */
    pthread_cond_t wait_cv; /* Clients wait on condition var to proceed */
} *barrier_t;


barrier_t barrier_init(int n_clients);
void barrier_destroy(barrier_t barrier);
int barrier_wait(barrier_t barrier, int increment);

#endif /* _BARRIER_H */

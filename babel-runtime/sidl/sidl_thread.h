/*
 * sidl_thread.h -- generic thread & mutex stuff used in Babel
 * 
 */


/*
 * Recursive Mutex --  Allow the same thread to lock the mutex 
 * recursively, but require a matching number of unlocks
 * 
 * Note: this is not recommended by Lewis & Berg (Multithreaded
 * Programming w/ pthreads).  I tried using the implementation they
 * provide anyway, but the liscencing is not clear... so I wrote my own.
 */

#ifndef SIDL_THREAD_H
#define SIDL_THREAD_H

#include "babel_config.h"

#ifdef HAVE_PTHREAD
#include <pthread.h>

#ifdef __cplusplus
extern "C" { /* } */
#endif

struct sidl_recursive_mutex_t {
  pthread_mutex_t lock;  /* lock for structure */
  pthread_cond_t  cv;    /* waiting list control */
  int             count; /* times locked recursively */
  pthread_t       owner; /* thread or NULL_TID */
};

#ifndef NULL_TID
#define NULL_TID (pthread_t) 0
#endif /* NULL_TID */

#define SIDL_RECURSIVE_MUTEX_INITIALIZER \
 {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, NULL_TID}

int sidl_recursive_mutex_init( struct sidl_recursive_mutex_t* m );
int sidl_recursive_mutex_destroy( struct sidl_recursive_mutex_t * m );
int sidl_recursive_mutex_lock( struct sidl_recursive_mutex_t* m );
int sidl_recursive_mutex_unlock( struct sidl_recursive_mutex_t * m );

#ifdef __cplusplus
}
#endif

#endif /* HAVE_PTHREAD */

#endif /* defined (SIDL_THREAD_H) */

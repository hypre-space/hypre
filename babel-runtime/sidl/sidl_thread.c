/*
 * sidl_thread.c -- generic thread & mutex stuff used in Babel
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
#include "sidl_thread.h"

#ifdef HAVE_PTHREAD

int sidl_recursive_mutex_init( struct sidl_recursive_mutex_t *m ) { 
  int err=0;
  m->owner = NULL_TID;
  m->count = 0;
  err = pthread_mutex_init(&(m->lock), NULL);
  if (err != 0 ) { return err; }
  return  pthread_cond_init(&(m->cv), NULL);
}

int sidl_recursive_mutex_destroy(struct sidl_recursive_mutex_t *m) {
  int err=0;
  err = pthread_mutex_destroy(&(m->lock));
  if ( err ) { return err; }
  err = pthread_cond_destroy(&(m->cv));
  if ( err ) { return err; }
  m->count = 0;
  m->owner = NULL_TID;
  return err;
}

int sidl_recursive_mutex_lock(struct sidl_recursive_mutex_t *m) {
  pthread_t mythread = pthread_self();
  int err;
  
  err = pthread_mutex_lock(&(m->lock));
  if ( err ) { return err; }
  while ((m->owner) && !pthread_equal(m->owner, mythread)) { 
    pthread_cond_wait(&(m->cv), &(m->lock));
  }
  m->owner = mythread;
  m->count++;
  err = pthread_mutex_unlock(&(m->lock));
  if ( err ) { return err; }
  return 0;
}

int sidl_recursive_mutex_unlock(struct sidl_recursive_mutex_t *m) {
  int err;

  err = pthread_mutex_lock(&(m->lock));
  if ( err ) { return err; }

  if (--(m->count) == 0) {
    m->owner = NULL_TID;
    err = pthread_cond_signal(&(m->cv));
    if ( err ) { return err; }
  }
  err = pthread_mutex_unlock(&(m->lock));
  if ( err ) { return err; }
  return 0;
}

#endif

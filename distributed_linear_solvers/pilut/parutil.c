/*
 * parutil.c
 *
 * This file contains utility functions
 *
 * Started 8/28/94
 * George
 *
 * $Id$
 *
 */

#include "ilu.h"
#include "./DistributedMatrixPilutSolver.h"


/*************************************************************************
* This function prints an error message and exits
**************************************************************************/
void errexit( char *f_str, ...)
{
  va_list argp;
  int nprow, npcol;

  /*fprintf(stdout,"[%3d]", mype);*/

  va_start(argp, f_str);
  vfprintf(stdout, f_str, argp);
  va_end(argp);

  fprintf(stdout,"\n");
  fflush(stdout);

  abort();
}


/*************************************************************************
* This makes life easier by aborting all threads together, and printing
* some diagnostic with the PE.
**************************************************************************/
void my_abort( int inSignal, hypre_PilutSolverGlobals *globals )
{
  printf( "PE %d caught sig %d\n", mype, inSignal );
  fflush(0);
  MPI_Abort( pilut_comm, inSignal );
}


/*************************************************************************
* The following function allocates an array of ints
**************************************************************************/
int *idx_malloc(int n, char *msg)
{
  int *ptr;

  if (n == 0)
    return NULL;

  ptr = (int *)malloc(sizeof(int)*n);
  if (ptr == NULL) {
    errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(int));
  }

  return ptr;

}


/*************************************************************************
* The follwoing function allocates an array of ints and initializes
**************************************************************************/
int *idx_malloc_init(int n, int ival, char *msg)
{
  int *ptr;
  int i;

  if (n == 0)
    return NULL;

  ptr = (int *)malloc(sizeof(int)*n);
  if (ptr == NULL) {
    errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(int));
  }

  for (i=0; i<n; i++)
    ptr[i] = ival;

  return ptr;
}


/*************************************************************************
* The following function allocates an array of floats
**************************************************************************/
double *fp_malloc(int n, char *msg)
{
  double *ptr;

  if (n == 0)
    return NULL;

  ptr = (double *)malloc(sizeof(double)*n);
  if (ptr == NULL) {
    errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(double));
  }

  return ptr;

}


/*************************************************************************
* The follwoing function allocates an array of floats and initializes
**************************************************************************/
double *fp_malloc_init(int n, double ival, char *msg)
{
  double *ptr;
  int i;

  if (n == 0)
    return NULL;

  ptr = (double *)malloc(sizeof(double)*n);
  if (ptr == NULL) {
    errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(double));
  }

  for (i=0; i<n; i++)
    ptr[i] = ival;

  return ptr;
}



/*************************************************************************
* This function is my wrapper around malloc.
**************************************************************************/
void *mymalloc(int nbytes, char *msg)
{
  void *ptr;

  if (nbytes == 0)
    return NULL;

  ptr = (void *)malloc(nbytes);
  if (ptr == NULL) {
    errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, nbytes);
  }

  return ptr;
}


/*************************************************************************
* This function is my wrapper around free, allows multiple pointers    
**************************************************************************/
void free_multi(void *ptr1,...)
{
  va_list plist;
  void *ptr;

  if (ptr1 != NULL)
    free(ptr1);
  ptr1 = NULL;

  va_start(plist, ptr1);

  while ((int)(ptr = va_arg(plist, void *)) != -1) {
    if (ptr != NULL)
      free(ptr);
    ptr = NULL;
  }

  va_end(plist);

}            


/*************************************************************************
* The following function copies an int (int) array
**************************************************************************/
void memcpy_int( int *dest, const int *src, size_t n )
{
  if (dest) memcpy(dest, src, n*sizeof(int));
}

/*************************************************************************
* The following function copies an int (int) array
**************************************************************************/
void memcpy_idx( int *dest, const int *src, size_t n )
{
  if (dest) memcpy(dest, src, n*sizeof(int));
}

/*************************************************************************
* The following function copies a floating point (double) array.
* Note this assumes BLAS 1 routine SCOPY. An alternative would be memcpy.
* There is a noticeable difference between this and just a for loop.
**************************************************************************/
void memcpy_fp( double *dest, const double *src, size_t n )
{
  int i, inc=1;

  /*SCOPY(&n, src, &inc, dest, &inc);*/
  for (i=0; i<n; i++) dest[i] = src[i];
}


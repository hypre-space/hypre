/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




/*
 * parutil.c
 *
 * This file contains utility functions
 *
 * Started 8/28/94
 * George
 *
 * $Id: parutil.c,v 2.7 2010/12/20 19:27:34 falgout Exp $
 *
 */

#include "ilu.h"
#include "DistributedMatrixPilutSolver.h"


/*************************************************************************
* This function prints an error message and exits
**************************************************************************/
void hypre_errexit( char *f_str, ...)
{
  va_list argp;

  /*hypre_fprintf(stdout,"[%3d]", mype);*/

  va_start(argp, f_str);
  vfprintf(stdout, f_str, argp);
  va_end(argp);

  hypre_fprintf(stdout,"\n");
  fflush(stdout);

  abort();
}


/*************************************************************************
* This makes life easier by aborting all threads together, and printing
* some diagnostic with the PE.
**************************************************************************/
void hypre_my_abort( HYPRE_Int inSignal, hypre_PilutSolverGlobals *globals )
{
  hypre_printf( "PE %d caught sig %d\n", mype, inSignal );
  fflush(stdout);
  hypre_MPI_Abort( pilut_comm, inSignal );
}


/*************************************************************************
* The following function allocates an array of ints
**************************************************************************/
HYPRE_Int *hypre_idx_malloc(HYPRE_Int n, char *msg)
{
  HYPRE_Int *ptr;

  if (n == 0)
    return NULL;

  ptr = (HYPRE_Int *)malloc(sizeof(HYPRE_Int)*n);
  if (ptr == NULL) {
    hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(HYPRE_Int));
  }

  return ptr;

}


/*************************************************************************
* The follwoing function allocates an array of ints and initializes
**************************************************************************/
HYPRE_Int *hypre_idx_malloc_init(HYPRE_Int n, HYPRE_Int ival, char *msg)
{
  HYPRE_Int *ptr;
  HYPRE_Int i;

  if (n == 0)
    return NULL;

  ptr = (HYPRE_Int *)malloc(sizeof(HYPRE_Int)*n);
  if (ptr == NULL) {
    hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(HYPRE_Int));
  }

  for (i=0; i<n; i++)
    ptr[i] = ival;

  return ptr;
}


/*************************************************************************
* The following function allocates an array of floats
**************************************************************************/
double *hypre_fp_malloc(HYPRE_Int n, char *msg)
{
  double *ptr;

  if (n == 0)
    return NULL;

  ptr = (double *)malloc(sizeof(double)*n);
  if (ptr == NULL) {
    hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(double));
  }

  return ptr;

}


/*************************************************************************
* The follwoing function allocates an array of floats and initializes
**************************************************************************/
double *hypre_fp_malloc_init(HYPRE_Int n, double ival, char *msg)
{
  double *ptr;
  HYPRE_Int i;

  if (n == 0)
    return NULL;

  ptr = (double *)malloc(sizeof(double)*n);
  if (ptr == NULL) {
    hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(double));
  }

  for (i=0; i<n; i++)
    ptr[i] = ival;

  return ptr;
}



/*************************************************************************
* This function is my wrapper around malloc.
**************************************************************************/
void *hypre_mymalloc(HYPRE_Int nbytes, char *msg)
{
  void *ptr;

  if (nbytes == 0)
    return NULL;

  ptr = (void *)malloc(nbytes);
  if (ptr == NULL) {
    hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, nbytes);
  }

  return ptr;
}


/*************************************************************************
* This function is my wrapper around free, allows multiple pointers    
**************************************************************************/
#if 0
void hypre_free_multi(void *ptr1,...)
{
  va_list plist;
  void *ptr;

  if (ptr1 != NULL)
    free(ptr1);
  ptr1 = NULL;

  va_start(plist, ptr1);

  while ( (ptr = va_arg(plist, void *)) != ((void *) -1) ) {
    if (ptr != NULL)
      free(ptr);
    ptr = NULL;
  }

  va_end(plist);

}    
#endif        

/*************************************************************************
* The following function copies an HYPRE_Int (HYPRE_Int) array
**************************************************************************/
void hypre_memcpy_int( HYPRE_Int *dest, const HYPRE_Int *src, size_t n )
{
  if (dest) memcpy(dest, src, n*sizeof(HYPRE_Int));
}

/*************************************************************************
* The following function copies an HYPRE_Int (HYPRE_Int) array
**************************************************************************/
void hypre_memcpy_idx( HYPRE_Int *dest, const HYPRE_Int *src, size_t n )
{
  if (dest) memcpy(dest, src, n*sizeof(HYPRE_Int));
}

/*************************************************************************
* The following function copies a floating point (double) array.
* Note this assumes BLAS 1 routine SCOPY. An alternative would be memcpy.
* There is a noticeable difference between this and just a for loop.
**************************************************************************/
void hypre_memcpy_fp( double *dest, const double *src, size_t n )
{
  HYPRE_Int i;

  /*SCOPY(&n, src, &inc, dest, &inc);*/
  for (i=0; i<n; i++) dest[i] = src[i];
}


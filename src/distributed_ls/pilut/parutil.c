/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
#include "DistributedMatrixPilutSolver.h"


/*************************************************************************
* This function prints an error message and exits
**************************************************************************/
void hypre_errexit(const char *f_str, ...)
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
HYPRE_Int *hypre_idx_malloc(HYPRE_Int n,const char *msg)
{
  HYPRE_Int *ptr;

  if (n == 0)
    return NULL;

  ptr = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
  if (ptr == NULL) {
    hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(HYPRE_Int));
  }

  return ptr;

}


/*************************************************************************
* The follwoing function allocates an array of ints and initializes
**************************************************************************/
HYPRE_Int *hypre_idx_malloc_init(HYPRE_Int n, HYPRE_Int ival,const char *msg)
{
  HYPRE_Int *ptr;
  HYPRE_Int i;

  if (n == 0)
    return NULL;

  ptr = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
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
HYPRE_Real *hypre_fp_malloc(HYPRE_Int n,const char *msg)
{
  HYPRE_Real *ptr;

  if (n == 0)
    return NULL;

  ptr = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
  if (ptr == NULL) {
    hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(HYPRE_Real));
  }

  return ptr;

}


/*************************************************************************
* The follwoing function allocates an array of floats and initializes
**************************************************************************/
HYPRE_Real *hypre_fp_malloc_init(HYPRE_Int n, HYPRE_Real ival,const char *msg)
{
  HYPRE_Real *ptr;
  HYPRE_Int i;

  if (n == 0)
    return NULL;

  ptr = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
  if (ptr == NULL) {
    hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(HYPRE_Real));
  }

  for (i=0; i<n; i++)
    ptr[i] = ival;

  return ptr;
}



/*************************************************************************
* This function is my wrapper around malloc.
**************************************************************************/
void *hypre_mymalloc(HYPRE_Int nbytes,const char *msg)
{
  void *ptr;

  if (nbytes == 0)
    return NULL;

  ptr = hypre_TAlloc(char, nbytes, HYPRE_MEMORY_HOST);
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

   hypre_TFree(ptr1, HYPRE_MEMORY_HOST);

   va_start(plist, ptr1);

   while ( (ptr = va_arg(plist, void *)) != ((void *) -1) ) {
      hypre_TFree(ptr, HYPRE_MEMORY_HOST);
   }

   va_end(plist);
}
#endif

/*************************************************************************
* The following function copies an HYPRE_Int (HYPRE_Int) array
**************************************************************************/
void hypre_memcpy_int( HYPRE_Int *dest, const HYPRE_Int *src, size_t n )
{
   if (dest) hypre_TMemcpy(dest,  src, HYPRE_Int, n, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
}

/*************************************************************************
* The following function copies an HYPRE_Int (HYPRE_Int) array
**************************************************************************/
void hypre_memcpy_idx( HYPRE_Int *dest, const HYPRE_Int *src, size_t n )
{
   if (dest) hypre_TMemcpy(dest,  src, HYPRE_Int, n, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
}

/*************************************************************************
* The following function copies a floating point (HYPRE_Real) array.
* Note this assumes BLAS 1 routine hypre_dcopy. An alternative would be memcpy.
* There is a noticeable difference between this and just a for loop.
**************************************************************************/
void hypre_memcpy_fp( HYPRE_Real *dest, const HYPRE_Real *src, size_t n )
{
  HYPRE_Int i, ni = (HYPRE_Int) n;

  /*hypre_dcopy(&n, src, &inc, dest, &inc);*/
  for (i=0; i<ni; i++) dest[i] = src[i];
}


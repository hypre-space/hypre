/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




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
void hypre_errexit( char *f_str, ...)
{
  va_list argp;

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
void hypre_my_abort( int inSignal, hypre_PilutSolverGlobals *globals )
{
  printf( "PE %d caught sig %d\n", mype, inSignal );
  fflush(stdout);
  MPI_Abort( pilut_comm, inSignal );
}


/*************************************************************************
* The following function allocates an array of ints
**************************************************************************/
int *hypre_idx_malloc(int n, char *msg)
{
  int *ptr;

  if (n == 0)
    return NULL;

  ptr = (int *)malloc(sizeof(int)*n);
  if (ptr == NULL) {
    hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(int));
  }

  return ptr;

}


/*************************************************************************
* The follwoing function allocates an array of ints and initializes
**************************************************************************/
int *hypre_idx_malloc_init(int n, int ival, char *msg)
{
  int *ptr;
  int i;

  if (n == 0)
    return NULL;

  ptr = (int *)malloc(sizeof(int)*n);
  if (ptr == NULL) {
    hypre_errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, n*sizeof(int));
  }

  for (i=0; i<n; i++)
    ptr[i] = ival;

  return ptr;
}


/*************************************************************************
* The following function allocates an array of floats
**************************************************************************/
double *hypre_fp_malloc(int n, char *msg)
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
double *hypre_fp_malloc_init(int n, double ival, char *msg)
{
  double *ptr;
  int i;

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
void *hypre_mymalloc(int nbytes, char *msg)
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
* The following function copies an int (int) array
**************************************************************************/
void hypre_memcpy_int( int *dest, const int *src, size_t n )
{
  if (dest) memcpy(dest, src, n*sizeof(int));
}

/*************************************************************************
* The following function copies an int (int) array
**************************************************************************/
void hypre_memcpy_idx( int *dest, const int *src, size_t n )
{
  if (dest) memcpy(dest, src, n*sizeof(int));
}

/*************************************************************************
* The following function copies a floating point (double) array.
* Note this assumes BLAS 1 routine SCOPY. An alternative would be memcpy.
* There is a noticeable difference between this and just a for loop.
**************************************************************************/
void hypre_memcpy_fp( double *dest, const double *src, size_t n )
{
  int i;

  /*SCOPY(&n, src, &inc, dest, &inc);*/
  for (i=0; i<n; i++) dest[i] = src[i];
}


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
 * util.c
 *
 * This function contains various utility routines
 *
 * Started 9/28/95
 * George
 *
 * $Id: util.c,v 2.7 2011/11/07 18:12:01 kolev1 Exp $
 */

#include "ilu.h"
#include "DistributedMatrixPilutSolver.h"

/*************************************************************************
* This function finds the minimum value in the array removes it and
* returns it. It decreases the size of the array.
**************************************************************************/
HYPRE_Int hypre_ExtractMinLR( hypre_PilutSolverGlobals *globals )
{
  HYPRE_Int i, j=0 ;

  for (i=1; i<lastlr; i++) {
    if (lr[i] < lr[j])
      j = i;
  }
  i = lr[j];

  /* Remove it */
  lastlr-- ;
  if (j < lastlr) 
    lr[j] = lr[lastlr];

  return i;
}


/*************************************************************************
* This function sort an (idx,val) array in increasing idx values
**************************************************************************/
void hypre_IdxIncSort(HYPRE_Int n, HYPRE_Int *idx, double *val)
{
  HYPRE_Int i, j, min;
  double tmpval;
  HYPRE_Int tmpidx;

  for (i=0; i<n; i++) {
    min = i;
    for (j=i+1; j<n; j++) {
      if (idx[j] < idx[min])
        min = j;
    }

    if (min != i) {
      SWAP(idx[i], idx[min], tmpidx);
      SWAP(val[i], val[min], tmpval);
    }
  }
}



/*************************************************************************
* This function sort an (idx,val) array in decreasing abs val 
**************************************************************************/
void hypre_ValDecSort(HYPRE_Int n, HYPRE_Int *idx, double *val)
{
  HYPRE_Int i, j, max;
  HYPRE_Int tmpidx;
  double tmpval;

  for (i=0; i<n; i++) {
    max = i;
    for (j=i+1; j<n; j++) {
      if (fabs(val[j]) > fabs(val[max]))
        max = j;
    }

    if (max != i) {
      SWAP(idx[i], idx[max], tmpidx);
      SWAP(val[i], val[max], tmpval);
    }
  }
}





/*************************************************************************
* This function takes an (idx, val) array and compacts it so that every 
* entry with idx[] = -1, gets removed. It returns the new count
**************************************************************************/
HYPRE_Int hypre_CompactIdx(HYPRE_Int n, HYPRE_Int *idx, double *val)
{
  HYPRE_Int i, j;

  j = n-1;
  for (i=0; i<n; i++) {
    if (idx[i] == -1) {
      while (j > i && idx[j] == -1)
        j--;
      if (j > i) {
        idx[i] = idx[j];
        val[i] = val[j];
        j--;
      }
      else {
        n = i;
        break;
      }
    }
    if (i == j) {
      n = i+1;
      break;
    }
  }

  return n;
}

/*************************************************************************
* This function prints an (idx, val) pair
**************************************************************************/
void hypre_PrintIdxVal(HYPRE_Int n, HYPRE_Int *idx, double *val)
{
  HYPRE_Int i;

  hypre_printf("%3d ", n);
  for (i=0; i<n; i++) 
    hypre_printf("(%3d, %3.1e) ", idx[i], val[i]);
  hypre_printf("\n");

}



/*************************************************************************
* This function compares 2 KeyValueType variables for sorting in inc order
**************************************************************************/
HYPRE_Int hypre_DecKeyValueCmp(const void *v1, const void *v2)
{
  KeyValueType *n1, *n2;

  n1 = (KeyValueType *)v1;
  n2 = (KeyValueType *)v2;

  return n2->key - n1->key;

}


/*************************************************************************
* This function sorts an array of type KeyValueType in increasing order
**************************************************************************/
void hypre_SortKeyValueNodesDec(KeyValueType *nodes, HYPRE_Int n)
{
  hypre_tex_qsort((void *)nodes, (size_t)n, (size_t)sizeof(KeyValueType), hypre_DecKeyValueCmp);
}


/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
HYPRE_Int hypre_sasum(HYPRE_Int n, HYPRE_Int *x)
{
  HYPRE_Int sum = 0;
  HYPRE_Int i;

  for (i=0; i<n; i++)
    sum += x[i];

  return sum;
}


/*************************************************************************
* This function compares 2 ints for sorting in inc order
**************************************************************************/
static HYPRE_Int incshort(const void *v1, const void *v2)
{
  return (*((HYPRE_Int *)v1) - *((HYPRE_Int *)v2));
}

/*************************************************************************
* This function compares 2 ints for sorting in dec order
**************************************************************************/
static HYPRE_Int decshort(const void *v1, const void *v2)
{
  return (*((HYPRE_Int *)v2) - *((HYPRE_Int *)v1));
}

/*************************************************************************
* These functions sorts an array of XXX
**************************************************************************/
void hypre_sincsort(HYPRE_Int n, HYPRE_Int *a)
{
  hypre_tex_qsort((void *)a, (size_t)n, (size_t)sizeof(HYPRE_Int), incshort);
}


void hypre_sdecsort(HYPRE_Int n, HYPRE_Int *a)
{
  hypre_tex_qsort((void *)a, (size_t)n, (size_t)sizeof(HYPRE_Int), decshort);
}





/*
 * util.c
 *
 * This function contains various utility routines
 *
 * Started 9/28/95
 * George
 *
 * $Id$
 */

#include "ilu.h"
#include "./DistributedMatrixPilutSolver.h"

/*************************************************************************
* This function finds the minimum value in the array removes it and
* returns it. It decreases the size of the array.
**************************************************************************/
int ExtractMinLR( hypre_PilutSolverGlobals *globals )
{
  int i, j=0 ;

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
void IdxIncSort(int n, int *idx, double *val)
{
  int i, j, min;
  double tmpval;
  int tmpidx;

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
void ValDecSort(int n, int *idx, double *val)
{
  int i, j, max;
  int tmpidx;
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
int CompactIdx(int n, int *idx, double *val)
{
  int i, j;

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
void PrintIdxVal(int n, int *idx, double *val)
{
  int i;

  printf("%3d ", n);
  for (i=0; i<n; i++) 
    printf("(%3d, %3.1e) ", idx[i], val[i]);
  printf("\n");

}



/*************************************************************************
* This function compares 2 KeyValueType variables for sorting in inc order
**************************************************************************/
int DecKeyValueCmp(const void *v1, const void *v2)
{
  KeyValueType *n1, *n2;

  n1 = (KeyValueType *)v1;
  n2 = (KeyValueType *)v2;

  return n2->key - n1->key;

}


/*************************************************************************
* This function sorts an array of type KeyValueType in increasing order
**************************************************************************/
void SortKeyValueNodesDec(KeyValueType *nodes, int n)
{
  tex_qsort((void *)nodes, (size_t)n, (size_t)sizeof(KeyValueType), DecKeyValueCmp);
}


/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
int sasum(int n, int *x)
{
  int sum = 0;
  int i;

  for (i=0; i<n; i++)
    sum += x[i];

  return sum;
}


/*************************************************************************
* This function compares 2 ints for sorting in inc order
**************************************************************************/
static int incshort(const void *v1, const void *v2)
{
  return (*((int *)v1) - *((int *)v2));
}

/*************************************************************************
* This function compares 2 ints for sorting in dec order
**************************************************************************/
static int decshort(const void *v1, const void *v2)
{
  return (*((int *)v2) - *((int *)v1));
}

/*************************************************************************
* These functions sorts an array of XXX
**************************************************************************/
void sincsort(int n, int *a)
{
  tex_qsort((void *)a, (size_t)n, (size_t)sizeof(int), incshort);
}


void sdecsort(int n, int *a)
{
  tex_qsort((void *)a, (size_t)n, (size_t)sizeof(int), decshort);
}





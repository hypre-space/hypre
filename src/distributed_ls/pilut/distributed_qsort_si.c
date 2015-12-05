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
 * distributed_qsort.c:
 * Our own version of the system qsort routine which is faster by an average
 * of 25%, with lows and highs of 10% and 50%.
 * The THRESHold below is the insertion sort threshold, and has been adjusted
 * for records of size 48 bytes.
 * The MTHREShold is where we stop finding a better median.
 */

#include <stdlib.h>			/* only for type declarations */
#include <stdio.h>			/* only for type declarations */

#include "ilu.h"

#define		THRESH		1	/* threshold for insertion */
#define		MTHRESH		6	/* threshold for median */

static void siqst(HYPRE_Int *, HYPRE_Int *);
static void sdqst(HYPRE_Int *, HYPRE_Int *);


/*
 * hypre_tex_qsort:
 * First, set up some global parameters for qst to share.  Then, quicksort
 * with qst(), and then a cleanup insertion sort ourselves.  Sound simple?
 * It's not...
 */

void hypre_sincsort_fast(HYPRE_Int n, HYPRE_Int *base)
{
  register HYPRE_Int *i;
  register HYPRE_Int *j;
  register HYPRE_Int *lo;
  register HYPRE_Int *hi;
  register HYPRE_Int *min;
  register HYPRE_Int c;
  HYPRE_Int *max;

  if (n <= 1)
    return;

  max = base + n;

  if (n >= THRESH) {
    siqst(base, max);
    hi = base + THRESH;
  }
  else 
    hi = max;


  /* First put smallest element, which must be in the first THRESH, in the
     first position as a sentinel.  This is done just by searching the
     first THRESH elements (or the first n if n < THRESH), finding the min,
     and swapping it into the first position. */
  for (j = lo = base; lo++ < hi;) {
    if (*j > *lo)
      j = lo;
  }
  if (j != base) { /* swap j into place */
    c = *base;
    *base = *j;
    *j = c;
  }

  /* With our sentinel in place, we now run the following hyper-fast
     insertion sort.  For each remaining element, min, from [1] to [n-1],
     set hi to the index of the element AFTER which this one goes. Then, do
     the standard insertion sort shift on a character at a time basis for
     each element in the frob. */
  for (min = base; (hi = min += 1) < max;) {
    while (*(--hi) > *min);
    if ((hi += 1) != min) {
      for (lo = min + 1; --lo >= min;) {
	c = *lo;
	for (i = j = lo; (j -= 1) >= hi; i = j)
	   *i = *j;
	*i = c;
      }
    }
  }
}



/*
 * qst:
 * Do a quicksort
 * First, find the median element, and put that one in the first place as the
 * discriminator.  (This "median" is just the median of the first, last and
 * middle elements).  (Using this median instead of the first element is a big
 * win).  Then, the usual partitioning/swapping, followed by moving the
 * discriminator into the right place.  Then, figure out the sizes of the two
 * partions, do the smaller one recursively and the larger one via a repeat of
 * this code.  Stopping when there are less than THRESH elements in a partition
 * and cleaning up with an insertion sort (in our caller) is a huge win.
 * All data swaps are done in-line, which is space-losing but time-saving.
 * (And there are only three places where this is done).
 */

static void siqst(HYPRE_Int *base, HYPRE_Int *max)
{
  register HYPRE_Int *i;
  register HYPRE_Int *j;
  register HYPRE_Int *jj;
  register HYPRE_Int *mid;
  register HYPRE_Int c;
  HYPRE_Int *tmp;
  HYPRE_Int lo;
  HYPRE_Int hi;

  lo = max - base;		/* number of elements as shorts */
  do {
    /* At the top here, lo is the number of characters of elements in the
       current partition.  (Which should be max - base). Find the median
       of the first, last, and middle element and make that the middle
       element.  Set j to largest of first and middle.  If max is larger
       than that guy, then it's that guy, else compare max with loser of
       first and take larger.  Things are set up to prefer the middle,
       then the first in case of ties. */
    mid = base + ((unsigned) lo>>1);
    if (lo >= MTHRESH) {
      j = (*base > *mid ? base : mid);
      tmp = max - 1;
      if (*j > *tmp) {
        j = (j == base ? mid : base); /* switch to first loser */
        if (*j < *tmp)
          j = tmp;
      }

      if (j != mid) {  /* SWAP */ 
        c = *mid;
        *mid = *j;
        *j = c;
      }
    }

    /* Semi-standard quicksort partitioning/swapping */
    for (i = base, j = max - 1;;) {
      while (i < mid && *i <= *mid)
        i++;
      while (j > mid) {
        if (*mid <= *j) {
          j--;
          continue;
        }
        tmp = i + 1;	/* value of i after swap */
        if (i == mid) 	/* j <-> mid, new mid is j */
          mid = jj = j;
        else 		/* i <-> j */
          jj = j--;
        goto swap;
      }

      if (i == mid) 
	break;
      else {		/* i <-> mid, new mid is i */
        jj = mid;
        tmp = mid = i;	/* value of i after swap */
        j--;
      }
swap:
      c = *i;
      *i = *jj;
      *jj = c;
      i = tmp;
    }

    /* Look at sizes of the two partitions, do the smaller one first by
       recursion, then do the larger one by making sure lo is its size,
       base and max are update correctly, and branching back. But only
       repeat (recursively or by branching) if the partition is of at
       least size THRESH. */
    i = (j = mid) + 1;
    if ((lo = j - base) <= (hi = max - i)) {
      if (lo >= THRESH)
        siqst(base, j);
      base = i;
      lo = hi;
    }
    else {
      if (hi >= THRESH)
        siqst(i, max);
      max = j;
    }
  } while (lo >= THRESH);
}


/*************************************************************************
* A decreasing sort of HYPRE_Int ints 
**************************************************************************/
void hypre_sdecsort_fast(HYPRE_Int n, HYPRE_Int *base)
{
  register HYPRE_Int *i;
  register HYPRE_Int *j;
  register HYPRE_Int *lo;
  register HYPRE_Int *hi;
  register HYPRE_Int *min;
  register HYPRE_Int c;
  HYPRE_Int *max;

  if (n <= 1)
    return;

  max = base + n;

  if (n >= THRESH) {
    sdqst(base, max);
    hi = base + THRESH;
  }
  else 
    hi = max;


  /* First put smallest element, which must be in the first THRESH, in the
     first position as a sentinel.  This is done just by searching the
     first THRESH elements (or the first n if n < THRESH), finding the min,
     and swapping it into the first position. */
  for (j = lo = base; lo++ < hi;) {
    if (*j < *lo)
      j = lo;
  }
  if (j != base) { /* swap j into place */
    c = *base;
    *base = *j;
    *j = c;
  }

  /* With our sentinel in place, we now run the following hyper-fast
     insertion sort.  For each remaining element, min, from [1] to [n-1],
     set hi to the index of the element AFTER which this one goes. Then, do
     the standard insertion sort shift on a character at a time basis for
     each element in the frob. */
  for (min = base; (hi = min += 1) < max;) {
    while (*(--hi) < *min);
    if ((hi += 1) != min) {
      for (lo = min + 1; --lo >= min;) {
	c = *lo;
	for (i = j = lo; (j -= 1) >= hi; i = j)
	   *i = *j;
	*i = c;
      }
    }
  }
}



static void sdqst(HYPRE_Int *base, HYPRE_Int *max)
{
  register HYPRE_Int *i;
  register HYPRE_Int *j;
  register HYPRE_Int *jj;
  register HYPRE_Int *mid;
  register HYPRE_Int c;
  HYPRE_Int *tmp;
  HYPRE_Int lo;
  HYPRE_Int hi;

  lo = max - base;		/* number of elements as shorts */
  do {
    /* At the top here, lo is the number of characters of elements in the
       current partition.  (Which should be max - base). Find the median
       of the first, last, and middle element and make that the middle
       element.  Set j to largest of first and middle.  If max is larger
       than that guy, then it's that guy, else compare max with loser of
       first and take larger.  Things are set up to prefer the middle,
       then the first in case of ties. */
    mid = base + ((unsigned) lo>>1);
    if (lo >= MTHRESH) {
      j = (*base < *mid ? base : mid);
      tmp = max - 1;
      if (*j < *tmp) {
        j = (j == base ? mid : base); /* switch to first loser */
        if (*j > *tmp)
          j = tmp;
      }

      if (j != mid) {  /* SWAP */ 
        c = *mid;
        *mid = *j;
        *j = c;
      }
    }

    /* Semi-standard quicksort partitioning/swapping */
    for (i = base, j = max - 1;;) {
      while (i < mid && *i >= *mid)
        i++;
      while (j > mid) {
        if (*mid >= *j) {
          j--;
          continue;
        }
        tmp = i + 1;	/* value of i after swap */
        if (i == mid) 	/* j <-> mid, new mid is j */
          mid = jj = j;
        else 		/* i <-> j */
          jj = j--;
        goto swap;
      }

      if (i == mid) 
	break;
      else {		/* i <-> mid, new mid is i */
        jj = mid;
        tmp = mid = i;	/* value of i after swap */
        j--;
      }
swap:
      c = *i;
      *i = *jj;
      *jj = c;
      i = tmp;
    }

    /* Look at sizes of the two partitions, do the smaller one first by
       recursion, then do the larger one by making sure lo is its size,
       base and max are update correctly, and branching back. But only
       repeat (recursively or by branching) if the partition is of at
       least size THRESH. */
    i = (j = mid) + 1;
    if ((lo = j - base) <= (hi = max - i)) {
      if (lo >= THRESH)
        sdqst(base, j);
      base = i;
      lo = hi;
    }
    else {
      if (hi >= THRESH)
        sdqst(i, max);
      max = j;
    }
  } while (lo >= THRESH);
}

/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * OrderStat - Utility functions for selecting the i-th order statistic,
 * i.e., the i-th smallest element in a list of n elements.  There is one
 * user function in this file:  randomized_select(a, p, r, i), which 
 * selects the i-th order statistic from the double precision array a[p:r].
   The contents of the array are altered by the function.
 *
 * Reference: Cormen, Leiserson, Rivest, Introduction to Algorithms, p. 187.
 *
 *****************************************************************************/

#include <stdlib.h>
#include "OrderStat.h"

/*--------------------------------------------------------------------------
 * partition - Return q such that a[p:q] has no element greater than 
 * elements in a[q+1:r].
 *--------------------------------------------------------------------------*/

static HYPRE_Int partition(double *a, HYPRE_Int p, HYPRE_Int r)
{
    double x, temp;
    HYPRE_Int i, j;

    x = a[p];
    i = p - 1;
    j = r + 1;

    while (1)
    {
	do
	    j--;
	while (a[j] > x);

	do
	    i++;
	while (a[i] < x);

	if (i < j)
	{
	    temp = a[i];
	    a[i] = a[j];
	    a[j] = temp;
	}
	else
	    return j;

    }
}

/*--------------------------------------------------------------------------
 * randomized_partition - Randomizies the partitioning function by selecting
 * a random pivot element.
 *--------------------------------------------------------------------------*/

static HYPRE_Int randomized_partition(double *a, HYPRE_Int p, HYPRE_Int r)
{
    double temp;
    HYPRE_Int i;

    /* select a random number in [p,r] */
    i = p + (rand() % (r-p+1));

    temp = a[i];
    a[i] = a[p];
    a[p] = temp;

    return partition(a, p, r);
}

/*--------------------------------------------------------------------------
 * randomized_select - Return the i-th smallest element of the double 
 * precision array a[p:r].  The contents of the array are altered on return.
 * "i" should range from 1 to r-p+1.
 *--------------------------------------------------------------------------*/

double randomized_select(double *a, HYPRE_Int p, HYPRE_Int r, HYPRE_Int i)
{
    HYPRE_Int q, k;

    if (p == r)
	return a[p];

    q = randomized_partition(a, p, r);

    /* number of elements in the first list */
    k = q - p + 1;

    if (i <= k)
	return randomized_select(a, p, q, i);
    else
	return randomized_select(a, q+1, r, i-k);
}

/*--------------------------------------------------------------------------
 * shell_sort - sorts x[0:n-1] in place, ascending order
 *--------------------------------------------------------------------------*/

void shell_sort(const HYPRE_Int n, HYPRE_Int x[])
{
    HYPRE_Int m, max, j, k, itemp;

    m = n/2;

    while (m > 0)
    {
        max = n - m;
        for (j=0; j<max; j++)
        {
            for (k=j; k>=0; k-=m)
            {
                if (x[k+m] >= x[k])
                    break;
                itemp = x[k+m];
                x[k+m] = x[k];
                x[k] = itemp;
            }
        }
        m = m/2;
    }
}


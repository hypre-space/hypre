/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * OrderStat - Utility functions for selecting the i-th order statistic,
 * i.e., the i-th smallest element in a list of n elements.  There is one
 * user function in this file:  randomized_select(a, p, r, i), which 
 * selects the i-th order statistic from the HYPRE_Real precision array a[p:r].
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

static HYPRE_Int partition(HYPRE_Real *a, HYPRE_Int p, HYPRE_Int r)
{
    HYPRE_Real x, temp;
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

static HYPRE_Int randomized_partition(HYPRE_Real *a, HYPRE_Int p, HYPRE_Int r)
{
    HYPRE_Real temp;
    HYPRE_Int i;

    /* select a random number in [p,r] */
    i = p + (rand() % (r-p+1));

    temp = a[i];
    a[i] = a[p];
    a[p] = temp;

    return partition(a, p, r);
}

/*--------------------------------------------------------------------------
 * randomized_select - Return the i-th smallest element of the HYPRE_Real 
 * precision array a[p:r].  The contents of the array are altered on return.
 * "i" should range from 1 to r-p+1.
 *--------------------------------------------------------------------------*/

HYPRE_Real randomized_select(HYPRE_Real *a, HYPRE_Int p, HYPRE_Int r, HYPRE_Int i)
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
 * hypre_shell_sort - sorts x[0:n-1] in place, ascending order
 *--------------------------------------------------------------------------*/

void hypre_shell_sort(const HYPRE_Int n, HYPRE_Int x[])
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


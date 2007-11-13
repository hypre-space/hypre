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

static int partition(double *a, int p, int r)
{
    double x, temp;
    int i, j;

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

static int randomized_partition(double *a, int p, int r)
{
    double temp;
    int i;

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

double randomized_select(double *a, int p, int r, int i)
{
    int q, k;

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

void shell_sort(const int n, int x[])
{
    int m, max, j, k, itemp;

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


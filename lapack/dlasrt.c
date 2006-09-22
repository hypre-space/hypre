/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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



#include "hypre_lapack.h"
#include "f2c.h"

/* Subroutine */ int dlasrt_(char *id, integer *n, doublereal *d__, integer *
	info)
{
/*  -- LAPACK routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       September 30, 1994   


    Purpose   
    =======   

    Sort the numbers in D in increasing order (if ID = 'I') or   
    in decreasing order (if ID = 'D' ).   

    Use Quick Sort, reverting to Insertion sort on arrays of   
    size <= 20. Dimension of STACK limits N to about 2**32.   

    Arguments   
    =========   

    ID      (input) CHARACTER*1   
            = 'I': sort D in increasing order;   
            = 'D': sort D in decreasing order.   

    N       (input) INTEGER   
            The length of the array D.   

    D       (input/output) DOUBLE PRECISION array, dimension (N)   
            On entry, the array to be sorted.   
            On exit, D has been sorted into increasing order   
            (D(1) <= ... <= D(N) ) or into decreasing order   
            (D(1) >= ... >= D(N) ), depending on ID.   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    =====================================================================   


       Test the input paramters.   

       Parameter adjustments */
    /* System generated locals */
    integer i__1, i__2;
    /* Local variables */
    static integer endd, i__, j;
    extern logical lsame_(char *, char *);
    static integer stack[64]	/* was [2][32] */;
    static doublereal dmnmx, d1, d2, d3;
    static integer start;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static integer stkpnt, dir;
    static doublereal tmp;
#define stack_ref(a_1,a_2) stack[(a_2)*2 + a_1 - 3]

    --d__;

    /* Function Body */
    *info = 0;
    dir = -1;
    if (lsame_(id, "D")) {
	dir = 0;
    } else if (lsame_(id, "I")) {
	dir = 1;
    }
    if (dir == -1) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLASRT", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 1) {
	return 0;
    }

    stkpnt = 1;
    stack_ref(1, 1) = 1;
    stack_ref(2, 1) = *n;
L10:
    start = stack_ref(1, stkpnt);
    endd = stack_ref(2, stkpnt);
    --stkpnt;
    if (endd - start <= 20 && endd - start > 0) {

/*        Do Insertion sort on D( START:ENDD ) */

	if (dir == 0) {

/*           Sort into decreasing order */

	    i__1 = endd;
	    for (i__ = start + 1; i__ <= i__1; ++i__) {
		i__2 = start + 1;
		for (j = i__; j >= i__2; --j) {
		    if (d__[j] > d__[j - 1]) {
			dmnmx = d__[j];
			d__[j] = d__[j - 1];
			d__[j - 1] = dmnmx;
		    } else {
			goto L30;
		    }
/* L20: */
		}
L30:
		;
	    }

	} else {

/*           Sort into increasing order */

	    i__1 = endd;
	    for (i__ = start + 1; i__ <= i__1; ++i__) {
		i__2 = start + 1;
		for (j = i__; j >= i__2; --j) {
		    if (d__[j] < d__[j - 1]) {
			dmnmx = d__[j];
			d__[j] = d__[j - 1];
			d__[j - 1] = dmnmx;
		    } else {
			goto L50;
		    }
/* L40: */
		}
L50:
		;
	    }

	}

    } else if (endd - start > 20) {

/*        Partition D( START:ENDD ) and stack parts, largest one first   

          Choose partition entry as median of 3 */

	d1 = d__[start];
	d2 = d__[endd];
	i__ = (start + endd) / 2;
	d3 = d__[i__];
	if (d1 < d2) {
	    if (d3 < d1) {
		dmnmx = d1;
	    } else if (d3 < d2) {
		dmnmx = d3;
	    } else {
		dmnmx = d2;
	    }
	} else {
	    if (d3 < d2) {
		dmnmx = d2;
	    } else if (d3 < d1) {
		dmnmx = d3;
	    } else {
		dmnmx = d1;
	    }
	}

	if (dir == 0) {

/*           Sort into decreasing order */

	    i__ = start - 1;
	    j = endd + 1;
L60:
L70:
	    --j;
	    if (d__[j] < dmnmx) {
		goto L70;
	    }
L80:
	    ++i__;
	    if (d__[i__] > dmnmx) {
		goto L80;
	    }
	    if (i__ < j) {
		tmp = d__[i__];
		d__[i__] = d__[j];
		d__[j] = tmp;
		goto L60;
	    }
	    if (j - start > endd - j - 1) {
		++stkpnt;
		stack_ref(1, stkpnt) = start;
		stack_ref(2, stkpnt) = j;
		++stkpnt;
		stack_ref(1, stkpnt) = j + 1;
		stack_ref(2, stkpnt) = endd;
	    } else {
		++stkpnt;
		stack_ref(1, stkpnt) = j + 1;
		stack_ref(2, stkpnt) = endd;
		++stkpnt;
		stack_ref(1, stkpnt) = start;
		stack_ref(2, stkpnt) = j;
	    }
	} else {

/*           Sort into increasing order */

	    i__ = start - 1;
	    j = endd + 1;
L90:
L100:
	    --j;
	    if (d__[j] > dmnmx) {
		goto L100;
	    }
L110:
	    ++i__;
	    if (d__[i__] < dmnmx) {
		goto L110;
	    }
	    if (i__ < j) {
		tmp = d__[i__];
		d__[i__] = d__[j];
		d__[j] = tmp;
		goto L90;
	    }
	    if (j - start > endd - j - 1) {
		++stkpnt;
		stack_ref(1, stkpnt) = start;
		stack_ref(2, stkpnt) = j;
		++stkpnt;
		stack_ref(1, stkpnt) = j + 1;
		stack_ref(2, stkpnt) = endd;
	    } else {
		++stkpnt;
		stack_ref(1, stkpnt) = j + 1;
		stack_ref(2, stkpnt) = endd;
		++stkpnt;
		stack_ref(1, stkpnt) = start;
		stack_ref(2, stkpnt) = j;
	    }
	}
    }
    if (stkpnt > 0) {
	goto L10;
    }
    return 0;

/*     End of DLASRT */

} /* dlasrt_ */

#undef stack_ref



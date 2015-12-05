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
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/


#include "../blas/hypre_blas.h"
#include "hypre_lapack.h"
#include "f2c.h"

/* Subroutine */ int dorg2l_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *info)
{
/*  -- LAPACK routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       February 29, 1992   


    Purpose   
    =======   

    DORG2L generates an m by n real matrix Q with orthonormal columns,   
    which is defined as the last n columns of a product of k elementary   
    reflectors of order m   

          Q  =  H(k) . . . H(2) H(1)   

    as returned by DGEQLF.   

    Arguments   
    =========   

    M       (input) INTEGER   
            The number of rows of the matrix Q. M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix Q. M >= N >= 0.   

    K       (input) INTEGER   
            The number of elementary reflectors whose product defines the   
            matrix Q. N >= K >= 0.   

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)   
            On entry, the (n-k+i)-th column must contain the vector which   
            defines the elementary reflector H(i), for i = 1,2,...,k, as   
            returned by DGEQLF in the last k columns of its array   
            argument A.   
            On exit, the m by n matrix Q.   

    LDA     (input) INTEGER   
            The first dimension of the array A. LDA >= max(1,M).   

    TAU     (input) DOUBLE PRECISION array, dimension (K)   
            TAU(i) must contain the scalar factor of the elementary   
            reflector H(i), as returned by DGEQLF.   

    WORK    (workspace) DOUBLE PRECISION array, dimension (N)   

    INFO    (output) INTEGER   
            = 0: successful exit   
            < 0: if INFO = -i, the i-th argument has an illegal value   

    =====================================================================   


       Test the input arguments   

       Parameter adjustments */
    /* Table of constant values */
    static integer c__1 = 1;
    
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublereal d__1;
    /* Local variables */
    static integer i__, j, l;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *), dlarf_(char *, integer *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, integer *, doublereal *);
    static integer ii;
    extern /* Subroutine */ int xerbla_(char *, integer *);
#define a_ref(a_1,a_2) a[(a_2)*a_dim1 + a_1]


    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0 || *n > *m) {
	*info = -2;
    } else if (*k < 0 || *k > *n) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORG2L", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 0) {
	return 0;
    }

/*     Initialise columns 1:n-k to columns of the unit matrix */

    i__1 = *n - *k;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (l = 1; l <= i__2; ++l) {
	    a_ref(l, j) = 0.;
/* L10: */
	}
	a_ref(*m - *n + j, j) = 1.;
/* L20: */
    }

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ii = *n - *k + i__;

/*        Apply H(i) to A(1:m-k+i,1:n-k+i) from the left */

	a_ref(*m - *n + ii, ii) = 1.;
	i__2 = *m - *n + ii;
	i__3 = ii - 1;
	dlarf_("Left", &i__2, &i__3, &a_ref(1, ii), &c__1, &tau[i__], &a[
		a_offset], lda, &work[1]);
	i__2 = *m - *n + ii - 1;
	d__1 = -tau[i__];
	dscal_(&i__2, &d__1, &a_ref(1, ii), &c__1);
	a_ref(*m - *n + ii, ii) = 1. - tau[i__];

/*        Set A(m-k+i+1:m,n-k+i) to zero */

	i__2 = *m;
	for (l = *m - *n + ii + 1; l <= i__2; ++l) {
	    a_ref(l, ii) = 0.;
/* L30: */
	}
/* L40: */
    }
    return 0;

/*     End of DORG2L */

} /* dorg2l_ */

#undef a_ref



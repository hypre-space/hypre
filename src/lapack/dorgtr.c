/* Copyright (c) 1992-2008 The University of Tennessee.  All rights reserved.
 * See file COPYING in this directory for details. */

#ifdef __cplusplus
extern "C" {
#endif

#include "f2c.h"
#include "hypre_lapack.h"

/* Subroutine */ integer dorgtr_(const char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info)
{
/*  -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    DORGTR generates a real orthogonal matrix Q which is defined as the
    product of n-1 elementary reflectors of order N, as returned by
    DSYTRD:

    if UPLO = 'U', Q = H(n-1) . . . H(2) H(1),

    if UPLO = 'L', Q = H(1) H(2) . . . H(n-1).

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            = 'U': Upper triangle of A contains elementary reflectors
                   from DSYTRD;
            = 'L': Lower triangle of A contains elementary reflectors
                   from DSYTRD.

    N       (input) INTEGER
            The order of the matrix Q. N >= 0.

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the vectors which define the elementary reflectors,
            as returned by DSYTRD.
            On exit, the N-by-N orthogonal matrix Q.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,N).

    TAU     (input) DOUBLE PRECISION array, dimension (N-1)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by DSYTRD.

    WORK    (workspace/output) DOUBLE PRECISION array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK. LWORK >= max(1,N-1).
            For optimum performance LWORK >= (N-1)*NB, where NB is
            the optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input arguments

       Parameter adjustments */
    /* Table of constant values */
    integer c__1 = 1;
    integer c_n1 = -1;

    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    /* Local variables */
    integer i__, j;
    extern logical lsame_(const char *,const char *);
    integer iinfo;
    logical upper;
    integer nb;
    extern /* Subroutine */ integer xerbla_(const char *, integer *);
    extern integer ilaenv_(integer *,const char *,const char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ integer dorgql_(integer *, integer *, integer *,
	    doublereal *, integer *, doublereal *, doublereal *, integer *,
	    integer *), dorgqr_(integer *, integer *, integer *, doublereal *,
	     integer *, doublereal *, doublereal *, integer *, integer *);
    integer lwkopt;
    logical lquery;
#define a_ref(a_1,a_2) a[(a_2)*a_dim1 + a_1]


    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    lquery = *lwork == -1;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__1 = 1, i__2 = *n - 1;
	if (*lwork < max(i__1,i__2) && ! lquery) {
	    *info = -7;
	}
    }

    if (*info == 0) {
	if (upper) {
	    i__1 = *n - 1;
	    i__2 = *n - 1;
	    i__3 = *n - 1;
	    nb = ilaenv_(&c__1, "DORGQL", " ", &i__1, &i__2, &i__3, &c_n1, (
		    ftnlen)6, (ftnlen)1);
	} else {
	    i__1 = *n - 1;
	    i__2 = *n - 1;
	    i__3 = *n - 1;
	    nb = ilaenv_(&c__1, "DORGQR", " ", &i__1, &i__2, &i__3, &c_n1, (
		    ftnlen)6, (ftnlen)1);
	}
/* Computing MAX */
	i__1 = 1, i__2 = *n - 1;
	lwkopt = max(i__1,i__2) * nb;
	work[1] = (doublereal) lwkopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORGTR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	work[1] = 1.;
	return 0;
    }

    if (upper) {

/*        Q was determined by a call to DSYTRD with UPLO = 'U'

          Shift the vectors which define the elementary reflectors one
          column to the left, and set the last row and column of Q to
          those of the unit matrix */

	i__1 = *n - 1;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a_ref(i__, j) = a_ref(i__, j + 1);
/* L10: */
	    }
	    a_ref(*n, j) = 0.;
/* L20: */
	}
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    a_ref(i__, *n) = 0.;
/* L30: */
	}
	a_ref(*n, *n) = 1.;

/*        Generate Q(1:n-1,1:n-1) */

	i__1 = *n - 1;
	i__2 = *n - 1;
	i__3 = *n - 1;
	dorgql_(&i__1, &i__2, &i__3, &a[a_offset], lda, &tau[1], &work[1],
		lwork, &iinfo);

    } else {

/*        Q was determined by a call to DSYTRD with UPLO = 'L'.

          Shift the vectors which define the elementary reflectors one
          column to the right, and set the first row and column of Q to
          those of the unit matrix */

	for (j = *n; j >= 2; --j) {
	    a_ref(1, j) = 0.;
	    i__1 = *n;
	    for (i__ = j + 1; i__ <= i__1; ++i__) {
		a_ref(i__, j) = a_ref(i__, j - 1);
/* L40: */
	    }
/* L50: */
	}
	a_ref(1, 1) = 1.;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    a_ref(i__, 1) = 0.;
/* L60: */
	}
	if (*n > 1) {

/*           Generate Q(2:n,2:n) */

	    i__1 = *n - 1;
	    i__2 = *n - 1;
	    i__3 = *n - 1;
	    dorgqr_(&i__1, &i__2, &i__3, &a_ref(2, 2), lda, &tau[1], &work[1],
		     lwork, &iinfo);
	}
    }
    work[1] = (doublereal) lwkopt;
    return 0;

/*     End of DORGTR */

} /* dorgtr_ */

#undef a_ref

#ifdef __cplusplus
}
#endif

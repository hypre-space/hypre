/* Copyright (c) 1992-2008 The University of Tennessee.  All rights reserved.
 * See file COPYING in this directory for details. */

#ifdef __cplusplus
extern "C" {
#endif

#include "f2c.h"
#include "hypre_lapack.h"

/* Subroutine */ integer dtrti2_(const char *uplo, const char *diag, integer *n, doublereal *
	a, integer *lda, integer *info)
{
/*  -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    DTRTI2 computes the inverse of a real upper or lower triangular
    matrix.

    This is the Level 2 BLAS version of the algorithm.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies whether the matrix A is upper or lower triangular.
            = 'U':  Upper triangular
            = 'L':  Lower triangular

    DIAG    (input) CHARACTER*1
            Specifies whether or not the matrix A is unit triangular.
            = 'N':  Non-unit triangular
            = 'U':  Unit triangular

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the triangular matrix A.  If UPLO = 'U', the
            leading n by n upper triangular part of the array A contains
            the upper triangular matrix, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading n by n lower triangular part of the array A contains
            the lower triangular matrix, and the strictly upper
            triangular part of A is not referenced.  If DIAG = 'U', the
            diagonal elements of A are also not referenced and are
            assumed to be 1.

            On exit, the (triangular) inverse of the original matrix, in
            the same storage format.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value

    =====================================================================


       Test the input parameters.

       Parameter adjustments */
    /* Table of constant values */
    integer c__1 = 1;

    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    /* Local variables */
    integer j;
    extern /* Subroutine */ integer dscal_(integer *, doublereal *, doublereal *,
	    integer *);
    extern logical lsame_(const char *, const char *);
    logical upper;
    extern /* Subroutine */ integer dtrmv_(const char *, const char *, const char *, integer *,
	    doublereal *, integer *, doublereal *, integer *), xerbla_(const char *, integer *);
    logical nounit;
    doublereal ajj;
#define a_ref(a_1,a_2) a[(a_2)*a_dim1 + a_1]


    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    nounit = lsame_(diag, "N");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (! nounit && ! lsame_(diag, "U")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DTRTI2", &i__1);
	return 0;
    }

    if (upper) {

/*        Compute inverse of upper triangular matrix. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (nounit) {
		a_ref(j, j) = 1. / a_ref(j, j);
		ajj = -a_ref(j, j);
	    } else {
		ajj = -1.;
	    }

/*           Compute elements 1:j-1 of j-th column. */

	    i__2 = j - 1;
	    dtrmv_("Upper", "No transpose", diag, &i__2, &a[a_offset], lda, &
		    a_ref(1, j), &c__1);
	    i__2 = j - 1;
	    dscal_(&i__2, &ajj, &a_ref(1, j), &c__1);
/* L10: */
	}
    } else {

/*        Compute inverse of lower triangular matrix. */

	for (j = *n; j >= 1; --j) {
	    if (nounit) {
		a_ref(j, j) = 1. / a_ref(j, j);
		ajj = -a_ref(j, j);
	    } else {
		ajj = -1.;
	    }
	    if (j < *n) {

/*              Compute elements j+1:n of j-th column. */

		i__1 = *n - j;
		dtrmv_("Lower", "No transpose", diag, &i__1, &a_ref(j + 1, j
			+ 1), lda, &a_ref(j + 1, j), &c__1);
		i__1 = *n - j;
		dscal_(&i__1, &ajj, &a_ref(j + 1, j), &c__1);
	    }
/* L20: */
	}
    }

    return 0;

/*     End of DTRTI2 */

} /* dtrti2_ */

#undef a_ref

#ifdef __cplusplus
}
#endif

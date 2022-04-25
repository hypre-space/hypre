/* Copyright (c) 1992-2008 The University of Tennessee.  All rights reserved.
 * See file COPYING in this directory for details. */

#ifdef __cplusplus
extern "C" {
#endif

#include "f2c.h"
#include "hypre_lapack.h"

/* Subroutine */ integer dsytd2_(const char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *d__, doublereal *e, doublereal *tau, integer *info)
{
/*  -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    DSYTD2 reduces a real symmetric matrix A to symmetric tridiagonal
    form T by an orthogonal similarity transformation: Q' * A * Q = T.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is stored:
            = 'U':  Upper triangular
            = 'L':  Lower triangular

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading
            n-by-n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading n-by-n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit, if UPLO = 'U', the diagonal and first superdiagonal
            of A are overwritten by the corresponding elements of the
            tridiagonal matrix T, and the elements above the first
            superdiagonal, with the array TAU, represent the orthogonal
            matrix Q as a product of elementary reflectors; if UPLO
            = 'L', the diagonal and first subdiagonal of A are over-
            written by the corresponding elements of the tridiagonal
            matrix T, and the elements below the first subdiagonal, with
            the array TAU, represent the orthogonal matrix Q as a product
            of elementary reflectors. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    D       (output) DOUBLE PRECISION array, dimension (N)
            The diagonal elements of the tridiagonal matrix T:
            D(i) = A(i,i).

    E       (output) DOUBLE PRECISION array, dimension (N-1)
            The off-diagonal elements of the tridiagonal matrix T:
            E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'.

    TAU     (output) DOUBLE PRECISION array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    If UPLO = 'U', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(n-1) . . . H(2) H(1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in
    A(1:i-1,i+1), and tau in TAU(i).

    If UPLO = 'L', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(1) H(2) . . . H(n-1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in A(i+2:n,i),
    and tau in TAU(i).

    The contents of A on exit are illustrated by the following examples
    with n = 5:

    if UPLO = 'U':                       if UPLO = 'L':

      (  d   e   v2  v3  v4 )              (  d                  )
      (      d   e   v3  v4 )              (  e   d              )
      (          d   e   v4 )              (  v1  e   d          )
      (              d   e  )              (  v1  v2  e   d      )
      (                  d  )              (  v1  v2  v3  e   d  )

    where d and e denote diagonal and off-diagonal elements of T, and vi
    denotes an element of the vector defining H(i).

    =====================================================================


       Test the input parameters

       Parameter adjustments */
    /* Table of constant values */
    integer c__1 = 1;
    doublereal c_b8 = 0.;
    doublereal c_b14 = -1.;

    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    /* Local variables */
    extern doublereal ddot_(integer *, doublereal *, integer *, doublereal *,
	    integer *);
    doublereal taui;
    extern /* Subroutine */ integer dsyr2_(const char *, integer *, doublereal *,
	    doublereal *, integer *, doublereal *, integer *, doublereal *,
	    integer *);
    integer i__;
    doublereal alpha;
    extern logical lsame_(const char *,const char *);
    extern /* Subroutine */ integer daxpy_(integer *, doublereal *, doublereal *,
	    integer *, doublereal *, integer *);
    logical upper;
    extern /* Subroutine */ integer dsymv_(const char *, integer *, doublereal *,
	    doublereal *, integer *, doublereal *, integer *, doublereal *,
	    doublereal *, integer *), dlarfg_(integer *, doublereal *,
	     doublereal *, integer *, doublereal *), xerbla_(const char *, integer *
	    );
#define a_ref(a_1,a_2) a[(a_2)*a_dim1 + a_1]


    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --d__;
    --e;
    --tau;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSYTD2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 0) {
	return 0;
    }

    if (upper) {

/*        Reduce the upper triangle of A */

	for (i__ = *n - 1; i__ >= 1; --i__) {

/*           Generate elementary reflector H(i) = I - tau * v * v'
             to annihilate A(1:i-1,i+1) */

	    dlarfg_(&i__, &a_ref(i__, i__ + 1), &a_ref(1, i__ + 1), &c__1, &
		    taui);
	    e[i__] = a_ref(i__, i__ + 1);

	    if (taui != 0.) {

/*              Apply H(i) from both sides to A(1:i,1:i) */

		a_ref(i__, i__ + 1) = 1.;

/*              Compute  x := tau * A * v  storing x in TAU(1:i) */

		dsymv_(uplo, &i__, &taui, &a[a_offset], lda, &a_ref(1, i__ +
			1), &c__1, &c_b8, &tau[1], &c__1);

/*              Compute  w := x - 1/2 * tau * (x'*v) * v */

		alpha = taui * -.5 * ddot_(&i__, &tau[1], &c__1, &a_ref(1,
			i__ + 1), &c__1);
		daxpy_(&i__, &alpha, &a_ref(1, i__ + 1), &c__1, &tau[1], &
			c__1);

/*              Apply the transformation as a rank-2 update:
                   A := A - v * w' - w * v' */

		dsyr2_(uplo, &i__, &c_b14, &a_ref(1, i__ + 1), &c__1, &tau[1],
			 &c__1, &a[a_offset], lda);

		a_ref(i__, i__ + 1) = e[i__];
	    }
	    d__[i__ + 1] = a_ref(i__ + 1, i__ + 1);
	    tau[i__] = taui;
/* L10: */
	}
	d__[1] = a_ref(1, 1);
    } else {

/*        Reduce the lower triangle of A */

	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Generate elementary reflector H(i) = I - tau * v * v'
             to annihilate A(i+2:n,i)

   Computing MIN */
	    i__2 = i__ + 2;
	    i__3 = *n - i__;
	    dlarfg_(&i__3, &a_ref(i__ + 1, i__), &a_ref(min(i__2,*n), i__), &
		    c__1, &taui);
	    e[i__] = a_ref(i__ + 1, i__);

	    if (taui != 0.) {

/*              Apply H(i) from both sides to A(i+1:n,i+1:n) */

		a_ref(i__ + 1, i__) = 1.;

/*              Compute  x := tau * A * v  storing y in TAU(i:n-1) */

		i__2 = *n - i__;
		dsymv_(uplo, &i__2, &taui, &a_ref(i__ + 1, i__ + 1), lda, &
			a_ref(i__ + 1, i__), &c__1, &c_b8, &tau[i__], &c__1);

/*              Compute  w := x - 1/2 * tau * (x'*v) * v */

		i__2 = *n - i__;
		alpha = taui * -.5 * ddot_(&i__2, &tau[i__], &c__1, &a_ref(
			i__ + 1, i__), &c__1);
		i__2 = *n - i__;
		daxpy_(&i__2, &alpha, &a_ref(i__ + 1, i__), &c__1, &tau[i__],
			&c__1);

/*              Apply the transformation as a rank-2 update:
                   A := A - v * w' - w * v' */

		i__2 = *n - i__;
		dsyr2_(uplo, &i__2, &c_b14, &a_ref(i__ + 1, i__), &c__1, &tau[
			i__], &c__1, &a_ref(i__ + 1, i__ + 1), lda)
			;

		a_ref(i__ + 1, i__) = e[i__];
	    }
	    d__[i__] = a_ref(i__, i__);
	    tau[i__] = taui;
/* L20: */
	}
	d__[*n] = a_ref(*n, *n);
    }

    return 0;

/*     End of DSYTD2 */

} /* dsytd2_ */

#undef a_ref

#ifdef __cplusplus
}
#endif

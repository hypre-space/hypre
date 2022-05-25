/* Copyright (c) 1992-2008 The University of Tennessee.  All rights reserved.
 * See file COPYING in this directory for details. */

#ifdef __cplusplus
extern "C" {
#endif

#include "f2c.h"
#include "hypre_lapack.h"

/* Subroutine */ integer dlatrd_(const char *uplo, integer *n, integer *nb, doublereal *
	a, integer *lda, doublereal *e, doublereal *tau, doublereal *w,
	integer *ldw)
{
/*  -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    DLATRD reduces NB rows and columns of a real symmetric matrix A to
    symmetric tridiagonal form by an orthogonal similarity
    transformation Q' * A * Q, and returns the matrices V and W which are
    needed to apply the transformation to the unreduced part of A.

    If UPLO = 'U', DLATRD reduces the last NB rows and columns of a
    matrix, of which the upper triangle is supplied;
    if UPLO = 'L', DLATRD reduces the first NB rows and columns of a
    matrix, of which the lower triangle is supplied.

    This is an auxiliary routine called by DSYTRD.

    Arguments
    =========

    UPLO    (input) CHARACTER
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is stored:
            = 'U': Upper triangular
            = 'L': Lower triangular

    N       (input) INTEGER
            The order of the matrix A.

    NB      (input) INTEGER
            The number of rows and columns to be reduced.

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading
            n-by-n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading n-by-n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit:
            if UPLO = 'U', the last NB columns have been reduced to
              tridiagonal form, with the diagonal elements overwriting
              the diagonal elements of A; the elements above the diagonal
              with the array TAU, represent the orthogonal matrix Q as a
              product of elementary reflectors;
            if UPLO = 'L', the first NB columns have been reduced to
              tridiagonal form, with the diagonal elements overwriting
              the diagonal elements of A; the elements below the diagonal
              with the array TAU, represent the  orthogonal matrix Q as a
              product of elementary reflectors.
            See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= (1,N).

    E       (output) DOUBLE PRECISION array, dimension (N-1)
            If UPLO = 'U', E(n-nb:n-1) contains the superdiagonal
            elements of the last NB columns of the reduced matrix;
            if UPLO = 'L', E(1:nb) contains the subdiagonal elements of
            the first NB columns of the reduced matrix.

    TAU     (output) DOUBLE PRECISION array, dimension (N-1)
            The scalar factors of the elementary reflectors, stored in
            TAU(n-nb:n-1) if UPLO = 'U', and in TAU(1:nb) if UPLO = 'L'.
            See Further Details.

    W       (output) DOUBLE PRECISION array, dimension (LDW,NB)
            The n-by-nb matrix W required to update the unreduced part
            of A.

    LDW     (input) INTEGER
            The leading dimension of the array W. LDW >= max(1,N).

    Further Details
    ===============

    If UPLO = 'U', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(n) H(n-1) . . . H(n-nb+1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(i:n) = 0 and v(i-1) = 1; v(1:i-1) is stored on exit in A(1:i-1,i),
    and tau in TAU(i-1).

    If UPLO = 'L', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(1) H(2) . . . H(nb).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i) = 0 and v(i+1) = 1; v(i+1:n) is stored on exit in A(i+1:n,i),
    and tau in TAU(i).

    The elements of the vectors v together form the n-by-nb matrix V
    which is needed, with W, to apply the transformation to the unreduced
    part of the matrix, using a symmetric rank-2k update of the form:
    A := A - V*W' - W*V'.

    The contents of A on exit are illustrated by the following examples
    with n = 5 and nb = 2:

    if UPLO = 'U':                       if UPLO = 'L':

      (  a   a   a   v4  v5 )              (  d                  )
      (      a   a   v4  v5 )              (  1   d              )
      (          a   1   v5 )              (  v1  1   a          )
      (              d   1  )              (  v1  v2  a   a      )
      (                  d  )              (  v1  v2  a   a   a  )

    where d denotes a diagonal element of the reduced matrix, a denotes
    an element of the original matrix that is unchanged, and vi denotes
    an element of the vector defining H(i).

    =====================================================================


       Quick return if possible

       Parameter adjustments */
    /* Table of constant values */
    doublereal c_b5 = -1.;
    doublereal c_b6 = 1.;
    integer c__1 = 1;
    doublereal c_b16 = 0.;

    /* System generated locals */
    integer a_dim1, a_offset, w_dim1, w_offset, i__1, i__2, i__3;
    /* Local variables */
    extern doublereal ddot_(integer *, doublereal *, integer *, doublereal *,
	    integer *);
    integer i__;
    doublereal alpha;
    extern /* Subroutine */ integer dscal_(integer *, doublereal *, doublereal *,
	    integer *);
    extern logical lsame_(const char *,const char *);
    extern /* Subroutine */ integer dgemv_(const char *, integer *, integer *,
	    doublereal *, doublereal *, integer *, doublereal *, integer *,
	    doublereal *, doublereal *, integer *), daxpy_(integer *,
	    doublereal *, doublereal *, integer *, doublereal *, integer *),
	    dsymv_(const char *, integer *, doublereal *, doublereal *, integer *,
	    doublereal *, integer *, doublereal *, doublereal *, integer *), dlarfg_(integer *, doublereal *, doublereal *, integer *,
	     doublereal *);
    integer iw;
#define a_ref(a_1,a_2) a[(a_2)*a_dim1 + a_1]
#define w_ref(a_1,a_2) w[(a_2)*w_dim1 + a_1]


    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --e;
    --tau;
    w_dim1 = *ldw;
    w_offset = 1 + w_dim1 * 1;
    w -= w_offset;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }

    if (lsame_(uplo, "U")) {

/*        Reduce last NB columns of upper triangle */

	i__1 = *n - *nb + 1;
	for (i__ = *n; i__ >= i__1; --i__) {
	    iw = i__ - *n + *nb;
	    if (i__ < *n) {

/*              Update A(1:i,i) */

		i__2 = *n - i__;
		dgemv_("No transpose", &i__, &i__2, &c_b5, &a_ref(1, i__ + 1),
			 lda, &w_ref(i__, iw + 1), ldw, &c_b6, &a_ref(1, i__),
			 &c__1);
		i__2 = *n - i__;
		dgemv_("No transpose", &i__, &i__2, &c_b5, &w_ref(1, iw + 1),
			ldw, &a_ref(i__, i__ + 1), lda, &c_b6, &a_ref(1, i__),
			 &c__1);
	    }
	    if (i__ > 1) {

/*              Generate elementary reflector H(i) to annihilate
                A(1:i-2,i) */

		i__2 = i__ - 1;
		dlarfg_(&i__2, &a_ref(i__ - 1, i__), &a_ref(1, i__), &c__1, &
			tau[i__ - 1]);
		e[i__ - 1] = a_ref(i__ - 1, i__);
		a_ref(i__ - 1, i__) = 1.;

/*              Compute W(1:i-1,i) */

		i__2 = i__ - 1;
		dsymv_("Upper", &i__2, &c_b6, &a[a_offset], lda, &a_ref(1,
			i__), &c__1, &c_b16, &w_ref(1, iw), &c__1);
		if (i__ < *n) {
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    dgemv_("Transpose", &i__2, &i__3, &c_b6, &w_ref(1, iw + 1)
			    , ldw, &a_ref(1, i__), &c__1, &c_b16, &w_ref(i__
			    + 1, iw), &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    dgemv_("No transpose", &i__2, &i__3, &c_b5, &a_ref(1, i__
			    + 1), lda, &w_ref(i__ + 1, iw), &c__1, &c_b6, &
			    w_ref(1, iw), &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    dgemv_("Transpose", &i__2, &i__3, &c_b6, &a_ref(1, i__ +
			    1), lda, &a_ref(1, i__), &c__1, &c_b16, &w_ref(
			    i__ + 1, iw), &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    dgemv_("No transpose", &i__2, &i__3, &c_b5, &w_ref(1, iw
			    + 1), ldw, &w_ref(i__ + 1, iw), &c__1, &c_b6, &
			    w_ref(1, iw), &c__1);
		}
		i__2 = i__ - 1;
		dscal_(&i__2, &tau[i__ - 1], &w_ref(1, iw), &c__1);
		i__2 = i__ - 1;
		alpha = tau[i__ - 1] * -.5 * ddot_(&i__2, &w_ref(1, iw), &
			c__1, &a_ref(1, i__), &c__1);
		i__2 = i__ - 1;
		daxpy_(&i__2, &alpha, &a_ref(1, i__), &c__1, &w_ref(1, iw), &
			c__1);
	    }

/* L10: */
	}
    } else {

/*        Reduce first NB columns of lower triangle */

	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Update A(i:n,i) */

	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    dgemv_("No transpose", &i__2, &i__3, &c_b5, &a_ref(i__, 1), lda, &
		    w_ref(i__, 1), ldw, &c_b6, &a_ref(i__, i__), &c__1);
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    dgemv_("No transpose", &i__2, &i__3, &c_b5, &w_ref(i__, 1), ldw, &
		    a_ref(i__, 1), lda, &c_b6, &a_ref(i__, i__), &c__1);
	    if (i__ < *n) {

/*              Generate elementary reflector H(i) to annihilate
                A(i+2:n,i)

   Computing MIN */
		i__2 = i__ + 2;
		i__3 = *n - i__;
		dlarfg_(&i__3, &a_ref(i__ + 1, i__), &a_ref(min(i__2,*n), i__)
			, &c__1, &tau[i__]);
		e[i__] = a_ref(i__ + 1, i__);
		a_ref(i__ + 1, i__) = 1.;

/*              Compute W(i+1:n,i) */

		i__2 = *n - i__;
		dsymv_("Lower", &i__2, &c_b6, &a_ref(i__ + 1, i__ + 1), lda, &
			a_ref(i__ + 1, i__), &c__1, &c_b16, &w_ref(i__ + 1,
			i__), &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		dgemv_("Transpose", &i__2, &i__3, &c_b6, &w_ref(i__ + 1, 1),
			ldw, &a_ref(i__ + 1, i__), &c__1, &c_b16, &w_ref(1,
			i__), &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b5, &a_ref(i__ + 1, 1)
			, lda, &w_ref(1, i__), &c__1, &c_b6, &w_ref(i__ + 1,
			i__), &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		dgemv_("Transpose", &i__2, &i__3, &c_b6, &a_ref(i__ + 1, 1),
			lda, &a_ref(i__ + 1, i__), &c__1, &c_b16, &w_ref(1,
			i__), &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b5, &w_ref(i__ + 1, 1)
			, ldw, &w_ref(1, i__), &c__1, &c_b6, &w_ref(i__ + 1,
			i__), &c__1);
		i__2 = *n - i__;
		dscal_(&i__2, &tau[i__], &w_ref(i__ + 1, i__), &c__1);
		i__2 = *n - i__;
		alpha = tau[i__] * -.5 * ddot_(&i__2, &w_ref(i__ + 1, i__), &
			c__1, &a_ref(i__ + 1, i__), &c__1);
		i__2 = *n - i__;
		daxpy_(&i__2, &alpha, &a_ref(i__ + 1, i__), &c__1, &w_ref(i__
			+ 1, i__), &c__1);
	    }

/* L20: */
	}
    }

    return 0;

/*     End of DLATRD */

} /* dlatrd_ */

#undef w_ref
#undef a_ref

#ifdef __cplusplus
}
#endif

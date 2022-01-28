/* Copyright (c) 1992-2008 The University of Tennessee.  All rights reserved.
 * See file COPYING in this directory for details. */

#ifdef __cplusplus
extern "C" {
#endif

#include "f2c.h"
#include "hypre_lapack.h"

/* Subroutine */ integer dsygst_(integer *itype,const char *uplo, integer *n,
	doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *
	info)
{
/*  -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    DSYGST reduces a real symmetric-definite generalized eigenproblem
    to standard form.

    If ITYPE = 1, the problem is A*x = lambda*B*x,
    and A is overwritten by inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T)

    If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
    B*A*x = lambda*x, and A is overwritten by U*A*U**T or L**T*A*L.

    B must have been previously factorized as U**T*U or L*L**T by DPOTRF.

    Arguments
    =========

    ITYPE   (input) INTEGER
            = 1: compute inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T);
            = 2 or 3: compute U*A*U**T or L**T*A*L.

    UPLO    (input) CHARACTER
            = 'U':  Upper triangle of A is stored and B is factored as
                    U**T*U;
            = 'L':  Lower triangle of A is stored and B is factored as
                    L*L**T.

    N       (input) INTEGER
            The order of the matrices A and B.  N >= 0.

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.

            On exit, if INFO = 0, the transformed matrix, stored in the
            same format as A.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    B       (input) DOUBLE PRECISION array, dimension (LDB,N)
            The triangular factor from the Cholesky factorization of B,
            as returned by DPOTRF.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input parameters.

       Parameter adjustments */
    /* Table of constant values */
    integer c__1 = 1;
    integer c_n1 = -1;
    doublereal c_b14 = 1.;
    doublereal c_b16 = -.5;
    doublereal c_b19 = -1.;
    doublereal c_b52 = .5;

    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;
    /* Local variables */
    integer k;
    extern logical lsame_(const char *,const char *);
    extern /* Subroutine */ integer dtrmm_(const char *,const char *,const char *,const char *,
	    integer *, integer *, doublereal *, doublereal *, integer *,
	    doublereal *, integer *), dsymm_(
	    const char *,const char *, integer *, integer *, doublereal *, doublereal *,
	    integer *, doublereal *, integer *, doublereal *, doublereal *,
	    integer *);
    logical upper;
    extern /* Subroutine */ integer dtrsm_(const char *,const char *,const char *,const char *,
	    integer *, integer *, doublereal *, doublereal *, integer *,
	    doublereal *, integer *), dsygs2_(
	    integer *,const char *, integer *, doublereal *, integer *, doublereal
	    *, integer *, integer *);
    integer kb;
    extern /* Subroutine */ integer dsyr2k_(const char *,const char *, integer *, integer *,
	    doublereal *, doublereal *, integer *, doublereal *, integer *,
	    doublereal *, doublereal *, integer *);
    integer nb;
    extern /* Subroutine */ integer xerbla_(const char *, integer *);
    extern integer ilaenv_(integer *,const char *,const char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
#define a_ref(a_1,a_2) a[(a_2)*a_dim1 + a_1]
#define b_ref(a_1,a_2) b[(a_2)*b_dim1 + a_1]


    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1 * 1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (*itype < 1 || *itype > 3) {
	*info = -1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*ldb < max(1,*n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSYGST", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "DSYGST", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)1);

    if (nb <= 1 || nb >= *n) {

/*        Use unblocked code */

	dsygs2_(itype, uplo, n, &a[a_offset], lda, &b[b_offset], ldb, info);
    } else {

/*        Use blocked code */

	if (*itype == 1) {
	    if (upper) {

/*              Compute inv(U')*A*inv(U) */

		i__1 = *n;
		i__2 = nb;
		for (k = 1; i__2 < 0 ? k >= i__1 : k <= i__1; k += i__2) {
/* Computing MIN */
		    i__3 = *n - k + 1;
		    kb = min(i__3,nb);

/*                 Update the upper triangle of A(k:n,k:n) */

		    dsygs2_(itype, uplo, &kb, &a_ref(k, k), lda, &b_ref(k, k),
			     ldb, info);
		    if (k + kb <= *n) {
			i__3 = *n - k - kb + 1;
			dtrsm_("Left", uplo, "Transpose", "Non-unit", &kb, &
				i__3, &c_b14, &b_ref(k, k), ldb, &a_ref(k, k
				+ kb), lda);
			i__3 = *n - k - kb + 1;
			dsymm_("Left", uplo, &kb, &i__3, &c_b16, &a_ref(k, k),
				 lda, &b_ref(k, k + kb), ldb, &c_b14, &a_ref(
				k, k + kb), lda);
			i__3 = *n - k - kb + 1;
			dsyr2k_(uplo, "Transpose", &i__3, &kb, &c_b19, &a_ref(
				k, k + kb), lda, &b_ref(k, k + kb), ldb, &
				c_b14, &a_ref(k + kb, k + kb), lda);
			i__3 = *n - k - kb + 1;
			dsymm_("Left", uplo, &kb, &i__3, &c_b16, &a_ref(k, k),
				 lda, &b_ref(k, k + kb), ldb, &c_b14, &a_ref(
				k, k + kb), lda);
			i__3 = *n - k - kb + 1;
			dtrsm_("Right", uplo, "No transpose", "Non-unit", &kb,
				 &i__3, &c_b14, &b_ref(k + kb, k + kb), ldb, &
				a_ref(k, k + kb), lda);
		    }
/* L10: */
		}
	    } else {

/*              Compute inv(L)*A*inv(L') */

		i__2 = *n;
		i__1 = nb;
		for (k = 1; i__1 < 0 ? k >= i__2 : k <= i__2; k += i__1) {
/* Computing MIN */
		    i__3 = *n - k + 1;
		    kb = min(i__3,nb);

/*                 Update the lower triangle of A(k:n,k:n) */

		    dsygs2_(itype, uplo, &kb, &a_ref(k, k), lda, &b_ref(k, k),
			     ldb, info);
		    if (k + kb <= *n) {
			i__3 = *n - k - kb + 1;
			dtrsm_("Right", uplo, "Transpose", "Non-unit", &i__3,
				&kb, &c_b14, &b_ref(k, k), ldb, &a_ref(k + kb,
				 k), lda);
			i__3 = *n - k - kb + 1;
			dsymm_("Right", uplo, &i__3, &kb, &c_b16, &a_ref(k, k)
				, lda, &b_ref(k + kb, k), ldb, &c_b14, &a_ref(
				k + kb, k), lda);
			i__3 = *n - k - kb + 1;
			dsyr2k_(uplo, "No transpose", &i__3, &kb, &c_b19, &
				a_ref(k + kb, k), lda, &b_ref(k + kb, k), ldb,
				 &c_b14, &a_ref(k + kb, k + kb), lda);
			i__3 = *n - k - kb + 1;
			dsymm_("Right", uplo, &i__3, &kb, &c_b16, &a_ref(k, k)
				, lda, &b_ref(k + kb, k), ldb, &c_b14, &a_ref(
				k + kb, k), lda);
			i__3 = *n - k - kb + 1;
			dtrsm_("Left", uplo, "No transpose", "Non-unit", &
				i__3, &kb, &c_b14, &b_ref(k + kb, k + kb),
				ldb, &a_ref(k + kb, k), lda);
		    }
/* L20: */
		}
	    }
	} else {
	    if (upper) {

/*              Compute U*A*U' */

		i__1 = *n;
		i__2 = nb;
		for (k = 1; i__2 < 0 ? k >= i__1 : k <= i__1; k += i__2) {
/* Computing MIN */
		    i__3 = *n - k + 1;
		    kb = min(i__3,nb);

/*                 Update the upper triangle of A(1:k+kb-1,1:k+kb-1) */

		    i__3 = k - 1;
		    dtrmm_("Left", uplo, "No transpose", "Non-unit", &i__3, &
			    kb, &c_b14, &b[b_offset], ldb, &a_ref(1, k), lda);
		    i__3 = k - 1;
		    dsymm_("Right", uplo, &i__3, &kb, &c_b52, &a_ref(k, k),
			    lda, &b_ref(1, k), ldb, &c_b14, &a_ref(1, k), lda);
		    i__3 = k - 1;
		    dsyr2k_(uplo, "No transpose", &i__3, &kb, &c_b14, &a_ref(
			    1, k), lda, &b_ref(1, k), ldb, &c_b14, &a[
			    a_offset], lda);
		    i__3 = k - 1;
		    dsymm_("Right", uplo, &i__3, &kb, &c_b52, &a_ref(k, k),
			    lda, &b_ref(1, k), ldb, &c_b14, &a_ref(1, k), lda);
		    i__3 = k - 1;
		    dtrmm_("Right", uplo, "Transpose", "Non-unit", &i__3, &kb,
			     &c_b14, &b_ref(k, k), ldb, &a_ref(1, k), lda);
		    dsygs2_(itype, uplo, &kb, &a_ref(k, k), lda, &b_ref(k, k),
			     ldb, info);
/* L30: */
		}
	    } else {

/*              Compute L'*A*L */

		i__2 = *n;
		i__1 = nb;
		for (k = 1; i__1 < 0 ? k >= i__2 : k <= i__2; k += i__1) {
/* Computing MIN */
		    i__3 = *n - k + 1;
		    kb = min(i__3,nb);

/*                 Update the lower triangle of A(1:k+kb-1,1:k+kb-1) */

		    i__3 = k - 1;
		    dtrmm_("Right", uplo, "No transpose", "Non-unit", &kb, &
			    i__3, &c_b14, &b[b_offset], ldb, &a_ref(k, 1),
			    lda);
		    i__3 = k - 1;
		    dsymm_("Left", uplo, &kb, &i__3, &c_b52, &a_ref(k, k),
			    lda, &b_ref(k, 1), ldb, &c_b14, &a_ref(k, 1), lda);
		    i__3 = k - 1;
		    dsyr2k_(uplo, "Transpose", &i__3, &kb, &c_b14, &a_ref(k,
			    1), lda, &b_ref(k, 1), ldb, &c_b14, &a[a_offset],
			    lda);
		    i__3 = k - 1;
		    dsymm_("Left", uplo, &kb, &i__3, &c_b52, &a_ref(k, k),
			    lda, &b_ref(k, 1), ldb, &c_b14, &a_ref(k, 1), lda);
		    i__3 = k - 1;
		    dtrmm_("Left", uplo, "Transpose", "Non-unit", &kb, &i__3,
			    &c_b14, &b_ref(k, k), ldb, &a_ref(k, 1), lda);
		    dsygs2_(itype, uplo, &kb, &a_ref(k, k), lda, &b_ref(k, k),
			     ldb, info);
/* L40: */
		}
	    }
	}
    }
    return 0;

/*     End of DSYGST */

} /* dsygst_ */

#undef b_ref
#undef a_ref

#ifdef __cplusplus
}
#endif

/* Copyright (c) 1992-2008 The University of Tennessee.  All rights reserved.
 * See file COPYING in this directory for details. */

#ifdef __cplusplus
extern "C" {
#endif

#include "f2c.h"
#include "hypre_lapack.h"

/* Subroutine */ integer dgetrs_(const char *trans, integer *n, integer *nrhs,
	doublereal *a, integer *lda, integer *ipiv, doublereal *b, integer *
	ldb, integer *info)
{
/*  -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       March 31, 1993


    Purpose
    =======

    DGETRS solves a system of linear equations
       A * X = B  or  A' * X = B
    with a general N-by-N matrix A using the LU factorization computed
    by DGETRF.

    Arguments
    =========

    TRANS   (input) CHARACTER*1
            Specifies the form of the system of equations:
            = 'N':  A * X = B  (No transpose)
            = 'T':  A'* X = B  (Transpose)
            = 'C':  A'* X = B  (Conjugate transpose = Transpose)

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    A       (input) DOUBLE PRECISION array, dimension (LDA,N)
            The factors L and U from the factorization A = P*L*U
            as computed by DGETRF.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (input) INTEGER array, dimension (N)
            The pivot indices from DGETRF; for 1<=i<=N, row i of the
            matrix was interchanged with row IPIV(i).

    B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

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
    doublereal c_b12 = 1.;
    integer c_n1 = -1;

    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1;
    /* Local variables */
    extern logical lsame_(const char *,const char *);
    extern /* Subroutine */ integer dtrsm_(const char *,const char *,const char *,const char *,
	    integer *, integer *, doublereal *, doublereal *, integer *,
	    doublereal *, integer *), xerbla_(
	    const char *, integer *), dlaswp_(integer *, doublereal *,
	    integer *, integer *, integer *, integer *, integer *);
    logical notran;


    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1 * 1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    notran = lsame_(trans, "N");
    if (! notran && ! lsame_(trans, "T") && ! lsame_(
	    trans, "C")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*ldb < max(1,*n)) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGETRS", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0 || *nrhs == 0) {
	return 0;
    }

    if (notran) {

/*        Solve A * X = B.

          Apply row interchanges to the right hand sides. */

	dlaswp_(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c__1);

/*        Solve L*X = B, overwriting B with X. */

	dtrsm_("Left", "Lower", "No transpose", "Unit", n, nrhs, &c_b12, &a[
		a_offset], lda, &b[b_offset], ldb);

/*        Solve U*X = B, overwriting B with X. */

	dtrsm_("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &c_b12, &
		a[a_offset], lda, &b[b_offset], ldb);
    } else {

/*        Solve A' * X = B.

          Solve U'*X = B, overwriting B with X. */

	dtrsm_("Left", "Upper", "Transpose", "Non-unit", n, nrhs, &c_b12, &a[
		a_offset], lda, &b[b_offset], ldb);

/*        Solve L'*X = B, overwriting B with X. */

	dtrsm_("Left", "Lower", "Transpose", "Unit", n, nrhs, &c_b12, &a[
		a_offset], lda, &b[b_offset], ldb);

/*        Apply row interchanges to the solution vectors. */

	dlaswp_(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c_n1);
    }

    return 0;

/*     End of DGETRS */

} /* dgetrs_ */

#ifdef __cplusplus
}
#endif

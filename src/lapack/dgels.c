/* Copyright (c) 1992-2008 The University of Tennessee.  All rights reserved.
 * See file COPYING in this directory for details. */

#ifdef __cplusplus
extern "C" {
#endif

#include "f2c.h"
#include "hypre_lapack.h"

/* Subroutine */ integer dgels_(char *trans, integer *m, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *b, integer *ldb,
	doublereal *work, integer *lwork, integer *info)
{
/*  -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    DGELS solves overdetermined or underdetermined real linear systems
    involving an M-by-N matrix A, or its transpose, using a QR or LQ
    factorization of A.  It is assumed that A has full rank.

    The following options are provided:

    1. If TRANS = 'N' and m >= n:  find the least squares solution of
       an overdetermined system, i.e., solve the least squares problem
                    minimize || B - A*X ||.

    2. If TRANS = 'N' and m < n:  find the minimum norm solution of
       an underdetermined system A * X = B.

    3. If TRANS = 'T' and m >= n:  find the minimum norm solution of
       an undetermined system A**T * X = B.

    4. If TRANS = 'T' and m < n:  find the least squares solution of
       an overdetermined system, i.e., solve the least squares problem
                    minimize || B - A**T * X ||.

    Several right hand side vectors b and solution vectors x can be
    handled in a single call; they are stored as the columns of the
    M-by-NRHS right hand side matrix B and the N-by-NRHS solution
    matrix X.

    Arguments
    =========

    TRANS   (input) CHARACTER
            = 'N': the linear system involves A;
            = 'T': the linear system involves A**T.

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of
            columns of the matrices B and X. NRHS >=0.

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit,
              if M >= N, A is overwritten by details of its QR
                         factorization as returned by DGEQRF;
              if M <  N, A is overwritten by details of its LQ
                         factorization as returned by DGELQF.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
            On entry, the matrix B of right hand side vectors, stored
            columnwise; B is M-by-NRHS if TRANS = 'N', or N-by-NRHS
            if TRANS = 'T'.
            On exit, B is overwritten by the solution vectors, stored
            columnwise:
            if TRANS = 'N' and m >= n, rows 1 to n of B contain the least
            squares solution vectors; the residual sum of squares for the
            solution in each column is given by the sum of squares of
            elements N+1 to M in that column;
            if TRANS = 'N' and m < n, rows 1 to N of B contain the
            minimum norm solution vectors;
            if TRANS = 'T' and m >= n, rows 1 to M of B contain the
            minimum norm solution vectors;
            if TRANS = 'T' and m < n, rows 1 to M of B contain the
            least squares solution vectors; the residual sum of squares
            for the solution in each column is given by the sum of
            squares of elements M+1 to N in that column.

    LDB     (input) INTEGER
            The leading dimension of the array B. LDB >= MAX(1,M,N).

    WORK    (workspace/output) DOUBLE PRECISION array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            LWORK >= max( 1, MN + max( MN, NRHS ) ).
            For optimal performance,
            LWORK >= max( 1, MN + max( MN, NRHS )*NB ).
            where MN = min(M,N) and NB is the optimum block size.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input arguments.

       Parameter adjustments */
    /* Table of constant values */
    integer c__1 = 1;
    integer c_n1 = -1;
    doublereal c_b33 = 0.;
    integer c__0 = 0;
    doublereal c_b61 = 1.;

    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;
    /* Local variables */
    doublereal anrm, bnrm;
    integer brow;
    logical tpsd;
    integer i__, j, iascl, ibscl;
    extern logical lsame_(const char *,const char *);
    extern /* Subroutine */ integer dtrsm_(const char *,const char *,const char *,const char *,
	    integer *, integer *, doublereal *, doublereal *, integer *,
	    doublereal *, integer *);
    integer wsize;
    doublereal rwork[1];
    extern /* Subroutine */ integer dlabad_(doublereal *, doublereal *);
    integer nb;
    extern doublereal dlamch_(const char *), dlange_(const char *, integer *,
	    integer *, doublereal *, integer *, doublereal *);
    integer mn;
    extern /* Subroutine */ integer dgelqf_(integer *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, integer *, integer *),
	    dlascl_(const char *, integer *, integer *, doublereal *, doublereal *,
	    integer *, integer *, doublereal *, integer *, integer *),
	     dgeqrf_(integer *, integer *, doublereal *, integer *,
	    doublereal *, doublereal *, integer *, integer *), dlaset_(const char *,
	     integer *, integer *, doublereal *, doublereal *, doublereal *,
	    integer *), xerbla_(const char *, integer *);
    extern integer ilaenv_(integer *,const char *,const char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    integer scllen;
    doublereal bignum;
    extern /* Subroutine */ integer dormlq_(const char *,const char *, integer *, integer *,
	    integer *, doublereal *, integer *, doublereal *, doublereal *,
	    integer *, doublereal *, integer *, integer *),
	    dormqr_(const char *,const char *, integer *, integer *, integer *,
	    doublereal *, integer *, doublereal *, doublereal *, integer *,
	    doublereal *, integer *, integer *);
    doublereal smlnum;
    logical lquery;
#define b_ref(a_1,a_2) b[(a_2)*b_dim1 + a_1]


    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1 * 1;
    b -= b_offset;
    --work;

    /* Function Body */
    *info = 0;
    mn = min(*m,*n);
    lquery = *lwork == -1;
    if (! (lsame_(trans, "N") || lsame_(trans, "T"))) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*nrhs < 0) {
	*info = -4;
    } else if (*lda < max(1,*m)) {
	*info = -6;
    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__1 = max(1,*m);
	if (*ldb < max(i__1,*n)) {
	    *info = -8;
	} else /* if(complicated condition) */ {
/* Computing MAX */
	    i__1 = 1, i__2 = mn + max(mn,*nrhs);
	    if (*lwork < max(i__1,i__2) && ! lquery) {
		*info = -10;
	    }
	}
    }

/*     Figure out optimal block size */

    if (*info == 0 || *info == -10) {

	tpsd = TRUE_;
	if (lsame_(trans, "N")) {
	    tpsd = FALSE_;
	}

	if (*m >= *n) {
	    nb = ilaenv_(&c__1, "DGEQRF", " ", m, n, &c_n1, &c_n1, (ftnlen)6,
		    (ftnlen)1);
	    if (tpsd) {
/* Computing MAX */
		i__1 = nb, i__2 = ilaenv_(&c__1, "DORMQR", "LN", m, nrhs, n, &
			c_n1, (ftnlen)6, (ftnlen)2);
		nb = max(i__1,i__2);
	    } else {
/* Computing MAX */
		i__1 = nb, i__2 = ilaenv_(&c__1, "DORMQR", "LT", m, nrhs, n, &
			c_n1, (ftnlen)6, (ftnlen)2);
		nb = max(i__1,i__2);
	    }
	} else {
	    nb = ilaenv_(&c__1, "DGELQF", " ", m, n, &c_n1, &c_n1, (ftnlen)6,
		    (ftnlen)1);
	    if (tpsd) {
/* Computing MAX */
		i__1 = nb, i__2 = ilaenv_(&c__1, "DORMLQ", "LT", n, nrhs, m, &
			c_n1, (ftnlen)6, (ftnlen)2);
		nb = max(i__1,i__2);
	    } else {
/* Computing MAX */
		i__1 = nb, i__2 = ilaenv_(&c__1, "DORMLQ", "LN", n, nrhs, m, &
			c_n1, (ftnlen)6, (ftnlen)2);
		nb = max(i__1,i__2);
	    }
	}

/* Computing MAX */
	i__1 = 1, i__2 = mn + max(mn,*nrhs) * nb;
	wsize = max(i__1,i__2);
	work[1] = (doublereal) wsize;

    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGELS ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible

   Computing MIN */
    i__1 = min(*m,*n);
    if (min(i__1,*nrhs) == 0) {
	i__1 = max(*m,*n);
	dlaset_("Full", &i__1, nrhs, &c_b33, &c_b33, &b[b_offset], ldb);
	return 0;
    }

/*     Get machine parameters */

    smlnum = dlamch_("S") / dlamch_("P");
    bignum = 1. / smlnum;
    dlabad_(&smlnum, &bignum);

/*     Scale A, B if max element outside range [SMLNUM,BIGNUM] */

    anrm = dlange_("M", m, n, &a[a_offset], lda, rwork);
    iascl = 0;
    if (anrm > 0. && anrm < smlnum) {

/*        Scale matrix norm up to SMLNUM */

	dlascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda,
		info);
	iascl = 1;
    } else if (anrm > bignum) {

/*        Scale matrix norm down to BIGNUM */

	dlascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda,
		info);
	iascl = 2;
    } else if (anrm == 0.) {

/*        Matrix all zero. Return zero solution. */

	i__1 = max(*m,*n);
	dlaset_("F", &i__1, nrhs, &c_b33, &c_b33, &b[b_offset], ldb);
	goto L50;
    }

    brow = *m;
    if (tpsd) {
	brow = *n;
    }
    bnrm = dlange_("M", &brow, nrhs, &b[b_offset], ldb, rwork);
    ibscl = 0;
    if (bnrm > 0. && bnrm < smlnum) {

/*        Scale matrix norm up to SMLNUM */

	dlascl_("G", &c__0, &c__0, &bnrm, &smlnum, &brow, nrhs, &b[b_offset],
		ldb, info);
	ibscl = 1;
    } else if (bnrm > bignum) {

/*        Scale matrix norm down to BIGNUM */

	dlascl_("G", &c__0, &c__0, &bnrm, &bignum, &brow, nrhs, &b[b_offset],
		ldb, info);
	ibscl = 2;
    }

    if (*m >= *n) {

/*        compute QR factorization of A */

	i__1 = *lwork - mn;
	dgeqrf_(m, n, &a[a_offset], lda, &work[1], &work[mn + 1], &i__1, info)
		;

/*        workspace at least N, optimally N*NB */

	if (! tpsd) {

/*           Least-Squares Problem min || A * X - B ||

             B(1:M,1:NRHS) := Q' * B(1:M,1:NRHS) */

	    i__1 = *lwork - mn;
	    dormqr_("Left", "Transpose", m, nrhs, n, &a[a_offset], lda, &work[
		    1], &b[b_offset], ldb, &work[mn + 1], &i__1, info);

/*           workspace at least NRHS, optimally NRHS*NB

             B(1:N,1:NRHS) := inv(R) * B(1:N,1:NRHS) */

	    dtrsm_("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &
		    c_b61, &a[a_offset], lda, &b[b_offset], ldb);

	    scllen = *n;

	} else {

/*           Overdetermined system of equations A' * X = B

             B(1:N,1:NRHS) := inv(R') * B(1:N,1:NRHS) */

	    dtrsm_("Left", "Upper", "Transpose", "Non-unit", n, nrhs, &c_b61,
		    &a[a_offset], lda, &b[b_offset], ldb);

/*           B(N+1:M,1:NRHS) = ZERO */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = *n + 1; i__ <= i__2; ++i__) {
		    b_ref(i__, j) = 0.;
/* L10: */
		}
/* L20: */
	    }

/*           B(1:M,1:NRHS) := Q(1:N,:) * B(1:N,1:NRHS) */

	    i__1 = *lwork - mn;
	    dormqr_("Left", "No transpose", m, nrhs, n, &a[a_offset], lda, &
		    work[1], &b[b_offset], ldb, &work[mn + 1], &i__1, info);

/*           workspace at least NRHS, optimally NRHS*NB */

	    scllen = *m;

	}

    } else {

/*        Compute LQ factorization of A */

	i__1 = *lwork - mn;
	dgelqf_(m, n, &a[a_offset], lda, &work[1], &work[mn + 1], &i__1, info)
		;

/*        workspace at least M, optimally M*NB. */

	if (! tpsd) {

/*           underdetermined system of equations A * X = B

             B(1:M,1:NRHS) := inv(L) * B(1:M,1:NRHS) */

	    dtrsm_("Left", "Lower", "No transpose", "Non-unit", m, nrhs, &
		    c_b61, &a[a_offset], lda, &b[b_offset], ldb);

/*           B(M+1:N,1:NRHS) = 0 */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n;
		for (i__ = *m + 1; i__ <= i__2; ++i__) {
		    b_ref(i__, j) = 0.;
/* L30: */
		}
/* L40: */
	    }

/*           B(1:N,1:NRHS) := Q(1:N,:)' * B(1:M,1:NRHS) */

	    i__1 = *lwork - mn;
	    dormlq_("Left", "Transpose", n, nrhs, m, &a[a_offset], lda, &work[
		    1], &b[b_offset], ldb, &work[mn + 1], &i__1, info);

/*           workspace at least NRHS, optimally NRHS*NB */

	    scllen = *n;

	} else {

/*           overdetermined system min || A' * X - B ||

             B(1:N,1:NRHS) := Q * B(1:N,1:NRHS) */

	    i__1 = *lwork - mn;
	    dormlq_("Left", "No transpose", n, nrhs, m, &a[a_offset], lda, &
		    work[1], &b[b_offset], ldb, &work[mn + 1], &i__1, info);

/*           workspace at least NRHS, optimally NRHS*NB

             B(1:M,1:NRHS) := inv(L') * B(1:M,1:NRHS) */

	    dtrsm_("Left", "Lower", "Transpose", "Non-unit", m, nrhs, &c_b61,
		    &a[a_offset], lda, &b[b_offset], ldb);

	    scllen = *m;

	}

    }

/*     Undo scaling */

    if (iascl == 1) {
	dlascl_("G", &c__0, &c__0, &anrm, &smlnum, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    } else if (iascl == 2) {
	dlascl_("G", &c__0, &c__0, &anrm, &bignum, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    }
    if (ibscl == 1) {
	dlascl_("G", &c__0, &c__0, &smlnum, &bnrm, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    } else if (ibscl == 2) {
	dlascl_("G", &c__0, &c__0, &bignum, &bnrm, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    }

L50:
    work[1] = (doublereal) wsize;

    return 0;

/*     End of DGELS */

} /* dgels_ */

#undef b_ref

#ifdef __cplusplus
}
#endif

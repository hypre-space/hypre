/* Copyright (c) 1992-2008 The University of Tennessee.  All rights reserved.
 * See file COPYING in this directory for details. */

#ifdef __cplusplus
extern "C" {
#endif

#include "f2c.h"
#include "hypre_lapack.h"

/* Subroutine */ integer dorm2r_(const char *side,const char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *info)
{
/*  -- LAPACK routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       February 29, 1992   


    Purpose   
    =======   

    DORM2R overwrites the general real m by n matrix C with   

          Q * C  if SIDE = 'L' and TRANS = 'N', or   

          Q'* C  if SIDE = 'L' and TRANS = 'T', or   

          C * Q  if SIDE = 'R' and TRANS = 'N', or   

          C * Q' if SIDE = 'R' and TRANS = 'T',   

    where Q is a real orthogonal matrix defined as the product of k   
    elementary reflectors   

          Q = H(1) H(2) . . . H(k)   

    as returned by DGEQRF. Q is of order m if SIDE = 'L' and of order n   
    if SIDE = 'R'.   

    Arguments   
    =========   

    SIDE    (input) CHARACTER*1   
            = 'L': apply Q or Q' from the Left   
            = 'R': apply Q or Q' from the Right   

    TRANS   (input) CHARACTER*1   
            = 'N': apply Q  (No transpose)   
            = 'T': apply Q' (Transpose)   

    M       (input) INTEGER   
            The number of rows of the matrix C. M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix C. N >= 0.   

    K       (input) INTEGER   
            The number of elementary reflectors whose product defines   
            the matrix Q.   
            If SIDE = 'L', M >= K >= 0;   
            if SIDE = 'R', N >= K >= 0.   

    A       (input) DOUBLE PRECISION array, dimension (LDA,K)   
            The i-th column must contain the vector which defines the   
            elementary reflector H(i), for i = 1,2,...,k, as returned by   
            DGEQRF in the first k columns of its array argument A.   
            A is modified by the routine but restored on exit.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.   
            If SIDE = 'L', LDA >= max(1,M);   
            if SIDE = 'R', LDA >= max(1,N).   

    TAU     (input) DOUBLE PRECISION array, dimension (K)   
            TAU(i) must contain the scalar factor of the elementary   
            reflector H(i), as returned by DGEQRF.   

    C       (input/output) DOUBLE PRECISION array, dimension (LDC,N)   
            On entry, the m by n matrix C.   
            On exit, C is overwritten by Q*C or Q'*C or C*Q' or C*Q.   

    LDC     (input) INTEGER   
            The leading dimension of the array C. LDC >= max(1,M).   

    WORK    (workspace) DOUBLE PRECISION array, dimension   
                                     (N) if SIDE = 'L',   
                                     (M) if SIDE = 'R'   

    INFO    (output) INTEGER   
            = 0: successful exit   
            < 0: if INFO = -i, the i-th argument had an illegal value   

    =====================================================================   


       Test the input arguments   

       Parameter adjustments */
    /* Table of constant values */
     integer c__1 = 1;
    
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2;
    /* Local variables */
     logical left;
     integer i__;
    extern /* Subroutine */ integer dlarf_(const char *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *, integer *, 
	    doublereal *);
    extern logical lsame_(const char *,const char *);
     integer i1, i2, i3, ic, jc, mi, ni, nq;
    extern /* Subroutine */ integer xerbla_(const char *, integer *);
     logical notran;
     doublereal aii;
#define a_ref(a_1,a_2) a[(a_2)*a_dim1 + a_1]
#define c___ref(a_1,a_2) c__[(a_2)*c_dim1 + a_1]


    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1 * 1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");

/*     NQ is the order of Q */

    if (left) {
	nq = *m;
    } else {
	nq = *n;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0 || *k > nq) {
	*info = -5;
    } else if (*lda < max(1,nq)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORM2R", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0 || *k == 0) {
	return 0;
    }

    if ((left && ! notran) || (! left && notran)) {
	i1 = 1;
	i2 = *k;
	i3 = 1;
    } else {
	i1 = *k;
	i2 = 1;
	i3 = -1;
    }

    if (left) {
	ni = *n;
	jc = 1;
    } else {
	mi = *m;
	ic = 1;
    }

    i__1 = i2;
    i__2 = i3;
    for (i__ = i1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	if (left) {

/*           H(i) is applied to C(i:m,1:n) */

	    mi = *m - i__ + 1;
	    ic = i__;
	} else {

/*           H(i) is applied to C(1:m,i:n) */

	    ni = *n - i__ + 1;
	    jc = i__;
	}

/*        Apply H(i) */

	aii = a_ref(i__, i__);
	a_ref(i__, i__) = 1.;
	dlarf_(side, &mi, &ni, &a_ref(i__, i__), &c__1, &tau[i__], &c___ref(
		ic, jc), ldc, &work[1]);
	a_ref(i__, i__) = aii;
/* L10: */
    }
    return 0;

/*     End of DORM2R */

} /* dorm2r_ */

#undef c___ref
#undef a_ref

#ifdef __cplusplus
}
#endif

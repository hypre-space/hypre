#include "../blas/hypre_blas.h"
#include "hypre_lapack.h"
#include "f2c.h"

/* Subroutine */ HYPRE_Int dsygs2_(integer *itype, char *uplo, integer *n, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *
	info)
{
/*  -- LAPACK routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       February 29, 1992   


    Purpose   
    =======   

    DSYGS2 reduces a real symmetric-definite generalized eigenproblem   
    to standard form.   

    If ITYPE = 1, the problem is A*x = lambda*B*x,   
    and A is overwritten by inv(U')*A*inv(U) or inv(L)*A*inv(L')   

    If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or   
    B*A*x = lambda*x, and A is overwritten by U*A*U` or L'*A*L.   

    B must have been previously factorized as U'*U or L*L' by DPOTRF.   

    Arguments   
    =========   

    ITYPE   (input) INTEGER   
            = 1: compute inv(U')*A*inv(U) or inv(L)*A*inv(L');   
            = 2 or 3: compute U*A*U' or L'*A*L.   

    UPLO    (input) CHARACTER   
            Specifies whether the upper or lower triangular part of the   
            symmetric matrix A is stored, and how B has been factorized.   
            = 'U':  Upper triangular   
            = 'L':  Lower triangular   

    N       (input) INTEGER   
            The order of the matrices A and B.  N >= 0.   

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)   
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading   
            n by n upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading n by n lower triangular part of A contains the lower   
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
            = 0:  successful exit.   
            < 0:  if INFO = -i, the i-th argument had an illegal value.   

    =====================================================================   


       Test the input parameters.   

       Parameter adjustments */
    /* Table of constant values */
    static doublereal c_b6 = -1.;
    static integer c__1 = 1;
    static doublereal c_b27 = 1.;
    
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;
    doublereal d__1;
    /* Local variables */
    extern /* Subroutine */ HYPRE_Int dsyr2_(char *, integer *, doublereal *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    static integer k;
    extern /* Subroutine */ HYPRE_Int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ HYPRE_Int daxpy_(integer *, doublereal *, doublereal *, 
	    integer *, doublereal *, integer *);
    static logical upper;
    extern /* Subroutine */ HYPRE_Int dtrmv_(char *, char *, char *, integer *, 
	    doublereal *, integer *, doublereal *, integer *), dtrsv_(char *, char *, char *, integer *, doublereal *, 
	    integer *, doublereal *, integer *);
    static doublereal ct;
    extern /* Subroutine */ HYPRE_Int xerbla_(char *, integer *);
    static doublereal akk, bkk;
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
	xerbla_("DSYGS2", &i__1);
	return 0;
    }

    if (*itype == 1) {
	if (upper) {

/*           Compute inv(U')*A*inv(U) */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the upper triangle of A(k:n,k:n) */

		akk = a_ref(k, k);
		bkk = b_ref(k, k);
/* Computing 2nd power */
		d__1 = bkk;
		akk /= d__1 * d__1;
		a_ref(k, k) = akk;
		if (k < *n) {
		    i__2 = *n - k;
		    d__1 = 1. / bkk;
		    dscal_(&i__2, &d__1, &a_ref(k, k + 1), lda);
		    ct = akk * -.5;
		    i__2 = *n - k;
		    daxpy_(&i__2, &ct, &b_ref(k, k + 1), ldb, &a_ref(k, k + 1)
			    , lda);
		    i__2 = *n - k;
		    dsyr2_(uplo, &i__2, &c_b6, &a_ref(k, k + 1), lda, &b_ref(
			    k, k + 1), ldb, &a_ref(k + 1, k + 1), lda);
		    i__2 = *n - k;
		    daxpy_(&i__2, &ct, &b_ref(k, k + 1), ldb, &a_ref(k, k + 1)
			    , lda);
		    i__2 = *n - k;
		    dtrsv_(uplo, "Transpose", "Non-unit", &i__2, &b_ref(k + 1,
			     k + 1), ldb, &a_ref(k, k + 1), lda);
		}
/* L10: */
	    }
	} else {

/*           Compute inv(L)*A*inv(L') */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the lower triangle of A(k:n,k:n) */

		akk = a_ref(k, k);
		bkk = b_ref(k, k);
/* Computing 2nd power */
		d__1 = bkk;
		akk /= d__1 * d__1;
		a_ref(k, k) = akk;
		if (k < *n) {
		    i__2 = *n - k;
		    d__1 = 1. / bkk;
		    dscal_(&i__2, &d__1, &a_ref(k + 1, k), &c__1);
		    ct = akk * -.5;
		    i__2 = *n - k;
		    daxpy_(&i__2, &ct, &b_ref(k + 1, k), &c__1, &a_ref(k + 1, 
			    k), &c__1);
		    i__2 = *n - k;
		    dsyr2_(uplo, &i__2, &c_b6, &a_ref(k + 1, k), &c__1, &
			    b_ref(k + 1, k), &c__1, &a_ref(k + 1, k + 1), lda);
		    i__2 = *n - k;
		    daxpy_(&i__2, &ct, &b_ref(k + 1, k), &c__1, &a_ref(k + 1, 
			    k), &c__1);
		    i__2 = *n - k;
		    dtrsv_(uplo, "No transpose", "Non-unit", &i__2, &b_ref(k 
			    + 1, k + 1), ldb, &a_ref(k + 1, k), &c__1);
		}
/* L20: */
	    }
	}
    } else {
	if (upper) {

/*           Compute U*A*U' */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the upper triangle of A(1:k,1:k) */

		akk = a_ref(k, k);
		bkk = b_ref(k, k);
		i__2 = k - 1;
		dtrmv_(uplo, "No transpose", "Non-unit", &i__2, &b[b_offset], 
			ldb, &a_ref(1, k), &c__1);
		ct = akk * .5;
		i__2 = k - 1;
		daxpy_(&i__2, &ct, &b_ref(1, k), &c__1, &a_ref(1, k), &c__1);
		i__2 = k - 1;
		dsyr2_(uplo, &i__2, &c_b27, &a_ref(1, k), &c__1, &b_ref(1, k),
			 &c__1, &a[a_offset], lda);
		i__2 = k - 1;
		daxpy_(&i__2, &ct, &b_ref(1, k), &c__1, &a_ref(1, k), &c__1);
		i__2 = k - 1;
		dscal_(&i__2, &bkk, &a_ref(1, k), &c__1);
/* Computing 2nd power */
		d__1 = bkk;
		a_ref(k, k) = akk * (d__1 * d__1);
/* L30: */
	    }
	} else {

/*           Compute L'*A*L */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the lower triangle of A(1:k,1:k) */

		akk = a_ref(k, k);
		bkk = b_ref(k, k);
		i__2 = k - 1;
		dtrmv_(uplo, "Transpose", "Non-unit", &i__2, &b[b_offset], 
			ldb, &a_ref(k, 1), lda);
		ct = akk * .5;
		i__2 = k - 1;
		daxpy_(&i__2, &ct, &b_ref(k, 1), ldb, &a_ref(k, 1), lda);
		i__2 = k - 1;
		dsyr2_(uplo, &i__2, &c_b27, &a_ref(k, 1), lda, &b_ref(k, 1), 
			ldb, &a[a_offset], lda);
		i__2 = k - 1;
		daxpy_(&i__2, &ct, &b_ref(k, 1), ldb, &a_ref(k, 1), lda);
		i__2 = k - 1;
		dscal_(&i__2, &bkk, &a_ref(k, 1), lda);
/* Computing 2nd power */
		d__1 = bkk;
		a_ref(k, k) = akk * (d__1 * d__1);
/* L40: */
	    }
	}
    }
    return 0;

/*     End of DSYGS2 */

} /* dsygs2_ */

#undef b_ref
#undef a_ref



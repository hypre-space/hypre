#include "hypre_blas.h"
#include "f2c.h"

/* Subroutine */ HYPRE_Int ztrmv_(char *uplo, char *trans, char *diag, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *x, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublecomplex z__1, z__2, z__3;
    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);
    /* Local variables */
    static integer info;
    static doublecomplex temp;
    static integer i__, j;
    extern logical hypre_lsame_(char *, char *);
    static integer ix, jx, kx;
    extern /* Subroutine */ HYPRE_Int hypre_xerbla_(char *, integer *);
    static logical noconj, nounit;
#define a_subscr(a_1,a_2) (a_2)*a_dim1 + a_1
#define a_ref(a_1,a_2) a[a_subscr(a_1,a_2)]
/*  Purpose   
    =======   
    ZTRMV  performs one of the matrix-vector operations   
       x := A*x,   or   x := A'*x,   or   x := conjg( A' )*x,   
    where x is an n element vector and  A is an n by n unit, or non-unit,   
    upper or lower triangular matrix.   
    Parameters   
    ==========   
    UPLO   - CHARACTER*1.   
             On entry, UPLO specifies whether the matrix is an upper or   
             lower triangular matrix as follows:   
                UPLO = 'U' or 'u'   A is an upper triangular matrix.   
                UPLO = 'L' or 'l'   A is a lower triangular matrix.   
             Unchanged on exit.   
    TRANS  - CHARACTER*1.   
             On entry, TRANS specifies the operation to be performed as   
             follows:   
                TRANS = 'N' or 'n'   x := A*x.   
                TRANS = 'T' or 't'   x := A'*x.   
                TRANS = 'C' or 'c'   x := conjg( A' )*x.   
             Unchanged on exit.   
    DIAG   - CHARACTER*1.   
             On entry, DIAG specifies whether or not A is unit   
             triangular as follows:   
                DIAG = 'U' or 'u'   A is assumed to be unit triangular.   
                DIAG = 'N' or 'n'   A is not assumed to be unit   
                                    triangular.   
             Unchanged on exit.   
    N      - INTEGER.   
             On entry, N specifies the order of the matrix A.   
             N must be at least zero.   
             Unchanged on exit.   
    A      - COMPLEX*16       array of DIMENSION ( LDA, n ).   
             Before entry with  UPLO = 'U' or 'u', the leading n by n   
             upper triangular part of the array A must contain the upper   
             triangular matrix and the strictly lower triangular part of   
             A is not referenced.   
             Before entry with UPLO = 'L' or 'l', the leading n by n   
             lower triangular part of the array A must contain the lower   
             triangular matrix and the strictly upper triangular part of   
             A is not referenced.   
             Note that when  DIAG = 'U' or 'u', the diagonal elements of   
             A are not referenced either, but are assumed to be unity.   
             Unchanged on exit.   
    LDA    - INTEGER.   
             On entry, LDA specifies the first dimension of A as declared   
             in the calling (sub) program. LDA must be at least   
             max( 1, n ).   
             Unchanged on exit.   
    X      - COMPLEX*16       array of dimension at least   
             ( 1 + ( n - 1 )*abs( INCX ) ).   
             Before entry, the incremented array X must contain the n   
             element vector x. On exit, X is overwritten with the   
             tranformed vector x.   
    INCX   - INTEGER.   
             On entry, INCX specifies the increment for the elements of   
             X. INCX must not be zero.   
             Unchanged on exit.   
    Level 2 Blas routine.   
    -- Written on 22-October-1986.   
       Jack Dongarra, Argonne National Lab.   
       Jeremy Du Croz, Nag Central Office.   
       Sven Hammarling, Nag Central Office.   
       Richard Hanson, Sandia National Labs.   
       Test the input parameters.   
       Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;
    /* Function Body */
    info = 0;
    if (! hypre_lsame_(uplo, "U") && ! hypre_lsame_(uplo, "L")) {
	info = 1;
    } else if (! hypre_lsame_(trans, "N") && ! hypre_lsame_(trans, 
	    "T") && ! hypre_lsame_(trans, "C")) {
	info = 2;
    } else if (! hypre_lsame_(diag, "U") && ! hypre_lsame_(diag, 
	    "N")) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*lda < max(1,*n)) {
	info = 6;
    } else if (*incx == 0) {
	info = 8;
    }
    if (info != 0) {
	hypre_xerbla_("ZTRMV ", &info);
	return 0;
    }
/*     Quick return if possible. */
    if (*n == 0) {
	return 0;
    }
    noconj = hypre_lsame_(trans, "T");
    nounit = hypre_lsame_(diag, "N");
/*     Set up the start point in X if the increment is not unity. This   
       will be  ( N - 1 )*INCX  too small for descending loops. */
    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }
/*     Start the operations. In this version the elements of A are   
       accessed sequentially with one pass through A. */
    if (hypre_lsame_(trans, "N")) {
/*        Form  x := A*x. */
	if (hypre_lsame_(uplo, "U")) {
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    if (x[i__2].r != 0. || x[i__2].i != 0.) {
			i__2 = j;
			temp.r = x[i__2].r, temp.i = x[i__2].i;
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    i__3 = i__;
			    i__4 = i__;
			    i__5 = a_subscr(i__, j);
			    z__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i, 
				    z__2.i = temp.r * a[i__5].i + temp.i * a[
				    i__5].r;
			    z__1.r = x[i__4].r + z__2.r, z__1.i = x[i__4].i + 
				    z__2.i;
			    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
/* L10: */
			}
			if (nounit) {
			    i__2 = j;
			    i__3 = j;
			    i__4 = a_subscr(j, j);
			    z__1.r = x[i__3].r * a[i__4].r - x[i__3].i * a[
				    i__4].i, z__1.i = x[i__3].r * a[i__4].i + 
				    x[i__3].i * a[i__4].r;
			    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
			}
		    }
/* L20: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = jx;
		    if (x[i__2].r != 0. || x[i__2].i != 0.) {
			i__2 = jx;
			temp.r = x[i__2].r, temp.i = x[i__2].i;
			ix = kx;
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    i__3 = ix;
			    i__4 = ix;
			    i__5 = a_subscr(i__, j);
			    z__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i, 
				    z__2.i = temp.r * a[i__5].i + temp.i * a[
				    i__5].r;
			    z__1.r = x[i__4].r + z__2.r, z__1.i = x[i__4].i + 
				    z__2.i;
			    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
			    ix += *incx;
/* L30: */
			}
			if (nounit) {
			    i__2 = jx;
			    i__3 = jx;
			    i__4 = a_subscr(j, j);
			    z__1.r = x[i__3].r * a[i__4].r - x[i__3].i * a[
				    i__4].i, z__1.i = x[i__3].r * a[i__4].i + 
				    x[i__3].i * a[i__4].r;
			    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
			}
		    }
		    jx += *incx;
/* L40: */
		}
	    }
	} else {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    if (x[i__1].r != 0. || x[i__1].i != 0.) {
			i__1 = j;
			temp.r = x[i__1].r, temp.i = x[i__1].i;
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    i__2 = i__;
			    i__3 = i__;
			    i__4 = a_subscr(i__, j);
			    z__2.r = temp.r * a[i__4].r - temp.i * a[i__4].i, 
				    z__2.i = temp.r * a[i__4].i + temp.i * a[
				    i__4].r;
			    z__1.r = x[i__3].r + z__2.r, z__1.i = x[i__3].i + 
				    z__2.i;
			    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
/* L50: */
			}
			if (nounit) {
			    i__1 = j;
			    i__2 = j;
			    i__3 = a_subscr(j, j);
			    z__1.r = x[i__2].r * a[i__3].r - x[i__2].i * a[
				    i__3].i, z__1.i = x[i__2].r * a[i__3].i + 
				    x[i__2].i * a[i__3].r;
			    x[i__1].r = z__1.r, x[i__1].i = z__1.i;
			}
		    }
/* L60: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    i__1 = jx;
		    if (x[i__1].r != 0. || x[i__1].i != 0.) {
			i__1 = jx;
			temp.r = x[i__1].r, temp.i = x[i__1].i;
			ix = kx;
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    i__2 = ix;
			    i__3 = ix;
			    i__4 = a_subscr(i__, j);
			    z__2.r = temp.r * a[i__4].r - temp.i * a[i__4].i, 
				    z__2.i = temp.r * a[i__4].i + temp.i * a[
				    i__4].r;
			    z__1.r = x[i__3].r + z__2.r, z__1.i = x[i__3].i + 
				    z__2.i;
			    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
			    ix -= *incx;
/* L70: */
			}
			if (nounit) {
			    i__1 = jx;
			    i__2 = jx;
			    i__3 = a_subscr(j, j);
			    z__1.r = x[i__2].r * a[i__3].r - x[i__2].i * a[
				    i__3].i, z__1.i = x[i__2].r * a[i__3].i + 
				    x[i__2].i * a[i__3].r;
			    x[i__1].r = z__1.r, x[i__1].i = z__1.i;
			}
		    }
		    jx -= *incx;
/* L80: */
		}
	    }
	}
    } else {
/*        Form  x := A'*x  or  x := conjg( A' )*x. */
	if (hypre_lsame_(uplo, "U")) {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    temp.r = x[i__1].r, temp.i = x[i__1].i;
		    if (noconj) {
			if (nounit) {
			    i__1 = a_subscr(j, j);
			    z__1.r = temp.r * a[i__1].r - temp.i * a[i__1].i, 
				    z__1.i = temp.r * a[i__1].i + temp.i * a[
				    i__1].r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			for (i__ = j - 1; i__ >= 1; --i__) {
			    i__1 = a_subscr(i__, j);
			    i__2 = i__;
			    z__2.r = a[i__1].r * x[i__2].r - a[i__1].i * x[
				    i__2].i, z__2.i = a[i__1].r * x[i__2].i + 
				    a[i__1].i * x[i__2].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L90: */
			}
		    } else {
			if (nounit) {
			    d_cnjg(&z__2, &a_ref(j, j));
			    z__1.r = temp.r * z__2.r - temp.i * z__2.i, 
				    z__1.i = temp.r * z__2.i + temp.i * 
				    z__2.r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			for (i__ = j - 1; i__ >= 1; --i__) {
			    d_cnjg(&z__3, &a_ref(i__, j));
			    i__1 = i__;
			    z__2.r = z__3.r * x[i__1].r - z__3.i * x[i__1].i, 
				    z__2.i = z__3.r * x[i__1].i + z__3.i * x[
				    i__1].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L100: */
			}
		    }
		    i__1 = j;
		    x[i__1].r = temp.r, x[i__1].i = temp.i;
/* L110: */
		}
	    } else {
		jx = kx + (*n - 1) * *incx;
		for (j = *n; j >= 1; --j) {
		    i__1 = jx;
		    temp.r = x[i__1].r, temp.i = x[i__1].i;
		    ix = jx;
		    if (noconj) {
			if (nounit) {
			    i__1 = a_subscr(j, j);
			    z__1.r = temp.r * a[i__1].r - temp.i * a[i__1].i, 
				    z__1.i = temp.r * a[i__1].i + temp.i * a[
				    i__1].r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			for (i__ = j - 1; i__ >= 1; --i__) {
			    ix -= *incx;
			    i__1 = a_subscr(i__, j);
			    i__2 = ix;
			    z__2.r = a[i__1].r * x[i__2].r - a[i__1].i * x[
				    i__2].i, z__2.i = a[i__1].r * x[i__2].i + 
				    a[i__1].i * x[i__2].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L120: */
			}
		    } else {
			if (nounit) {
			    d_cnjg(&z__2, &a_ref(j, j));
			    z__1.r = temp.r * z__2.r - temp.i * z__2.i, 
				    z__1.i = temp.r * z__2.i + temp.i * 
				    z__2.r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			for (i__ = j - 1; i__ >= 1; --i__) {
			    ix -= *incx;
			    d_cnjg(&z__3, &a_ref(i__, j));
			    i__1 = ix;
			    z__2.r = z__3.r * x[i__1].r - z__3.i * x[i__1].i, 
				    z__2.i = z__3.r * x[i__1].i + z__3.i * x[
				    i__1].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L130: */
			}
		    }
		    i__1 = jx;
		    x[i__1].r = temp.r, x[i__1].i = temp.i;
		    jx -= *incx;
/* L140: */
		}
	    }
	} else {
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    temp.r = x[i__2].r, temp.i = x[i__2].i;
		    if (noconj) {
			if (nounit) {
			    i__2 = a_subscr(j, j);
			    z__1.r = temp.r * a[i__2].r - temp.i * a[i__2].i, 
				    z__1.i = temp.r * a[i__2].i + temp.i * a[
				    i__2].r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    i__3 = a_subscr(i__, j);
			    i__4 = i__;
			    z__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[
				    i__4].i, z__2.i = a[i__3].r * x[i__4].i + 
				    a[i__3].i * x[i__4].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L150: */
			}
		    } else {
			if (nounit) {
			    d_cnjg(&z__2, &a_ref(j, j));
			    z__1.r = temp.r * z__2.r - temp.i * z__2.i, 
				    z__1.i = temp.r * z__2.i + temp.i * 
				    z__2.r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    d_cnjg(&z__3, &a_ref(i__, j));
			    i__3 = i__;
			    z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i, 
				    z__2.i = z__3.r * x[i__3].i + z__3.i * x[
				    i__3].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L160: */
			}
		    }
		    i__2 = j;
		    x[i__2].r = temp.r, x[i__2].i = temp.i;
/* L170: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = jx;
		    temp.r = x[i__2].r, temp.i = x[i__2].i;
		    ix = jx;
		    if (noconj) {
			if (nounit) {
			    i__2 = a_subscr(j, j);
			    z__1.r = temp.r * a[i__2].r - temp.i * a[i__2].i, 
				    z__1.i = temp.r * a[i__2].i + temp.i * a[
				    i__2].r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    ix += *incx;
			    i__3 = a_subscr(i__, j);
			    i__4 = ix;
			    z__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[
				    i__4].i, z__2.i = a[i__3].r * x[i__4].i + 
				    a[i__3].i * x[i__4].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L180: */
			}
		    } else {
			if (nounit) {
			    d_cnjg(&z__2, &a_ref(j, j));
			    z__1.r = temp.r * z__2.r - temp.i * z__2.i, 
				    z__1.i = temp.r * z__2.i + temp.i * 
				    z__2.r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    ix += *incx;
			    d_cnjg(&z__3, &a_ref(i__, j));
			    i__3 = ix;
			    z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i, 
				    z__2.i = z__3.r * x[i__3].i + z__3.i * x[
				    i__3].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L190: */
			}
		    }
		    i__2 = jx;
		    x[i__2].r = temp.r, x[i__2].i = temp.i;
		    jx += *incx;
/* L200: */
		}
	    }
	}
    }
    return 0;
/*     End of ZTRMV . */
} /* ztrmv_ */
#undef a_ref
#undef a_subscr


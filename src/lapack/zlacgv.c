#include "hypre_lapack.h"
#include "../blas/hypre_blas.h"
#include "f2c.h"

/* Subroutine */ HYPRE_Int zlacgv_(integer *n, doublecomplex *x, integer *incx)
{
/*  -- LAPACK auxiliary routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       October 31, 1992   


    Purpose   
    =======   

    ZLACGV conjugates a complex vector of length N.   

    Arguments   
    =========   

    N       (input) INTEGER   
            The length of the vector X.  N >= 0.   

    X       (input/output) COMPLEX*16 array, dimension   
                           (1+(N-1)*abs(INCX))   
            On entry, the vector of length N to be conjugated.   
            On exit, X is overwritten with conjg(X).   

    INCX    (input) INTEGER   
            The spacing between successive elements of X.   

   =====================================================================   


       Parameter adjustments */
    /* System generated locals */
    integer i__1, i__2;
    doublecomplex z__1;
    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);
    /* Local variables */
    static integer ioff, i__;

    --x;

    /* Function Body */
    if (*incx == 1) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__;
	    d_cnjg(&z__1, &x[i__]);
	    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
/* L10: */
	}
    } else {
	ioff = 1;
	if (*incx < 0) {
	    ioff = 1 - (*n - 1) * *incx;
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = ioff;
	    d_cnjg(&z__1, &x[ioff]);
	    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
	    ioff += *incx;
/* L20: */
	}
    }
    return 0;

/*     End of ZLACGV */

} /* zlacgv_ */


#include "../blas/hypre_blas.h"
#include "f2c.h"

/* Double Complex */ VOID zladiv_(doublecomplex * ret_val, doublecomplex *x, 
	doublecomplex *y)
{
/*  -- LAPACK auxiliary routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       October 31, 1992   


    Purpose   
    =======   

    ZLADIV := X / Y, where X and Y are complex.  The computation of X / Y   
    will not overflow on an intermediary step unless the results   
    overflows.   

    Arguments   
    =========   

    X       (input) COMPLEX*16   
    Y       (input) COMPLEX*16   
            The complex scalars X and Y.   

    ===================================================================== */
    /* System generated locals */
    doublereal d__1, d__2, d__3, d__4;
    doublecomplex z__1;
    /* Builtin functions */
    double d_imag(doublecomplex *);
    /* Local variables */
    static doublereal zi;
    extern /* Subroutine */ HYPRE_Int dladiv_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *);
    static doublereal zr;



    d__1 = x->r;
    d__2 = d_imag(x);
    d__3 = y->r;
    d__4 = d_imag(y);
    dladiv_(&d__1, &d__2, &d__3, &d__4, &zr, &zi);
    z__1.r = zr, z__1.i = zi;
     ret_val->r = z__1.r,  ret_val->i = z__1.i;

    return ;

/*     End of ZLADIV */

} /* zladiv_ */


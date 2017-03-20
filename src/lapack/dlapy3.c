#include "../blas/hypre_blas.h"
#include "f2c.h"

doublereal dlapy3_(doublereal *x, doublereal *y, doublereal *z__)
{
/*  -- LAPACK auxiliary routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       October 31, 1992   


    Purpose   
    =======   

    DLAPY3 returns sqrt(x**2+y**2+z**2), taking care not to cause   
    unnecessary overflow.   

    Arguments   
    =========   

    X       (input) DOUBLE PRECISION   
    Y       (input) DOUBLE PRECISION   
    Z       (input) DOUBLE PRECISION   
            X, Y and Z specify the values x, y and z.   

    ===================================================================== */
    /* System generated locals */
    doublereal ret_val, d__1, d__2, d__3;
    /* Builtin functions */
    double sqrt(doublereal);
    /* Local variables */
    static doublereal xabs, yabs, zabs, w;



    xabs = abs(*x);
    yabs = abs(*y);
    zabs = abs(*z__);
/* Computing MAX */
    d__1 = max(xabs,yabs);
    w = max(d__1,zabs);
    if (w == 0.) {
	ret_val = 0.;
    } else {
/* Computing 2nd power */
	d__1 = xabs / w;
/* Computing 2nd power */
	d__2 = yabs / w;
/* Computing 2nd power */
	d__3 = zabs / w;
	ret_val = w * sqrt(d__1 * d__1 + d__2 * d__2 + d__3 * d__3);
    }
    return ret_val;

/*     End of DLAPY3 */

} /* dlapy3_ */


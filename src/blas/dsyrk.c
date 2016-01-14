

/*  -- translated by f2c (version 19940927).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "f2c.h"
#include "hypre_blas.h"

/* Subroutine */ HYPRE_Int dsyrk_(char *uplo, char *trans, integer *n, integer *k, 
	doublereal *alpha, doublereal *a, integer *lda, doublereal *beta, 
	doublereal *c, integer *ldc)
{


    /* System generated locals */

    /* Local variables */
    static integer info;
    static doublereal temp;
    static integer i, j, l;
    extern logical hypre_lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ HYPRE_Int hypre_xerbla_(char *, integer *);


/*  Purpose   
    =======   

    DSYRK  performs one of the symmetric rank k operations   

       C := alpha*A*A' + beta*C,   

    or   

       C := alpha*A'*A + beta*C,   

    where  alpha and beta  are scalars, C is an  n by n  symmetric matrix 
  
    and  A  is an  n by k  matrix in the first case and a  k by n  matrix 
  
    in the second case.   

    Parameters   
    ==========   

    UPLO   - CHARACTER*1.   
             On  entry,   UPLO  specifies  whether  the  upper  or  lower 
  
             triangular  part  of the  array  C  is to be  referenced  as 
  
             follows:   

                UPLO = 'U' or 'u'   Only the  upper triangular part of  C 
  
                                    is to be referenced.   

                UPLO = 'L' or 'l'   Only the  lower triangular part of  C 
  
                                    is to be referenced.   

             Unchanged on exit.   

    TRANS  - CHARACTER*1.   
             On entry,  TRANS  specifies the operation to be performed as 
  
             follows:   

                TRANS = 'N' or 'n'   C := alpha*A*A' + beta*C.   

                TRANS = 'T' or 't'   C := alpha*A'*A + beta*C.   

                TRANS = 'C' or 'c'   C := alpha*A'*A + beta*C.   

             Unchanged on exit.   

    N      - INTEGER.   
             On entry,  N specifies the order of the matrix C.  N must be 
  
             at least zero.   
             Unchanged on exit.   

    K      - INTEGER.   
             On entry with  TRANS = 'N' or 'n',  K  specifies  the number 
  
             of  columns   of  the   matrix   A,   and  on   entry   with 
  
             TRANS = 'T' or 't' or 'C' or 'c',  K  specifies  the  number 
  
             of rows of the matrix  A.  K must be at least zero.   
             Unchanged on exit.   

    ALPHA  - DOUBLE PRECISION.   
             On entry, ALPHA specifies the scalar alpha.   
             Unchanged on exit.   

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is 
  
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.   
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k 
  
             part of the array  A  must contain the matrix  A,  otherwise 
  
             the leading  k by n  part of the array  A  must contain  the 
  
             matrix A.   
             Unchanged on exit.   

    LDA    - INTEGER.   
             On entry, LDA specifies the first dimension of A as declared 
  
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n' 
  
             then  LDA must be at least  max( 1, n ), otherwise  LDA must 
  
             be at least  max( 1, k ).   
             Unchanged on exit.   

    BETA   - DOUBLE PRECISION.   
             On entry, BETA specifies the scalar beta.   
             Unchanged on exit.   

    C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).   
             Before entry  with  UPLO = 'U' or 'u',  the leading  n by n 
  
             upper triangular part of the array C must contain the upper 
  
             triangular part  of the  symmetric matrix  and the strictly 
  
             lower triangular part of C is not referenced.  On exit, the 
  
             upper triangular part of the array  C is overwritten by the 
  
             upper triangular part of the updated matrix.   
             Before entry  with  UPLO = 'L' or 'l',  the leading  n by n 
  
             lower triangular part of the array C must contain the lower 
  
             triangular part  of the  symmetric matrix  and the strictly 
  
             upper triangular part of C is not referenced.  On exit, the 
  
             lower triangular part of the array  C is overwritten by the 
  
             lower triangular part of the updated matrix.   

    LDC    - INTEGER.   
             On entry, LDC specifies the first dimension of C as declared 
  
             in  the  calling  (sub)  program.   LDC  must  be  at  least 
  
             max( 1, n ).   
             Unchanged on exit.   


    Level 3 Blas routine.   

    -- Written on 8-February-1989.   
       Jack Dongarra, Argonne National Laboratory.   
       Iain Duff, AERE Harwell.   
       Jeremy Du Croz, Numerical Algorithms Group Ltd.   
       Sven Hammarling, Numerical Algorithms Group Ltd.   



       Test the input parameters.   

    
   Parameter adjustments   
       Function Body */

#define A(I,J) a[(I)-1 + ((J)-1)* ( *lda)]
#define C(I,J) c[(I)-1 + ((J)-1)* ( *ldc)]

    if (hypre_lsame_(trans, "N")) {
	nrowa = *n;
    } else {
	nrowa = *k;
    }
    upper = hypre_lsame_(uplo, "U");

    info = 0;
    if (! upper && ! hypre_lsame_(uplo, "L")) {
	info = 1;
    } else if (! hypre_lsame_(trans, "N") && ! hypre_lsame_(trans, "T") &&
	     ! hypre_lsame_(trans, "C")) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*k < 0) {
	info = 4;
    } else if (*lda < max(1,nrowa)) {
	info = 7;
    } else if (*ldc < max(1,*n)) {
	info = 10;
    }
    if (info != 0) {
	hypre_xerbla_("DSYRK ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || ((*alpha == 0. || *k == 0) && (*beta == 1.))) {
	return 0;
    }

/*     And when  alpha.eq.zero. */

    if (*alpha == 0.) {
	if (upper) {
	    if (*beta == 0.) {
		for (j = 1; j <= *n; ++j) {
		    for (i = 1; i <= j; ++i) {
			C(i,j) = 0.;
/* L10: */
		    }
/* L20: */
		}
	    } else {
		for (j = 1; j <= *n; ++j) {
		    for (i = 1; i <= j; ++i) {
			C(i,j) = *beta * C(i,j);
/* L30: */
		    }
/* L40: */
		}
	    }
	} else {
	    if (*beta == 0.) {
		for (j = 1; j <= *n; ++j) {
		    for (i = j; i <= *n; ++i) {
			C(i,j) = 0.;
/* L50: */
		    }
/* L60: */
		}
	    } else {
		for (j = 1; j <= *n; ++j) {
		    for (i = j; i <= *n; ++i) {
			C(i,j) = *beta * C(i,j);
/* L70: */
		    }
/* L80: */
		}
	    }
	}
	return 0;
    }

/*     Start the operations. */

    if (hypre_lsame_(trans, "N")) {

/*        Form  C := alpha*A*A' + beta*C. */

	if (upper) {
	    for (j = 1; j <= *n; ++j) {
		if (*beta == 0.) {
		    for (i = 1; i <= j; ++i) {
			C(i,j) = 0.;
/* L90: */
		    }
		} else if (*beta != 1.) {
		    for (i = 1; i <= j; ++i) {
			C(i,j) = *beta * C(i,j);
/* L100: */
		    }
		}
		for (l = 1; l <= *k; ++l) {
		    if (A(j,l) != 0.) {
			temp = *alpha * A(j,l);
			for (i = 1; i <= j; ++i) {
			    C(i,j) += temp * A(i,l);
/* L110: */
			}
		    }
/* L120: */
		}
/* L130: */
	    }
	} else {
	    for (j = 1; j <= *n; ++j) {
		if (*beta == 0.) {
		    for (i = j; i <= *n; ++i) {
			C(i,j) = 0.;
/* L140: */
		    }
		} else if (*beta != 1.) {
		    for (i = j; i <= *n; ++i) {
			C(i,j) = *beta * C(i,j);
/* L150: */
		    }
		}
		for (l = 1; l <= *k; ++l) {
		    if (A(j,l) != 0.) {
			temp = *alpha * A(j,l);
			for (i = j; i <= *n; ++i) {
			    C(i,j) += temp * A(i,l);
/* L160: */
			}
		    }
/* L170: */
		}
/* L180: */
	    }
	}
    } else {

/*        Form  C := alpha*A'*A + beta*C. */

	if (upper) {
	    for (j = 1; j <= *n; ++j) {
		for (i = 1; i <= j; ++i) {
		    temp = 0.;
		    for (l = 1; l <= *k; ++l) {
			temp += A(l,i) * A(l,j);
/* L190: */
		    }
		    if (*beta == 0.) {
			C(i,j) = *alpha * temp;
		    } else {
			C(i,j) = *alpha * temp + *beta * C(i,j);
		    }
/* L200: */
		}
/* L210: */
	    }
	} else {
	    for (j = 1; j <= *n; ++j) {
		for (i = j; i <= *n; ++i) {
		    temp = 0.;
		    for (l = 1; l <= *k; ++l) {
			temp += A(l,i) * A(l,j);
/* L220: */
		    }
		    if (*beta == 0.) {
			C(i,j) = *alpha * temp;
		    } else {
			C(i,j) = *alpha * temp + *beta * C(i,j);
		    }
/* L230: */
		}
/* L240: */
	    }
	}
    }

    return 0;

/*     End of DSYRK . */

} /* dsyrk_ */

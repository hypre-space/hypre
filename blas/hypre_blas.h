/* blas.h  --  Contains BLAS prototypes needed by Hypre */

#ifndef HYPRE_BLAS_H
#define HYPRE_BLAS_H
#include "f2c.h"

/* --------------------------------------------------------------------------
 *   Change all names to  to avoid link conflicts
 * --------------------------------------------------------------------------*/

#define dasum   dasum
#define daxpy   daxpy
#define dcopy   dcopy
#define ddot    ddot
#define dgemm   dgemm
#define dgemv   dgemv
#define dger    dger
#define dnrm2   dnrm2
#define drot    drot
#define dscal   dscal
#define dswap   dswap
#define dsymm   dsymm
#define dsymv   dsymv
#define dsyr2   dsyr2
#define dsyr2k  dsyr2k
#define dsyrk   dsyrk
#define dtrmm   dtrmm
#define dtrmv   dtrmv
#define dtrsm   dtrsm
#define dtrsv   dtrsv
#define idamax  idamax

/* blas_utils.c */
logical lsame_ ( char *ca , char *cb );
int xerbla_ ( char *srname , integer *info );
integer s_cmp ( char *a0 , char *b0 , ftnlen la , ftnlen lb );
VOID s_copy ( char *a , char *b , ftnlen la , ftnlen lb );

/* dasum.c */
doublereal dasum_ ( integer *n , doublereal *dx , integer *incx );

/* daxpy.c */
int daxpy_ ( integer *n , doublereal *da , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dcopy.c */
int dcopy_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* ddot.c */
doublereal ddot_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dgemm.c */
int dgemm_ ( char *transa , char *transb , integer *m , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c , integer *ldc );

/* dgemv.c */
int dgemv_ ( char *trans , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *x , integer *incx , doublereal *beta , doublereal *y , integer *incy );

/* dger.c */
int dger_ ( integer *m , integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *y , integer *incy , doublereal *a , integer *lda );

/* dnrm2.c */
doublereal dnrm2_ ( integer *n , doublereal *dx , integer *incx );

/* drot.c */
int drot_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy , doublereal *c , doublereal *s );

/* dscal.c */
int dscal_ ( integer *n , doublereal *da , doublereal *dx , integer *incx );

/* dswap.c */
int dswap_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dsymm.c */
int dsymm_ ( char *side , char *uplo , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c__ , integer *ldc );

/* dsymv.c */
int dsymv_ ( char *uplo , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *x , integer *incx , doublereal *beta , doublereal *y , integer *incy );

/* dsyr2.c */
int dsyr2_ ( char *uplo , integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *y , integer *incy , doublereal *a , integer *lda );

/* dsyr2k.c */
int dsyr2k_ ( char *uplo , char *trans , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c__ , integer *ldc );

/* dsyrk.c */
int dsyrk_ ( char *uplo , char *trans , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *beta , doublereal *c , integer *ldc );

/* dtrmm.c */
int dtrmm_ ( char *side , char *uplo , char *transa , char *diag , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb );

/* dtrmv.c */
int dtrmv_ ( char *uplo , char *trans , char *diag , integer *n , doublereal *a , integer *lda , doublereal *x , integer *incx );

/* dtrsm.c */
int dtrsm_ ( char *side , char *uplo , char *transa , char *diag , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb );

/* dtrsv.c */
int dtrsv_ ( char *uplo , char *trans , char *diag , integer *n , doublereal *a , integer *lda , doublereal *x , integer *incx );

/* idamax.c */
integer idamax_ ( integer *n , doublereal *dx , integer *incx );

#endif

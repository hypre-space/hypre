/* hypre_blas.h  --  Contains BLAS prototypes needed by Hypre */

#ifndef HYPRE_BLAS_H
#define HYPRE_BLAS_H

/* dasum.c */
doublereal dasum_ ( integer *n , doublereal *dx , integer *incx );

/* daxpy.c */
int daxpy_ ( integer *n , doublereal *da , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dcopy.c */
int dcopy_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* ddot.c */
doublereal ddot_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dgels_all.c */
double d_lg10 ( doublereal *x );
double d_sign ( doublereal *a , doublereal *b );
int dgelq2_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );
int dgelqf_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );
int dgels_ ( char *trans , integer *m , integer *n , integer *nrhs , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *work , integer *lwork , integer *info );
int dgeqr2_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );
int dgeqrf_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );
int dger_ ( integer *m , integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *y , integer *incy , doublereal *a , integer *lda );
int dlabad_ ( doublereal *small , doublereal *large );
doublereal dlamch_ ( char *cmach );
int dlamc1_ ( integer *beta , integer *t , logical *rnd , logical *ieee1 );
int dlamc2_ ( integer *beta , integer *t , logical *rnd , doublereal *eps , integer *emin , doublereal *rmin , integer *emax , doublereal *rmax );
doublereal dlamc3_ ( doublereal *a , doublereal *b );
int dlamc4_ ( integer *emin , doublereal *start , integer *base );
int dlamc5_ ( integer *beta , integer *p , integer *emin , logical *ieee , integer *emax , doublereal *rmax );
doublereal dlange_ ( char *norm , integer *m , integer *n , doublereal *a , integer *lda , doublereal *work );
doublereal dlapy2_ ( doublereal *x , doublereal *y );
int dlarf_ ( char *side , integer *m , integer *n , doublereal *v , integer *incv , doublereal *tau , doublereal *c , integer *ldc , doublereal *work );
int dlarfb_ ( char *side , char *trans , char *direct , char *storev , integer *m , integer *n , integer *k , doublereal *v , integer *ldv , doublereal *t , integer *ldt , doublereal *c , integer *ldc , doublereal *work , integer *ldwork );
int dlarfg_ ( integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *tau );
int dlarft_ ( char *direct , char *storev , integer *n , integer *k , doublereal *v , integer *ldv , doublereal *tau , doublereal *t , integer *ldt );
int dlascl_ ( char *type , integer *kl , integer *ku , doublereal *cfrom , doublereal *cto , integer *m , integer *n , doublereal *a , integer *lda , integer *info );
int dlaset_ ( char *uplo , integer *m , integer *n , doublereal *alpha , doublereal *beta , doublereal *a , integer *lda );
int dlassq_ ( integer *n , doublereal *x , integer *incx , doublereal *scale , doublereal *sumsq );
int dorm2r_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c , integer *ldc , doublereal *work , integer *info );
int dorml2_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c , integer *ldc , doublereal *work , integer *info );
int dormlq_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c , integer *ldc , doublereal *work , integer *lwork , integer *info );
int dormqr_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c , integer *ldc , doublereal *work , integer *lwork , integer *info );
int dtrmm_ ( char *side , char *uplo , char *transa , char *diag , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb );
int dtrmv_ ( char *uplo , char *trans , char *diag , integer *n , doublereal *a , integer *lda , doublereal *x , integer *incx );
double pow_di ( doublereal *ap , integer *bp );
int s_cat ( char *lp , char *rpp [], ftnlen rnp [], ftnlen *np , ftnlen ll );

/* dgemm.c */
int dgemm_ ( char *transa , char *transb , integer *m , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c , integer *ldc );

/* dgemv.c */
int dgemv_ ( char *trans , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *x , integer *incx , doublereal *beta , doublereal *y , integer *incy );

/* dnrm2.c */
doublereal dnrm2_ ( integer *n , doublereal *dx , integer *incx );

/* dpotrf.c */
int dpotrf_ ( char *uplo , integer *n , doublereal *a , integer *lda , integer *info );
int dpotf2_ ( char *uplo , integer *n , doublereal *a , integer *lda , integer *info );
integer ilaenv_ ( integer *ispec , char *name , char *opts , integer *n1 , integer *n2 , integer *n3 , integer *n4 , ftnlen name_len , ftnlen opts_len );
logical lsame_ ( char *ca , char *cb );
int xerbla_ ( char *srname , integer *info );

/* dpotrs.c */
int dpotrs_ ( char *uplo , integer *n , integer *nrhs , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );

/* drot.c */
int drot_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy , doublereal *c , doublereal *s );

/* dscal.c */
int dscal_ ( integer *n , doublereal *da , doublereal *dx , integer *incx );

/* dsygv.c */
int dswap_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );
int dsymm_ ( char *side , char *uplo , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c__ , integer *ldc );
int dsyr2k_ ( char *uplo , char *trans , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c__ , integer *ldc );
int dlasr_ ( char *side , char *pivot , char *direct , integer *m , integer *n , doublereal *c__ , doublereal *s , doublereal *a , integer *lda );
int dorg2r_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );
int dlatrd_ ( char *uplo , integer *n , integer *nb , doublereal *a , integer *lda , doublereal *e , doublereal *tau , doublereal *w , integer *ldw );
int dsygs2_ ( integer *itype , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );
int dlaev2_ ( doublereal *a , doublereal *b , doublereal *c__ , doublereal *rt1 , doublereal *rt2 , doublereal *cs1 , doublereal *sn1 );
int dlartg_ ( doublereal *f , doublereal *g , doublereal *cs , doublereal *sn , doublereal *r__ );
int dsytd2_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tau , integer *info );
doublereal dlanst_ ( char *norm , integer *n , doublereal *d__ , doublereal *e );
int dsteqr_ ( char *compz , integer *n , doublereal *d__ , doublereal *e , doublereal *z__ , integer *ldz , doublereal *work , integer *info );
int dsygst_ ( integer *itype , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );
int dsytrd_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tau , doublereal *work , integer *lwork , integer *info );
int dorgqr_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );
int dlae2_ ( doublereal *a , doublereal *b , doublereal *c__ , doublereal *rt1 , doublereal *rt2 );
doublereal dlansy_ ( char *norm , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *work );
int dlasrt_ ( char *id , integer *n , doublereal *d__ , integer *info );
int dorg2l_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );
int dorgql_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );
int dorgtr_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );
int dsterf_ ( integer *n , doublereal *d__ , doublereal *e , integer *info );
int dsyev_ ( char *jobz , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *w , doublereal *work , integer *lwork , integer *info );
int dsygv_ ( integer *itype , char *jobz , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *w , doublereal *work , integer *lwork , integer *info );

/* dsymv.c */
int dsymv_ ( char *uplo , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *x , integer *incx , doublereal *beta , doublereal *y , integer *incy );

/* dsyr2.c */
int dsyr2_ ( char *uplo , integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *y , integer *incy , doublereal *a , integer *lda );

/* dsyrk.c */
int dsyrk_ ( char *uplo , char *trans , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *beta , doublereal *c , integer *ldc );

/* dtrsm.c */
int dtrsm_ ( char *side , char *uplo , char *transa , char *diag , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb );

/* dtrsv.c */
int dtrsv_ ( char *uplo , char *trans , char *diag , integer *n , doublereal *a , integer *lda , doublereal *x , integer *incx );

/* idamax.c */
integer idamax_ ( integer *n , doublereal *dx , integer *incx );

/* s_cmp.c */
integer s_cmp ( char *a0 , char *b0 , ftnlen la , ftnlen lb );

/* s_copy.c */
VOID s_copy ( char *a , char *b , ftnlen la , ftnlen lb );

#endif

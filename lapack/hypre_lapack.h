/* hypre_lapack.h  --  Contains LAPACK prototypes needed by Hypre */

#ifndef HYPRE_LAPACK_H
#define HYPRE_LAPACK_H

/* dgels.c */
int dgels_ ( char *trans , integer *m , integer *n , integer *nrhs , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *work , integer *lwork , integer *info );

/* dsyev.c */
int dsyev_ ( char *jobz , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *w , doublereal *work , integer *lwork , integer *info );

/* dsygv.c */
int dsygv_ ( integer *itype , char *jobz , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *w , doublereal *work , integer *lwork , integer *info );

/* lapack_utils.c */
int xerbla_ ( char *srname , integer *info );
integer ilaenv_ ( integer *ispec , char *name__ , char *opts , integer *n1 , integer *n2 , integer *n3 , integer *n4 , ftnlen name_len , ftnlen opts_len );
logical lsame_ ( char *ca , char *cb );
doublereal dlamch_ ( char *cmach );
int dlamc1_ ( integer *beta , integer *t , logical *rnd , logical *ieee1 );
int dlamc2_ ( integer *beta , integer *t , logical *rnd , doublereal *eps , integer *emin , doublereal *rmin , integer *emax , doublereal *rmax );
doublereal dlamc3_ ( doublereal *a , doublereal *b );
int dlamc4_ ( integer *emin , doublereal *start , integer *base );
int dlamc5_ ( integer *beta , integer *p , integer *emin , logical *ieee , integer *emax , doublereal *rmax );
int dgelq2_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );
int dgelqf_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );
int dgeqr2_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );
int dgeqrf_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );
int dlabad_ ( doublereal *small , doublereal *large );
int dlae2_ ( doublereal *a , doublereal *b , doublereal *c__ , doublereal *rt1 , doublereal *rt2 );
int dlaev2_ ( doublereal *a , doublereal *b , doublereal *c__ , doublereal *rt1 , doublereal *rt2 , doublereal *cs1 , doublereal *sn1 );
doublereal dlange_ ( char *norm , integer *m , integer *n , doublereal *a , integer *lda , doublereal *work );
doublereal dlanst_ ( char *norm , integer *n , doublereal *d__ , doublereal *e );
doublereal dlansy_ ( char *norm , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *work );
doublereal dlapy2_ ( doublereal *x , doublereal *y );
int dlarf_ ( char *side , integer *m , integer *n , doublereal *v , integer *incv , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work );
int dlarfb_ ( char *side , char *trans , char *direct , char *storev , integer *m , integer *n , integer *k , doublereal *v , integer *ldv , doublereal *t , integer *ldt , doublereal *c__ , integer *ldc , doublereal *work , integer *ldwork );
int dlarfg_ ( integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *tau );
int dlarft_ ( char *direct , char *storev , integer *n , integer *k , doublereal *v , integer *ldv , doublereal *tau , doublereal *t , integer *ldt );
int dlartg_ ( doublereal *f , doublereal *g , doublereal *cs , doublereal *sn , doublereal *r__ );
int dlascl_ ( char *type__ , integer *kl , integer *ku , doublereal *cfrom , doublereal *cto , integer *m , integer *n , doublereal *a , integer *lda , integer *info );
int dlaset_ ( char *uplo , integer *m , integer *n , doublereal *alpha , doublereal *beta , doublereal *a , integer *lda );
int dlasr_ ( char *side , char *pivot , char *direct , integer *m , integer *n , doublereal *c__ , doublereal *s , doublereal *a , integer *lda );
int dlasrt_ ( char *id , integer *n , doublereal *d__ , integer *info );
int dlassq_ ( integer *n , doublereal *x , integer *incx , doublereal *scale , doublereal *sumsq );
int dlatrd_ ( char *uplo , integer *n , integer *nb , doublereal *a , integer *lda , doublereal *e , doublereal *tau , doublereal *w , integer *ldw );
int dorg2l_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );
int dorg2r_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );
int dorgql_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );
int dorgqr_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );
int dorgtr_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );
int dorm2r_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *info );
int dorml2_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *info );
int dormlq_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *lwork , integer *info );
int dormqr_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *lwork , integer *info );
int dpotf2_ ( char *uplo , integer *n , doublereal *a , integer *lda , integer *info );
int dpotrf_ ( char *uplo , integer *n , doublereal *a , integer *lda , integer *info );
int dpotrs_ ( char *uplo , integer *n , integer *nrhs , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );
int dsteqr_ ( char *compz , integer *n , doublereal *d__ , doublereal *e , doublereal *z__ , integer *ldz , doublereal *work , integer *info );
int dsterf_ ( integer *n , doublereal *d__ , doublereal *e , integer *info );
int dsygs2_ ( integer *itype , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );
int dsygst_ ( integer *itype , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );
int dsytd2_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tau , integer *info );
int dsytrd_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tau , doublereal *work , integer *lwork , integer *info );

#endif

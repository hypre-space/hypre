/* hypre_lapack.h  --  Contains LAPACK prototypes needed by Hypre */

#ifndef HYPRE_LAPACK_H
#define HYPRE_LAPACK_H
#include "f2c.h"

/* --------------------------------------------------------------------------
 *  Change all names to hypre_ to avoid link conflicts
 * --------------------------------------------------------------------------*/

#define dgebd2  hypre_dgebd2
#define dgebrd  hypre_dgebrd
#define dgelq2  hypre_dgelq2
#define dgelqf  hypre_dgelqf
#define dgels   hypre_dgels
#define dgeqr2  hypre_dgeqr2
#define dgeqrf  hypre_dgeqrf
#define dgesvd  hypre_dgesvd
#define dlabad  hypre_dlabad
#define dlabrd  hypre_dlabrd
#define dlae2   hypre_dlae2
#define dlaev2  hypre_dlaev2
#define dlamch  hypre_dlamch
#define dlamc1  hypre_dlamc1
#define dlamc2  hypre_dlamc2
#define dlamc3  hypre_dlamc3
#define dlamc4  hypre_dlamc4
#define dlamc5  hypre_dlamc5
#define dlange  hypre_dlange
#define dlanst  hypre_dlanst
#define dlansy  hypre_dlansy
#define dlapy2  hypre_dlapy2
#define dlarf   hypre_dlarf
#define dlarfb  hypre_dlarfb
#define dlarfg  hypre_dlarfg
#define dlarft  hypre_dlarft
#define dlartg  hypre_dlartg
#define dlascl  hypre_dlascl
#define dlaset  hypre_dlaset
#define dlasr   hypre_dlasr
#define dlasrt  hypre_dlasrt
#define dlassq  hypre_dlassq
#define dlatrd  hypre_dlatrd
#define dorg2l  hypre_dorg2l
#define dorg2r  hypre_dorg2r
#define dorgql  hypre_dorgql
#define dorgqr  hypre_dorgqr
#define dorgtr  hypre_dorgtr
#define dorm2r  hypre_dorm2r
#define dorml2  hypre_dorml2
#define dormlq  hypre_dormlq
#define dormqr  hypre_dormqr
#define dpotf2  hypre_dpotf2
#define dpotrf  hypre_dpotrf
#define dpotrs  hypre_dpotrs
#define dsteqr  hypre_dsteqr
#define dsterf  hypre_dsterf
#define dsyev   hypre_dsyev
#define dsygst  hypre_dsygst
#define dsygv   hypre_dsygv
#define dsytd2  hypre_dsytd2
#define dsytrd  hypre_dsytrd
#define ieeeck  hypre_ieeeck
#define ilaenv  hypre_ilaenv
#define d_lg10  hypre_d_lg10
#define d_sign  hypre_d_sign
#define pow_di  hypre_pow_di
#define pow_dd  hypre_pow_dd
#define s_cat   hypre_s_cat
#define lsame   hypre_lsame
#define xerbla  hypre_xerbla
#define dbdsqr  hypre_dbdsqr
#define dorgbr  hypre_dorgbr
#define dsygs2  hypre_dsygs2
#define dorglq  hypre_dorglq
#define dlacpy  hypre_dlacpy
#define dormbr  hypre_dormbr
#define dlasq1  hypre_dlasq1
#define dlas2   hypre_dlas2
#define dlasv2  hypre_dlasv2
#define dorgl2  hypre_dorgl2
#define dlasq2  hypre_dlasq2
#define dlasq3  hypre_dlasq3
#define dlasq4  hypre_dlasq4
#define dlasq5  hypre_dlasq5
#define dlasq6  hypre_dlasq6

/* --------------------------------------------------------------------------
 *           Prototypes
 * --------------------------------------------------------------------------*/

/* dgebd2.c */
int hypre_dgebd2_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tauq , doublereal *taup , doublereal *work , integer *info );

/* dgebrd.c */
int hypre_dgebrd_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tauq , doublereal *taup , doublereal *work , integer *lwork , integer *info );

/* dgelq2.c */
int hypre_dgelq2_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dgelqf.c */
int hypre_dgelqf_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dgels.c */
int hypre_dgels_ ( char *trans , integer *m , integer *n , integer *nrhs , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *work , integer *lwork , integer *info );

/* dgeqr2.c */
int hypre_dgeqr2_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dgeqrf.c */
int hypre_dgeqrf_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dgesvd.c */
int hypre_dgesvd_ ( char *jobu , char *jobvt , integer *m , integer *n , doublereal *a , integer *lda , doublereal *s , doublereal *u , integer *ldu , doublereal *vt , integer *ldvt , doublereal *work , integer *lwork , integer *info );

/* dlabad.c */
int hypre_dlabad_ ( doublereal *small , doublereal *large );

/* dlabrd.c */
int hypre_dlabrd_ ( integer *m , integer *n , integer *nb , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tauq , doublereal *taup , doublereal *x , integer *ldx , doublereal *y , integer *ldy );

/* dlae2.c */
int hypre_dlae2_ ( doublereal *a , doublereal *b , doublereal *c__ , doublereal *rt1 , doublereal *rt2 );

/* dlaev2.c */
int hypre_dlaev2_ ( doublereal *a , doublereal *b , doublereal *c__ , doublereal *rt1 , doublereal *rt2 , doublereal *cs1 , doublereal *sn1 );

/* dlamch.c */
doublereal hypre_dlamch_ ( char *cmach );
int hypre_dlamc1_ ( integer *beta , integer *t , logical *rnd , logical *ieee1 );
int hypre_dlamc2_ ( integer *beta , integer *t , logical *rnd , doublereal *eps , integer *emin , doublereal *rmin , integer *emax , doublereal *rmax );
doublereal hypre_dlamc3_ ( doublereal *a , doublereal *b );
int hypre_dlamc4_ ( integer *emin , doublereal *start , integer *base );
int hypre_dlamc5_ ( integer *beta , integer *p , integer *emin , logical *ieee , integer *emax , doublereal *rmax );

/* dlange.c */
doublereal hypre_dlange_ ( char *norm , integer *m , integer *n , doublereal *a , integer *lda , doublereal *work );

/* dlanst.c */
doublereal hypre_dlanst_ ( char *norm , integer *n , doublereal *d__ , doublereal *e );

/* dlansy.c */
doublereal hypre_dlansy_ ( char *norm , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *work );

/* dlapy2.c */
doublereal hypre_dlapy2_ ( doublereal *x , doublereal *y );

/* dlarf.c */
int hypre_dlarf_ ( char *side , integer *m , integer *n , doublereal *v , integer *incv , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work );

/* dlarfb.c */
int hypre_dlarfb_ ( char *side , char *trans , char *direct , char *storev , integer *m , integer *n , integer *k , doublereal *v , integer *ldv , doublereal *t , integer *ldt , doublereal *c__ , integer *ldc , doublereal *work , integer *ldwork );

/* dlarfg.c */
int hypre_dlarfg_ ( integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *tau );

/* dlarft.c */
int hypre_dlarft_ ( char *direct , char *storev , integer *n , integer *k , doublereal *v , integer *ldv , doublereal *tau , doublereal *t , integer *ldt );

/* dlartg.c */
int hypre_dlartg_ ( doublereal *f , doublereal *g , doublereal *cs , doublereal *sn , doublereal *r__ );

/* dlascl.c */
int hypre_dlascl_ ( char *type__ , integer *kl , integer *ku , doublereal *cfrom , doublereal *cto , integer *m , integer *n , doublereal *a , integer *lda , integer *info );

/* dlaset.c */
int hypre_dlaset_ ( char *uplo , integer *m , integer *n , doublereal *alpha , doublereal *beta , doublereal *a , integer *lda );

/* dlasr.c */
int hypre_dlasr_ ( char *side , char *pivot , char *direct , integer *m , integer *n , doublereal *c__ , doublereal *s , doublereal *a , integer *lda );

/* dlasrt.c */
int hypre_dlasrt_ ( char *id , integer *n , doublereal *d__ , integer *info );

/* dlassq.c */
int hypre_dlassq_ ( integer *n , doublereal *x , integer *incx , doublereal *scale , doublereal *sumsq );

/* dlatrd.c */
int hypre_dlatrd_ ( char *uplo , integer *n , integer *nb , doublereal *a , integer *lda , doublereal *e , doublereal *tau , doublereal *w , integer *ldw );

/* dorg2l.c */
int hypre_dorg2l_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dorg2r.c */
int hypre_dorg2r_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dorgql.c */
int hypre_dorgql_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dorgqr.c */
int hypre_dorgqr_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dorgtr.c */
int hypre_dorgtr_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dorm2r.c */
int hypre_dorm2r_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *info );

/* dorml2.c */
int hypre_dorml2_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *info );

/* dormlq.c */
int hypre_dormlq_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *lwork , integer *info );

/* dormqr.c */
int hypre_dormqr_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *lwork , integer *info );

/* dpotf2.c */
int hypre_dpotf2_ ( char *uplo , integer *n , doublereal *a , integer *lda , integer *info );

/* dpotrf.c */
int hypre_dpotrf_ ( char *uplo , integer *n , doublereal *a , integer *lda , integer *info );

/* dpotrs.c */
int hypre_dpotrs_ ( char *uplo , integer *n , integer *nrhs , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );

/* dsteqr.c */
int hypre_dsteqr_ ( char *compz , integer *n , doublereal *d__ , doublereal *e , doublereal *z__ , integer *ldz , doublereal *work , integer *info );

/* dsterf.c */
int hypre_dsterf_ ( integer *n , doublereal *d__ , doublereal *e , integer *info );

/* dsyev.c */
int hypre_dsyev_ ( char *jobz , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *w , doublereal *work , integer *lwork , integer *info );

/* dsygst.c */
int hypre_dsygst_ ( integer *itype , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );

/* dsygv.c */
int hypre_dsygv_ ( integer *itype , char *jobz , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *w , doublereal *work , integer *lwork , integer *info );

/* dsytd2.c */
int hypre_dsytd2_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tau , integer *info );

/* dsytrd.c */
int hypre_dsytrd_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* ieeeck.c */
integer hypre_ieeeck_ ( integer *ispec , real *zero , real *one );

/* ilaenv.c */
integer hypre_ilaenv_ ( integer *ispec , char *name__ , char *opts , integer *n1 , integer *n2 , integer *n3 , integer *n4 , ftnlen name_len , ftnlen opts_len );

/* lapack_utils.c */
double hypre_d_lg10 ( doublereal *x );
double hypre_d_sign ( doublereal *a , doublereal *b );
double hypre_pow_di ( doublereal *ap , integer *bp );
double hypre_pow_dd ( doublereal *ap , doublereal *bp );
int hypre_s_cat ( char *lp , char *rpp [], ftnlen rnp [], ftnlen *np , ftnlen ll );

/* lsame.c */
logical hypre_lsame_ ( char *ca , char *cb );

/* xerbla.c */
int hypre_xerbla_ ( char *srname , integer *info );

/* dbdsqr.c */
int hypre_dbdsqr_ ( char *uplo , integer *n , integer *ncvt , integer *nru , integer *ncc , doublereal *d__ , doublereal *e , doublereal *vt , integer *ldvt , doublereal *u , integer *ldu , doublereal *c__ , integer *ldc , doublereal *work , integer *info );

/* dorgbr.c */
int hypre_dorgbr_ ( char *vect , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dsygs2.c */
int hypre_dsygs2_ ( integer *itype , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );

/* dorglq.c */
int hypre_dorglq_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dlacpy.c */
int hypre_dlacpy_ ( char *uplo , integer *m , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb );

/* dormbr.c */
int hypre_dormbr_ ( char *vect , char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *lwork , integer *info );

/* dlasq1.c */
int hypre_dlasq1_ ( integer *n , doublereal *d__ , doublereal *e , doublereal *work , integer *info );

/* dlas2.c */
int hypre_dlas2_ ( doublereal *f , doublereal *g , doublereal *h__ , doublereal *ssmin , doublereal *ssmax );

/* dlasv2.c */
int hypre_dlasv2_ ( doublereal *f , doublereal *g , doublereal *h__ , doublereal *ssmin , doublereal *ssmax , doublereal *snr , doublereal *csr , doublereal *snl , doublereal *csl );

/* dorgl2.c */
int hypre_dorgl2_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dlasq2.c */
int hypre_dlasq2_ ( integer *n , doublereal *z__ , integer *info );

/* dlasq3.c */
int hypre_dlasq3_ ( integer *i0 , integer *n0 , doublereal *z__ , integer *pp , doublereal *dmin__ , doublereal *sigma , doublereal *desig , doublereal *qmax , integer *nfail , integer *iter , integer *ndiv , logical *ieee );

/* dlasq4.c */
int hypre_dlasq4_ ( integer *i0 , integer *n0 , doublereal *z__ , integer *pp , integer *n0in , doublereal *dmin__ , doublereal *dmin1 , doublereal *dmin2 , doublereal *dn , doublereal *dn1 , doublereal *dn2 , doublereal *tau , integer *ttype );

/* dlasq5.c */
int hypre_dlasq5_ ( integer *i0 , integer *n0 , doublereal *z__ , integer *pp , doublereal *tau , doublereal *dmin__ , doublereal *dmin1 , doublereal *dmin2 , doublereal *dn , doublereal *dnm1 , doublereal *dnm2 , logical *ieee );

/* dlasq6.c */
int hypre_dlasq6_ ( integer *i0 , integer *n0 , doublereal *z__ , integer *pp , doublereal *dmin__ , doublereal *dmin1 , doublereal *dmin2 , doublereal *dn , doublereal *dnm1 , doublereal *dnm2 );

#endif

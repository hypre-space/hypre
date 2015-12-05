/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 1.14 $
 ***********************************************************************EHEADER*/



/* hypre_lapack.h  --  Contains LAPACK prototypes needed by Hypre */

#ifndef HYPRE_LAPACK_H
#define HYPRE_LAPACK_H
#include "f2c.h"

/* --------------------------------------------------------------------------
 *  Change all names to hypre_ to avoid link conflicts
 * --------------------------------------------------------------------------*/

#define dgebd2_  hypre_dgebd2
#define dgebrd_  hypre_dgebrd
#define dgelq2_  hypre_dgelq2
#define dgelqf_  hypre_dgelqf
#define dgels_   hypre_dgels
#define dgeqr2_  hypre_dgeqr2
#define dgeqrf_  hypre_dgeqrf
#define dgesvd_  hypre_dgesvd
#define dgetf2_  hypre_dgetf2
#define dgetrf_  hypre_dgetrf
#define dgetrs_  hypre_dgetrs
#define dlabad_  hypre_dlabad
#define dlabrd_  hypre_dlabrd
#define dlae2_   hypre_dlae2
#define dlaev2_  hypre_dlaev2
#define dlamch_  hypre_dlamch
#define dlamc1_  hypre_dlamc1
#define dlamc2_  hypre_dlamc2
#define dlamc3_  hypre_dlamc3
#define dlamc4_  hypre_dlamc4
#define dlamc5_  hypre_dlamc5
#define dlaswp_  hypre_dlaswp
#define dlange_  hypre_dlange
#define dlanst_  hypre_dlanst
#define dlansy_  hypre_dlansy
#define dlapy2_  hypre_dlapy2
#define dlarf_   hypre_dlarf
#define dlarfb_  hypre_dlarfb
#define dlarfg_  hypre_dlarfg
#define dlarft_  hypre_dlarft
#define dlartg_  hypre_dlartg
#define dlascl_  hypre_dlascl
#define dlaset_  hypre_dlaset
#define dlasr_   hypre_dlasr
#define dlasrt_  hypre_dlasrt
#define dlassq_  hypre_dlassq
#define dlatrd_  hypre_dlatrd
#define dorg2l_  hypre_dorg2l
#define dorg2r_  hypre_dorg2r
#define dorgql_  hypre_dorgql
#define dorgqr_  hypre_dorgqr
#define dorgtr_  hypre_dorgtr
#define dorm2r_  hypre_dorm2r
#define dorml2_  hypre_dorml2
#define dormlq_  hypre_dormlq
#define dormqr_  hypre_dormqr
#define dpotf2_  hypre_dpotf2
#define dpotrf_  hypre_dpotrf
#define dpotrs_  hypre_dpotrs
#define dsteqr_  hypre_dsteqr
#define dsterf_  hypre_dsterf
#define dsyev_   hypre_dsyev
#define dsygst_  hypre_dsygst
#define dsygv_   hypre_dsygv
#define dsytd2_  hypre_dsytd2
#define dsytrd_  hypre_dsytrd
#define ieeeck_  hypre_ieeeck
#define ilaenv_  hypre_ilaenv
#define d_lg10_  hypre_d_lg10
#define d_sign_  hypre_d_sign
#define pow_di_  hypre_pow_di
#define pow_dd_  hypre_pow_dd
#define s_cat_   hypre_s_cat
#define lsame_   hypre_lsame
#define xerbla_  hypre_xerbla
#define dbdsqr_  hypre_dbdsqr
#define dorgbr_  hypre_dorgbr
#define dsygs2_  hypre_dsygs2
#define dorglq_  hypre_dorglq
#define dlacpy_  hypre_dlacpy
#define dormbr_  hypre_dormbr
#define dlasq1_  hypre_dlasq1
#define dlas2_   hypre_dlas2
#define dlasv2_  hypre_dlasv2
#define dorgl2_  hypre_dorgl2
#define dlasq2_  hypre_dlasq2
#define dlasq3_  hypre_dlasq3
#define dlasq4_  hypre_dlasq4
#define dlasq5_  hypre_dlasq5
#define dlasq6_  hypre_dlasq6

/* --------------------------------------------------------------------------
 *           Prototypes
 * --------------------------------------------------------------------------*/

/* dgebd2.c */
int dgebd2_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tauq , doublereal *taup , doublereal *work , integer *info );

/* dgebrd.c */
int dgebrd_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tauq , doublereal *taup , doublereal *work , integer *lwork , integer *info );

/* dgelq2.c */
int dgelq2_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dgelqf.c */
int dgelqf_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dgels.c */
int dgels_ ( char *trans , integer *m , integer *n , integer *nrhs , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *work , integer *lwork , integer *info );

/* dgeqr2.c */
int dgeqr2_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dgeqrf.c */
int dgeqrf_ ( integer *m , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dgesvd.c */
int dgesvd_ ( char *jobu , char *jobvt , integer *m , integer *n , doublereal *a , integer *lda , doublereal *s , doublereal *u , integer *ldu , doublereal *vt , integer *ldvt , doublereal *work , integer *lwork , integer *info );


/* dgetf2.c */
int dgetf2_( integer *m , integer *n , doublereal *a , integer *lda , integer *ipiv , integer *info );

/* dgetrf.c */
int dgetrf_( integer *m , integer *n , doublereal *a , integer *lda , integer *ipiv , integer *info );

/* dgetrs.c */
int dgetrs_( char *trans , integer *n , integer *nrhs , doublereal *a , integer *lda , integer *ipiv , doublereal *b , integer *ldb , integer *info );

/* dlabad.c */
int dlabad_ ( doublereal *small , doublereal *large );

/* dlabrd.c */
int dlabrd_ ( integer *m , integer *n , integer *nb , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tauq , doublereal *taup , doublereal *x , integer *ldx , doublereal *y , integer *ldy );

/* dlae2.c */
int dlae2_ ( doublereal *a , doublereal *b , doublereal *c__ , doublereal *rt1 , doublereal *rt2 );

/* dlaev2.c */
int dlaev2_ ( doublereal *a , doublereal *b , doublereal *c__ , doublereal *rt1 , doublereal *rt2 , doublereal *cs1 , doublereal *sn1 );

/* dlamch.c */
doublereal dlamch_ ( char *cmach );
int dlamc1_ ( integer *beta , integer *t , logical *rnd , logical *ieee1 );
int dlamc2_ ( integer *beta , integer *t , logical *rnd , doublereal *eps , integer *emin , doublereal *rmin , integer *emax , doublereal *rmax );
doublereal dlamc3_ ( doublereal *a , doublereal *b );
int dlamc4_ ( integer *emin , doublereal *start , integer *base );
int dlamc5_ ( integer *beta , integer *p , integer *emin , logical *ieee , integer *emax , doublereal *rmax );


/* dlaswp.c */
int dlaswp_( integer *n , doublereal *a , integer *lda , integer *k1 , integer *k2 , integer *ipiv , integer *incx );

/* dlange.c */
doublereal dlange_ ( char *norm , integer *m , integer *n , doublereal *a , integer *lda , doublereal *work );

/* dlanst.c */
doublereal dlanst_ ( char *norm , integer *n , doublereal *d__ , doublereal *e );

/* dlansy.c */
doublereal dlansy_ ( char *norm , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *work );

/* dlapy2.c */
doublereal dlapy2_ ( doublereal *x , doublereal *y );

/* dlarf.c */
int dlarf_ ( char *side , integer *m , integer *n , doublereal *v , integer *incv , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work );

/* dlarfb.c */
int dlarfb_ ( char *side , char *trans , char *direct , char *storev , integer *m , integer *n , integer *k , doublereal *v , integer *ldv , doublereal *t , integer *ldt , doublereal *c__ , integer *ldc , doublereal *work , integer *ldwork );

/* dlarfg.c */
int dlarfg_ ( integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *tau );

/* dlarft.c */
int dlarft_ ( char *direct , char *storev , integer *n , integer *k , doublereal *v , integer *ldv , doublereal *tau , doublereal *t , integer *ldt );

/* dlartg.c */
int dlartg_ ( doublereal *f , doublereal *g , doublereal *cs , doublereal *sn , doublereal *r__ );

/* dlascl.c */
int dlascl_ ( char *type__ , integer *kl , integer *ku , doublereal *cfrom , doublereal *cto , integer *m , integer *n , doublereal *a , integer *lda , integer *info );

/* dlaset.c */
int dlaset_ ( char *uplo , integer *m , integer *n , doublereal *alpha , doublereal *beta , doublereal *a , integer *lda );

/* dlasr.c */
int dlasr_ ( char *side , char *pivot , char *direct , integer *m , integer *n , doublereal *c__ , doublereal *s , doublereal *a , integer *lda );

/* dlasrt.c */
int dlasrt_ ( char *id , integer *n , doublereal *d__ , integer *info );

/* dlassq.c */
int dlassq_ ( integer *n , doublereal *x , integer *incx , doublereal *scale , doublereal *sumsq );

/* dlatrd.c */
int dlatrd_ ( char *uplo , integer *n , integer *nb , doublereal *a , integer *lda , doublereal *e , doublereal *tau , doublereal *w , integer *ldw );

/* dorg2l.c */
int dorg2l_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dorg2r.c */
int dorg2r_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dorgql.c */
int dorgql_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dorgqr.c */
int dorgqr_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dorgtr.c */
int dorgtr_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dorm2r.c */
int dorm2r_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *info );

/* dorml2.c */
int dorml2_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *info );

/* dormlq.c */
int dormlq_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *lwork , integer *info );

/* dormqr.c */
int dormqr_ ( char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *lwork , integer *info );

/* dpotf2.c */
int dpotf2_ ( char *uplo , integer *n , doublereal *a , integer *lda , integer *info );

/* dpotrf.c */
int dpotrf_ ( char *uplo , integer *n , doublereal *a , integer *lda , integer *info );

/* dpotrs.c */
int dpotrs_ ( char *uplo , integer *n , integer *nrhs , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );

/* dsteqr.c */
int dsteqr_ ( char *compz , integer *n , doublereal *d__ , doublereal *e , doublereal *z__ , integer *ldz , doublereal *work , integer *info );

/* dsterf.c */
int dsterf_ ( integer *n , doublereal *d__ , doublereal *e , integer *info );

/* dsyev.c */
int dsyev_ ( char *jobz , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *w , doublereal *work , integer *lwork , integer *info );

/* dsygst.c */
int dsygst_ ( integer *itype , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );

/* dsygv.c */
int dsygv_ ( integer *itype , char *jobz , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *w , doublereal *work , integer *lwork , integer *info );

/* dsytd2.c */
int dsytd2_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tau , integer *info );

/* dsytrd.c */
int dsytrd_ ( char *uplo , integer *n , doublereal *a , integer *lda , doublereal *d__ , doublereal *e , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* ieeeck.c */
integer ieeeck_ ( integer *ispec , real *zero , real *one );

/* ilaenv.c */
integer ilaenv_ ( integer *ispec , char *name__ , char *opts , integer *n1 , integer *n2 , integer *n3 , integer *n4 , ftnlen name_len , ftnlen opts_len );

/* lapack_utils.c */
double d_lg10 ( doublereal *x );
double d_sign ( doublereal *a , doublereal *b );
double pow_di ( doublereal *ap , integer *bp );
double pow_dd ( doublereal *ap , doublereal *bp );
int s_cat ( char *lp , char *rpp [], ftnlen rnp [], ftnlen *np , ftnlen ll );

/* lsame.c */
logical lsame_ ( char *ca , char *cb );

/* xerbla.c */
int xerbla_ ( char *srname , integer *info );

/* dbdsqr.c */
int dbdsqr_ ( char *uplo , integer *n , integer *ncvt , integer *nru , integer *ncc , doublereal *d__ , doublereal *e , doublereal *vt , integer *ldvt , doublereal *u , integer *ldu , doublereal *c__ , integer *ldc , doublereal *work , integer *info );

/* dorgbr.c */
int dorgbr_ ( char *vect , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dsygs2.c */
int dsygs2_ ( integer *itype , char *uplo , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb , integer *info );

/* dorglq.c */
int dorglq_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *lwork , integer *info );

/* dlacpy.c */
int dlacpy_ ( char *uplo , integer *m , integer *n , doublereal *a , integer *lda , doublereal *b , integer *ldb );

/* dormbr.c */
int dormbr_ ( char *vect , char *side , char *trans , integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *c__ , integer *ldc , doublereal *work , integer *lwork , integer *info );

/* dlasq1.c */
int dlasq1_ ( integer *n , doublereal *d__ , doublereal *e , doublereal *work , integer *info );

/* dlas2.c */
int dlas2_ ( doublereal *f , doublereal *g , doublereal *h__ , doublereal *ssmin , doublereal *ssmax );

/* dlasv2.c */
int dlasv2_ ( doublereal *f , doublereal *g , doublereal *h__ , doublereal *ssmin , doublereal *ssmax , doublereal *snr , doublereal *csr , doublereal *snl , doublereal *csl );

/* dorgl2.c */
int dorgl2_ ( integer *m , integer *n , integer *k , doublereal *a , integer *lda , doublereal *tau , doublereal *work , integer *info );

/* dlasq2.c */
int dlasq2_ ( integer *n , doublereal *z__ , integer *info );

/* dlasq3.c */
int dlasq3_ ( integer *i0 , integer *n0 , doublereal *z__ , integer *pp , doublereal *dmin__ , doublereal *sigma , doublereal *desig , doublereal *qmax , integer *nfail , integer *iter , integer *ndiv , logical *ieee );

/* dlasq4.c */
int dlasq4_ ( integer *i0 , integer *n0 , doublereal *z__ , integer *pp , integer *n0in , doublereal *dmin__ , doublereal *dmin1 , doublereal *dmin2 , doublereal *dn , doublereal *dn1 , doublereal *dn2 , doublereal *tau , integer *ttype );

/* dlasq5.c */
int dlasq5_ ( integer *i0 , integer *n0 , doublereal *z__ , integer *pp , doublereal *tau , doublereal *dmin__ , doublereal *dmin1 , doublereal *dmin2 , doublereal *dn , doublereal *dnm1 , doublereal *dnm2 , logical *ieee );

/* dlasq6.c */
int dlasq6_ ( integer *i0 , integer *n0 , doublereal *z__ , integer *pp , doublereal *dmin__ , doublereal *dmin1 , doublereal *dmin2 , doublereal *dn , doublereal *dnm1 , doublereal *dnm2 );

#endif

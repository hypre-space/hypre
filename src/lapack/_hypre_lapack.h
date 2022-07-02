/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for HYPRE LAPACK
 *
 *****************************************************************************/

#ifndef HYPRE_LAPACK_H
#define HYPRE_LAPACK_H

#include "_hypre_utilities.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Change all 'hypre_' names based on using HYPRE or external library
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_USING_HYPRE_LAPACK

#define hypre_dbdsqr  hypre_F90_NAME_LAPACK(dbdsqr,DBDSQR)
#define hypre_dgebd2  hypre_F90_NAME_LAPACK(dgebd2,DGEBD2)
#define hypre_dgebrd  hypre_F90_NAME_LAPACK(dgebrd,DGEBRD)
#define hypre_dgelq2  hypre_F90_NAME_LAPACK(dgelq2,DGELQ2)
#define hypre_dgelqf  hypre_F90_NAME_LAPACK(dgelqf,DGELQF)
#define hypre_dgels   hypre_F90_NAME_LAPACK(dgels ,DGELS )
#define hypre_dgeqr2  hypre_F90_NAME_LAPACK(dgeqr2,DGEQR2)
#define hypre_dgeqrf  hypre_F90_NAME_LAPACK(dgeqrf,DGEQRF)
#define hypre_dgesvd  hypre_F90_NAME_LAPACK(dgesvd,DGESVD)
#define hypre_dgetf2  hypre_F90_NAME_LAPACK(dgetf2,DGETF2)
#define hypre_dgetrf  hypre_F90_NAME_LAPACK(dgetrf,DGETRF)
#define hypre_dgetri  hypre_F90_NAME_LAPACK(dgetri,DGETRI)
#define hypre_dgetrs  hypre_F90_NAME_LAPACK(dgetrs,DGETRS)
#define hypre_dlasq1  hypre_F90_NAME_LAPACK(dlasq1,DLASQ1)
#define hypre_dlasq2  hypre_F90_NAME_LAPACK(dlasq2,DLASQ2)
#define hypre_dlasrt  hypre_F90_NAME_LAPACK(dlasrt,DLASRT)
#define hypre_dorg2l  hypre_F90_NAME_LAPACK(dorg2l,DORG2L)
#define hypre_dorg2r  hypre_F90_NAME_LAPACK(dorg2r,DORG2R)
#define hypre_dorgbr  hypre_F90_NAME_LAPACK(dorgbr,DORGBR)
#define hypre_dorgl2  hypre_F90_NAME_LAPACK(dorgl2,DORGL2)
#define hypre_dorglq  hypre_F90_NAME_LAPACK(dorglq,DORGLQ)
#define hypre_dorgql  hypre_F90_NAME_LAPACK(dorgql,DORGQL)
#define hypre_dorgqr  hypre_F90_NAME_LAPACK(dorgqr,DORGQR)
#define hypre_dorgtr  hypre_F90_NAME_LAPACK(dorgtr,DORGTR)
#define hypre_dorm2r  hypre_F90_NAME_LAPACK(dorm2r,DORM2R)
#define hypre_dormbr  hypre_F90_NAME_LAPACK(dormbr,DORMBR)
#define hypre_dorml2  hypre_F90_NAME_LAPACK(dorml2,DORML2)
#define hypre_dormlq  hypre_F90_NAME_LAPACK(dormlq,DORMLQ)
#define hypre_dormqr  hypre_F90_NAME_LAPACK(dormqr,DORMQR)
#define hypre_dpotf2  hypre_F90_NAME_LAPACK(dpotf2,DPOTF2)
#define hypre_dpotrf  hypre_F90_NAME_LAPACK(dpotrf,DPOTRF)
#define hypre_dpotrs  hypre_F90_NAME_LAPACK(dpotrs,DPOTRS)
#define hypre_dsteqr  hypre_F90_NAME_LAPACK(dsteqr,DSTEQR)
#define hypre_dsterf  hypre_F90_NAME_LAPACK(dsterf,DSTERF)
#define hypre_dsyev   hypre_F90_NAME_LAPACK(dsyev ,DSYEV )
#define hypre_dsygs2  hypre_F90_NAME_LAPACK(dsygs2,DSYGS2)
#define hypre_dsygst  hypre_F90_NAME_LAPACK(dsygst,DSYGST)
#define hypre_dsygv   hypre_F90_NAME_LAPACK(dsygv ,DSYGV )
#define hypre_dsytd2  hypre_F90_NAME_LAPACK(dsytd2,DSYTD2)
#define hypre_dsytrd  hypre_F90_NAME_LAPACK(dsytrd,DSYTRD)
#define hypre_dtrti2  hypre_F90_NAME_LAPACK(dtrtri,DTRTI2)
#define hypre_dtrtri  hypre_F90_NAME_LAPACK(dtrtri,DTRTRI)

#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* dbdsqr.c */
HYPRE_Int hypre_dbdsqr (const char *uplo , HYPRE_Int *n , HYPRE_Int *ncvt , HYPRE_Int *nru , HYPRE_Int *ncc , HYPRE_Real *d__ , HYPRE_Real *e , HYPRE_Real *vt , HYPRE_Int *ldvt , HYPRE_Real *u , HYPRE_Int *ldu , HYPRE_Real *c__ , HYPRE_Int *ldc , HYPRE_Real *work , HYPRE_Int *info );

/* dgebd2.c */
HYPRE_Int hypre_dgebd2 ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *d__ , HYPRE_Real *e , HYPRE_Real *tauq , HYPRE_Real *taup , HYPRE_Real *work , HYPRE_Int *info );

/* dgebrd.c */
HYPRE_Int hypre_dgebrd ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *d__ , HYPRE_Real *e , HYPRE_Real *tauq , HYPRE_Real *taup , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dgelq2.c */
HYPRE_Int hypre_dgelq2 ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *work , HYPRE_Int *info );

/* dgelqf.c */
HYPRE_Int hypre_dgelqf ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dgels.c */
HYPRE_Int hypre_dgels ( char *trans , HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *nrhs , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *b , HYPRE_Int *ldb , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dgeqr2.c */
HYPRE_Int hypre_dgeqr2 ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *work , HYPRE_Int *info );

/* dgeqrf.c */
HYPRE_Int hypre_dgeqrf ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dgesvd.c */
HYPRE_Int hypre_dgesvd ( char *jobu , char *jobvt , HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *s , HYPRE_Real *u , HYPRE_Int *ldu , HYPRE_Real *vt , HYPRE_Int *ldvt , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dgetf2.c */
HYPRE_Int hypre_dgetf2 ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Int *ipiv , HYPRE_Int *info );

/* dgetrf.c */
HYPRE_Int hypre_dgetrf ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Int *ipiv , HYPRE_Int *info );

/* dgetri.c */
HYPRE_Int hypre_dgetri ( HYPRE_Int *n, HYPRE_Real *a, HYPRE_Int *lda, HYPRE_Int *ipiv, HYPRE_Real *work, HYPRE_Int *lwork, HYPRE_Int *info);

/* dgetrs.c */
HYPRE_Int hypre_dgetrs ( const char *trans , HYPRE_Int *n , HYPRE_Int *nrhs , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Int *ipiv , HYPRE_Real *b , HYPRE_Int *ldb , HYPRE_Int *info );

/* dlasq1.c */
HYPRE_Int hypre_dlasq1 ( HYPRE_Int *n , HYPRE_Real *d__ , HYPRE_Real *e , HYPRE_Real *work , HYPRE_Int *info );

/* dlasq2.c */
HYPRE_Int hypre_dlasq2 ( HYPRE_Int *n , HYPRE_Real *z__ , HYPRE_Int *info );

/* dlasrt.c */
HYPRE_Int hypre_dlasrt (const char *id , HYPRE_Int *n , HYPRE_Real *d__ , HYPRE_Int *info );

/* dorg2l.c */
HYPRE_Int hypre_dorg2l ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *work , HYPRE_Int *info );

/* dorg2r.c */
HYPRE_Int hypre_dorg2r ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *work , HYPRE_Int *info );

/* dorgbr.c */
HYPRE_Int hypre_dorgbr (const char *vect , HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dorgl2.c */
HYPRE_Int hypre_dorgl2 ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *work , HYPRE_Int *info );

/* dorglq.c */
HYPRE_Int hypre_dorglq ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dorgql.c */
HYPRE_Int hypre_dorgql ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dorgqr.c */
HYPRE_Int hypre_dorgqr ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dorgtr.c */
HYPRE_Int hypre_dorgtr (const char *uplo , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dorm2r.c */
HYPRE_Int hypre_dorm2r (const char *side ,const char *trans , HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *c__ , HYPRE_Int *ldc , HYPRE_Real *work , HYPRE_Int *info );

/* dormbr.c */
HYPRE_Int hypre_dormbr (const char *vect ,const char *side ,const char *trans , HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *c__ , HYPRE_Int *ldc , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dorml2.c */
HYPRE_Int hypre_dorml2 (const char *side ,const char *trans , HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *c__ , HYPRE_Int *ldc , HYPRE_Real *work , HYPRE_Int *info );

/* dormlq.c */
HYPRE_Int hypre_dormlq (const char *side ,const char *trans , HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *c__ , HYPRE_Int *ldc , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dormqr.c */
HYPRE_Int hypre_dormqr (const char *side ,const char *trans , HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *tau , HYPRE_Real *c__ , HYPRE_Int *ldc , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dpotf2.c */
HYPRE_Int hypre_dpotf2 (const char *uplo , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Int *info );

/* dpotrf.c */
HYPRE_Int hypre_dpotrf (const char *uplo , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Int *info );

/* dpotrs.c */
HYPRE_Int hypre_dpotrs ( char *uplo , HYPRE_Int *n , HYPRE_Int *nrhs , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *b , HYPRE_Int *ldb , HYPRE_Int *info );

/* dsteqr.c */
HYPRE_Int hypre_dsteqr (const char *compz , HYPRE_Int *n , HYPRE_Real *d__ , HYPRE_Real *e , HYPRE_Real *z__ , HYPRE_Int *ldz , HYPRE_Real *work , HYPRE_Int *info );

/* dsterf.c */
HYPRE_Int hypre_dsterf ( HYPRE_Int *n , HYPRE_Real *d__ , HYPRE_Real *e , HYPRE_Int *info );

/* dsyev.c */
HYPRE_Int hypre_dsyev (const char *jobz ,const char *uplo , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *w , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dsygs2.c */
HYPRE_Int hypre_dsygs2 ( HYPRE_Int *itype ,const char *uplo , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *b , HYPRE_Int *ldb , HYPRE_Int *info );

/* dsygst.c */
HYPRE_Int hypre_dsygst ( HYPRE_Int *itype ,const char *uplo , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *b , HYPRE_Int *ldb , HYPRE_Int *info );

/* dsygv.c */
HYPRE_Int hypre_dsygv ( HYPRE_Int *itype , char *jobz , char *uplo , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *b , HYPRE_Int *ldb , HYPRE_Real *w , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dsytd2.c */
HYPRE_Int hypre_dsytd2 (const char *uplo , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *d__ , HYPRE_Real *e , HYPRE_Real *tau , HYPRE_Int *info );

/* dsytrd.c */
HYPRE_Int hypre_dsytrd (const char *uplo , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *d__ , HYPRE_Real *e , HYPRE_Real *tau , HYPRE_Real *work , HYPRE_Int *lwork , HYPRE_Int *info );

/* dtrti2.c */
HYPRE_Int hypre_dtrti2 (const char *uplo, const char *diag, HYPRE_Int *n, HYPRE_Real *a, HYPRE_Int *lda, HYPRE_Int *info);

/* dtrtri.c */
HYPRE_Int hypre_dtrtri (const char *uplo, const char *diag, HYPRE_Int *n, HYPRE_Real *a, HYPRE_Int *lda, HYPRE_Int *info);


#ifdef __cplusplus
}
#endif

#endif

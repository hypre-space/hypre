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
#include "_hypre_fortran.h"

#ifdef HYPRE_MIXED_PRECISION
#include "_hypre_lapack_mup_def.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Change all 'hypre_' names based on using HYPRE or external library
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_USING_HYPRE_LAPACK

#if defined(HYPRE_SINGLE)
#define hypre_LAPACK_NAME(name, NAME) hypre_F90_NAME_LAPACK(s ## name, S ## NAME)
#else
#define hypre_LAPACK_NAME(name, NAME) hypre_F90_NAME_LAPACK(d ## name, D ## NAME)
#endif

#define hypre_dbdsqr  hypre_LAPACK_NAME(bdsqr, BDSQR)
#define hypre_dgebd2  hypre_LAPACK_NAME(gebd2, GEBD2)
#define hypre_dgebrd  hypre_LAPACK_NAME(gebrd, GEBRD)
#define hypre_dgelq2  hypre_LAPACK_NAME(gelq2, GELQ2)
#define hypre_dgelqf  hypre_LAPACK_NAME(gelqf, GELQF)
#define hypre_dgels   hypre_LAPACK_NAME(gels, GELS)
#define hypre_dgeqr2  hypre_LAPACK_NAME(geqr2, GEQR2)
#define hypre_dgeqrf  hypre_LAPACK_NAME(geqrf, GEQRF)
#define hypre_dgesvd  hypre_LAPACK_NAME(gesvd, GESVD)
#define hypre_dgetf2  hypre_LAPACK_NAME(getf2, GETF2)
#define hypre_dgetrf  hypre_LAPACK_NAME(getrf, GETRF)
#define hypre_dgetri  hypre_LAPACK_NAME(getri, GETRI)
#define hypre_dgetrs  hypre_LAPACK_NAME(getrs, GETRS)
#define hypre_dlasq1  hypre_LAPACK_NAME(lasq1, LASQ1)
#define hypre_dlasq2  hypre_LAPACK_NAME(lasq2, LASQ2)
#define hypre_dlasrt  hypre_LAPACK_NAME(lasrt, LASRT)
#define hypre_dorg2l  hypre_LAPACK_NAME(org2l, ORG2L)
#define hypre_dorg2r  hypre_LAPACK_NAME(org2r, ORG2R)
#define hypre_dorgbr  hypre_LAPACK_NAME(orgbr, ORGBR)
#define hypre_dorgl2  hypre_LAPACK_NAME(orgl2, ORGL2)
#define hypre_dorglq  hypre_LAPACK_NAME(orglq, ORGLQ)
#define hypre_dorgql  hypre_LAPACK_NAME(orgql, ORGQL)
#define hypre_dorgqr  hypre_LAPACK_NAME(orgqr, ORGQR)
#define hypre_dorgtr  hypre_LAPACK_NAME(orgtr, ORGTR)
#define hypre_dorm2r  hypre_LAPACK_NAME(orm2r, ORM2R)
#define hypre_dormbr  hypre_LAPACK_NAME(ormbr, ORMBR)
#define hypre_dorml2  hypre_LAPACK_NAME(orml2, ORML2)
#define hypre_dormlq  hypre_LAPACK_NAME(ormlq, ORMLQ)
#define hypre_dormqr  hypre_LAPACK_NAME(ormqr, ORMQR)
#define hypre_dpotf2  hypre_LAPACK_NAME(potf2, POTF2)
#define hypre_dpotrf  hypre_LAPACK_NAME(potrf, POTRF)
#define hypre_dpotrs  hypre_LAPACK_NAME(potrs, POTRS)
#define hypre_dsteqr  hypre_LAPACK_NAME(steqr, STEQR)
#define hypre_dsterf  hypre_LAPACK_NAME(sterf, STERF)
#define hypre_dsyev   hypre_LAPACK_NAME(syev, SYEV)
#define hypre_dsygs2  hypre_LAPACK_NAME(sygs2, SYGS2)
#define hypre_dsygst  hypre_LAPACK_NAME(sygst, SYGST)
#define hypre_dsygv   hypre_LAPACK_NAME(sygv, SYGV)
#define hypre_dsytd2  hypre_LAPACK_NAME(sytd2, SYTD2)
#define hypre_dsytrd  hypre_LAPACK_NAME(sytrd, SYTRD)
#define hypre_dtrti2  hypre_LAPACK_NAME(trti2, TRTI2)
#define hypre_dtrtri  hypre_LAPACK_NAME(trtri, TRTRI)
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

#ifdef HYPRE_MIXED_PRECISION
/* The following is for user compiles and the order is important.  The first
 * header ensures that we do not change prototype names in user files or in the
 * second header file.  The second header contains all the prototypes needed by
 * users for mixed precision. */
#ifndef hypre_MP_BUILD
#include "_hypre_lapack_mup_undef.h"
#include "_hypre_lapack_mup.h"
#endif
#endif

#endif

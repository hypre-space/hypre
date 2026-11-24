/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for HYPRE BLAS
 *
 *****************************************************************************/

#ifndef HYPRE_BLAS_H
#define HYPRE_BLAS_H

#include "_hypre_utilities.h"
#include "_hypre_fortran.h"

#include <HYPRE_config.h>

#ifdef HYPRE_MIXED_PRECISION
#include "_hypre_blas_mup_def.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Change all 'hypre_' names based on using HYPRE or external library
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_USING_HYPRE_BLAS

#if defined(HYPRE_SINGLE)
#define hypre_BLAS_NAME(name, NAME) hypre_F90_NAME_BLAS(s ## name, S ## NAME)
#define hypre_idamax  hypre_F90_NAME_BLAS(isamax, ISAMAX)
#else
#define hypre_BLAS_NAME(name, NAME) hypre_F90_NAME_BLAS(d ## name, D ## NAME)
#define hypre_idamax  hypre_F90_NAME_BLAS(idamax, IDAMAX)
#endif

#define hypre_dasum   hypre_BLAS_NAME(asum, ASUM)
#define hypre_daxpy   hypre_BLAS_NAME(axpy, AXPY)
#define hypre_dcopy   hypre_BLAS_NAME(copy, COPY)
#define hypre_ddot    hypre_BLAS_NAME(dot, DOT)
#define hypre_dgemm   hypre_BLAS_NAME(gemm, GEMM)
#define hypre_dgemv   hypre_BLAS_NAME(gemv, GEMV)
#define hypre_dger    hypre_BLAS_NAME(ger, GER)
#define hypre_dnrm2   hypre_BLAS_NAME(nrm2, NRM2)
#define hypre_drot    hypre_BLAS_NAME(rot, ROT)
#define hypre_dscal   hypre_BLAS_NAME(scal, SCAL)
#define hypre_dswap   hypre_BLAS_NAME(swap, SWAP)
#define hypre_dsymm   hypre_BLAS_NAME(symm, SYMM)
#define hypre_dsymv   hypre_BLAS_NAME(symv, SYMV)
#define hypre_dsyr2   hypre_BLAS_NAME(syr2, SYR2)
#define hypre_dsyr2k  hypre_BLAS_NAME(syr2k, SYR2K)
#define hypre_dsyrk   hypre_BLAS_NAME(syrk, SYRK)
#define hypre_dtrmm   hypre_BLAS_NAME(trmm, TRMM)
#define hypre_dtrmv   hypre_BLAS_NAME(trmv, TRMV)
#define hypre_dtrsm   hypre_BLAS_NAME(trsm, TRSM)
#define hypre_dtrsv   hypre_BLAS_NAME(trsv, TRSV)

#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* dasum.c */
HYPRE_Real hypre_dasum ( HYPRE_Int *n , HYPRE_Real *dx , HYPRE_Int *incx );

/* daxpy.c */
HYPRE_Int hypre_daxpy ( HYPRE_Int *n , HYPRE_Real *da , HYPRE_Real *dx , HYPRE_Int *incx , HYPRE_Real *dy , HYPRE_Int *incy );

/* dcopy.c */
HYPRE_Int hypre_dcopy ( HYPRE_Int *n , HYPRE_Real *dx , HYPRE_Int *incx , HYPRE_Real *dy , HYPRE_Int *incy );

/* ddot.c */
HYPRE_Real hypre_ddot ( HYPRE_Int *n , HYPRE_Real *dx , HYPRE_Int *incx , HYPRE_Real *dy , HYPRE_Int *incy );

/* dgemm.c */
HYPRE_Int hypre_dgemm ( const char *transa , const char *transb , HYPRE_Int *m , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *alpha , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *b , HYPRE_Int *ldb , HYPRE_Real *beta , HYPRE_Real *c , HYPRE_Int *ldc );

/* dgemv.c */
HYPRE_Int hypre_dgemv ( const char *trans , HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *alpha , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *x , HYPRE_Int *incx , HYPRE_Real *beta , HYPRE_Real *y , HYPRE_Int *incy );

/* dger.c */
HYPRE_Int hypre_dger ( HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *alpha , HYPRE_Real *x , HYPRE_Int *incx , HYPRE_Real *y , HYPRE_Int *incy , HYPRE_Real *a , HYPRE_Int *lda );

/* dnrm2.c */
HYPRE_Real hypre_dnrm2 ( HYPRE_Int *n , HYPRE_Real *dx , HYPRE_Int *incx );

/* drot.c */
HYPRE_Int hypre_drot ( HYPRE_Int *n , HYPRE_Real *dx , HYPRE_Int *incx , HYPRE_Real *dy , HYPRE_Int *incy , HYPRE_Real *c , HYPRE_Real *s );

/* dscal.c */
HYPRE_Int hypre_dscal ( HYPRE_Int *n , HYPRE_Real *da , HYPRE_Real *dx , HYPRE_Int *incx );

/* dswap.c */
HYPRE_Int hypre_dswap ( HYPRE_Int *n , HYPRE_Real *dx , HYPRE_Int *incx , HYPRE_Real *dy , HYPRE_Int *incy );

/* dsymm.c */
HYPRE_Int hypre_dsymm ( const char *side , const char *uplo , HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *alpha , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *b , HYPRE_Int *ldb , HYPRE_Real *beta , HYPRE_Real *c__ , HYPRE_Int *ldc );

/* dsymv.c */
HYPRE_Int hypre_dsymv ( const char *uplo , HYPRE_Int *n , HYPRE_Real *alpha , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *x , HYPRE_Int *incx , HYPRE_Real *beta , HYPRE_Real *y , HYPRE_Int *incy );

/* dsyr2.c */
HYPRE_Int hypre_dsyr2 ( const char *uplo , HYPRE_Int *n , HYPRE_Real *alpha , HYPRE_Real *x , HYPRE_Int *incx , HYPRE_Real *y , HYPRE_Int *incy , HYPRE_Real *a , HYPRE_Int *lda );

/* dsyr2k.c */
HYPRE_Int hypre_dsyr2k ( const char *uplo , const char *trans , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *alpha , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *b , HYPRE_Int *ldb , HYPRE_Real *beta , HYPRE_Real *c__ , HYPRE_Int *ldc );

/* dsyrk.c */
HYPRE_Int hypre_dsyrk ( const char *uplo , const char *trans , HYPRE_Int *n , HYPRE_Int *k , HYPRE_Real *alpha , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *beta , HYPRE_Real *c , HYPRE_Int *ldc );

/* dtrmm.c */
HYPRE_Int hypre_dtrmm ( const char *side , const char *uplo , const char *transa , const char *diag , HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *alpha , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *b , HYPRE_Int *ldb );

/* dtrmv.c */
HYPRE_Int hypre_dtrmv ( const char *uplo , const char *trans , const char *diag , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *x , HYPRE_Int *incx );

/* dtrsm.c */
HYPRE_Int hypre_dtrsm ( const char *side , const char *uplo , const char *transa , const char *diag , HYPRE_Int *m , HYPRE_Int *n , HYPRE_Real *alpha , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *b , HYPRE_Int *ldb );

/* dtrsv.c */
HYPRE_Int hypre_dtrsv ( const char *uplo , const char *trans , const char *diag , HYPRE_Int *n , HYPRE_Real *a , HYPRE_Int *lda , HYPRE_Real *x , HYPRE_Int *incx );

/* idamax.c */
HYPRE_Int hypre_idamax ( HYPRE_Int *n , HYPRE_Real *dx , HYPRE_Int *incx );

#ifdef __cplusplus
}
#endif

#ifdef HYPRE_MIXED_PRECISION
/* The following is for user compiles and the order is important.  The first
 * header ensures that we do not change prototype names in user files or in the
 * second header file.  The second header contains all the prototypes needed by
 * users for mixed precision. */
#ifndef hypre_MP_BUILD
#include "_hypre_blas_mup_undef.h"
#include "_hypre_blas_mup.h"
#endif
#endif

#endif

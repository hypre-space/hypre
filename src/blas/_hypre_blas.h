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
#include "fortran.h"

#include <HYPRE_config.h>

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Change all 'hypre_' names based on using HYPRE or external library
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_USING_HYPRE_BLAS

#if defined(HYPRE_SINGLE)
#define hypre_dasum   hypre_F90_NAME_BLAS(sasum ,SASUM )
#define hypre_daxpy   hypre_F90_NAME_BLAS(saxpy ,SAXPY )
#define hypre_dcopy   hypre_F90_NAME_BLAS(scopy ,SCOPY )
#define hypre_ddot    hypre_F90_NAME_BLAS(sdot  ,SDOT  )
#define hypre_dgemm   hypre_F90_NAME_BLAS(sgemm ,SGEMM )
#define hypre_dgemv   hypre_F90_NAME_BLAS(sgemv ,SGEMV )
#define hypre_dger    hypre_F90_NAME_BLAS(sger  ,SGER  )
#define hypre_dnrm2   hypre_F90_NAME_BLAS(snrm2 ,SNRM2 )
#define hypre_drot    hypre_F90_NAME_BLAS(srot  ,SROT  )
#define hypre_dscal   hypre_F90_NAME_BLAS(sscal ,SSCAL )
#define hypre_dswap   hypre_F90_NAME_BLAS(sswap ,SSWAP )
#define hypre_dsymm   hypre_F90_NAME_BLAS(ssymm ,SSYMM )
#define hypre_dsymv   hypre_F90_NAME_BLAS(ssymv ,SSYMV )
#define hypre_dsyr2   hypre_F90_NAME_BLAS(ssyr2 ,SSYR2 )
#define hypre_dsyr2k  hypre_F90_NAME_BLAS(ssyr2k,SSYR2K)
#define hypre_dsyrk   hypre_F90_NAME_BLAS(ssyrk ,SSYRK )
#define hypre_dtrmm   hypre_F90_NAME_BLAS(strmm ,STRMM )
#define hypre_dtrmv   hypre_F90_NAME_BLAS(strmv ,STRMV )
#define hypre_dtrsm   hypre_F90_NAME_BLAS(strsm ,STRSM )
#define hypre_dtrsv   hypre_F90_NAME_BLAS(strsv ,STRSV )
#define hypre_idamax  hypre_F90_NAME_BLAS(isamax,ISAMAX)
#else
#define hypre_dasum   hypre_F90_NAME_BLAS(dasum ,DASUM )
#define hypre_daxpy   hypre_F90_NAME_BLAS(daxpy ,DAXPY )
#define hypre_dcopy   hypre_F90_NAME_BLAS(dcopy ,DCOPY )
#define hypre_ddot    hypre_F90_NAME_BLAS(ddot  ,DDOT  )
#define hypre_dgemm   hypre_F90_NAME_BLAS(dgemm ,DGEMM )
#define hypre_dgemv   hypre_F90_NAME_BLAS(dgemv ,DGEMV )
#define hypre_dger    hypre_F90_NAME_BLAS(dger  ,DGER  )
#define hypre_dnrm2   hypre_F90_NAME_BLAS(dnrm2 ,DNRM2 )
#define hypre_drot    hypre_F90_NAME_BLAS(drot  ,DROT  )
#define hypre_dscal   hypre_F90_NAME_BLAS(dscal ,DSCAL )
#define hypre_dswap   hypre_F90_NAME_BLAS(dswap ,DSWAP )
#define hypre_dsymm   hypre_F90_NAME_BLAS(dsymm ,DSYMM )
#define hypre_dsymv   hypre_F90_NAME_BLAS(dsymv ,DSYMV )
#define hypre_dsyr2   hypre_F90_NAME_BLAS(dsyr2 ,DSYR2 )
#define hypre_dsyr2k  hypre_F90_NAME_BLAS(dsyr2k,DSYR2K)
#define hypre_dsyrk   hypre_F90_NAME_BLAS(dsyrk ,DSYRK )
#define hypre_dtrmm   hypre_F90_NAME_BLAS(dtrmm ,DTRMM )
#define hypre_dtrmv   hypre_F90_NAME_BLAS(dtrmv ,DTRMV )
#define hypre_dtrsm   hypre_F90_NAME_BLAS(dtrsm ,DTRSM )
#define hypre_dtrsv   hypre_F90_NAME_BLAS(dtrsv ,DTRSV )
#define hypre_idamax  hypre_F90_NAME_BLAS(idamax,IDAMAX)
#endif

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

#endif

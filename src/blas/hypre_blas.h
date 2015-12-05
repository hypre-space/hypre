/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/




/* hypre_blas.h  --  Contains BLAS prototypes needed by Hypre */

#ifndef HYPRE_BLAS_H
#define HYPRE_BLAS_H
#include "f2c.h"
#include "fortran.h"

/* --------------------------------------------------------------------------
 *   Change all names to hypre_ to avoid link conflicts
 * --------------------------------------------------------------------------*/

#define dasum_   hypre_F90_NAME_BLAS(dasum,DASUM)
#define daxpy_   hypre_F90_NAME_BLAS(daxpy,DAXPY)
#define dcopy_   hypre_F90_NAME_BLAS(dcopy,DCOPY)
#define ddot_    hypre_F90_NAME_BLAS(ddot,DDOT)
#define dgemm_   hypre_F90_NAME_BLAS(dgemm,DGEMM)
#define dgemv_   hypre_F90_NAME_BLAS(dgemv,DGEMV)
#define dger_    hypre_F90_NAME_BLAS(dger,DGER)
#define dnrm2_   hypre_F90_NAME_BLAS(dnrm2,DNRM2)
#define drot_    hypre_F90_NAME_BLAS(drot,DROT)
#define dscal_   hypre_F90_NAME_BLAS(dscal,DSCAL)
#define dswap_   hypre_F90_NAME_BLAS(dswap,DSWAP)
#define dsymm_   hypre_F90_NAME_BLAS(dsymm,DSYMM)
#define dsymv_   hypre_F90_NAME_BLAS(dsymv,DSYMV)
#define dsyr2_   hypre_F90_NAME_BLAS(dsyr2,DSYR2)
#define dsyr2k_  hypre_F90_NAME_BLAS(dsyr2k,DSYR2K)
#define dsyrk_   hypre_F90_NAME_BLAS(dsyrk,DSYRK)
#define dtrmm_   hypre_F90_NAME_BLAS(dtrmm,DTRMM)
#define dtrmv_   hypre_F90_NAME_BLAS(dtrmv,DTRMV)
#define dtrsm_   hypre_F90_NAME_BLAS(dtrsm,DTRSM)
#define dtrsv_   hypre_F90_NAME_BLAS(dtrsv,DTRSV)
#define idamax_  hypre_F90_NAME_BLAS(idamax,IDAMAX)
#define s_cmp    hypre_F90_NAME_BLAS(s_cmp,S_CMP)
#define s_copy   hypre_F90_NAME_BLAS(s_copy,S_COPY)

/* blas_utils.c */
logical lsame_ ( char *ca , char *cb );
HYPRE_Int xerbla_ ( char *srname , integer *info );
integer s_cmp ( char *a0 , char *b0 , ftnlen la , ftnlen lb );
VOID s_copy ( char *a , char *b , ftnlen la , ftnlen lb );

/* dasum.c */
doublereal dasum_ ( integer *n , doublereal *dx , integer *incx );

/* daxpy.c */
HYPRE_Int daxpy_ ( integer *n , doublereal *da , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dcopy.c */
HYPRE_Int dcopy_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* ddot.c */
doublereal ddot_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dgemm.c */
HYPRE_Int dgemm_ ( char *transa , char *transb , integer *m , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c , integer *ldc );

/* dgemv.c */
HYPRE_Int dgemv_ ( char *trans , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *x , integer *incx , doublereal *beta , doublereal *y , integer *incy );

/* dger.c */
HYPRE_Int dger_ ( integer *m , integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *y , integer *incy , doublereal *a , integer *lda );

/* dnrm2.c */
doublereal dnrm2_ ( integer *n , doublereal *dx , integer *incx );

/* drot.c */
HYPRE_Int drot_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy , doublereal *c , doublereal *s );

/* dscal.c */
HYPRE_Int dscal_ ( integer *n , doublereal *da , doublereal *dx , integer *incx );

/* dswap.c */
HYPRE_Int dswap_ ( integer *n , doublereal *dx , integer *incx , doublereal *dy , integer *incy );

/* dsymm.c */
HYPRE_Int dsymm_ ( char *side , char *uplo , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c__ , integer *ldc );

/* dsymv.c */
HYPRE_Int dsymv_ ( char *uplo , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *x , integer *incx , doublereal *beta , doublereal *y , integer *incy );

/* dsyr2.c */
HYPRE_Int dsyr2_ ( char *uplo , integer *n , doublereal *alpha , doublereal *x , integer *incx , doublereal *y , integer *incy , doublereal *a , integer *lda );

/* dsyr2k.c */
HYPRE_Int dsyr2k_ ( char *uplo , char *trans , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb , doublereal *beta , doublereal *c__ , integer *ldc );

/* dsyrk.c */
HYPRE_Int dsyrk_ ( char *uplo , char *trans , integer *n , integer *k , doublereal *alpha , doublereal *a , integer *lda , doublereal *beta , doublereal *c , integer *ldc );

/* dtrmm.c */
HYPRE_Int dtrmm_ ( char *side , char *uplo , char *transa , char *diag , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb );

/* dtrmv.c */
HYPRE_Int dtrmv_ ( char *uplo , char *trans , char *diag , integer *n , doublereal *a , integer *lda , doublereal *x , integer *incx );

/* dtrsm.c */
HYPRE_Int dtrsm_ ( char *side , char *uplo , char *transa , char *diag , integer *m , integer *n , doublereal *alpha , doublereal *a , integer *lda , doublereal *b , integer *ldb );

/* dtrsv.c */
HYPRE_Int dtrsv_ ( char *uplo , char *trans , char *diag , integer *n , doublereal *a , integer *lda , doublereal *x , integer *incx );

/* idamax.c */
integer idamax_ ( integer *n , doublereal *dx , integer *incx );

#endif

/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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
 * $Revision$
 ***********************************************************************EHEADER*/




/* hypre_blas.h  --  Contains BLAS prototypes needed by Hypre */

#ifndef HYPRE_BLAS_H
#define HYPRE_BLAS_H
#include "f2c.h"

/* --------------------------------------------------------------------------
 *   Change all names to hypre_ to avoid link conflicts
 * --------------------------------------------------------------------------*/

#define dasum_   hypre_dasum
#define daxpy_   hypre_daxpy
#define dcopy_   hypre_dcopy
#define ddot_    hypre_ddot
#define dgemm_   hypre_dgemm
#define dgemv_   hypre_dgemv
#define dger_    hypre_dger
#define dnrm2_   hypre_dnrm2
#define drot_    hypre_drot
#define dscal_   hypre_dscal
#define dswap_   hypre_dswap
#define dsymm_   hypre_dsymm
#define dsymv_   hypre_dsymv
#define dsyr2_   hypre_dsyr2
#define dsyr2k_  hypre_dsyr2k
#define dsyrk_   hypre_dsyrk
#define dtrmm_   hypre_dtrmm
#define dtrmv_   hypre_dtrmv
#define dtrsm_   hypre_dtrsm
#define dtrsv_   hypre_dtrsv
#define idamax_  hypre_idamax

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

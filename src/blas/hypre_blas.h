/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/***** DO NOT use this file outside of the BLAS directory *****/

/*--------------------------------------------------------------------------
 * This header renames the functions in BLAS to avoid conflicts
 *--------------------------------------------------------------------------*/

/* blas */
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

/* f2c library routines */
#define s_cmp    hypre_s_cmp
#define s_copy   hypre_s_copy
#define s_cat    hypre_s_cat
#define d_lg10   hypre_d_lg10
#define d_sign   hypre_d_sign
#define pow_dd   hypre_pow_dd
#define pow_di   hypre_pow_di

/* these auxiliary routines have a different definition in LAPACK */
#define lsame_   hypre_blas_lsame
#define xerbla_  hypre_blas_xerbla


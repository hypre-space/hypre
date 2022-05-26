/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/***** DO NOT use this file outside of the LAPACK directory *****/

/*--------------------------------------------------------------------------
 * This header renames the functions in LAPACK to avoid conflicts
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

/* lapack */
#define dbdsqr_  hypre_dbdsqr
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
#define dgetri_  hypre_dgetri
#define dgetrs_  hypre_dgetrs
#define dlasq1_  hypre_dlasq1
#define dlasq2_  hypre_dlasq2
#define dlasrt_  hypre_dlasrt
#define dorg2l_  hypre_dorg2l
#define dorg2r_  hypre_dorg2r
#define dorgbr_  hypre_dorgbr
#define dorgl2_  hypre_dorgl2
#define dorglq_  hypre_dorglq
#define dorgql_  hypre_dorgql
#define dorgqr_  hypre_dorgqr
#define dorgtr_  hypre_dorgtr
#define dorm2r_  hypre_dorm2r
#define dormbr_  hypre_dormbr
#define dorml2_  hypre_dorml2
#define dormlq_  hypre_dormlq
#define dormqr_  hypre_dormqr
#define dpotf2_  hypre_dpotf2
#define dpotrf_  hypre_dpotrf
#define dpotrs_  hypre_dpotrs
#define dsteqr_  hypre_dsteqr
#define dsterf_  hypre_dsterf
#define dsyev_   hypre_dsyev
#define dsygs2_  hypre_dsygs2
#define dsygst_  hypre_dsygst
#define dsygv_   hypre_dsygv
#define dsytd2_  hypre_dsytd2
#define dsytrd_  hypre_dsytrd
#define dtrti2_  hypre_dtrti2
#define dtrtri_  hypre_dtrtri

/* lapack auxiliary routines */
#define dlabad_  hypre_dlabad
#define dlabrd_  hypre_dlabrd
#define dlacpy_  hypre_dlacpy
#define dlae2_   hypre_dlae2
#define dlaev2_  hypre_dlaev2
#define dlamch_  hypre_dlamch
#define dlamc1_  hypre_dlamc1
#define dlamc2_  hypre_dlamc2
#define dlamc3_  hypre_dlamc3
#define dlamc4_  hypre_dlamc4
#define dlamc5_  hypre_dlamc5
#define dlange_  hypre_dlange
#define dlanst_  hypre_dlanst
#define dlansy_  hypre_dlansy
#define dlapy2_  hypre_dlapy2
#define dlarf_   hypre_dlarf
#define dlarfb_  hypre_dlarfb
#define dlarfg_  hypre_dlarfg
#define dlarft_  hypre_dlarft
#define dlartg_  hypre_dlartg
#define dlas2_   hypre_dlas2
#define dlascl_  hypre_dlascl
#define dlaset_  hypre_dlaset
#define dlasq3_  hypre_dlasq3
#define dlasq4_  hypre_dlasq4
#define dlasq5_  hypre_dlasq5
#define dlasq6_  hypre_dlasq6
#define dlasr_   hypre_dlasr
#define dlassq_  hypre_dlassq
#define dlasv2_  hypre_dlasv2
#define dlaswp_  hypre_dlaswp
#define dlatrd_  hypre_dlatrd
#define ieeeck_  hypre_ieeeck
#define ilaenv_  hypre_ilaenv

/* these auxiliary routines have a different definition in BLAS */
#define lsame_   hypre_lapack_lsame
#define xerbla_  hypre_lapack_xerbla

/* this is needed so that lapack can call external BLAS */
#include "_hypre_blas.h"

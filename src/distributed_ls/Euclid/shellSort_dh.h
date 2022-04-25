/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef SUPPORT_DH
#define SUPPORT_DH

/* #include "euclid_common.h" */

extern void shellSort_int(const HYPRE_Int n, HYPRE_Int *x);
extern void shellSort_float(HYPRE_Int n, HYPRE_Real *v);

/*
extern void shellSort_int_int(const HYPRE_Int n, HYPRE_Int *x, HYPRE_Int *y);
extern void shellSort_int_float(HYPRE_Int n, HYPRE_Int *x, HYPRE_Real *v);
extern void shellSort_int_int_float(HYPRE_Int n, HYPRE_Int *x, HYPRE_Int *y, HYPRE_Real *v);
*/

#endif

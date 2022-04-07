/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef THREADED_KRYLOV_H
#define THREADED_KRYLOV_H

/* #include "blas_dh.h" */

extern void bicgstab_euclid(Mat_dh A, Euclid_dh ctx, HYPRE_Real *x, HYPRE_Real *b, 
                                                              HYPRE_Int *itsOUT);

extern void cg_euclid(Mat_dh A, Euclid_dh ctx, HYPRE_Real *x, HYPRE_Real *b, 
                                                              HYPRE_Int *itsOUT);

#endif

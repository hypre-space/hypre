/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ConjGrad.h header file.
 *
 *****************************************************************************/

#ifndef _CONJGRAD_H
#define _CONJGRAD_H

void PCG_ParaSails(Matrix *mat, ParaSails *ps, HYPRE_Real *b, HYPRE_Real *x,
   HYPRE_Real tol, HYPRE_Int max_iter);
void FGMRES_ParaSails(Matrix *mat, ParaSails *ps, HYPRE_Real *b, HYPRE_Real *x,
   HYPRE_Int dim, HYPRE_Real tol, HYPRE_Int max_iter);

#endif /* _CONJGRAD_H */

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_AME_HEADER
#define hypre_AME_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary space Maxwell Eigensolver
 *--------------------------------------------------------------------------*/
typedef struct
{
   /* The AMS preconditioner */
   hypre_AMSData *precond;

   /* The edge element mass matrix */
   hypre_ParCSRMatrix *M;

   /* Discrete gradient matrix with eliminated boundary */
   hypre_ParCSRMatrix *G;
   /* The Laplacian matrix G^t M G */
   hypre_ParCSRMatrix *A_G;
   /* AMG preconditioner for A_G */
   HYPRE_Solver B1_G;
   /* PCG-AMG solver for A_G */
   HYPRE_Solver B2_G;

   /* Eigensystem for A x = lambda M x, G^t x = 0 */
   HYPRE_Int block_size;
   void *eigenvectors;
   HYPRE_Real *eigenvalues;

   /* Eigensolver (LOBPCG) options */
   HYPRE_Int pcg_maxit;
   HYPRE_Int maxit;
   HYPRE_Real atol;
   HYPRE_Real rtol;
   HYPRE_Int print_level;

   /* Matrix-vector interface interpreter */
   void *interpreter;

   /* Temporary vectors */
   hypre_ParVector *t1, *t2, *t3;

} hypre_AMEData;

#include "_hypre_lapack.h"

#endif

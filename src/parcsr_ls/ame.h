/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/





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
   int block_size;
   void *eigenvectors;
   double *eigenvalues;

   /* Eigensolver (LOBPCG) options */
   int maxit;
   double tol;
   int print_level;

   /* Matrix-vector interface interpreter */
   void *interpreter;

   /* Temporary vectors */
   hypre_ParVector *t1, *t2, *t3;

} hypre_AMEData;

#include "fortran.h"

int hypre_F90_NAME_LAPACK(dpotrf,DPOTRF)(char *, int *, double *, int *, int *);
int hypre_F90_NAME_LAPACK(dsygv,DSYGV)(int *, char *, char *, int *, double *, int *,
                                       double *, int *, double *, double *, int *, int *);

#endif

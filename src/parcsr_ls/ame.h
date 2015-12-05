/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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
 * $Revision: 2.2 $
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

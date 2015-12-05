/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




#include "multivector.h"

#ifndef LOCALLY_OPTIMAL_BLOCK_PRECONDITIONED_CONJUGATE_GRADIENTS
#define LOCALLY_OPTIMAL_BLOCK_PRECONDITIONED_CONJUGATE_GRADIENTS

#define PROBLEM_SIZE_TOO_SMALL			       	1
#define WRONG_BLOCK_SIZE			       	2
#define WRONG_CONSTRAINTS                               3
#define REQUESTED_ACCURACY_NOT_ACHIEVED			-1

typedef struct {

  double	absolute;
  double	relative;

} lobpcg_Tolerance;

typedef struct {

/* these pointers should point to 2 functions providing standard lapack  functionality */
   HYPRE_Int   (*dpotrf) (char *uplo, HYPRE_Int *n, double *a, HYPRE_Int *
        lda, HYPRE_Int *info);
   HYPRE_Int   (*dsygv) (HYPRE_Int *itype, char *jobz, char *uplo, HYPRE_Int *
        n, double *a, HYPRE_Int *lda, double *b, HYPRE_Int *ldb,
        double *w, double *work, HYPRE_Int *lwork, HYPRE_Int *info);

} lobpcg_BLASLAPACKFunctions;

#ifdef __cplusplus
extern "C" {
#endif

HYPRE_Int
lobpcg_solve( mv_MultiVectorPtr blockVectorX,
	      void* operatorAData,
	      void (*operatorA)( void*, void*, void* ),
	      void* operatorBData,
	      void (*operatorB)( void*, void*, void* ),
	      void* operatorTData,
	      void (*operatorT)( void*, void*, void* ),
	      mv_MultiVectorPtr blockVectorY,
              lobpcg_BLASLAPACKFunctions blap_fn,
	      lobpcg_Tolerance tolerance,
	      HYPRE_Int maxIterations,
	      HYPRE_Int verbosityLevel,
	      HYPRE_Int* iterationNumber,

/* eigenvalues; "lambda_values" should point to array  containing <blocksize> doubles where <blocksi
ze> is the width of multivector "blockVectorX" */
              double * lambda_values,

/* eigenvalues history; a pointer to the entries of the  <blocksize>-by-(<maxIterations>+1) matrix s
tored
in  fortran-style. (i.e. column-wise) The matrix may be  a submatrix of a larger matrix, see next
argument; If you don't need eigenvalues history, provide NULL in this entry */
              double * lambdaHistory_values,

/* global height of the matrix (stored in fotran-style)  specified by previous argument */
              HYPRE_Int lambdaHistory_gh,

/* residual norms; argument should point to array of <blocksize> doubles */
              double * residualNorms_values,

/* residual norms history; a pointer to the entries of the  <blocksize>-by-(<maxIterations>+1) matri
x
stored in  fortran-style. (i.e. column-wise) The matrix may be  a submatrix of a larger matrix, see
next
argument If you don't need residual norms history, provide NULL in this entry */
              double * residualNormsHistory_values ,

/* global height of the matrix (stored in fotran-style)  specified by previous argument */
              HYPRE_Int residualNormsHistory_gh

);

#ifdef __cplusplus
}
#endif

#endif /* LOCALLY_OPTIMAL_BLOCK_PRECONDITIONED_CONJUGATE_GRADIENTS */

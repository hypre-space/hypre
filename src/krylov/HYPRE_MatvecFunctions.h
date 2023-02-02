/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_MATVEC_FUNCTIONS
#define HYPRE_MATVEC_FUNCTIONS

typedef struct
{
   void*  (*MatvecCreate)     ( void *A, void *x );
   HYPRE_Int (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                void *x, HYPRE_Complex beta, void *y );
   HYPRE_Int (*MatvecDestroy) ( void *matvec_data );

   void*  (*MatMultiVecCreate)     ( void *A, void *x );
   HYPRE_Int (*MatMultiVec)        ( void *data, HYPRE_Complex alpha, void *A,
                                     void *x, HYPRE_Complex beta, void *y );
   HYPRE_Int (*MatMultiVecDestroy) ( void *data );

} HYPRE_MatvecFunctions;

#endif

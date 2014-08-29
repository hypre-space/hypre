/*BHEADER**********************************************************************
 * Copyright (c) 2014,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#ifndef hypre_STRUCT_STMATRIX_HEADER
#define hypre_STRUCT_STMATRIX_HEADER

#include <assert.h>

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int     id;      /* Stencil matrix id for this term */
   HYPRE_Int     entry;   /* Stencil entry number */
   hypre_Index   offset;  /* Offset from center index */

} hypre_StCoeffTerm;

typedef struct coeff_link
{
   HYPRE_Int          nterms;  /* Number of terms */
   hypre_StCoeffTerm *terms;   /* Array of terms */

   struct coeff_link *prev;
   struct coeff_link *next;

} hypre_StCoeff;

typedef struct
{
   HYPRE_Int       id;         /* Matrix ID */
   HYPRE_Int       size;       /* Number of stencil entries */

   hypre_Index     rmap;       /* Range map */
   hypre_Index     dmap;       /* Domain map */

   hypre_Index    *shapes;     /* Offsets describing the stencil's shape */
   hypre_StCoeff **coeffs;     /* Description of coefficients */

} hypre_StMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros:
 *--------------------------------------------------------------------------*/

/* TODO: Use accessor macros */
#define hypre_StCoeffTermElt(struct, elt) ((struct) -> elt)
#define hypre_StCoeffElt(struct, elt)     ((struct) -> elt)
#define hypre_StMatrixElt(struct, elt)    ((struct) -> elt)

/*--------------------------------------------------------------------------
 * Prototypes:
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StIndexCopy( hypre_Index index1,
                   hypre_Index index2,
                   HYPRE_Int   ndim );
HYPRE_Int
hypre_StIndexNegate( hypre_Index index,
                     HYPRE_Int   ndim );
HYPRE_Int
hypre_StIndexShift( hypre_Index index,
                    hypre_Index shift,
                    HYPRE_Int   ndim );
HYPRE_Int
hypre_StIndexPrint( hypre_Index index,
                    char        lchar,
                    char        rchar,
                    HYPRE_Int   ndim );
HYPRE_Int
hypre_StCoeffTermCopy( hypre_StCoeffTerm *term1,
                       hypre_StCoeffTerm *term2,
                       HYPRE_Int          ndim );
HYPRE_Int
hypre_StCoeffTermPrint( hypre_StCoeffTerm *term,
                        char              *matnames,
                        HYPRE_Int          ndim );
HYPRE_Int
hypre_StCoeffCreate( HYPRE_Int       nterms,
                     hypre_StCoeff **coeff_ptr );
HYPRE_Int
hypre_StCoeffClone( hypre_StCoeff  *coeff,
                    HYPRE_Int       ndim,
                    hypre_StCoeff **clone_ptr );
HYPRE_Int
hypre_StCoeffDestroy( hypre_StCoeff *coeff );
HYPRE_Int
hypre_StCoeffShift( hypre_StCoeff *coeff,
                    hypre_Index    shift,
                    HYPRE_Int      ndim );
HYPRE_Int
hypre_StCoeffPush( hypre_StCoeff **stack_ptr,
                   hypre_StCoeff  *coeff );
HYPRE_Int
hypre_StCoeffMult( hypre_StCoeff  *Acoeff,
                   hypre_StCoeff  *Bcoeff,
                   HYPRE_Int       ndim,
                   hypre_StCoeff **Ccoeff_ptr );
HYPRE_Int
hypre_StCoeffPrint( hypre_StCoeff *coeff,
                    char          *matnames,
                    HYPRE_Int      ndim );
HYPRE_Int
hypre_StMatrixCreate( HYPRE_Int        id,
                      HYPRE_Int        size,
                      HYPRE_Int        ndim,
                      hypre_StMatrix **matrix_ptr );
HYPRE_Int
hypre_StMatrixClone( hypre_StMatrix  *matrix,
                     HYPRE_Int        ndim,
                     hypre_StMatrix **mclone_ptr );
HYPRE_Int
hypre_StMatrixDestroy( hypre_StMatrix *matrix );
HYPRE_Int
hypre_StMatrixTranspose( hypre_StMatrix *matrix,
                         HYPRE_Int       ndim );
HYPRE_Int
hypre_StMatrixMatmat( hypre_StMatrix  *A,
                      hypre_StMatrix  *B,
                      HYPRE_Int        Cid,
                      HYPRE_Int        ndim,
                      hypre_StMatrix **C_ptr );
HYPRE_Int
hypre_StMatrixMatmult( HYPRE_Int        nmatrices,
                       hypre_StMatrix **matrices,
                       HYPRE_Int       *transposes,
                       HYPRE_Int        Cid,
                       HYPRE_Int        ndim,
                       hypre_StMatrix **C_ptr );
HYPRE_Int
hypre_StMatrixPrint( hypre_StMatrix *matrix,
                     char           *matnames,
                     HYPRE_Int       ndim );

#endif

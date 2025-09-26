/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_STRUCT_STMATRIX_HEADER
#define hypre_STRUCT_STMATRIX_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int     id;      /* Stencil matrix id for this term */
   HYPRE_Int     entry;   /* Stencil entry number */
   hypre_Index   shift;   /* Stencil shift from center */

} hypre_StTerm;

typedef struct coeff_link
{
   HYPRE_Int          nterms;  /* Number of terms */
   hypre_StTerm      *terms;   /* Array of terms */

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

#define hypre_StTermID(term)           ((term) -> id)
#define hypre_StTermEntry(term)        ((term) -> entry)
#define hypre_StTermShift(term)        ((term) -> shift)

#define hypre_StCoeffNTerms(coeff)     ((coeff) -> nterms)
#define hypre_StCoeffTerms(coeff)      ((coeff) -> terms)
#define hypre_StCoeffTerm(coeff, t)   &((coeff) -> terms[t])
#define hypre_StCoeffPrev(coeff)       ((coeff) -> prev)
#define hypre_StCoeffNext(coeff)       ((coeff) -> next)

#define hypre_StMatrixID(stmat)        ((stmat) -> id)
#define hypre_StMatrixSize(stmat)      ((stmat) -> size)
#define hypre_StMatrixRMap(stmat)      ((stmat) -> rmap)
#define hypre_StMatrixDMap(stmat)      ((stmat) -> dmap)
#define hypre_StMatrixShapes(stmat)    ((stmat) -> shapes)
#define hypre_StMatrixOffset(stmat, e) ((stmat) -> shapes[e])
#define hypre_StMatrixCoeffs(stmat)    ((stmat) -> coeffs)
#define hypre_StMatrixCoeff(stmat, e)  ((stmat) -> coeffs[e])

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
hypre_StTermCopy( hypre_StTerm *term1,
                  hypre_StTerm *term2,
                  HYPRE_Int     ndim );
HYPRE_Int
hypre_StTermPrint( hypre_StTerm *term,
                   char         *matnames,
                   HYPRE_Int     ndim );
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

HYPRE_Int
hypre_StMatrixNEntryCoeffs( hypre_StMatrix *matrix,
                            HYPRE_Int       entry );

#endif

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_parcsr interface mixed precision functions
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"

#ifdef HYPRE_MIXED_PRECISION

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_ParVectorCopy
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ParVectorCopy_mp( HYPRE_ParVector x,
                        HYPRE_ParVector y )
{
   return ( hypre_ParVectorCopy_mp( (hypre_ParVector *) x,
                                    (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_ParVectorAxpy
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ParVectorAxpy_mp( hypre_long_double alpha, HYPRE_ParVector x,
                        HYPRE_ParVector y )
{
   return ( hypre_ParVectorAxpy_mp( alpha, (hypre_ParVector *) x,
                                    (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_ParVectorConvert
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ParVectorConvert_mp( HYPRE_ParVector v,
                           HYPRE_Precision new_precision)
{
   return (hypre_ParVectorConvert_mp( (hypre_ParVector *) v,
                                      new_precision ));
}

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_ParCSRMatrixConvert
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ParCSRMatrixConvert_mp( HYPRE_ParCSRMatrix A,
                              HYPRE_Precision new_precision)
{
   return (hypre_ParCSRMatrixConvert_mp( (hypre_ParCSRMatrix *) A,
                                         new_precision ));
}

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_ParCSRMatrixClone
 *--------------------------------------------------------------------------*/
HYPRE_ParCSRMatrix
HYPRE_ParCSRMatrixClone_mp(HYPRE_ParCSRMatrix   A, HYPRE_Precision new_precision)
{
   return ((HYPRE_ParCSRMatrix)(hypre_ParCSRMatrixClone_mp((hypre_ParCSRMatrix   *)A, new_precision)));
}

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_ParCSRMatrixCopy
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ParCSRMatrixCopy_mp( HYPRE_ParCSRMatrix A, HYPRE_ParCSRMatrix B )
{
   return (hypre_ParCSRMatrixCopy_mp( (hypre_ParCSRMatrix *)A, (hypre_ParCSRMatrix *)B ));
}

#endif

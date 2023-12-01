/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_MultiblockMatrix interface
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_MultiblockMatrix
HYPRE_MultiblockMatrixCreate( void )
{
   return ( (HYPRE_MultiblockMatrix)
            hypre_MultiblockMatrixCreate(  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MultiblockMatrixDestroy( HYPRE_MultiblockMatrix matrix )
{
   return ( hypre_MultiblockMatrixDestroy( (hypre_MultiblockMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MultiblockMatrixLimitedDestroy( HYPRE_MultiblockMatrix matrix )
{
   return ( hypre_MultiblockMatrixLimitedDestroy( (hypre_MultiblockMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MultiblockMatrixInitialize( HYPRE_MultiblockMatrix matrix )
{
   return ( hypre_MultiblockMatrixInitialize( (hypre_MultiblockMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixAssemble
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MultiblockMatrixAssemble( HYPRE_MultiblockMatrix matrix )
{
   return ( hypre_MultiblockMatrixAssemble( (hypre_MultiblockMatrix *) matrix ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixPrint
 *--------------------------------------------------------------------------*/

void
HYPRE_MultiblockMatrixPrint( HYPRE_MultiblockMatrix matrix )
{
   hypre_MultiblockMatrixPrint( (hypre_MultiblockMatrix *) matrix );
}

/****************************************************************************
 END OF ROUTINES THAT ARE ESSENTIALLY JUST CALLS THROUGH TO OTHER ROUTINES
 AND THAT ARE INDEPENDENT OF THE PARTICULAR MATRIX TYPE (except for names)
 ***************************************************************************/

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixSetNumSubmatrices
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MultiblockMatrixSetNumSubmatrices( HYPRE_MultiblockMatrix matrix, HYPRE_Int n )
{
   return ( hypre_MultiblockMatrixSetNumSubmatrices(
               (hypre_MultiblockMatrix *) matrix, n ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixSetSubmatrixType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MultiblockMatrixSetSubmatrixType( HYPRE_MultiblockMatrix matrix,
                                        HYPRE_Int j,
                                        HYPRE_Int type )
{
   return ( hypre_MultiblockMatrixSetSubmatrixType(
               (hypre_MultiblockMatrix *) matrix, j, type ) );
}

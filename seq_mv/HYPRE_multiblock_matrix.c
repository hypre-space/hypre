/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_MultiblockMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_MultiblockMatrix 
HYPRE_MultiblockMatrixCreate( )
{
   return ( (HYPRE_MultiblockMatrix)
            hypre_MultiblockMatrixCreate(  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_MultiblockMatrixDestroy( HYPRE_MultiblockMatrix matrix )
{
   return( hypre_MultiblockMatrixDestroy( (hypre_MultiblockMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_MultiblockMatrixLimitedDestroy( HYPRE_MultiblockMatrix matrix )
{
   return( hypre_MultiblockMatrixLimitedDestroy( (hypre_MultiblockMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixInitialize
 *--------------------------------------------------------------------------*/

int
HYPRE_MultiblockMatrixInitialize( HYPRE_MultiblockMatrix matrix )
{
   return ( hypre_MultiblockMatrixInitialize( (hypre_MultiblockMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_MultiblockMatrixAssemble( HYPRE_MultiblockMatrix matrix )
{
   return( hypre_MultiblockMatrixAssemble( (hypre_MultiblockMatrix *) matrix ) );
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

int 
HYPRE_MultiblockMatrixSetNumSubmatrices( HYPRE_MultiblockMatrix matrix, int n )
{
   return( hypre_MultiblockMatrixSetNumSubmatrices( 
             (hypre_MultiblockMatrix *) matrix, n ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MultiblockMatrixSetSubmatrixType
 *--------------------------------------------------------------------------*/

int 
HYPRE_MultiblockMatrixSetSubmatrixType( HYPRE_MultiblockMatrix matrix, 
                                      int j,
                                      int type )
{
   return( hypre_MultiblockMatrixSetSubmatrixType( 
             (hypre_MultiblockMatrix *) matrix, j, type ) );
}

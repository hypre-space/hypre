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
 * HYPRE_NewMultiblockMatrix
 *--------------------------------------------------------------------------*/

HYPRE_MultiblockMatrix 
HYPRE_NewMultiblockMatrix( )
{
   return ( (HYPRE_MultiblockMatrix)
            hypre_NewMultiblockMatrix(  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeMultiblockMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_FreeMultiblockMatrix( HYPRE_MultiblockMatrix matrix )
{
   return( hypre_FreeMultiblockMatrix( (hypre_MultiblockMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LimitedFreeMultiblockMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_LimitedFreeMultiblockMatrix( HYPRE_MultiblockMatrix matrix )
{
   return( hypre_LimitedFreeMultiblockMatrix( (hypre_MultiblockMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeMultiblockMatrix
 *--------------------------------------------------------------------------*/

int
HYPRE_InitializeMultiblockMatrix( HYPRE_MultiblockMatrix matrix )
{
   return ( hypre_InitializeMultiblockMatrix( (hypre_MultiblockMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_AssembleMultiblockMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_AssembleMultiblockMatrix( HYPRE_MultiblockMatrix matrix )
{
   return( hypre_AssembleMultiblockMatrix( (hypre_MultiblockMatrix *) matrix ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_PrintMultiblockMatrix
 *--------------------------------------------------------------------------*/

void 
HYPRE_PrintMultiblockMatrix( HYPRE_MultiblockMatrix matrix )
{
   hypre_PrintMultiblockMatrix( (hypre_MultiblockMatrix *) matrix );
}

/****************************************************************************
 END OF ROUTINES THAT ARE ESSENTIALLY JUST CALLS THROUGH TO OTHER ROUTINES
 AND THAT ARE INDEPENDENT OF THE PARTICULAR MATRIX TYPE (except for names)
 ***************************************************************************/

/*--------------------------------------------------------------------------
 * HYPRE_SetMultiblockMatrixNumSubmatrices
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetMultiblockMatrixNumSubmatrices( HYPRE_MultiblockMatrix matrix, int n )
{
   return( hypre_SetMultiblockMatrixNumSubmatrices( 
             (hypre_MultiblockMatrix *) matrix, n ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetMultiblockMatrixSubmatrixType
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetMultiblockMatrixSubmatrixType( HYPRE_MultiblockMatrix matrix, 
                                      int j,
                                      int type )
{
   return( hypre_SetMultiblockMatrixSubmatrixType( 
             (hypre_MultiblockMatrix *) matrix, j, type ) );
}

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
 * HYPRE_CSRMatrix interface
 *
 *****************************************************************************/

#include "general.h"
#include "HYPRE_seq_matrix.h"
#include "./seq_matrix_vector_internal.h"

/*--------------------------------------------------------------------------
 * HYPRE_NewCSRMatrix
 *--------------------------------------------------------------------------*/

HYPRE_CSRMatrix 
HYPRE_NewCSRMatrix( )
{
   return ( (HYPRE_CSRMatrix)
            hypre_NewCSRMatrix(  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeCSRMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_FreeCSRMatrix( HYPRE_CSRMatrix matrix )
{
   return( hypre_FreeCSRMatrix( (hypre_CSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LimitedFreeCSRMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_LimitedFreeCSRMatrix( HYPRE_CSRMatrix matrix )
{
   return( hypre_LimitedFreeCSRMatrix( (hypre_CSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeCSRMatrix
 *--------------------------------------------------------------------------*/

int
HYPRE_InitializeCSRMatrix( HYPRE_CSRMatrix matrix )
{
   return ( hypre_InitializeCSRMatrix( (hypre_CSRMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_AssembleCSRMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_AssembleCSRMatrix( HYPRE_CSRMatrix matrix )
{
   return( hypre_AssembleCSRMatrix( (hypre_CSRMatrix *) matrix ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_PrintCSRMatrix
 *--------------------------------------------------------------------------*/

void 
HYPRE_PrintCSRMatrix( HYPRE_CSRMatrix matrix )
{
   hypre_PrintCSRMatrix( (hypre_CSRMatrix *) matrix );
}

/****************************************************************************
 END OF ROUTINES THAT ARE ESSENTIALLY JUST CALLS THROUGH TO OTHER ROUTINES
 AND THAT ARE INDEPENDENT OF THE PARTICULAR MATRIX TYPE (except for names)
 ***************************************************************************/

/*--------------------------------------------------------------------------
 * HYPRE_SetCSRMatrixData
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetCSRMatrixData( HYPRE_CSRMatrix  matrix, double *data )
{
   return ( hypre_SetCSRMatrixData( (hypre_CSRMatrix *) matrix, data) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetCSRMatrixIA
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetCSRMatrixIA( HYPRE_CSRMatrix  matrix, int *ia )
{
   return ( hypre_SetCSRMatrixIA( (hypre_CSRMatrix *) matrix, ia) );
}


/*--------------------------------------------------------------------------
 * HYPRE_SetCSRMatrixJA
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetCSRMatrixJA( HYPRE_CSRMatrix  matrix, int *ja )
{
   return ( hypre_SetCSRMatrixJA( (hypre_CSRMatrix *) matrix, ja) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetCSRMatrixN
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetCSRMatrixN( HYPRE_CSRMatrix  matrix, int n )
{
   return ( hypre_SetCSRMatrixN( (hypre_CSRMatrix *) matrix, n) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetCSRMatrixData
 *--------------------------------------------------------------------------*/

double *
HYPRE_GetCSRMatrixData( HYPRE_CSRMatrix  matrix )
{
   return ( hypre_GetCSRMatrixData( (hypre_CSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetCSRMatrixIA
 *--------------------------------------------------------------------------*/

int *
HYPRE_GetCSRMatrixIA( HYPRE_CSRMatrix  matrix )
{
   return ( hypre_GetCSRMatrixIA( (hypre_CSRMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_GetCSRMatrixJA
 *--------------------------------------------------------------------------*/

int *
HYPRE_GetCSRMatrixJA( HYPRE_CSRMatrix  matrix )
{
   return ( hypre_GetCSRMatrixJA( (hypre_CSRMatrix *) matrix) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetCSRMatrixN
 *--------------------------------------------------------------------------*/

int 
HYPRE_GetCSRMatrixN( HYPRE_CSRMatrix  matrix )
{
   return ( hypre_GetCSRMatrixN( (hypre_CSRMatrix *) matrix) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetCSRMatrixNNZ
 *--------------------------------------------------------------------------*/

int 
HYPRE_GetCSRMatrixNNZ( HYPRE_CSRMatrix  matrix )
{
   return ( hypre_GetCSRMatrixNNZ( (hypre_CSRMatrix *) matrix) );
}

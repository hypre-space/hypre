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
 * Member functions for hypre_MultiblockMatrix class.
 *
 *****************************************************************************/

#include "general.h"
#include "HYPRE_seq_matrix.h"
#include "./multiblock_matrix.h"

/*--------------------------------------------------------------------------
 * hypre_NewMultiblockMatrix
 *--------------------------------------------------------------------------*/

hypre_MultiblockMatrix *
hypre_NewMultiblockMatrix( )
{
   hypre_MultiblockMatrix  *matrix;

   matrix = hypre_CTAlloc(hypre_MultiblockMatrix, 1);

   return ( matrix );
}

/*--------------------------------------------------------------------------
 * hypre_FreeMultiblockMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_FreeMultiblockMatrix( hypre_MultiblockMatrix *matrix )
{
   int  ierr=0, i;

   if (matrix)
   {
      for(i=0; i < hypre_MultiblockMatrixNumSubmatrices(matrix); i++)
         hypre_Tfree(hypre_MultiblockMatrixSubmatrix(matrix,i));
      hypre_Tfree(hypre_MultiblockMatrixSubmatrices(matrix));
      hypre_Tfree(hypre_MultiblockMatrixSubmatrixTypes(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_LimitedFreeMultiblockMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_LimitedFreeMultiblockMatrix( hypre_MultiblockMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      hypre_Tfree(hypre_MultiblockMatrixSubmatrices(matrix));
      hypre_Tfree(hypre_MultiblockMatrixSubmatrixTypes(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_InitializeMultiblockMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_InitializeMultiblockMatrix( hypre_MultiblockMatrix *matrix )
{
   int    ierr=0;

   if( hypre_MultiblockMatrixNumSubmatrices(matrix) <= 0 )
      return(-1);

   hypre_MultiblockMatrixSubmatrixTypes(matrix) = 
      hypre_CTAlloc( int, hypre_MultiblockMatrixNumSubmatrices(matrix) );

   hypre_MultiblockMatrixSubmatrices(matrix) = 
      hypre_CTAlloc( void *, hypre_MultiblockMatrixNumSubmatrices(matrix) );

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_AssembleMultiblockMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleMultiblockMatrix( hypre_MultiblockMatrix *matrix )
{
   int    ierr=0;

   return(ierr);
}


/*--------------------------------------------------------------------------
 * hypre_PrintMultiblockMatrix
 *--------------------------------------------------------------------------*/

void
hypre_PrintMultiblockMatrix(hypre_MultiblockMatrix *matrix  )
{
   printf("Stub for hypre_MultiblockMatrix\n");
}

/*--------------------------------------------------------------------------
 * hypre_SetMultiblockMatrixNumSubmatrices
 *--------------------------------------------------------------------------*/

int
hypre_SetMultiblockMatrixNumSubmatrices(hypre_MultiblockMatrix *matrix, int n  )
{
   int ierr = 0;

   hypre_MultiblockMatrixNumSubmatrices(matrix) = n;
   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_SetMultiblockMatrixSubmatrixType
 *--------------------------------------------------------------------------*/

int
hypre_SetMultiblockMatrixSubmatrixType(hypre_MultiblockMatrix *matrix, 
                                     int j,
                                     int type  )
{
   int ierr = 0;

   if ( (j<0) || 
         (j >= hypre_MultiblockMatrixNumSubmatrices(matrix)) )
      return(-1);

   hypre_MultiblockMatrixSubmatrixType(matrix,j) = type;

   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_SetMultiblockMatrixSubmatrix
 *--------------------------------------------------------------------------*/

int
hypre_SetMultiblockMatrixSubmatrix(hypre_MultiblockMatrix *matrix, 
                                     int j,
                                     void *submatrix  )
{
   int ierr = 0;

   if ( (j<0) || 
         (j >= hypre_MultiblockMatrixNumSubmatrices(matrix)) )
      return(-1);

   hypre_MultiblockMatrixSubmatrix(matrix,j) = submatrix;

   return( ierr );
}



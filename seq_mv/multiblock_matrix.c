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

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_MultiblockMatrix *
hypre_MultiblockMatrixCreate( )
{
   hypre_MultiblockMatrix  *matrix;

   matrix = hypre_CTAlloc(hypre_MultiblockMatrix, 1);

   return ( matrix );
}

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_MultiblockMatrixDestroy( hypre_MultiblockMatrix *matrix )
{
   int  ierr=0, i;

   if (matrix)
   {
      for(i=0; i < hypre_MultiblockMatrixNumSubmatrices(matrix); i++)
         hypre_TFree(hypre_MultiblockMatrixSubmatrix(matrix,i));
      hypre_TFree(hypre_MultiblockMatrixSubmatrices(matrix));
      hypre_TFree(hypre_MultiblockMatrixSubmatrixTypes(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_MultiblockMatrixLimitedDestroy( hypre_MultiblockMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      hypre_TFree(hypre_MultiblockMatrixSubmatrices(matrix));
      hypre_TFree(hypre_MultiblockMatrixSubmatrixTypes(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_MultiblockMatrixInitialize( hypre_MultiblockMatrix *matrix )
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
 * hypre_MultiblockMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
hypre_MultiblockMatrixAssemble( hypre_MultiblockMatrix *matrix )
{
   int    ierr=0;

   return(ierr);
}


/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_MultiblockMatrixPrint(hypre_MultiblockMatrix *matrix  )
{
   printf("Stub for hypre_MultiblockMatrix\n");
}

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixSetNumSubmatrices
 *--------------------------------------------------------------------------*/

int
hypre_MultiblockMatrixSetNumSubmatrices(hypre_MultiblockMatrix *matrix, int n  )
{
   int ierr = 0;

   hypre_MultiblockMatrixNumSubmatrices(matrix) = n;
   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixSetSubmatrixType
 *--------------------------------------------------------------------------*/

int
hypre_MultiblockMatrixSetSubmatrixType(hypre_MultiblockMatrix *matrix, 
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
 * hypre_MultiblockMatrixSetSubmatrix
 *--------------------------------------------------------------------------*/

int
hypre_MultiblockMatrixSetSubmatrix(hypre_MultiblockMatrix *matrix, 
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



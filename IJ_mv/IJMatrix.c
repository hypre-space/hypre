/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * hypre_IJMatrix interface
 *
 *****************************************************************************/

#include "./IJ_mv.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * hypre_IJMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the row partitioning 

@return integer error code
@param IJMatrix [IN]
The ijmatrix to be pointed to.
*/

int
hypre_IJMatrixGetRowPartitioning( HYPRE_IJMatrix matrix ,
				  int    **row_partitioning )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- hypre_IJMatrixGetRowPartitioning\n");
      exit(1);
   }

   if ( hypre_IJMatrixRowPartitioning(ijmatrix))
      *row_partitioning = hypre_IJMatrixRowPartitioning(ijmatrix);
   else
      return -1;

   return -99;
}
/*--------------------------------------------------------------------------
 * hypre_IJMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the column partitioning

@return integer error code
@param IJMatrix [IN]
The ijmatrix to be pointed to.
*/

int
hypre_IJMatrixGetColPartitioning( HYPRE_IJMatrix matrix ,
				  int    **col_partitioning )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- hypre_IJMatrixGetColPartitioning\n");
      exit(1);
   }

   if ( hypre_IJMatrixColPartitioning(ijmatrix))
      *col_partitioning = hypre_IJMatrixColPartitioning(ijmatrix);
   else
      return -1;

   return -99;
}
/*--------------------------------------------------------------------------
 * hypre_IJMatrixSetObject
 *--------------------------------------------------------------------------*/

int 
hypre_IJMatrixSetObject( HYPRE_IJMatrix  matrix, 
                         void           *object )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (hypre_IJMatrixObject(ijmatrix) != NULL)
   {
      printf("Referencing a new IJMatrix object can orphan an old -- ");
      printf("hypre_IJMatrixSetObject\n");
      exit(1);
   }

   hypre_IJMatrixObject(ijmatrix) = object;

   return 0;
}

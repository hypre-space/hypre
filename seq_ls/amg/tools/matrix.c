/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Constructors and destructors for matrix structure.
 *
 *****************************************************************************/

#include "general.h"
#include "matrix.h"


/*--------------------------------------------------------------------------
 * NewMatrix
 *--------------------------------------------------------------------------*/

Matrix  *NewMatrix(data, ia, ja, size)
double  *data;
int     *ia;
int     *ja;
int      size;
{
   Matrix     *new;


   new = talloc(Matrix, 1);

   MatrixData(new) = data;
   MatrixIA(new)   = ia;
   MatrixJA(new)   = ja;
   MatrixSize(new) = size;

   return new;
}

/*--------------------------------------------------------------------------
 * FreeMatrix
 *--------------------------------------------------------------------------*/

void     FreeMatrix(matrix)
Matrix  *matrix;
{
   if (matrix)
   {
      tfree(MatrixData(matrix));
      tfree(MatrixIA(matrix));
      tfree(MatrixJA(matrix));
      tfree(matrix);
   }
}


/*--------------------------------------------------------------------------
 * hypre_GetIJMatrixISISMatrix
 *--------------------------------------------------------------------------*/

#include "headers.h"

#ifdef ISIS_AVAILABLE

#include "iostream.h"
#include "RowMatrix.h" // ISIS++ header file

/**
Returns a reference to the ParCsrMatrix used to implement IJMatrix
in the case that the local storage type for IJMatrix is HYPRE_ISIS.

@return integer error code
@param IJMatrix [IN]
The assembled HYPRE_IJMatrix.
@param reference [OUT]
The pointer to be set to point to IJMatrix.
*/
int 
hypre_GetIJMatrixISISMatrix( HYPRE_IJMatrix IJmatrix, RowMatrix *reference )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   RowMatrix isis_matrix = hypre_IJMatrixLocalStorage( matrix );

   *reference = isis_matrix

   return(ierr);
}
#endif

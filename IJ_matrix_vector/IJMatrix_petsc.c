/*--------------------------------------------------------------------------
 * hypre_GetIJMatrixPETScMatrix
 *--------------------------------------------------------------------------*/

#include "headers.h"

#ifdef PETSC_AVAILABLE

/* Matrix structure from PETSc */
#include "sles.h"

/**
Returns a reference to the ParCsrMatrix used to implement IJMatrix
in the case that the local storage type for IJMatrix is HYPRE_PETSC

@return integer error code
@param IJMatrix [IN]
The assembled HYPRE_IJMatrix.
@param reference [OUT]
The pointer to be set to point to IJMatrix.
*/
int 
hypre_GetIJMatrixParCSRMatrix( HYPRE_IJMatrix IJmatrix, Mat *reference )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* Note: the following line only works if local_storage is used to point
      directly at a parcsr matrix. If that is not true, this would need to
      be modified. -AJC */
   Mat petsc_matrix = hypre_IJMatrixLocalStorage( matrix );

   *reference = petsc_matrix;

   return(ierr);
}

#endif

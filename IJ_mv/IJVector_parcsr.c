/*--------------------------------------------------------------------------
 * hypre_IJVectorGetParVector
 *--------------------------------------------------------------------------*/

#include "headers.h"

/**
Returns a reference to the ParVector used to implement IJVector
in the case that the local storage type for IJVector is HYPRE_PARCSR.

@return integer error code
@param IJVector [IN]
The assembled HYPRE_IJVector.
@param reference [OUT]
The pointer to be set to point to IJVector.
*/
int 
hypre_IJVectorGetParVector( HYPRE_IJVector IJvector, HYPRE_ParVector *reference )
{
   int ierr = 0;
   hypre_IJVector *vector = (hypre_IJVector *) IJvector;

   HYPRE_ParVector par_vector = hypre_IJVectorLocalStorage( vector );

   ierr = hypre_RefParVector( par_vector, reference);

   return(ierr);
}


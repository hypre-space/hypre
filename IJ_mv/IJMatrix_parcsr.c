/*--------------------------------------------------------------------------
 * hypre_GetIJMatrixParCsrMatrix
 *--------------------------------------------------------------------------*/

/**
Returns a reference to the ParCsrMatrix used to implement IJMatrix
in the case that the local storage type for IJMatrix is HYPRE_PARCSR_MATRIX.

@return integer error code
@param IJMatrix [IN]
The assembled HYPRE_IJMatrix.
@param reference [OUT]
The pointer to be set to point to IJMatrix.
*/
int 
hypre_RefIJMatrix( HYPRE_IJMatrix IJmatrix, HYPRE_ParCSRMatrix *reference )
{
   int ierr = 0;
   hypre_IJMatrix *matrix = (hypre_IJMatrix *) IJmatrix;

   /* Note: the following line only works if local_storage is used to point
      directly at a parcsr matrix. If that is not true, this would need to
      be modified. -AJC */
   HYPRE_ParCsrMatrix parcsr_matrix = hypre_IJMatrixLocalStorage( matrix );

   /* This assume a routine in ParCsrMatrix that gives a reference. In the
      absence of such a routine, we would just *reference = parcsr_matrix */
   ierr = hypre_RefParCsrMatrix( parcsr_matrix, reference);

   return(ierr);
}


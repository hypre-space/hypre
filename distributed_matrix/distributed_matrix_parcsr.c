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
 * Member functions for hypre_DistributedMatrix class for par_csr storage scheme.
 *
 *****************************************************************************/

#include "./distributed_matrix.h"

#include "../parcsr_matrix_vector/HYPRE_parcsr_mv.h"

typedef struct
{
   int dummy;

} hypre_DistributedMatrixParcsrAuxiliaryData;

/*--------------------------------------------------------------------------
 * hypre_FreeDistributedMatrixParcsr
 *   Internal routine for freeing a matrix stored in Parcsr form.
 *--------------------------------------------------------------------------*/

int 
hypre_FreeDistributedMatrixParcsr( hypre_DistributedMatrix *distributed_matrix )
{
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(distributed_matrix);

   HYPRE_DestroyParCSRMatrix( Parcsr_matrix );

   hypre_TFree(hypre_DistributedMatrixAuxiliaryData( distributed_matrix ) );

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_InitializeDistributedMatrixParcsr
 *--------------------------------------------------------------------------*/

  /* matrix must be set before calling this function*/

int 
hypre_InitializeDistributedMatrixParcsr(hypre_DistributedMatrix *matrix)
{
   
   hypre_DistributedMatrixAuxiliaryData( distributed_matrix ) = 
      hypre_CTAlloc( hypre_DistributedMatrixParcsrAuxiliaryData, 1 );

   return 0;
}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_PrintDistributedMatrixParcsr
 *   Internal routine for printing a matrix stored in Parcsr form.
 *--------------------------------------------------------------------------*/

int 
hypre_PrintDistributedMatrixParcsr( hypre_DistributedMatrix *matrix )
{
   int  ierr=0;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(matrix);

   ierr = HYPRE_PrintParCSRMatrix( Parcsr_matrix, "STDOUT" );
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixLocalRangeParcsr
 *--------------------------------------------------------------------------*/

int 
hypre_GetDistributedMatrixLocalRangeParcsr( hypre_DistributedMatrix *matrix,
                             int *start,
                             int *end )
{
   int ierr=0;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(matrix);

   if (!Parcsr_matrix) return(-1);


   ierr = HYPRE_ParCSRMatrixGetOwnershipRange( Parcsr_matrix, start, end ); CHKERRA(ierr);

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixRowParcsr
 *--------------------------------------------------------------------------*/

int 
hypre_GetDistributedMatrixRowParcsr( hypre_DistributedMatrix *matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   int ierr;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(matrix);

   if (!Parcsr_matrix) return(-1);

   ierr = HYPRE_ParCSRMatrixGetRow( Parcsr_matrix, row, size, col_ind, values); CHKERRA(ierr);

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_RestoreDistributedMatrixRowParcsr
 *--------------------------------------------------------------------------*/

int 
hypre_RestoreDistributedMatrixRowParcsr( hypre_DistributedMatrix *matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   int ierr;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(matrix);

   if (Parcsr_matrix == NULL) return(-1);

   ierr = HYPRE_ParCSRMatrixRestoreRow( Parcsr_matrix, row, size, col_ind, values); CHKERRA(ierr);

   return(ierr);
}

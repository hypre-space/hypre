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
 * par_vector Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * hypre_SetParVectorDataOwner
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectordataowner)( long int *vector,
                                              int      *owns_data,
                                              int      *ierr       )
{
   *ierr = (int) ( hypre_SetParVectorDataOwner ( (hypre_ParVector *)  vector,
                                                 (int)               *owns_data ) );
}

/*--------------------------------------------------------------------------
 * hypre_SetParVectorPartitOwner
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectorpartitowner)( long int *vector,
                                                int      *owns_partitioning,
                                                int      *ierr    )
{
   *ierr = (int) ( hypre_SetParVectorPartitioningOwner
                         ( (HYPRE_ParCSRMatrix) *vector,
                           (int)                *owns_partitioning ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetDimsParCSR
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_getdimsparcsr)( long int *matrix,
                                      int      *M,
                                      int      *N,
                                      int      *ierr    )
{
   *ierr = (int) ( HYPRE_GetDimsParCSR( (HYPRE_ParCSRMatrix) *matrix,
                                        (int *)               M,
                                        (int *)               N       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetLocalRangeParcsr
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_getlocalrangeparcsr)( long int *matrix,
                                            int      *start,
                                            int      *end,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_GetLocalRangeParcsr( (HYPRE_ParCSRMatrix) *matrix,
                                              (int *)              start,
                                              (int *)              end      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetRowParCSRMatrix
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_getrowparcsrmatrix)( long int *matrix,
                                           int      *row,
                                           int      *size,
                                           long int *col_ind_ptr,
                                           long int *values_ptr,
                                           int      *ierr )
{
   int    **col_ind;
   double **values;

   *ierr = (int) ( HYPRE_GetRowParCSRMatrix( (HYPRE_ParCSRMatrix) *matrix,
                                             (int)                *row,
                                             (int *)              size,
                                             (int **)             col_ind,
                                             (double **)          values   ) );

   *col_ind_ptr = (long int) *col_ind;
   *values_ptr  = (long int) *values;
}

/*--------------------------------------------------------------------------
 * HYPRE_RestoreRowParCSRMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_restorerowparcsrmatrix)( long int *matrix,
                                               int      *row,
                                               int      *size,
                                               long int *col_ind_ptr,
                                               long int *values_ptr,
                                               int      *ierr         )
{
   int    **col_ind;  
   double **values;

   *ierr = (int) ( HYPRE_RestoreRowParCSRMatrix( (HYPRE_ParCSRMatrix) *matrix,
                                                 (int)                *row,
                                                 (int *)              size,
                                                 (int **)             col_ind,
                                                 (double **)          values   ) );

   *col_ind_ptr = (long int) *col_ind;
   *values_ptr  = (long int) *values;

}

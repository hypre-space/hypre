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
 * HYPRE_ParCSRMatrix Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_NewParCSRMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_newparcsrmatrix)( int      *comm,
                                        int      *global_num_rows,
                                        int      *global_num_cols,
                                        int      *row_starts,
                                        int      *col_starts,
                                        int      *num_cols_offd,
                                        int      *num_nonzeros_diag,
                                        int      *num_nonzeros_offd,
                                        long int *matrix,
                                        int      *ierr               )
{
   *matrix = (long int)
             ( HYPRE_CreateParCSRMatrix( (MPI_Comm) *comm,
                                         (int)      *global_num_rows,
                                         (int)      *global_num_cols,
                                         (int *)     row_starts,
                                         (int *)     col_starts,
                                         (int)      *num_cols_offd,
                                         (int)      *num_nonzeros_diag,
                                         (int)      *num_nonzeros_offd  ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_DestroyParCSRMatrix
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_destroyparcsrmatrix)( long int *matrix,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_DestroyParCSRMatrix( (HYPRE_ParCSRMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeParCSRMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_initializeparcsrmatrix)( long int *matrix,
                                               int      *ierr    )
{
   *ierr = (int) ( HYPRE_InitializeParCSRMatrix( (HYPRE_ParCSRMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PrintParCSRMatrix
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_printparcsrmatrix)( long int *matrix,
                                          char     *file_name,
                                          int      *ierr       )
{
   *ierr = (int) ( HYPRE_PrintParCSRMatrix ( (HYPRE_ParCSRMatrix) *matrix,
                                             (char *)             file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GetCommParCSR
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_getcomparcsr)( long int *matrix,
                                     int      *comm,
                                     int      *ierr    )
{
   *ierr = (int) ( HYPRE_GetCommParCSR( (HYPRE_ParCSRMatrix) *matrix,
                                        (MPI_Comm *)         comm     ) );
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
                                        (int *)              M,
                                        (int *)              N        ) );
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

   *col_ind_ptr = *col_ind;
   *values_ptr  = *values;
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

   *col_ind_ptr = *col_ind;
   *values_ptr  = *values;

}

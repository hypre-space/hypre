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
 * HYPRE_ParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixcreate)( int      *comm,
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
   *ierr = (int)
             ( HYPRE_ParCSRMatrixCreate( (MPI_Comm) *comm,
                                         (int)      *global_num_rows,
                                         (int)      *global_num_cols,
                                         (int *)     row_starts,
                                         (int *)     col_starts,
                                         (int)      *num_cols_offd,
                                         (int)      *num_nonzeros_diag,
                                         (int)      *num_nonzeros_offd,
                                         (HYPRE_ParCSRMatrix *) matrix  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixdestroy)( long int *matrix,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRMatrixDestroy( (HYPRE_ParCSRMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixinitialize)( long int *matrix,
                                               int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRMatrixInitialize( (HYPRE_ParCSRMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixRead
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixread)( int      *comm,
                                         char     *file_name,
                                         long int *matrix,
                                         int      *ierr       )
{
   *ierr = (int) ( HYPRE_ParCSRMatrixRead( (MPI_Comm) *comm,
                                (char *)    file_name,
				(HYPRE_ParCSRMatrix *) matrix ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixPrint
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixprint)( long int *matrix,
                                          char     *file_name,
                                          int      *ierr       )
{
   *ierr = (int) ( HYPRE_ParCSRMatrixPrint ( (HYPRE_ParCSRMatrix) *matrix,
                             (char *)              file_name ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetComm
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixgetcomm)( long int *matrix,
                                            int      *comm,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRMatrixGetComm( (HYPRE_ParCSRMatrix) *matrix,
                                        (MPI_Comm *)          comm    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetDims
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixgetdims)( long int *matrix,
                                            int      *M,
                                            int      *N,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRMatrixGetDims( (HYPRE_ParCSRMatrix) *matrix,
                                        (int *)               M,
                                        (int *)               N       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixgetlocalrange)( long int *matrix,
                                                  int      *row_start,
                                                  int      *row_end,
                                                  int      *col_start,
                                                  int      *col_end,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRMatrixGetLocalRange( (HYPRE_ParCSRMatrix) *matrix,
                                              (int *)               row_start,
                                              (int *)               row_end,
                                              (int *)               col_start,
                                              (int *)               col_end) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetRow
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_parcsrmatrixgetrow)( long int *matrix,
                                           int      *row,
                                           int      *size,
                                           long int *col_ind_ptr,
                                           long int *values_ptr,
                                           int      *ierr )
{
   int    **col_ind;
   double **values;

   *ierr = (int) ( HYPRE_ParCSRMatrixGetRow( (HYPRE_ParCSRMatrix) *matrix,
                                             (int)                *row,
                                             (int *)              size,
                                             (int **)             col_ind,
                                             (double **)          values   ) );

   *col_ind_ptr = (long int) *col_ind;
   *values_ptr  = (long int) *values;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixRestoreRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixrestorerow)( long int *matrix,
                                               int      *row,
                                               int      *size,
                                               long int *col_ind_ptr,
                                               long int *values_ptr,
                                               int      *ierr         )
{
   int    **col_ind;  
   double **values;

   *ierr = (int) ( HYPRE_ParCSRMatrixRestoreRow( (HYPRE_ParCSRMatrix) *matrix,
                                                 (int)                *row,
                                                 (int *)               size,
                                                 (int **)              col_ind,
                                                 (double **)           values   ) );

   *col_ind_ptr = (long int) *col_ind;
   *values_ptr  = (long int) *values;

}

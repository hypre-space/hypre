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
hypre_F90_IFACE(hypre_parcsrmatrixcreate, HYPRE_PARCSRMATRIXCREATE)( int      *comm,
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
hypre_F90_IFACE(hypre_parcsrmatrixdestroy, HYPRE_PARCSRMATRIXDESTROY)( long int *matrix,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRMatrixDestroy( (HYPRE_ParCSRMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixinitialize, HYPRE_PARCSRMATRIXINITIALIZE)( long int *matrix,
                                               int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRMatrixInitialize( (HYPRE_ParCSRMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixRead
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixread, HYPRE_PARCSRMATRIXREAD)( int      *comm,
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
hypre_F90_IFACE(hypre_parcsrmatrixprint, HYPRE_PARCSRMATRIXPRINT)( long int *matrix,
                                          char     *fort_file_name,
                                          int      *fort_file_name_size,
                                          int      *ierr       )
{
   int i;
   char *c_file_name;

   c_file_name = hypre_CTAlloc(char, *fort_file_name_size);

   for (i = 0; i < *fort_file_name_size; i++)
     c_file_name[i] = fort_file_name[i];

   *ierr = (int) ( HYPRE_ParCSRMatrixPrint ( (HYPRE_ParCSRMatrix) *matrix,
                                             (char *)              c_file_name ) );

   hypre_TFree(c_file_name);

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetComm
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixgetcomm, HYPRE_PARCSRMATRIXGETCOMM)( long int *matrix,
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
hypre_F90_IFACE(hypre_parcsrmatrixgetdims, HYPRE_PARCSRMATRIXGETDIMS)( long int *matrix,
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
hypre_F90_IFACE(hypre_parcsrmatrixgetlocalrange, HYPRE_PARCSRMATRIXGETLOCALRANGE)( long int *matrix,
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
hypre_F90_IFACE(hypre_parcsrmatrixgetrow, HYPRE_PARCSRMATRIXGETROW)( long int *matrix,
                                           int      *row,
                                           int      *size,
                                           long int *col_ind_ptr,
                                           long int *values_ptr,
                                           int      *ierr )
{
   int    *col_ind;
   double *values;

   *ierr = (int) ( HYPRE_ParCSRMatrixGetRow( (HYPRE_ParCSRMatrix) *matrix,
                                             (int)                *row,
                                             (int *)               size,
                                             (int **)             &col_ind,
                                             (double **)          &values   ) );

   *col_ind_ptr = (long int) col_ind;
   *values_ptr  = (long int) values;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixRestoreRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixrestorerow, HYPRE_PARCSRMATRIXRESTOREROW)( long int *matrix,
                                               int      *row,
                                               int      *size,
                                               long int *col_ind_ptr,
                                               long int *values_ptr,
                                               int      *ierr         )
{
   int    *col_ind;  
   double *values;

   *ierr = (int) ( HYPRE_ParCSRMatrixRestoreRow( (HYPRE_ParCSRMatrix) *matrix,
                                                 (int)                *row,
                                                 (int *)               size,
                                                 (int **)             &col_ind,
                                                 (double **)          &values   ) );

   *col_ind_ptr = (long int) col_ind;
   *values_ptr  = (long int) values;

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixmatvec, HYPRE_PARCSRMATRIXMATVEC)( double   *alpha,
                                           long int *A,
                                           long int *x,
                                           double   *beta,
                                           long int *y,  
                                           int      *ierr   )
{

   *ierr = (int) ( HYPRE_ParCSRMatrixMatvec( (double)             *alpha,
                                             (HYPRE_ParCSRMatrix) *A,
                                             (HYPRE_ParVector)    *x,
                                             (double)             *beta,
                                             (HYPRE_ParVector)    *y      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixMatvecT
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixmatvect, HYPRE_PARCSRMATRIXMATVECT)( double   *alpha,
                                            long int *A,
                                            long int *x,
                                            double   *beta,
                                            long int *y,
                                            int      *ierr    )
{

   *ierr = (int) ( HYPRE_ParCSRMatrixMatvecT( (double)             *alpha,
                                              (HYPRE_ParCSRMatrix) *A,
                                              (HYPRE_ParVector)    *x,
                                              (double)             *beta,
                                              (HYPRE_ParVector)    *y      ) );
}


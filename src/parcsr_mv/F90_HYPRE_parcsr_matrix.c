/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/



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
 * HYPRE_ParCSRMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_parcsrmatrixgetrowpartiti, HYPRE_PARCSRMATRIXGETROWPARTITI)
                                         ( long int *matrix,
                                           long int *row_partitioning_ptr,
                                           int      *ierr )
{
   int    *row_partitioning;

   *ierr = (int) 
         ( HYPRE_ParCSRMatrixGetRowPartitioning( (HYPRE_ParCSRMatrix) *matrix,
                                                 (int **)    &row_partitioning  ) );

   *row_partitioning_ptr = (long int) row_partitioning;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_parcsrmatrixgetcolpartiti, HYPRE_PARCSRMATRIXGETCOLPARTITI)
                                         ( long int *matrix,
                                           long int *col_partitioning_ptr,
                                           int      *ierr )
{
   int    *col_partitioning;

   *ierr = (int) 
         ( HYPRE_ParCSRMatrixGetColPartitioning( (HYPRE_ParCSRMatrix) *matrix,
                                                 (int **)    &col_partitioning  ) );

   *col_partitioning_ptr = (long int) col_partitioning;
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
 * HYPRE_CSRMatrixToParCSRMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_csrmatrixtoparcsrmatrix, HYPRE_CSRMATRIXTOPARCSRMATRIX)
                                          (int      *comm,
                                           long int *A_CSR,
                                           int      *row_partitioning,  
                                           int      *col_partitioning,  
                                           long int *matrix,
                                           int      *ierr   )
{

   *ierr = (int) ( HYPRE_CSRMatrixToParCSRMatrix( (MPI_Comm)  *comm,
                                             (HYPRE_CSRMatrix) *A_CSR,
                                             (int *)            row_partitioning,
                                             (int *)            col_partitioning,
                                             (HYPRE_ParCSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_csrmatrixtoparcsrmatrix_withnewpartitioning, HYPRE_CSRMATRIXTOPARCSRMATRIX_WITHNEWPARTITIONING)
                                          (int      *comm,
                                           long int *A_CSR,
                                           long int *matrix,
                                           int      *ierr   )
{

   *ierr = (int) ( HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning(
                      (MPI_Comm)  *comm,
                      (HYPRE_CSRMatrix) *A_CSR,
                      (HYPRE_ParCSRMatrix *) matrix ) );
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

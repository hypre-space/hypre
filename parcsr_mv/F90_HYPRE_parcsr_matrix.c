/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
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
hypre_F90_IFACE(hypre_parcsrmatrixcreate, HYPRE_PARCSRMATRIXCREATE)( hypre_F90_Comm *comm,
                                        HYPRE_Int      *global_num_rows,
                                        HYPRE_Int      *global_num_cols,
                                        HYPRE_Int      *row_starts,
                                        HYPRE_Int      *col_starts,
                                        HYPRE_Int      *num_cols_offd,
                                        HYPRE_Int      *num_nonzeros_diag,
                                        HYPRE_Int      *num_nonzeros_offd,
                                        hypre_F90_Obj *matrix,
                                        HYPRE_Int      *ierr               )
{
   *ierr = (HYPRE_Int)
             ( HYPRE_ParCSRMatrixCreate( (MPI_Comm) *comm,
                                         (HYPRE_Int)      *global_num_rows,
                                         (HYPRE_Int)      *global_num_cols,
                                         (HYPRE_Int *)     row_starts,
                                         (HYPRE_Int *)     col_starts,
                                         (HYPRE_Int)      *num_cols_offd,
                                         (HYPRE_Int)      *num_nonzeros_diag,
                                         (HYPRE_Int)      *num_nonzeros_offd,
                                         (HYPRE_ParCSRMatrix *) matrix  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixdestroy, HYPRE_PARCSRMATRIXDESTROY)( hypre_F90_Obj *matrix,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRMatrixDestroy( (HYPRE_ParCSRMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixinitialize, HYPRE_PARCSRMATRIXINITIALIZE)( hypre_F90_Obj *matrix,
                                               HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRMatrixInitialize( (HYPRE_ParCSRMatrix) *matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixRead
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixread, HYPRE_PARCSRMATRIXREAD)( hypre_F90_Comm *comm,
                                         char     *file_name,
                                         hypre_F90_Obj *matrix,
                                         HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRMatrixRead( (MPI_Comm) *comm,
                                (char *)    file_name,
				(HYPRE_ParCSRMatrix *) matrix ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixprint, HYPRE_PARCSRMATRIXPRINT)( hypre_F90_Obj *matrix,
                                          char     *fort_file_name,
                                          HYPRE_Int      *fort_file_name_size,
                                          HYPRE_Int      *ierr       )
{
   HYPRE_Int i;
   char *c_file_name;

   c_file_name = hypre_CTAlloc(char, *fort_file_name_size);

   for (i = 0; i < *fort_file_name_size; i++)
     c_file_name[i] = fort_file_name[i];

   *ierr = (HYPRE_Int) ( HYPRE_ParCSRMatrixPrint ( (HYPRE_ParCSRMatrix) *matrix,
                                             (char *)              c_file_name ) );

   hypre_TFree(c_file_name);

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetComm
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixgetcomm, HYPRE_PARCSRMATRIXGETCOMM)( hypre_F90_Obj *matrix,
                                            hypre_F90_Comm *comm,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRMatrixGetComm( (HYPRE_ParCSRMatrix) *matrix,
                                        (MPI_Comm *)          comm    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetDims
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixgetdims, HYPRE_PARCSRMATRIXGETDIMS)( hypre_F90_Obj *matrix,
                                            HYPRE_Int      *M,
                                            HYPRE_Int      *N,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRMatrixGetDims( (HYPRE_ParCSRMatrix) *matrix,
                                        (HYPRE_Int *)               M,
                                        (HYPRE_Int *)               N       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_parcsrmatrixgetrowpartiti, HYPRE_PARCSRMATRIXGETROWPARTITI)
                                         ( hypre_F90_Obj *matrix,
                                           hypre_F90_Obj *row_partitioning_ptr,
                                           HYPRE_Int      *ierr )
{
   HYPRE_Int    *row_partitioning;

   *ierr = (HYPRE_Int) 
         ( HYPRE_ParCSRMatrixGetRowPartitioning( (HYPRE_ParCSRMatrix) *matrix,
                                                 (HYPRE_Int **)    &row_partitioning  ) );

   *row_partitioning_ptr = (hypre_F90_Obj) row_partitioning;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_parcsrmatrixgetcolpartiti, HYPRE_PARCSRMATRIXGETCOLPARTITI)
                                         ( hypre_F90_Obj *matrix,
                                           hypre_F90_Obj *col_partitioning_ptr,
                                           HYPRE_Int      *ierr )
{
   HYPRE_Int    *col_partitioning;

   *ierr = (HYPRE_Int) 
         ( HYPRE_ParCSRMatrixGetColPartitioning( (HYPRE_ParCSRMatrix) *matrix,
                                                 (HYPRE_Int **)    &col_partitioning  ) );

   *col_partitioning_ptr = (hypre_F90_Obj) col_partitioning;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixgetlocalrange, HYPRE_PARCSRMATRIXGETLOCALRANGE)( hypre_F90_Obj *matrix,
                                                  HYPRE_Int      *row_start,
                                                  HYPRE_Int      *row_end,
                                                  HYPRE_Int      *col_start,
                                                  HYPRE_Int      *col_end,
                                                  HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRMatrixGetLocalRange( (HYPRE_ParCSRMatrix) *matrix,
                                              (HYPRE_Int *)               row_start,
                                              (HYPRE_Int *)               row_end,
                                              (HYPRE_Int *)               col_start,
                                              (HYPRE_Int *)               col_end) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetRow
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_parcsrmatrixgetrow, HYPRE_PARCSRMATRIXGETROW)( hypre_F90_Obj *matrix,
                                           HYPRE_Int      *row,
                                           HYPRE_Int      *size,
                                           hypre_F90_Obj *col_ind_ptr,
                                           hypre_F90_Obj *values_ptr,
                                           HYPRE_Int      *ierr )
{
   HYPRE_Int    *col_ind;
   double *values;

   *ierr = (HYPRE_Int) ( HYPRE_ParCSRMatrixGetRow( (HYPRE_ParCSRMatrix) *matrix,
                                             (HYPRE_Int)                *row,
                                             (HYPRE_Int *)               size,
                                             (HYPRE_Int **)             &col_ind,
                                             (double **)          &values   ) );

   *col_ind_ptr = (hypre_F90_Obj) col_ind;
   *values_ptr  = (hypre_F90_Obj) values;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixRestoreRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixrestorerow, HYPRE_PARCSRMATRIXRESTOREROW)( hypre_F90_Obj *matrix,
                                               HYPRE_Int      *row,
                                               HYPRE_Int      *size,
                                               hypre_F90_Obj *col_ind_ptr,
                                               hypre_F90_Obj *values_ptr,
                                               HYPRE_Int      *ierr         )
{
   HYPRE_Int    *col_ind;  
   double *values;

   *ierr = (HYPRE_Int) ( HYPRE_ParCSRMatrixRestoreRow( (HYPRE_ParCSRMatrix) *matrix,
                                                 (HYPRE_Int)                *row,
                                                 (HYPRE_Int *)               size,
                                                 (HYPRE_Int **)             &col_ind,
                                                 (double **)          &values   ) );

   *col_ind_ptr = (hypre_F90_Obj) col_ind;
   *values_ptr  = (hypre_F90_Obj) values;

}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixToParCSRMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_csrmatrixtoparcsrmatrix, HYPRE_CSRMATRIXTOPARCSRMATRIX)
                                          (hypre_F90_Comm *comm,
                                           hypre_F90_Obj *A_CSR,
                                           HYPRE_Int      *row_partitioning,  
                                           HYPRE_Int      *col_partitioning,  
                                           hypre_F90_Obj *matrix,
                                           HYPRE_Int      *ierr   )
{

   *ierr = (HYPRE_Int) ( HYPRE_CSRMatrixToParCSRMatrix( (MPI_Comm)  *comm,
                                             (HYPRE_CSRMatrix) *A_CSR,
                                             (HYPRE_Int *)            row_partitioning,
                                             (HYPRE_Int *)            col_partitioning,
                                             (HYPRE_ParCSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_csrmatrixtoparcsrmatrix_withnewpartitioning, HYPRE_CSRMATRIXTOPARCSRMATRIX_WITHNEWPARTITIONING)
                                          (hypre_F90_Comm *comm,
                                           hypre_F90_Obj *A_CSR,
                                           hypre_F90_Obj *matrix,
                                           HYPRE_Int      *ierr   )
{

   *ierr = (HYPRE_Int) ( HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning(
                      (MPI_Comm)  *comm,
                      (HYPRE_CSRMatrix) *A_CSR,
                      (HYPRE_ParCSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixmatvec, HYPRE_PARCSRMATRIXMATVEC)( double   *alpha,
                                           hypre_F90_Obj *A,
                                           hypre_F90_Obj *x,
                                           double   *beta,
                                           hypre_F90_Obj *y,  
                                           HYPRE_Int      *ierr   )
{

   *ierr = (HYPRE_Int) ( HYPRE_ParCSRMatrixMatvec( (double)             *alpha,
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
                                            hypre_F90_Obj *A,
                                            hypre_F90_Obj *x,
                                            double   *beta,
                                            hypre_F90_Obj *y,
                                            HYPRE_Int      *ierr    )
{

   *ierr = (HYPRE_Int) ( HYPRE_ParCSRMatrixMatvecT( (double)             *alpha,
                                              (HYPRE_ParCSRMatrix) *A,
                                              (HYPRE_ParVector)    *x,
                                              (double)             *beta,
                                              (HYPRE_ParVector)    *y      ) );
}

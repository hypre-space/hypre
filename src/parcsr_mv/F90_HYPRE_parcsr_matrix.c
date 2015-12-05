/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
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
hypre_F90_IFACE(hypre_parcsrmatrixcreate, HYPRE_PARCSRMATRIXCREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Int *global_num_rows,
     hypre_F90_Int *global_num_cols,
     hypre_F90_IntArray *row_starts,
     hypre_F90_IntArray *col_starts,
     hypre_F90_Int *num_cols_offd,
     hypre_F90_Int *num_nonzeros_diag,
     hypre_F90_Int *num_nonzeros_offd,
     hypre_F90_Obj *matrix,
     hypre_F90_Int *ierr               )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRMatrixCreate(
           hypre_F90_PassComm (comm),
           hypre_F90_PassInt (global_num_rows),
           hypre_F90_PassInt (global_num_cols),
           hypre_F90_PassIntArray (row_starts),
           hypre_F90_PassIntArray (col_starts),
           hypre_F90_PassInt (num_cols_offd),
           hypre_F90_PassInt (num_nonzeros_diag),
           hypre_F90_PassInt (num_nonzeros_offd),
           hypre_F90_PassObjRef (HYPRE_ParCSRMatrix, matrix)  ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixdestroy, HYPRE_PARCSRMATRIXDESTROY)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRMatrixDestroy(
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixinitialize, HYPRE_PARCSRMATRIXINITIALIZE)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRMatrixInitialize(
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixRead
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixread, HYPRE_PARCSRMATRIXREAD)
   ( hypre_F90_Comm *comm,
     char     *file_name,
     hypre_F90_Obj *matrix,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRMatrixRead(
           hypre_F90_PassComm (comm),
           (char *)    file_name,
           hypre_F90_PassObjRef (HYPRE_ParCSRMatrix, matrix) ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixprint, HYPRE_PARCSRMATRIXPRINT)
   ( hypre_F90_Obj *matrix,
     char     *fort_file_name,
     hypre_F90_Int *fort_file_name_size,
     hypre_F90_Int *ierr       )
{
   HYPRE_Int i;
   char *c_file_name;

   c_file_name = hypre_CTAlloc(char, *fort_file_name_size);

   for (i = 0; i < *fort_file_name_size; i++)
   {
      c_file_name[i] = fort_file_name[i];
   }

   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRMatrixPrint(
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, matrix),
           (char *)              c_file_name ) );

   hypre_TFree(c_file_name);

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetComm
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixgetcomm, HYPRE_PARCSRMATRIXGETCOMM)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Comm *comm,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRMatrixGetComm(
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, matrix),
           (MPI_Comm *)          comm    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetDims
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixgetdims, HYPRE_PARCSRMATRIXGETDIMS)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *M,
     hypre_F90_Int *N,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRMatrixGetDims(
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, matrix),
           hypre_F90_PassIntRef (M),
           hypre_F90_PassIntRef (N)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_parcsrmatrixgetrowpartiti, HYPRE_PARCSRMATRIXGETROWPARTITI)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Obj *row_partitioning_ptr,
     hypre_F90_Int *ierr )
{
   HYPRE_Int *row_partitioning;

   *ierr = (hypre_F90_Int) HYPRE_ParCSRMatrixGetRowPartitioning(
      hypre_F90_PassObj (HYPRE_ParCSRMatrix, matrix),
      (HYPRE_Int **)    &row_partitioning  );

   *row_partitioning_ptr = (hypre_F90_Obj) row_partitioning;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_parcsrmatrixgetcolpartiti, HYPRE_PARCSRMATRIXGETCOLPARTITI)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Obj *col_partitioning_ptr,
     hypre_F90_Int *ierr )
{
   HYPRE_Int *col_partitioning;

   *ierr = (hypre_F90_Int) HYPRE_ParCSRMatrixGetColPartitioning(
      hypre_F90_PassObj (HYPRE_ParCSRMatrix, matrix),
      (HYPRE_Int **)    &col_partitioning  );

   *col_partitioning_ptr = (hypre_F90_Obj) col_partitioning;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrmatrixgetlocalrange, HYPRE_PARCSRMATRIXGETLOCALRANGE)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *row_start,
     hypre_F90_Int *row_end,
     hypre_F90_Int *col_start,
     hypre_F90_Int *col_end,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRMatrixGetLocalRange(
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, matrix),
           hypre_F90_PassIntRef (row_start),
           hypre_F90_PassIntRef (row_end),
           hypre_F90_PassIntRef (col_start),
           hypre_F90_PassIntRef (col_end)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixGetRow
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_parcsrmatrixgetrow, HYPRE_PARCSRMATRIXGETROW)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *row,
     hypre_F90_Int *size,
     hypre_F90_Obj *col_ind_ptr,
     hypre_F90_Obj *values_ptr,
     hypre_F90_Int *ierr )
{
   HYPRE_Int *col_ind;
   double    *values;

   *ierr = (hypre_F90_Int) HYPRE_ParCSRMatrixGetRow(
      hypre_F90_PassObj      (HYPRE_ParCSRMatrix, matrix),
      hypre_F90_PassInt      (row),
      hypre_F90_PassIntRef (size),
      (HYPRE_Int **)         &col_ind,
      (double **)            &values );

   *col_ind_ptr = (hypre_F90_Obj) col_ind;
   *values_ptr  = (hypre_F90_Obj) values;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixRestoreRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixrestorerow, HYPRE_PARCSRMATRIXRESTOREROW)
   ( hypre_F90_Obj *matrix,
     hypre_F90_Int *row,
     hypre_F90_Int *size,
     hypre_F90_Obj *col_ind_ptr,
     hypre_F90_Obj *values_ptr,
     hypre_F90_Int *ierr )
{
   HYPRE_Int *col_ind;  
   double    *values;

   *ierr = (hypre_F90_Int) HYPRE_ParCSRMatrixRestoreRow(
      hypre_F90_PassObj      (HYPRE_ParCSRMatrix, matrix),
      hypre_F90_PassInt      (row),
      hypre_F90_PassIntRef (size),
      (HYPRE_Int **)         &col_ind,
      (double **)            &values );

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
    hypre_F90_IntArray *row_partitioning,  
    hypre_F90_IntArray *col_partitioning,  
    hypre_F90_Obj *matrix,
    hypre_F90_Int *ierr   )
{

   *ierr = (hypre_F90_Int)
      ( HYPRE_CSRMatrixToParCSRMatrix(
           hypre_F90_PassComm (comm),
           hypre_F90_PassObj (HYPRE_CSRMatrix, A_CSR),
           hypre_F90_PassIntArray (row_partitioning),
           hypre_F90_PassIntArray (col_partitioning),
           hypre_F90_PassObjRef (HYPRE_ParCSRMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_csrmatrixtoparcsrmatrix_withnewpartitioning, HYPRE_CSRMATRIXTOPARCSRMATRIX_WITHNEWPARTITIONING)
   (hypre_F90_Comm *comm,
    hypre_F90_Obj *A_CSR,
    hypre_F90_Obj *matrix,
    hypre_F90_Int *ierr   )
{

   *ierr = (hypre_F90_Int)
      ( HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning(
           hypre_F90_PassComm (comm),
           hypre_F90_PassObj (HYPRE_CSRMatrix, A_CSR),
           hypre_F90_PassObjRef (HYPRE_ParCSRMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixmatvec, HYPRE_PARCSRMATRIXMATVEC)
   ( hypre_F90_Dbl *alpha,
     hypre_F90_Obj *A,
     hypre_F90_Obj *x,
     hypre_F90_Dbl *beta,
     hypre_F90_Obj *y,  
     hypre_F90_Int *ierr   )
{

   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRMatrixMatvec(
           hypre_F90_PassDbl (alpha),
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
           hypre_F90_PassObj (HYPRE_ParVector, x),
           hypre_F90_PassDbl (beta),
           hypre_F90_PassObj (HYPRE_ParVector, y)      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRMatrixMatvecT
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixmatvect, HYPRE_PARCSRMATRIXMATVECT)
   ( hypre_F90_Dbl *alpha,
     hypre_F90_Obj *A,
     hypre_F90_Obj *x,
     hypre_F90_Dbl *beta,
     hypre_F90_Obj *y,
     hypre_F90_Int *ierr    )
{

   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRMatrixMatvecT(
           hypre_F90_PassDbl (alpha),
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
           hypre_F90_PassObj (HYPRE_ParVector, x),
           hypre_F90_PassDbl (beta),
           hypre_F90_PassObj (HYPRE_ParVector, y)      ) );
}

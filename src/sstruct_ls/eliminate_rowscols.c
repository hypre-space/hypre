/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




#include "_hypre_parcsr_mv.h"
#include "eliminate_rowscols.h"

int hypre_ParCSRMatrixEliminateRowsCols (hypre_ParCSRMatrix *A,
                                         int nrows_to_eliminate,
                                         int *rows_to_eliminate)
{
   int ierr = 0;

   MPI_Comm         comm      = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *diag      = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd      = hypre_ParCSRMatrixOffd(A);
   int diag_nrows             = hypre_CSRMatrixNumRows(diag);
   int offd_ncols             = hypre_CSRMatrixNumCols(offd);

   int ncols_to_eliminate;
   int *cols_to_eliminate;

   int             myproc;
   int             ibeg;

   MPI_Comm_rank(comm, &myproc);
   ibeg= 0;


   /* take care of the diagonal part (sequential elimination) */
   hypre_CSRMatrixEliminateRowsColsDiag (A, nrows_to_eliminate,
                                         rows_to_eliminate);

   /* eliminate the off-diagonal rows */
   hypre_CSRMatrixEliminateRowsOffd (A, nrows_to_eliminate,
                                     rows_to_eliminate);

   /* figure out which offd cols should be eliminated */
   {
      hypre_ParCSRCommHandle *comm_handle;
      hypre_ParCSRCommPkg *comm_pkg;
      int num_sends, *int_buf_data;
      int index, start;
      int i, j, k;

      int *eliminate_row = hypre_CTAlloc(int, diag_nrows);
      int *eliminate_col = hypre_CTAlloc(int, offd_ncols);

      /* make sure A has a communication package */
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      }

      /* which of the local rows are to be eliminated */
      for (i = 0; i < diag_nrows; i++)
         eliminate_row[i] = 0;
      for (i = 0; i < nrows_to_eliminate; i++)
         eliminate_row[rows_to_eliminate[i]-ibeg] = 1;

      /* use a Matvec communication pattern to find (in eliminate_col)
         which of the local offd columns are to be eliminated */
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      int_buf_data = hypre_CTAlloc(int,
                                   hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                   num_sends));
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            k = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            int_buf_data[index++] = eliminate_row[k];
         }
      }
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg,
                                                 int_buf_data, eliminate_col);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* set the array cols_to_eliminate */
      ncols_to_eliminate = 0;
      for (i = 0; i < offd_ncols; i++)
         if (eliminate_col[i])
            ncols_to_eliminate++;

      cols_to_eliminate = hypre_CTAlloc(int, ncols_to_eliminate);

      ncols_to_eliminate = 0;
      for (i = 0; i < offd_ncols; i++)
         if (eliminate_col[i])
            cols_to_eliminate[ncols_to_eliminate++] = i;

      hypre_TFree(int_buf_data);
      hypre_TFree(eliminate_row);
      hypre_TFree(eliminate_col);
   }

   /* eliminate the off-diagonal columns */
   hypre_CSRMatrixEliminateColsOffd (offd, ncols_to_eliminate,
                                     cols_to_eliminate);

   hypre_TFree(cols_to_eliminate);

   return ierr;
}


int hypre_CSRMatrixEliminateRowsColsDiag (hypre_ParCSRMatrix *A,
                                          int nrows_to_eliminate,
                                          int *rows_to_eliminate)
{
   int ierr = 0;
  
   MPI_Comm          comm      = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix  *Adiag     = hypre_ParCSRMatrixDiag(A);

   int               i, j;
   int               irow, ibeg, iend;

   int               nnz       = hypre_CSRMatrixNumNonzeros(Adiag);
   int              *Ai        = hypre_CSRMatrixI(Adiag);
   int              *Aj        = hypre_CSRMatrixJ(Adiag);
   double           *Adata     = hypre_CSRMatrixData(Adiag);

   int              *local_rows;
  
   int               myproc;

   MPI_Comm_rank(comm, &myproc);
   ibeg= 0;

   /* grab local rows to eliminate */
   local_rows= hypre_TAlloc(int, nrows_to_eliminate);
   for (i= 0; i< nrows_to_eliminate; i++)
   {
      local_rows[i]= rows_to_eliminate[i]-ibeg;
   }
      
   /* remove the columns */
   for (i = 0; i < nnz; i++)
   {
      irow = hypre_BinarySearch(local_rows, Aj[i],
                                nrows_to_eliminate);
      if (irow != -1)
         Adata[i] = 0.0;
   }

   /* remove the rows and set the diagonal equal to 1 */
   for (i = 0; i < nrows_to_eliminate; i++)
   {
      irow = local_rows[i];
      ibeg = Ai[irow];
      iend = Ai[irow+1];
      for (j = ibeg; j < iend; j++)
         if (Aj[j] == irow)
            Adata[j] = 1.0;
         else
            Adata[j] = 0.0;
   }

   hypre_TFree(local_rows);

   return ierr;
}

int hypre_CSRMatrixEliminateRowsOffd (hypre_ParCSRMatrix *A,
                                      int  nrows_to_eliminate,
                                      int *rows_to_eliminate)
{
   int ierr = 0;

   MPI_Comm         comm      = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *Aoffd     = hypre_ParCSRMatrixOffd(A);
   int             *Ai        = hypre_CSRMatrixI(Aoffd);

   double          *Adata     = hypre_CSRMatrixData(Aoffd);

   int i, j;
   int ibeg, iend;

   int *local_rows;
   int myproc;

   MPI_Comm_rank(comm, &myproc);
   ibeg= 0;

   /* grab local rows to eliminate */
   local_rows= hypre_TAlloc(int, nrows_to_eliminate);
   for (i= 0; i< nrows_to_eliminate; i++)
   {
      local_rows[i]= rows_to_eliminate[i]-ibeg;
   }

   for (i = 0; i < nrows_to_eliminate; i++)
   {
      ibeg = Ai[local_rows[i]];
      iend = Ai[local_rows[i]+1];
      for (j = ibeg; j < iend; j++)
         Adata[j] = 0.0;
   }

   hypre_TFree(local_rows);

   return ierr;
}

int hypre_CSRMatrixEliminateColsOffd (hypre_CSRMatrix *Aoffd,
                                      int ncols_to_eliminate,
                                      int *cols_to_eliminate)
{
   int ierr = 0;

   int i;
   int icol;

   int nnz = hypre_CSRMatrixNumNonzeros(Aoffd);
   int *Aj = hypre_CSRMatrixJ(Aoffd);
   double *Adata = hypre_CSRMatrixData(Aoffd);

   for (i = 0; i < nnz; i++)
   {
      icol = hypre_BinarySearch(cols_to_eliminate, Aj[i],
                                ncols_to_eliminate);
      if (icol != -1)
         Adata[i] = 0.0;
   }

   return ierr;
}

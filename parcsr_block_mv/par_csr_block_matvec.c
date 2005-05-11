/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "HYPRE.h"
#include "par_csr_block_matrix.h"
#include "parcsr_mv/parcsr_mv.h"
#include "seq_mv/seq_mv.h"
#include <assert.h>

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRBlockMatrixMatvec(double alpha, hypre_ParCSRBlockMatrix *A,
                              hypre_ParVector *x, double beta,
                              hypre_ParVector *y)
{
   hypre_ParCSRCommHandle *comm_handle;
/*
   hypre_ParCSRCommPkg	  *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   hypre_CSRBlockMatrix   *diag   = hypre_ParCSRBlockMatrixDiag(A);
   hypre_CSRBlockMatrix   *offd   = hypre_ParCSRBlockMatrixOffd(A);
   hypre_Vector           *x_local  = hypre_ParVectorLocalVector(x);   
   hypre_Vector           *y_local  = hypre_ParVectorLocalVector(y);   
   int                    num_rows = hypre_ParCSRBlockMatrixGlobalNumRows(A);
   int                    num_cols = hypre_ParCSRBlockMatrixGlobalNumCols(A);
   int                    blk_size = hypre_ParCSRBlockMatrixBlockSize(A);
*/
   hypre_ParCSRCommPkg	  *comm_pkg;
   hypre_CSRBlockMatrix   *diag;
   hypre_CSRBlockMatrix   *offd;
   hypre_Vector           *x_local;   
   hypre_Vector           *y_local;   
   int                    num_rows;
   int                    num_cols;
   int                    blk_size;

   hypre_Vector           *x_tmp;
/*
   int                    x_size = hypre_ParVectorGlobalSize(x);
   int                    y_size = hypre_ParVectorGlobalSize(y);
*/
   int                    x_size;
   int                    y_size;

/*
   int	                  num_cols_offd = hypre_CSRBlockMatrixNumCols(offd);
*/
   int	                  num_cols_offd;

   int                    ierr = 0, nprocs;
   int	                  num_sends, i, j, k, index, start;
   double                 *x_tmp_data, *x_buf_data;
/*
   double                 *x_local_data = hypre_VectorData(x_local);
*/
   double                 *x_local_data;

   MPI_Comm_size(hypre_ParCSRBlockMatrixComm(A), &nprocs);
printf("matvec 1 %d\n", nprocs);
   comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   diag   = hypre_ParCSRBlockMatrixDiag(A);
   offd   = hypre_ParCSRBlockMatrixOffd(A);
   num_cols_offd = hypre_CSRBlockMatrixNumCols(offd);
   x_local  = hypre_ParVectorLocalVector(x);   
   y_local  = hypre_ParVectorLocalVector(y);   
   num_rows = hypre_ParCSRBlockMatrixGlobalNumRows(A);
   num_cols = hypre_ParCSRBlockMatrixGlobalNumCols(A);
   blk_size = hypre_ParCSRBlockMatrixBlockSize(A);
   x_size = hypre_ParVectorGlobalSize(x);
   y_size = hypre_ParVectorGlobalSize(y);
   x_local_data = hypre_VectorData(x_local);
printf("matvec 2 %d %d\n", blk_size, num_cols_offd);
fflush(stdout);

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  
    *--------------------------------------------------------------------*/
 
   if (num_cols*blk_size != x_size) ierr = 11;
   if (num_rows*blk_size != y_size) ierr = 12;
   if (num_cols*blk_size != x_size && num_rows*blk_size != y_size) ierr = 13;

   if (nprocs > 1)
   {
      x_tmp = hypre_SeqVectorCreate(num_cols_offd*blk_size);
      hypre_SeqVectorInitialize(x_tmp);
      x_tmp_data = hypre_VectorData(x_tmp);
      comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle,1);
      if (!comm_pkg)
      {
         hypre_BlockMatvecCommPkgCreate(A);
         comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A); 
      }
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      x_buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart
                                 (comm_pkg, num_sends*blk_size));

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            for (k = 0; j < blk_size; k++)
               x_buf_data[index++] 
                  = x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)*blk_size+k];
      }
      comm_handle = hypre_ParCSRCommHandleCreate(1,comm_pkg,x_buf_data,x_tmp_data);
   }
printf("matvec 5\n");
fflush(stdout);
   hypre_CSRBlockMatrixMatvec(alpha, diag, x_local, beta, y_local);
   if (nprocs > 1)
   {
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
      if (num_cols_offd) hypre_CSRBlockMatrixMatvec(alpha,offd,x_tmp,1.0,y_local);    
      hypre_SeqVectorDestroy(x_tmp);
      x_tmp = NULL;
      hypre_TFree(x_buf_data);
   }
   return ierr;
}


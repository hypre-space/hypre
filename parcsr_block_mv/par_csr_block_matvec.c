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

extern 
hypre_ParCSRCommHandle *hypre_ParCSRBlockCommHandleCreate(int,
                              hypre_ParCSRCommPkg *, double *, double *);

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRBlockMatrixMatvec(double alpha, hypre_ParCSRBlockMatrix *A,
                              hypre_ParVector *x, double beta,
                              hypre_ParVector *y)
{
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg	  *comm_pkg;
   hypre_CSRBlockMatrix   *diag, *offd;
   hypre_Vector           *x_local, *y_local, *x_tmp;
   int                    i, j, k, index, num_rows, num_cols;
   int                    blk_size, x_size, y_size, size;
   int	                  num_cols_offd, start, finish, elem;
   int                    ierr = 0, nprocs, num_sends, mypid;
   double                 *x_tmp_data, *x_buf_data, *x_local_data;

   MPI_Comm_size(hypre_ParCSRBlockMatrixComm(A), &nprocs);
   MPI_Comm_rank(hypre_ParCSRBlockMatrixComm(A), &mypid);
   comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   num_rows = hypre_ParCSRBlockMatrixGlobalNumRows(A);
   num_cols = hypre_ParCSRBlockMatrixGlobalNumCols(A);
   blk_size = hypre_ParCSRBlockMatrixBlockSize(A);
   diag   = hypre_ParCSRBlockMatrixDiag(A);
   offd   = hypre_ParCSRBlockMatrixOffd(A);
   num_cols_offd = hypre_CSRBlockMatrixNumCols(offd);
   x_local  = hypre_ParVectorLocalVector(x);   
   y_local  = hypre_ParVectorLocalVector(y);   
   x_size = hypre_ParVectorGlobalSize(x);
   y_size = hypre_ParVectorGlobalSize(y);
   x_local_data = hypre_VectorData(x_local);

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
      size = hypre_ParCSRCommPkgSendMapStart(comm_pkg,num_sends)*blk_size;
      x_buf_data = hypre_CTAlloc(double, size);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         finish = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
         for (j = start; j < finish; j++)
         {
            elem = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)*blk_size;
            for (k = 0; k < blk_size; k++)
               x_buf_data[index++] = x_local_data[elem++];
         }
      }
      comm_handle = hypre_ParCSRBlockCommHandleCreate(blk_size,comm_pkg,
                                 x_buf_data, x_tmp_data);
   }
   hypre_CSRBlockMatrixMatvec(alpha, diag, x_local, beta, y_local);
   if (nprocs > 1)
   {
      hypre_ParCSRBlockCommHandleDestroy(comm_handle);
      comm_handle = NULL;
      if (num_cols_offd) 
         hypre_CSRBlockMatrixMatvec(alpha,offd,x_tmp,1.0,y_local);    
      hypre_SeqVectorDestroy(x_tmp);
      x_tmp = NULL;
      hypre_TFree(x_buf_data);
   }
   return ierr;
}


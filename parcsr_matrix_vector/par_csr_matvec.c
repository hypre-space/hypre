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

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixMatvec( double           alpha,
              	 hypre_ParCSRMatrix *A,
                 hypre_ParVector    *x,
                 double           beta,
                 hypre_ParVector    *y     )
{
   hypre_ParCSRCommHandle	*comm_handle;
   hypre_ParCSRCommPkg	*comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix      *diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix      *offd   = hypre_ParCSRMatrixOffd(A);
   hypre_Vector         *x_local  = hypre_ParVectorLocalVector(x);   
   hypre_Vector         *y_local  = hypre_ParVectorLocalVector(y);   
   int         num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   int         num_cols = hypre_ParCSRMatrixGlobalNumCols(A);

   hypre_Vector      *x_tmp;
   int        x_size = hypre_ParVectorGlobalSize(x);
   int        y_size = hypre_ParVectorGlobalSize(y);
   int	      num_cols_offd = hypre_CSRMatrixNumCols(offd);
   int        ierr = 0;
   int	      num_sends, i, j, index, start;

   double     *x_tmp_data, *x_buf_data;
   double     *x_local_data = hypre_VectorData(x_local);
   /*---------------------------------------------------------------------
    *  Check for size compatibility.  ParMatvec returns ierr = 11 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 12 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 13 if both are true.
    *
    *  Because temporary vectors are often used in ParMatvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
 
    if (num_cols != x_size)
              ierr = 11;

    if (num_rows != y_size)
              ierr = 12;

    if (num_cols != x_size && num_rows != y_size)
              ierr = 13;

   x_tmp = hypre_VectorCreate(num_cols_offd);
   hypre_VectorInitialize(x_tmp);
   x_tmp_data = hypre_VectorData(x_tmp);
   
   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
	hypre_MatvecCommPkgCreate(A);
	comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   x_buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends));

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		x_buf_data[index++] 
		 = x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }
	
   comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, x_buf_data, 
	x_tmp_data);

   hypre_CSRMatrixMatvec( alpha, diag, x_local, beta, y_local);
   
   hypre_ParCSRCommHandleDestroy(comm_handle);
	
   if (num_cols_offd) hypre_CSRMatrixMatvec( alpha, offd, x_tmp, 1.0, y_local);    

   hypre_VectorDestroy(x_tmp);
   hypre_TFree(x_buf_data);
  
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixMatvecT( double           alpha,
                  hypre_ParCSRMatrix *A,
                  hypre_ParVector    *x,
                  double           beta,
                  hypre_ParVector    *y     )
{
   hypre_ParCSRCommHandle	*comm_handle;
   hypre_ParCSRCommPkg	*comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   hypre_Vector *y_tmp;
   double       *y_tmp_data, *y_buf_data;
   double       *y_local_data = hypre_VectorData(y_local);

   int         num_rows  = hypre_ParCSRMatrixGlobalNumRows(A);
   int         num_cols  = hypre_ParCSRMatrixGlobalNumCols(A);
   int	       num_cols_offd = hypre_CSRMatrixNumCols(offd);
   int         x_size = hypre_ParVectorGlobalSize(x);
   int         y_size = hypre_ParVectorGlobalSize(y);

   int         i, j, index, start, num_sends;

   int         ierr  = 0;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of 
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
 
    if (num_rows != x_size)
              ierr = 1;

    if (num_cols != y_size)
              ierr = 2;

    if (num_rows != x_size && num_cols != y_size)
              ierr = 3;
   /*-----------------------------------------------------------------------
    *-----------------------------------------------------------------------*/

   y_tmp = hypre_VectorCreate(num_cols_offd);
   hypre_VectorInitialize(y_tmp);

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
	hypre_MatvecCommPkgCreate(A);
	comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   y_buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends));
   y_tmp_data = hypre_VectorData(y_tmp);
   y_local_data = hypre_VectorData(y_local);

   if (num_cols_offd) hypre_CSRMatrixMatvecT(alpha, offd, x_local, 0.0, y_tmp);

   comm_handle = hypre_ParCSRCommHandleCreate( 2, comm_pkg, y_tmp_data, 
	y_buf_data);

   hypre_CSRMatrixMatvecT(alpha, diag, x_local, beta, y_local);

   hypre_ParCSRCommHandleDestroy(comm_handle);   

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		y_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
			+= y_buf_data[index++];
   }
	
   hypre_VectorDestroy(y_tmp);
   hypre_TFree(y_buf_data);

   return ierr;
}

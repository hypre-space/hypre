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
 * computes |D^-1/2 A D^-1/2 |_sup where D diagonal matrix 
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixScaledNorm
 *--------------------------------------------------------------------------*/

int
hypre_ParCSRMatrixScaledNorm( hypre_ParCSRMatrix *A, double *scnorm)
{
   hypre_ParCSRCommHandle	*comm_handle;
   hypre_ParCSRCommPkg	*comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   MPI_Comm		 comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix      *diag   = hypre_ParCSRMatrixDiag(A);
   int			*diag_i = hypre_CSRMatrixI(diag);
   int			*diag_j = hypre_CSRMatrixJ(diag);
   double		*diag_data = hypre_CSRMatrixData(diag);
   hypre_CSRMatrix      *offd   = hypre_ParCSRMatrixOffd(A);
   int			*offd_i = hypre_CSRMatrixI(offd);
   int			*offd_j = hypre_CSRMatrixJ(offd);
   double		*offd_data = hypre_CSRMatrixData(offd);
   int         		 global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   int         		 global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
   int	                *row_starts = hypre_ParCSRMatrixRowStarts(A);
   int			 num_rows = hypre_CSRMatrixNumRows(diag);
   int			 num_cols = hypre_CSRMatrixNumCols(diag);

   hypre_ParVector      *dinvsqrt;
   double		*dis_data;
   hypre_Vector      	*dis_ext;
   double 		*dis_ext_data;
   hypre_Vector         *sum;
   double		*sum_data;
  
   int	      num_cols_offd = hypre_CSRMatrixNumCols(offd);
   int	      num_sends, i, j, index, start;

   double     *d_buf_data;
   double      mat_norm, max_row_sum;

   dinvsqrt = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
   hypre_ParVectorInitialize(dinvsqrt);
   dis_data = hypre_VectorData(hypre_ParVectorLocalVector(dinvsqrt));
   hypre_ParVectorSetPartitioningOwner(dinvsqrt,0);
   dis_ext = hypre_VectorCreate(num_cols_offd);
   hypre_VectorInitialize(dis_ext);
   dis_ext_data = hypre_VectorData(dis_ext);
   sum = hypre_VectorCreate(num_rows);
   hypre_VectorInitialize(sum);
   sum_data = hypre_VectorData(sum);

   /* generate dinvsqrt */
   for (i=0; i < num_rows; i++)
   {
      dis_data[i] = 1.0/sqrt(fabs(diag_data[diag_i[i]]));
   }
   
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
   d_buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
						num_sends));

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		d_buf_data[index++] 
		 = dis_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }
	
   comm_handle = hypre_InitializeCommunication( 1, comm_pkg, d_buf_data, 
	dis_ext_data);

   for (i=0; i < num_rows; i++)
   {
      for (j=diag_i[i]; j < diag_i[i+1]; j++)
      {
	 sum_data[i] += fabs(diag_data[j])*dis_data[i]*dis_data[diag_j[j]];
      }
   }   
   hypre_FinalizeCommunication(comm_handle);

   for (i=0; i < num_rows; i++)
   {
      for (j=offd_i[i]; j < offd_i[i+1]; j++)
      {
	 sum_data[i] += fabs(offd_data[j])*dis_data[i]*dis_ext_data[offd_j[j]];
      }
   }   

   max_row_sum = 0;
   for (i=0; i < num_rows; i++)
   {
      if (max_row_sum < sum_data[i]) 
	 max_row_sum = sum_data[i];
   }	

   MPI_Allreduce(&max_row_sum, &mat_norm, 1, MPI_DOUBLE, MPI_MAX, comm);

   hypre_ParVectorDestroy(dinvsqrt);
   hypre_VectorDestroy(sum);
   hypre_VectorDestroy(dis_ext);
   hypre_TFree(d_buf_data);

   *scnorm = mat_norm;  
   return 0;
}

/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Relaxation scheme
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_ParAMGRelax
 *--------------------------------------------------------------------------*/

int  hypre_ParAMGRelax( hypre_ParCSRMatrix *A,
                        hypre_ParVector    *f,
                        int                *cf_marker,
                        int                 relax_type,
                        int                 relax_points,
                        double              relax_weight,
                        hypre_ParVector    *u,
                        hypre_ParVector    *Vtemp )
{
   MPI_Comm	   comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   double         *A_diag_data  = hypre_CSRMatrixData(A_diag);
   int            *A_diag_i     = hypre_CSRMatrixI(A_diag);
   int            *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   int            *A_offd_i     = hypre_CSRMatrixI(A_offd);
   double         *A_offd_data  = hypre_CSRMatrixData(A_offd);
   int            *A_offd_j     = hypre_CSRMatrixJ(A_offd);
   hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;

   int             n_global= hypre_ParCSRMatrixGlobalNumRows(A);
   int             n       = hypre_CSRMatrixNumRows(A_diag);
   int             num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   int	      	   first_index = hypre_ParVectorFirstIndex(u);
   
   hypre_Vector   *u_local = hypre_ParVectorLocalVector(u);
   double         *u_data  = hypre_VectorData(u_local);

   hypre_Vector   *f_local = hypre_ParVectorLocalVector(f);
   double         *f_data  = hypre_VectorData(f_local);

   hypre_Vector   *Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   double         *Vtemp_data = hypre_VectorData(Vtemp_local);
   double 	  *Vext_data;
   double 	  *v_buf_data;

   hypre_CSRMatrix *A_CSR;
   int		   *A_CSR_i;   
   int		   *A_CSR_j;
   double	   *A_CSR_data;
   
   hypre_Vector    *f_vector;
   double	   *f_vector_data;

   int             i, j, jr;
   int             ii, jj;
   int             column;
   int             relax_error = 0;
   int		   num_sends;
   int		   num_recvs;
   int		   index, start;
   int		   num_procs, my_id, ip, p;
   int		   vec_start, vec_len;
   int		   send_flag = 1;
   int		   recv_flag = 0;
   MPI_Status     *status;
   MPI_Request    *requests;

   double         *A_mat;
   double         *b_vec;

   double          zero = 0.0;
   double	   res;
   double          one_minus_weight;

   one_minus_weight = 1.0 - relax_weight;
   MPI_Comm_size(comm,&num_procs);  
   MPI_Comm_rank(comm,&my_id);  
   /*-----------------------------------------------------------------------
    * Switch statement to direct control based on relax_type:
    *     relax_type = 0 -> Jacobi or CF-Jacobi
    *     relax_type = 2 -> Jacobi (uses ParMatvec)
    *     relax_type = 1 -> Gauss-Seidel <--- very slow, sequential
    *     relax_type = 3 -> hybrid: GS-J mix off-processor, GS on-processor
    *     relax_type = 4 -> Gauss_Seidel: interior points in parallel ,
    *			 	   	  boundary sequential 
    *     relax_type = 9 -> Direct Solve
    *-----------------------------------------------------------------------*/
   
   switch (relax_type)
   {            
      case 0: /* Weighted Jacobi */
      {
   	num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   	v_buf_data = hypre_CTAlloc(double, 
			hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

	Vext_data = hypre_CTAlloc(double,num_cols_offd);
        
	if (num_cols_offd)
	{
		A_offd_j = hypre_CSRMatrixJ(A_offd);
		A_offd_data = hypre_CSRMatrixData(A_offd);
	}
 
   	index = 0;
   	for (i = 0; i < num_sends; i++)
   	{
        	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        	for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                	v_buf_data[index++] 
                 	= u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   	}
 
   	comm_handle = hypre_InitializeCommunication( 1, comm_pkg, v_buf_data, 
        	Vext_data);

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
         
         for (i = 0; i < n; i++)
         {
            Vtemp_data[i] = u_data[i];
         }
 
   	 hypre_FinalizeCommunication(comm_handle);

         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

         if (relax_points == 0)
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= one_minus_weight; 
                  u_data[i] += relax_weight * res / A_diag_data[A_diag_i[i]];
               }
            }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points 
				&& A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= one_minus_weight; 
                  u_data[i] += relax_weight * res / A_diag_data[A_diag_i[i]];
               }
            }     
         }
	 hypre_TFree(Vext_data);
	 hypre_TFree(v_buf_data);
      }
      break;

      case 2: /* Jacobi (uses ParMatvec) */
      {
 
         /*-----------------------------------------------------------------
          * Copy f into temporary vector.
          *-----------------------------------------------------------------*/
        
         hypre_ParVectorCopy(f,Vtemp); 
 
         /*-----------------------------------------------------------------
          * Perform Matvec Vtemp=f-Au
          *-----------------------------------------------------------------*/
 
            hypre_ParCSRMatrixMatvec(-1.0,A, u, 1.0, Vtemp);
            for (i = 0; i < n; i++)
            {
 
               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
           
               if (A_diag_data[A_diag_i[i]] != zero)
               {
                  u_data[i] += relax_weight * Vtemp_data[i] 
				/ A_diag_data[A_diag_i[i]];
               }
            }
      }
      break;
      
      
      case 3: /* Hybrid: Jacobi off-processor, 
                         Gauss-Seidel on-processor       */
      {
   	num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   	v_buf_data = hypre_CTAlloc(double, 
			hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

	Vext_data = hypre_CTAlloc(double,num_cols_offd);
        
	if (num_cols_offd)
	{
		A_offd_j = hypre_CSRMatrixJ(A_offd);
		A_offd_data = hypre_CSRMatrixData(A_offd);
	}
 
   	index = 0;
   	for (i = 0; i < num_sends; i++)
   	{
        	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        	for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1); j++)
                	v_buf_data[index++] 
                 	= u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   	}
 
   	comm_handle = hypre_InitializeCommunication( 1, comm_pkg, v_buf_data, 
        	Vext_data);

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
   	 hypre_FinalizeCommunication(comm_handle);

         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

         if (relax_points == 0)
         {
            for (i = 0; i < n; i++)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if ( A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
            for (i = 0; i < n; i++) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points 
				&& A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }     
         }
	 hypre_TFree(Vext_data);
	 hypre_TFree(v_buf_data);
      }
      break;

      case 1: /* Gauss-Seidel VERY SLOW */
      {
   	num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   	num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);

   	v_buf_data = hypre_CTAlloc(double, 
			hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

	Vext_data = hypre_CTAlloc(double,num_cols_offd);
        
	status = hypre_CTAlloc(MPI_Status,num_recvs);
	requests = hypre_CTAlloc(MPI_Request, num_recvs);

	if (num_cols_offd)
	{
		A_offd_j = hypre_CSRMatrixJ(A_offd);
		A_offd_data = hypre_CSRMatrixData(A_offd);
	}
 
         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
        /* 
         for (i = 0; i < n; i++)
         {
            Vtemp_data[i] = u_data[i];
         } */
 
         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/
	for (p = 0; p < num_procs; p++)
	{
	jr = 0;
	if (p != my_id)
	{
   	  for (i = 0; i < num_sends; i++)
   	  {
            ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
	    if (ip == p)
	    {
               vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	       vec_len = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1)-vec_start;
               for (j=vec_start; j < vec_start+vec_len; j++)
                  v_buf_data[j] = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
	       MPI_Isend(&v_buf_data[vec_start], vec_len, MPI_DOUBLE,
                        ip, 0, comm, &requests[jr++]);
	    }
   	  }
	  MPI_Waitall(jr,requests,status);
	  MPI_Barrier(comm);
        }
	else
        {
          for (i = 0; i < num_recvs; i++)
          {
             ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
             vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
             vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
             MPI_Irecv(&Vext_data[vec_start], vec_len, MPI_DOUBLE,
                        ip, 0, comm, &requests[jr++]);
	  }
	  MPI_Waitall(jr,requests,status);
          if (relax_points == 0)
          {
            for (i = 0; i < n; i++)	
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if ( A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points 
				&& A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }     
          }
	  MPI_Barrier(comm);
	 }
	}
	hypre_TFree(Vext_data);
	hypre_TFree(v_buf_data);
	hypre_TFree(status);
	hypre_TFree(requests);
      }
      break;

      case 4: /* Gauss-Seidel: relax interior points in parallel, boundary
				sequentially */
      {
   	num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   	num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);

   	v_buf_data = hypre_CTAlloc(double, 
			hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

	Vext_data = hypre_CTAlloc(double,num_cols_offd);
        
	status = hypre_CTAlloc(MPI_Status,num_recvs);
	requests = hypre_CTAlloc(MPI_Request, num_recvs);

	if (num_cols_offd)
	{
		A_offd_j = hypre_CSRMatrixJ(A_offd);
		A_offd_data = hypre_CSRMatrixData(A_offd);
	}
 
         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
        /* 
         for (i = 0; i < n; i++)
         {
            Vtemp_data[i] = u_data[i];
         } */
 
         /*-----------------------------------------------------------------
          * Relax interior points first
          *-----------------------------------------------------------------*/
          if (relax_points == 0)
          {
            for (i = 0; i < n; i++)	
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if ((A_offd_i[i+1]-A_offd_i[i]) == zero &&
               		A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }
          }
          else
          {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points 
               			&& (A_offd_i[i+1]-A_offd_i[i]) == zero 
				&& A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }     
          }
	for (p = 0; p < num_procs; p++)
	{
	jr = 0;
	if (p != my_id)
	{
   	  for (i = 0; i < num_sends; i++)
   	  {
            ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
	    if (ip == p)
	    {
               vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	       vec_len = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1)-vec_start;
               for (j=vec_start; j < vec_start+vec_len; j++)
                  v_buf_data[j] = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
	       MPI_Isend(&v_buf_data[vec_start], vec_len, MPI_DOUBLE,
                        ip, 0, comm, &requests[jr++]);
	    }
   	  }
	  MPI_Waitall(jr,requests,status);
	  MPI_Barrier(comm);
        }
	else
        {
          for (i = 0; i < num_recvs; i++)
          {
             ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
             vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
             vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
             MPI_Irecv(&Vext_data[vec_start], vec_len, MPI_DOUBLE,
                        ip, 0, comm, &requests[jr++]);
	  }
	  MPI_Waitall(jr,requests,status);
          if (relax_points == 0)
          {
            for (i = 0; i < n; i++)	
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if ((A_offd_i[i+1]-A_offd_i[i]) != zero &&
               		A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
            for (i = 0; i < n; i++)
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points 
               			&& (A_offd_i[i+1]-A_offd_i[i]) != zero 
				&& A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }     
          }
	  MPI_Barrier(comm);
	 }
	}
	hypre_TFree(Vext_data);
	hypre_TFree(v_buf_data);
	hypre_TFree(status);
	hypre_TFree(requests);
      }
      break;

      case 9: /* Direct solve: use gaussian elimination */
      {

         /*-----------------------------------------------------------------
          *  Generate CSR matrix from ParCSRMatrix A
          *-----------------------------------------------------------------*/

	 if (n)
	 {
	    A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
	    f_vector = hypre_ParVectorToVectorAll(f);
 	    A_CSR_i = hypre_CSRMatrixI(A_CSR);
 	    A_CSR_j = hypre_CSRMatrixJ(A_CSR);
 	    A_CSR_data = hypre_CSRMatrixData(A_CSR);
   	    f_vector_data = hypre_VectorData(f_vector);

            A_mat = hypre_CTAlloc(double, n_global*n_global);
            b_vec = hypre_CTAlloc(double, n_global);    

            /*---------------------------------------------------------------
             *  Load CSR matrix into A_mat.
             *---------------------------------------------------------------*/

            for (i = 0; i < n_global; i++)
            {
               for (jj = A_CSR_i[i]; jj < A_CSR_i[i+1]; jj++)
               {
                  column = A_CSR_j[jj];
                  A_mat[i*n_global+column] = A_CSR_data[jj];
               }
               b_vec[i] = f_vector_data[i];
            }

            relax_error = gselim(A_mat,b_vec,n_global);

            for (i = 0; i < n; i++)
            {
               u_data[i] = b_vec[first_index+i];
            }

	    hypre_TFree(A_mat); 
            hypre_TFree(b_vec);
            hypre_CSRMatrixDestroy(A_CSR);
            hypre_VectorDestroy(f_vector);
         
         }
      }
      break;   
   }

   return(relax_error); 
}

/*-------------------------------------------------------------------------
 *
 *                      Gaussian Elimination
 *
 *------------------------------------------------------------------------ */

int gselim(A,x,n)
double *A;
double *x;
int n;
{
   int    err_flag = 0;
   int    j,k,m;
   double factor;
   
   if (n==1)                           /* A is 1x1 */  
   {
      if (A[0] != 0.0)
      {
         x[0] = x[0]/A[0];
         return(err_flag);
      }
      else
      {
         err_flag = 1;
         return(err_flag);
      }
   }
   else                               /* A is nxn.  Forward elimination */ 
   {
      for (k = 0; k < n-1; k++)
      {
          if (A[k*n+k] != 0.0)
          {          
             for (j = k+1; j < n; j++)
             {
                 if (A[j*n+k] != 0.0)
                 {
                    factor = A[j*n+k]/A[k*n+k];
                    for (m = k+1; m < n; m++)
                    {
                        A[j*n+m]  -= factor * A[k*n+m];
                    }
                                     /* Elimination step for rhs */ 
                    x[j] -= factor * x[k];              
                 }
             }
          }
       }
                                    /* Back Substitution  */
       for (k = n-1; k > 0; --k)
       {
           x[k] /= A[k*n+k];
           for (j = 0; j < k; j++)
           {
               if (A[j*n+k] != 0.0)
               {
                  x[j] -= x[k] * A[j*n+k];
               }
           }
       }
       x[0] /= A[0];
       return(err_flag);
    }
}
 

         



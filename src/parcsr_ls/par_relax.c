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
 * Relaxation scheme
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "Common.h"

#ifdef HYPRE_USING_ESSL
#include <essl.h>
#else
/* RDF: This needs to be integrated with the hypre blas/lapack stuff */
#ifdef __cplusplus
extern "C" {
#endif
HYPRE_Int hypre_F90_NAME_LAPACK(dgetrf, DGETRF) (HYPRE_Int *, HYPRE_Int *, HYPRE_Real *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *);
HYPRE_Int hypre_F90_NAME_LAPACK(dgetrs, DGETRS) (const char *, HYPRE_Int *, HYPRE_Int *, HYPRE_Real *, HYPRE_Int *, HYPRE_Int *, HYPRE_Real *b, HYPRE_Int*, HYPRE_Int *);
#ifdef __cplusplus
}
#endif
#endif

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int  hypre_BoomerAMGRelax( hypre_ParCSRMatrix *A,
                           hypre_ParVector    *f,
                           HYPRE_Int                *cf_marker,
                           HYPRE_Int                 relax_type,
                           HYPRE_Int                 relax_points,
                           HYPRE_Real          relax_weight,
                           HYPRE_Real          omega,
                           HYPRE_Real         *l1_norms,
                           hypre_ParVector    *u,
                           hypre_ParVector    *Vtemp,
                           hypre_ParVector    *Ztemp )
{
   MPI_Comm	   comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real     *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int            *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int            *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int            *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Real     *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int            *A_offd_j     = hypre_CSRMatrixJ(A_offd);
   hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;

   HYPRE_Int             n_global= hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int             n       = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int             num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int	      	   first_index = hypre_ParVectorFirstIndex(u);
   
   hypre_Vector   *u_local = hypre_ParVectorLocalVector(u);
   HYPRE_Real     *u_data  = hypre_VectorData(u_local);

   hypre_Vector   *f_local = hypre_ParVectorLocalVector(f);
   HYPRE_Real     *f_data  = hypre_VectorData(f_local);

   hypre_Vector   *Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   HYPRE_Real     *Vtemp_data = hypre_VectorData(Vtemp_local);
   HYPRE_Real 	  *Vext_data = NULL;
   HYPRE_Real 	  *v_buf_data;
   HYPRE_Real 	  *tmp_data;

   hypre_Vector   *Ztemp_local;
   HYPRE_Real     *Ztemp_data;

   hypre_CSRMatrix *A_CSR;
   HYPRE_Int		   *A_CSR_i;   
   HYPRE_Int		   *A_CSR_j;
   HYPRE_Real	   *A_CSR_data;
   
   hypre_Vector    *f_vector;
   HYPRE_Real	   *f_vector_data;

   HYPRE_Int             i, j, jr;
   HYPRE_Int             ii, jj;
   HYPRE_Int             ns, ne, size, rest;
   HYPRE_Int             column;
   HYPRE_Int             relax_error = 0;
   HYPRE_Int		   num_sends;
   HYPRE_Int		   num_recvs;
   HYPRE_Int		   index, start;
   HYPRE_Int		   num_procs, num_threads, my_id, ip, p;
   HYPRE_Int		   vec_start, vec_len;
   hypre_MPI_Status     *status;
   hypre_MPI_Request    *requests;

   HYPRE_Real     *A_mat;
   HYPRE_Real     *b_vec;

   HYPRE_Real      zero = 0.0;
   HYPRE_Real	   res, res0, res2;
   HYPRE_Real      one_minus_weight;
   HYPRE_Real      one_minus_omega;
   HYPRE_Real      prod;

   one_minus_weight = 1.0 - relax_weight;
   one_minus_omega = 1.0 - omega;
   hypre_MPI_Comm_size(comm,&num_procs);  
   hypre_MPI_Comm_rank(comm,&my_id);  
   num_threads = hypre_NumThreads();
   /*-----------------------------------------------------------------------
    * Switch statement to direct control based on relax_type:
    *     relax_type = 0 -> Jacobi or CF-Jacobi
    *     relax_type = 1 -> Gauss-Seidel <--- very slow, sequential
    *     relax_type = 2 -> Gauss_Seidel: interior points in parallel ,
    *			 	   	  boundary sequential 
    *     relax_type = 3 -> hybrid: SOR-J mix off-processor, SOR on-processor
    *     		    with outer relaxation parameters (forward solve)
    *     relax_type = 4 -> hybrid: SOR-J mix off-processor, SOR on-processor
    *     		    with outer relaxation parameters (backward solve)
    *     relax_type = 5 -> hybrid: GS-J mix off-processor, chaotic GS on-node
    *     relax_type = 6 -> hybrid: SSOR-J mix off-processor, SSOR on-processor
    *     		    with outer relaxation parameters 
    *     relax_type = 7 -> Jacobi (uses Matvec), only needed in CGNR
    *     relax_type = 19-> Direct Solve, (old version)
    *     relax_type = 29-> Direct solve: use gaussian elimination & BLAS 
    *			    (with pivoting) (old version)
    *-----------------------------------------------------------------------*/
   switch (relax_type)
   {            
      case 0: /* Weighted Jacobi */
      {
	if (num_procs > 1)
	{
   	num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   	v_buf_data = hypre_CTAlloc(HYPRE_Real, 
			hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

	Vext_data = hypre_CTAlloc(HYPRE_Real,num_cols_offd);
        
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
 
   	comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, 
        	Vext_data);
	}
         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
         
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < n; i++)
         {
            Vtemp_data[i] = u_data[i];
         }
	 if (num_procs > 1)
	 { 
   	 hypre_ParCSRCommHandleDestroy(comm_handle);
         comm_handle = NULL;
	 } 

         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

         if (relax_points == 0)
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,res) HYPRE_SMP_SCHEDULE
#endif
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
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,res) HYPRE_SMP_SCHEDULE
#endif
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
	 if (num_procs > 1)
         {
	 hypre_TFree(Vext_data);
	 hypre_TFree(v_buf_data);
         }
      }
      break;

      case 5: /* Hybrid: Jacobi off-processor, 
                         chaotic Gauss-Seidel on-processor       */
      {
	if (num_procs > 1)
	{
   	num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   	v_buf_data = hypre_CTAlloc(HYPRE_Real, 
			hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

	Vext_data = hypre_CTAlloc(HYPRE_Real,num_cols_offd);
        
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
 
   	comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, 
        	Vext_data);

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
   	 hypre_ParCSRCommHandleDestroy(comm_handle);
         comm_handle = NULL;
	}

         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

         if (relax_points == 0)
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,res) HYPRE_SMP_SCHEDULE
#endif
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
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,res) HYPRE_SMP_SCHEDULE
#endif
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
         if (num_procs > 1)
         {
	   hypre_TFree(Vext_data);
	   hypre_TFree(v_buf_data);
         }
      }
      break;

      case 3: /* Hybrid: Jacobi off-processor, 
                         Gauss-Seidel on-processor       
                         (forward loop) */
      {

         if (num_threads > 1)
         {
            Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
            Ztemp_data = hypre_VectorData(Ztemp_local);
         }
         
#ifdef HYPRE_USING_PERSISTENT_COMM
         // JSP: persistent comm can be similarly used for other smoothers
         hypre_ParCSRPersistentCommHandle *persistent_comm_handle;
#endif
         
         if (num_procs > 1)
         {
#ifdef HYPRE_PROFILE
            hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
            
#ifdef HYPRE_USING_PERSISTENT_COMM
            persistent_comm_handle = hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);
            v_buf_data = (HYPRE_Real *)persistent_comm_handle->send_data;
            Vext_data = (HYPRE_Real *)persistent_comm_handle->recv_data;
#else
            v_buf_data = hypre_CTAlloc(HYPRE_Real, 
                                       hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
            
            Vext_data = hypre_CTAlloc(HYPRE_Real,num_cols_offd);
#endif
            
            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRMatrixJ(A_offd);
               A_offd_data = hypre_CSRMatrixData(A_offd);
            }

            HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
            HYPRE_Int end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
            for (i = begin; i < end; i++)
            {
               v_buf_data[i - begin]
                  = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)];
            }
            
#ifdef HYPRE_PROFILE
            hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
            hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

#ifdef HYPRE_USING_PERSISTENT_COMM
            hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle);
#else
            comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, 
                                                        Vext_data);
#endif
            
            /*-----------------------------------------------------------------
             * Copy current approximation into temporary vector.
             *-----------------------------------------------------------------*/
#ifdef HYPRE_USING_PERSISTENT_COMM
            hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle);
#else
            hypre_ParCSRCommHandleDestroy(comm_handle);
#endif
            comm_handle = NULL;

#ifdef HYPRE_PROFILE
            hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif
         }

        /*-----------------------------------------------------------------
         * Relax all points.
         *-----------------------------------------------------------------*/
#ifdef HYPRE_PROFILE
        hypre_profile_times[HYPRE_TIMER_ID_RELAX] -= hypre_MPI_Wtime();
#endif

	if (relax_weight == 1 && omega == 1)
        {
         if (relax_points == 0)
         {
	  if (num_threads > 1)
          {
	   tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
	   {
	    size = n/num_threads;
	    rest = n - size*num_threads;
	    if (j < rest)
	    {
	       ns = j*size+j;
	       ne = (j+1)*size+j+1;
	    }
	    else
	    {
	       ns = j*size+rest;
	       ne = (j+1)*size+rest;
	    }
            for (i = ns; i < ne; i++)	/* interior points first */
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
		     if (ii >= ns && ii < ne)
                        res -= A_diag_data[jj] * u_data[ii];
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
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

          }
	  else
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
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
	  if (num_threads > 1)
	  {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
	   {
	    size = n/num_threads;
	    rest = n - size*num_threads;
	    if (j < rest)
	    {
	       ns = j*size+j;
	       ne = (j+1)*size+j+1;
	    }
	    else
	    {
	       ns = j*size+rest;
	       ne = (j+1)*size+rest;
	    }
            for (i = ns; i < ne; i++) /* relax interior points */
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
		     if (ii >= ns && ii < ne)
                        res -= A_diag_data[jj] * u_data[ii];
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
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

	  }
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
         }
        }
	else
        {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < n; i++)
         {
            Vtemp_data[i] = u_data[i];
         }
         prod = (1.0-relax_weight*omega);
         if (relax_points == 0)
         {
	  if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
	   {
	    size = n/num_threads;
	    rest = n - size*num_threads;
	    if (j < rest)
	    {
	       ns = j*size+j;
	       ne = (j+1)*size+j+1;
	    }
	    else
	    {
	       ns = j*size+rest;
	       ne = (j+1)*size+rest;
	    }
            for (i = ns; i < ne; i++)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if ( A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  res0 = 0.0;
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
		     if (ii >= ns && ii < ne)
		     {
                        res0 -= A_diag_data[jj] * u_data[ii];
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
		     }
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }
           }

          }
	  else
          {
            for (i = 0; i < n; i++)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if ( A_diag_data[A_diag_i[i]] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }
          }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
	  if (num_threads > 1)
	  {
             tmp_data = Ztemp_data;

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
	   {
	    size = n/num_threads;
	    rest = n - size*num_threads;
	    if (j < rest)
	    {
	       ns = j*size+j;
	       ne = (j+1)*size+j+1;
	    }
	    else
	    {
	       ns = j*size+rest;
	       ne = (j+1)*size+rest;
	    }
            for (i = ns; i < ne; i++) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points 
				&& A_diag_data[A_diag_i[i]] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
		     if (ii >= ns && ii < ne)
		     {
                        res0 -= A_diag_data[jj] * u_data[ii];
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
		     }
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }     
           }     

           
	  }
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
                  res0 = 0.0;
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }     
	  }
         }
        }
#ifndef HYPRE_USING_PERSISTENT_COMM
        if (num_procs > 1)
        {
	   hypre_TFree(Vext_data);
	   hypre_TFree(v_buf_data);
        }
#endif
#ifdef HYPRE_PROFILE
        hypre_profile_times[HYPRE_TIMER_ID_RELAX] += hypre_MPI_Wtime();
#endif
      }
      break;

      case 1: /* Gauss-Seidel VERY SLOW */
      {
        if (num_procs > 1)
        {
   	num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   	num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);

   	v_buf_data = hypre_CTAlloc(HYPRE_Real, 
			hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

	Vext_data = hypre_CTAlloc(HYPRE_Real,num_cols_offd);
        
	status  = hypre_CTAlloc(hypre_MPI_Status,num_recvs+num_sends);
	requests= hypre_CTAlloc(hypre_MPI_Request, num_recvs+num_sends);

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
 
        } 
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
	       hypre_MPI_Isend(&v_buf_data[vec_start], vec_len, HYPRE_MPI_REAL,
                        ip, 0, comm, &requests[jr++]);
	    }
   	  }
	  hypre_MPI_Waitall(jr,requests,status);
	  hypre_MPI_Barrier(comm);
        }
	else
        {
          if (num_procs > 1)
	  {
	  for (i = 0; i < num_recvs; i++)
          {
             ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
             vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
             vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
             hypre_MPI_Irecv(&Vext_data[vec_start], vec_len, HYPRE_MPI_REAL,
                        ip, 0, comm, &requests[jr++]);
	  }
	  hypre_MPI_Waitall(jr,requests,status);
	  }
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
	  if (num_procs > 1)
	  hypre_MPI_Barrier(comm);
	 }
	}
	if (num_procs > 1)
	{
	hypre_TFree(Vext_data);
	hypre_TFree(v_buf_data);
	hypre_TFree(status);
	hypre_TFree(requests);
	}
      }
      break;

      case 2: /* Gauss-Seidel: relax interior points in parallel, boundary
				sequentially */
      {
	if (num_procs > 1)
	{
   	num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   	num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);

   	v_buf_data = hypre_CTAlloc(HYPRE_Real, 
			hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

	Vext_data = hypre_CTAlloc(HYPRE_Real,num_cols_offd);
        
	status  = hypre_CTAlloc(hypre_MPI_Status,num_recvs+num_sends);
	requests= hypre_CTAlloc(hypre_MPI_Request, num_recvs+num_sends);

	if (num_cols_offd)
	{
		A_offd_j = hypre_CSRMatrixJ(A_offd);
		A_offd_data = hypre_CSRMatrixData(A_offd);
	}
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
	       hypre_MPI_Isend(&v_buf_data[vec_start], vec_len, HYPRE_MPI_REAL,
                        ip, 0, comm, &requests[jr++]);
	    }
   	  }
	  hypre_MPI_Waitall(jr,requests,status);
	  hypre_MPI_Barrier(comm);
        }
	else
        {
	  if (num_procs > 1)
  	  {
          for (i = 0; i < num_recvs; i++)
          {
             ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
             vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
             vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
             hypre_MPI_Irecv(&Vext_data[vec_start], vec_len, HYPRE_MPI_REAL,
                        ip, 0, comm, &requests[jr++]);
	  }
	  hypre_MPI_Waitall(jr,requests,status);
	  }
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
	  if (num_procs > 1)
	  hypre_MPI_Barrier(comm);
	 }
	}
	if (num_procs > 1)
	{
	hypre_TFree(Vext_data);
	hypre_TFree(v_buf_data);
	hypre_TFree(status);
	hypre_TFree(requests);
	}
      }
      break;

      case 4: /* Hybrid: Jacobi off-processor, 
                         Gauss-Seidel/SOR on-processor 
                         (backward loop) */
      {
	if (num_procs > 1)
	{
   	num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   	v_buf_data = hypre_CTAlloc(HYPRE_Real, 
			hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

	Vext_data = hypre_CTAlloc(HYPRE_Real,num_cols_offd);
        
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
 
   	comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, 
        	Vext_data);

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
   	 hypre_ParCSRCommHandleDestroy(comm_handle);
         comm_handle = NULL;
	}

         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

	 if (relax_weight == 1 && omega == 1)
         {
         if (relax_points == 0)
         {
	  if (num_threads > 1)
          {
	   tmp_data = hypre_CTAlloc(HYPRE_Real,n);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
	   {
	    size = n/num_threads;
	    rest = n - size*num_threads;
	    if (j < rest)
	    {
	       ns = j*size+j;
	       ne = (j+1)*size+j+1;
	    }
	    else
	    {
	       ns = j*size+rest;
	       ne = (j+1)*size+rest;
	    }
            for (i = ne-1; i > ns-1; i--)	/* interior points first */
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
		     if (ii >= ns && ii < ne)
                        res -= A_diag_data[jj] * u_data[ii];
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
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
           hypre_TFree(tmp_data);
          }
	  else
          {
            for (i = n-1; i > -1; i--)	/* interior points first */
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
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
	  if (num_threads > 1)
	  {
	   tmp_data = hypre_CTAlloc(HYPRE_Real,n);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
	   {
	    size = n/num_threads;
	    rest = n - size*num_threads;
	    if (j < rest)
	    {
	       ns = j*size+j;
	       ne = (j+1)*size+j+1;
	    }
	    else
	    {
	       ns = j*size+rest;
	       ne = (j+1)*size+rest;
	    }
            for (i = ne-1; i > ns-1; i--) /* relax interior points */
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
		     if (ii >= ns && ii < ne)
                        res -= A_diag_data[jj] * u_data[ii];
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
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
           hypre_TFree(tmp_data);
           
	  }
	  else
	  {
            for (i = n-1; i > -1; i--) /* relax interior points */
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
         }
         }
	 else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < n; i++)
         {
            Vtemp_data[i] = u_data[i];
         }
         prod = (1.0-relax_weight*omega);
         if (relax_points == 0)
         {
	  if (num_threads > 1)
          {
	   tmp_data = hypre_CTAlloc(HYPRE_Real,n);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
	   {
	    size = n/num_threads;
	    rest = n - size*num_threads;
	    if (j < rest)
	    {
	       ns = j*size+j;
	       ne = (j+1)*size+j+1;
	    }
	    else
	    {
	       ns = j*size+rest;
	       ne = (j+1)*size+rest;
	    }
            for (i = ne-1; i > ns-1; i--)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if ( A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  res0 = 0.0;
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
		     if (ii >= ns && ii < ne)
		     {
                        res0 -= A_diag_data[jj] * u_data[ii];
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
		     }
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }
           }
           hypre_TFree(tmp_data);
           
          }
	  else
          {
            for (i = n-1; i > -1; i--)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if ( A_diag_data[A_diag_i[i]] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }
          }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
	  if (num_threads > 1)
	  {
	   tmp_data = hypre_CTAlloc(HYPRE_Real,n);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
	   {
	    size = n/num_threads;
	    rest = n - size*num_threads;
	    if (j < rest)
	    {
	       ns = j*size+j;
	       ne = (j+1)*size+j+1;
	    }
	    else
	    {
	       ns = j*size+rest;
	       ne = (j+1)*size+rest;
	    }
            for (i = ne-1; i > ns-1; i--) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points 
				&& A_diag_data[A_diag_i[i]] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
		     if (ii >= ns && ii < ne)
		     {
                        res0 -= A_diag_data[jj] * u_data[ii];
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
		     }
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }     
           }     
           hypre_TFree(tmp_data);
	  }
	  else
	  {
            for (i = n-1; i > -1; i--) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
      
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points 
				&& A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  res0 = 0.0;
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }     
	  }
         }
         }
         if (num_procs > 1)
         {
	   hypre_TFree(Vext_data);
	   hypre_TFree(v_buf_data);
         }
      }
      break;

      case 6: /* Hybrid: Jacobi off-processor, 
                         Symm. Gauss-Seidel/ SSOR on-processor
			with outer relaxation parameter */
      {

         if (num_threads > 1)
         {
            Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
            Ztemp_data = hypre_VectorData(Ztemp_local);
         }
         
         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
	if (num_procs > 1)
	{
   	num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   	v_buf_data = hypre_CTAlloc(HYPRE_Real, 
			hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

	Vext_data = hypre_CTAlloc(HYPRE_Real,num_cols_offd);
        
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
 
   	comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, 
        	Vext_data);

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
   	 hypre_ParCSRCommHandleDestroy(comm_handle);
         comm_handle = NULL;
	}

        /*-----------------------------------------------------------------
         * Relax all points.
         *-----------------------------------------------------------------*/

	if (relax_weight == 1 && omega == 1)
        {
         if (relax_points == 0)
         {
	  if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
	   {
	    size = n/num_threads;
	    rest = n - size*num_threads;
	    if (j < rest)
	    {
	       ns = j*size+j;
	       ne = (j+1)*size+j+1;
	    }
	    else
	    {
	       ns = j*size+rest;
	       ne = (j+1)*size+rest;
	    }
            for (i = ns; i < ne; i++)	/* interior points first */
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
		     if (ii >= ns && ii < ne)
		     {
                        res -= A_diag_data[jj] * u_data[ii];
		     }
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }
            for (i = ne-1; i > ns-1; i--)	/* interior points first */
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
		     if (ii >= ns && ii < ne)
		     {
                        res -= A_diag_data[jj] * u_data[ii];
		     }
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
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

          }
	  else
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
            for (i = n-1; i > -1; i--)	/* interior points first */
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
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
	  if (num_threads > 1)
	  {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
	   {
	    size = n/num_threads;
	    rest = n - size*num_threads;
	    if (j < rest)
	    {
	       ns = j*size+j;
	       ne = (j+1)*size+j+1;
	    }
	    else
	    {
	       ns = j*size+rest;
	       ne = (j+1)*size+rest;
	    }
            for (i = ns; i < ne; i++) /* relax interior points */
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
		     if (ii >= ns && ii < ne)
		     {
                        res -= A_diag_data[jj] * u_data[ii];
		     }
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }     
            for (i = ne-1; i > ns-1; i--) /* relax interior points */
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
		     if (ii >= ns && ii < ne)
		     {
                        res -= A_diag_data[jj] * u_data[ii];
		     }
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
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

	  }
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
            for (i = n-1; i > -1; i--) /* relax interior points */
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
         }
        }
        else
        {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < n; i++)
         {
            Vtemp_data[i] = u_data[i];
         }
         prod = (1.0-relax_weight*omega);
         if (relax_points == 0)
         {
	  if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
	   {
	    size = n/num_threads;
	    rest = n - size*num_threads;
	    if (j < rest)
	    {
	       ns = j*size+j;
	       ne = (j+1)*size+j+1;
	    }
	    else
	    {
	       ns = j*size+rest;
	       ne = (j+1)*size+rest;
	    }
            for (i = ns; i < ne; i++)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if ( A_diag_data[A_diag_i[i]] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
		     if (ii >= ns && ii < ne)
		     {
                        res0 -= A_diag_data[jj] * u_data[ii];
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
		     }
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }
            for (i = ne-1; i > ns-1; i--)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if ( A_diag_data[A_diag_i[i]] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
		     if (ii >= ns && ii < ne)
		     {
                        res0 -= A_diag_data[jj] * u_data[ii];
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
		     }
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }
           }

          }
	  else
          {
            for (i = 0; i < n; i++)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if ( A_diag_data[A_diag_i[i]] != zero)
               {
                  res0 = 0.0;
                  res = f_data[i];
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }
            for (i = n-1; i > -1; i--)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if ( A_diag_data[A_diag_i[i]] != zero)
               {
                  res0 = 0.0;
                  res = f_data[i];
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }
          }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
	  if (num_threads > 1)
	  {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
	   {
	    size = n/num_threads;
	    rest = n - size*num_threads;
	    if (j < rest)
	    {
	       ns = j*size+j;
	       ne = (j+1)*size+j+1;
	    }
	    else
	    {
	       ns = j*size+rest;
	       ne = (j+1)*size+rest;
	    }
            for (i = ns; i < ne; i++) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points 
				&& A_diag_data[A_diag_i[i]] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
		     if (ii >= ns && ii < ne)
		     {
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
                        res0 -= A_diag_data[jj] * u_data[ii];
		     }
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }     
            for (i = ne-1; i > ns-1; i--) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points 
				&& A_diag_data[A_diag_i[i]] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
		     if (ii >= ns && ii < ne)
		     {
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
                        res0 -= A_diag_data[jj] * u_data[ii];
		     }
		     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }     
           }     

	  }
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
                  res0 = 0.0;
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }     
            for (i = n-1; i > -1; i--) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
      
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
             
               if (cf_marker[i] == relax_points 
				&& A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  res0 = 0.0;
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
			one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
			one_minus_weight*res2) / A_diag_data[A_diag_i[i]];*/
               }
            }     
	  }
         }
        }
        if (num_procs > 1)
        {
	   hypre_TFree(Vext_data);
	   hypre_TFree(v_buf_data);
        }
      }
      break;

      case 7: /* Jacobi (uses ParMatvec) */
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

      case 8: /* hybrid L1 Symm. Gauss-Seidel */
      {

         if (num_threads > 1)
         {
            Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
            Ztemp_data = hypre_VectorData(Ztemp_local);
         }

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
        if (num_procs > 1)
        {
        num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

        v_buf_data = hypre_CTAlloc(HYPRE_Real,
                        hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

        Vext_data = hypre_CTAlloc(HYPRE_Real,num_cols_offd);

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

        comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data,
                Vext_data);

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
         hypre_ParCSRCommHandleDestroy(comm_handle);
         comm_handle = NULL;
        }

        /*-----------------------------------------------------------------
         * Relax all points.
         *-----------------------------------------------------------------*/

        if (relax_weight == 1 && omega == 1)
        {
         if (relax_points == 0)
         {
          if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
           {
            size = n/num_threads;
            rest = n - size*num_threads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ns; i < ne; i++)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res -= A_diag_data[jj] * u_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
            for (i = ne-1; i > ns-1; i--)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res -= A_diag_data[jj] * u_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
           }

          }
          else
          {
            for (i = 0; i < n; i++)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
            for (i = n-1; i > -1; i--)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
          }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
          if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
           {
            size = n/num_threads;
            rest = n - size*num_threads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ns; i < ne; i++) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res -= A_diag_data[jj] * u_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
            for (i = ne-1; i > ns-1; i--) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res -= A_diag_data[jj] * u_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
           }

          }
          else
          {
            for (i = 0; i < n; i++) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is

                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
            for (i = n-1; i > -1; i--) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is

                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
          }
         }
        }
        else
        {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < n; i++)
         {
            Vtemp_data[i] = u_data[i];
         }
         prod = (1.0-relax_weight*omega);
         if (relax_points == 0)
         {
          if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
           {
            size = n/num_threads;
            rest = n - size*num_threads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ns; i < ne; i++)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res0 -= A_diag_data[jj] * u_data[ii];
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
            for (i = ne-1; i > ns-1; i--)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res0 -= A_diag_data[jj] * u_data[ii];
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
           }

          }
          else
          {
            for (i = 0; i < n; i++)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res0 = 0.0;
                  res = f_data[i];
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
            for (i = n-1; i > -1; i--)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res0 = 0.0;
                  res = f_data[i];
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
          }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
          if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
           {
            size = n/num_threads;
            rest = n - size*num_threads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ns; i < ne; i++) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
                        res0 -= A_diag_data[jj] * u_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
            for (i = ne-1; i > ns-1; i--) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
                        res0 -= A_diag_data[jj] * u_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
           }

          }
          else
          {
            for (i = 0; i < n; i++) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is

                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res = f_data[i];
                  res0 = 0.0;
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
            for (i = n-1; i > -1; i--) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is

                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res = f_data[i];
                  res0 = 0.0;
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
          }
         }
        }
        if (num_procs > 1)
        {
           hypre_TFree(Vext_data);
           hypre_TFree(v_buf_data);
        }
      }
      break;

      case 13: /* hybrid L1 Gauss-Seidel forward solve */
      {

         if (num_threads > 1)
         {
            Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
            Ztemp_data = hypre_VectorData(Ztemp_local);
         }

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
        if (num_procs > 1)
        {
         num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

         v_buf_data = hypre_CTAlloc(HYPRE_Real,
                        hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

         Vext_data = hypre_CTAlloc(HYPRE_Real,num_cols_offd);

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

         comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data,
                Vext_data);

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
         hypre_ParCSRCommHandleDestroy(comm_handle);
         comm_handle = NULL;
        }

        /*-----------------------------------------------------------------
         * Relax all points.
         *-----------------------------------------------------------------*/

        if (relax_weight == 1 && omega == 1)
        {
         if (relax_points == 0)
         {
          if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
           {
            size = n/num_threads;
            rest = n - size*num_threads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ns; i < ne; i++)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res -= A_diag_data[jj] * u_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
           }

          }
          else
          {
            for (i = 0; i < n; i++)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
          }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
          if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
           {
            size = n/num_threads;
            rest = n - size*num_threads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ns; i < ne; i++) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res -= A_diag_data[jj] * u_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
           }

          }
          else
          {
            for (i = 0; i < n; i++) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is

                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
          }
         }
        }
        else
        {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < n; i++)
         {
            Vtemp_data[i] = u_data[i];
         }
         prod = (1.0-relax_weight*omega);
         if (relax_points == 0)
         {
          if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
           {
            size = n/num_threads;
            rest = n - size*num_threads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ns; i < ne; i++)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res0 -= A_diag_data[jj] * u_data[ii];
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
           }

          }
          else
          {
            for (i = 0; i < n; i++)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res0 = 0.0;
                  res = f_data[i];
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
          }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
          if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
           {
            size = n/num_threads;
            rest = n - size*num_threads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ns; i < ne; i++) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
                        res0 -= A_diag_data[jj] * u_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
           }

          }
          else
          {
            for (i = 0; i < n; i++) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is

                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res = f_data[i];
                  res0 = 0.0;
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
          }
         }
        }
        if (num_procs > 1)
        {
           hypre_TFree(Vext_data);
           hypre_TFree(v_buf_data);
        }
      }
      break;

      case 14: /* hybrid L1 Gauss-Seidel backward solve */
      {

         if (num_threads > 1)
         {
            Ztemp_local = hypre_ParVectorLocalVector(Ztemp);
            Ztemp_data = hypre_VectorData(Ztemp_local);
         }

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
        if (num_procs > 1)
        {
        num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

        v_buf_data = hypre_CTAlloc(HYPRE_Real,
                        hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

        Vext_data = hypre_CTAlloc(HYPRE_Real,num_cols_offd);

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

        comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data,
                Vext_data);

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
         hypre_ParCSRCommHandleDestroy(comm_handle);
         comm_handle = NULL;
        }

        /*-----------------------------------------------------------------
         * Relax all points.
         *-----------------------------------------------------------------*/

        if (relax_weight == 1 && omega == 1)
        {
         if (relax_points == 0)
         {
          if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
           {
            size = n/num_threads;
            rest = n - size*num_threads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ne-1; i > ns-1; i--)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res -= A_diag_data[jj] * u_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
           }

          }
          else
          {
            for (i = n-1; i > -1; i--)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
          }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
          if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
           {
            size = n/num_threads;
            rest = n - size*num_threads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ne-1; i > ns-1; i--) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res -= A_diag_data[jj] * u_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
           }

          }
          else
          {
            for (i = n-1; i > -1; i--) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is

                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] += res / l1_norms[i];
               }
            }
          }
         }
        }
        else
        {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < n; i++)
         {
            Vtemp_data[i] = u_data[i];
         }
         prod = (1.0-relax_weight*omega);
         if (relax_points == 0)
         {
          if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
           {
            size = n/num_threads;
            rest = n - size*num_threads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ne-1; i > ns-1; i--)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res0 -= A_diag_data[jj] * u_data[ii];
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
           }

          }
          else
          {
            for (i = n-1; i > -1; i--)	/* interior points first */
            {

               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if ( l1_norms[i] != zero)
               {
                  res0 = 0.0;
                  res = f_data[i];
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
          }
         }

         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/

         else
         {
          if (num_threads > 1)
          {
             tmp_data = Ztemp_data;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,ns,ne,res,rest,size) HYPRE_SMP_SCHEDULE
#endif
           for (j = 0; j < num_threads; j++)
           {
            size = n/num_threads;
            rest = n - size*num_threads;
            if (j < rest)
            {
               ns = j*size+j;
               ne = (j+1)*size+j+1;
            }
            else
            {
               ns = j*size+rest;
               ne = (j+1)*size+rest;
            }
            for (i = ne-1; i > ns-1; i--) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res0 = 0.0;
                  res2 = 0.0;
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     if (ii >= ns && ii < ne)
                     {
                        res2 += A_diag_data[jj] * Vtemp_data[ii];
                        res0 -= A_diag_data[jj] * u_data[ii];
                     }
                     else
                        res -= A_diag_data[jj] * tmp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
           }

          }
          else
          {
            for (i = n-1; i > -1; i--) /* relax interior points */
            {

               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is

                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/

               if (cf_marker[i] == relax_points
                                && l1_norms[i] != zero)
               {
                  res = f_data[i];
                  res0 = 0.0;
                  res2 = 0.0;
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     res0 -= A_diag_data[jj] * u_data[ii];
                     res2 += A_diag_data[jj] * Vtemp_data[ii];
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     res -= A_offd_data[jj] * Vext_data[ii];
                  }
                  u_data[i] *= prod;
                  u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / l1_norms[i];
                  /*u_data[i] += omega*(relax_weight*res + res0 +
                        one_minus_weight*res2) / l1_norms[i];*/
               }
            }
          }
         }
        }
        if (num_procs > 1)
        {
           hypre_TFree(Vext_data);
           hypre_TFree(v_buf_data);
        }
      }
      break;

      case 19: /* Direct solve: use gaussian elimination */
      {

         /*-----------------------------------------------------------------
          *  Generate CSR matrix from ParCSRMatrix A
          *-----------------------------------------------------------------*/
#ifdef HYPRE_NO_GLOBAL_PARTITION
         /* all processors are needed for these routines */
         A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
         f_vector = hypre_ParVectorToVectorAll(f);
	 if (n)
	 {
	 
#else
	 if (n)
	 {
	    A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
	    f_vector = hypre_ParVectorToVectorAll(f);
#endif
 	    A_CSR_i = hypre_CSRMatrixI(A_CSR);
 	    A_CSR_j = hypre_CSRMatrixJ(A_CSR);
 	    A_CSR_data = hypre_CSRMatrixData(A_CSR);
   	    f_vector_data = hypre_VectorData(f_vector);

            A_mat = hypre_CTAlloc(HYPRE_Real, n_global*n_global);
            b_vec = hypre_CTAlloc(HYPRE_Real, n_global);    

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
            A_CSR = NULL;
            hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         
         }
#ifdef HYPRE_NO_GLOBAL_PARTITION
         else
         {
            
            hypre_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         }
#endif

      }
      break;   
      case 98: /* Direct solve: use gaussian elimination & BLAS (with pivoting) */
      {

         HYPRE_Int info;
         HYPRE_Int one_i = 1;
         HYPRE_Int *piv;
         /*-----------------------------------------------------------------
          *  Generate CSR matrix from ParCSRMatrix A
          *-----------------------------------------------------------------*/
#ifdef HYPRE_NO_GLOBAL_PARTITION
         /* all processors are needed for these routines */
         A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
         f_vector = hypre_ParVectorToVectorAll(f);
	 if (n)
	 {
	 
#else
	 if (n)
	 {
	    A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
	    f_vector = hypre_ParVectorToVectorAll(f);
#endif
 	    A_CSR_i = hypre_CSRMatrixI(A_CSR);
 	    A_CSR_j = hypre_CSRMatrixJ(A_CSR);
 	    A_CSR_data = hypre_CSRMatrixData(A_CSR);
   	    f_vector_data = hypre_VectorData(f_vector);

            A_mat = hypre_CTAlloc(HYPRE_Real, n_global*n_global);
            b_vec = hypre_CTAlloc(HYPRE_Real, n_global);    

            /*---------------------------------------------------------------
             *  Load CSR matrix into A_mat.
             *---------------------------------------------------------------*/

            for (i = 0; i < n_global; i++)
            {
               for (jj = A_CSR_i[i]; jj < A_CSR_i[i+1]; jj++)
               {
             
                  /* need col major */
                  column = A_CSR_j[jj];
                  A_mat[i + n_global*column] = A_CSR_data[jj];
               }
               b_vec[i] = f_vector_data[i];
            }

            piv = hypre_CTAlloc(HYPRE_Int, n_global);

           /* write over A with LU */
#ifdef HYPRE_USING_ESSL
            dgetrf(n_global, n_global, A_mat, n_global, piv, &info);

#else
            hypre_F90_NAME_LAPACK(dgetrf, DGETRF)(&n_global, &n_global, 
                                             A_mat, &n_global, piv, &info);
#endif
            
           /*now b_vec = inv(A)*b_vec  */
#ifdef HYPRE_USING_ESSL
            dgetrs("N", n_global, &one_i, A_mat, 
                   n_global, piv, b_vec, 
                   n_global, &info);

#else
            hypre_F90_NAME_LAPACK(dgetrs, DGETRS)("N", &n_global, &one_i, A_mat, 
                                             &n_global, piv, b_vec, 
                                             &n_global, &info);
#endif
            hypre_TFree(piv);
            

            for (i = 0; i < n; i++)
            {
               u_data[i] = b_vec[first_index+i];
            }

	    hypre_TFree(A_mat); 
            hypre_TFree(b_vec);
            hypre_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         
         }
#ifdef HYPRE_NO_GLOBAL_PARTITION
         else
         {
            
            hypre_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         }
#endif

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

HYPRE_Int hypre_GaussElimSetup (hypre_ParAMGData *amg_data, HYPRE_Int level, HYPRE_Int relax_type)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_GS_ELIM_SETUP] -= hypre_MPI_Wtime();
#endif

   /* Par Data Structure variables */
   hypre_ParCSRMatrix *A = hypre_ParAMGDataAArray(amg_data)[level];
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   MPI_Comm           comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm           new_comm;

   /* Generate sub communicator */
   hypre_GenerateSubComm(comm, num_rows, &new_comm);

   if (num_rows)
   {
      hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
      HYPRE_Int *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
      HYPRE_Int *A_diag_i = hypre_CSRMatrixI(A_diag);
      HYPRE_Int *A_offd_i = hypre_CSRMatrixI(A_offd);
      HYPRE_Int *A_diag_j = hypre_CSRMatrixJ(A_diag);
      HYPRE_Int *A_offd_j = hypre_CSRMatrixJ(A_offd);
      HYPRE_Real *A_diag_data = hypre_CSRMatrixData(A_diag);
      HYPRE_Real *A_offd_data = hypre_CSRMatrixData(A_offd);
      HYPRE_Real *A_mat, *A_mat_local;
      HYPRE_Int *comm_info, *info, *displs;
      HYPRE_Int *mat_info, *mat_displs;
      HYPRE_Int new_num_procs, A_mat_local_size, i, jj, column;
      HYPRE_Int first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);

      hypre_MPI_Comm_size(new_comm, &new_num_procs);
      comm_info = hypre_CTAlloc(HYPRE_Int, 2*new_num_procs+1);
      mat_info = hypre_CTAlloc(HYPRE_Int, new_num_procs);
      mat_displs = hypre_CTAlloc(HYPRE_Int, new_num_procs+1);
      info = &comm_info[0];
      displs = &comm_info[new_num_procs];
      hypre_MPI_Allgather(&num_rows, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, new_comm);
      displs[0] = 0;
      mat_displs[0] = 0;
      for (i=0; i < new_num_procs; i++)
      {
         displs[i+1] = displs[i]+info[i];
         mat_displs[i+1] = global_num_rows*displs[i+1];
         mat_info[i] = global_num_rows*info[i];
      }
      hypre_ParAMGDataBVec(amg_data) = hypre_CTAlloc(HYPRE_Real, global_num_rows);
      A_mat_local_size =  global_num_rows*num_rows;
      A_mat_local = hypre_CTAlloc(HYPRE_Real, A_mat_local_size);
      A_mat = hypre_CTAlloc(HYPRE_Real, global_num_rows*global_num_rows);
      /* load local matrix into A_mat_local */
      for (i = 0; i < num_rows; i++)
      {
         for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
         {
             /* need col major */
             column = A_diag_j[jj]+first_row_index;
             A_mat_local[i*global_num_rows + column] = A_diag_data[jj];
         }
         for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
         {
             /* need col major */
             column = col_map_offd[A_offd_j[jj]];
             A_mat_local[i*global_num_rows + column] = A_offd_data[jj];
         }
      }
      hypre_MPI_Allgatherv( A_mat_local, A_mat_local_size, HYPRE_MPI_REAL, A_mat, 
		mat_info, mat_displs, HYPRE_MPI_REAL, new_comm);
      if (relax_type == 99)
      {
          HYPRE_Real *AT_mat;
	  AT_mat = hypre_CTAlloc(HYPRE_Real, global_num_rows*global_num_rows);
          for (i=0; i < global_num_rows; i++)
	     for (jj=0; jj < global_num_rows; jj++)
                 AT_mat[i*global_num_rows + jj] = A_mat[i+ jj*global_num_rows];
          hypre_ParAMGDataAMat(amg_data) = AT_mat;
          hypre_TFree (A_mat);
      }
      else
         hypre_ParAMGDataAMat(amg_data) = A_mat;
      hypre_ParAMGDataCommInfo(amg_data) = comm_info;
      hypre_ParAMGDataNewComm(amg_data) = new_comm;
      hypre_TFree(mat_info);
      hypre_TFree(mat_displs);
      hypre_TFree(A_mat_local);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_GS_ELIM_SETUP] += hypre_MPI_Wtime();
#endif
   
   return hypre_error_flag;
}


HYPRE_Int hypre_GaussElimSolve (hypre_ParAMGData *amg_data, HYPRE_Int level, HYPRE_Int relax_type)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_GS_ELIM_SOLVE] -= hypre_MPI_Wtime();
#endif

   hypre_ParCSRMatrix *A = hypre_ParAMGDataAArray(amg_data)[level];
   HYPRE_Int  n        = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int  error_flag = 0;

   if (n)
   {
      MPI_Comm new_comm = hypre_ParAMGDataNewComm(amg_data);
      hypre_ParVector *f = hypre_ParAMGDataFArray(amg_data)[level]; 
      hypre_ParVector *u = hypre_ParAMGDataUArray(amg_data)[level]; 
      HYPRE_Real *A_mat = hypre_ParAMGDataAMat(amg_data);
      HYPRE_Real *b_vec = hypre_ParAMGDataBVec(amg_data);
      HYPRE_Real *f_data = hypre_VectorData(hypre_ParVectorLocalVector(f)); 
      HYPRE_Real *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u)); 
      HYPRE_Real *A_tmp;
      HYPRE_Int *comm_info = hypre_ParAMGDataCommInfo(amg_data);
      HYPRE_Int *displs, *info;
      HYPRE_Int  n_global = hypre_ParCSRMatrixGlobalNumRows(A);
      HYPRE_Int new_num_procs, i, my_info;
      HYPRE_Int first_index = hypre_ParCSRMatrixFirstRowIndex(A);
      HYPRE_Int one_i = 1;

      hypre_MPI_Comm_size(new_comm, &new_num_procs);
      info = &comm_info[0];
      displs = &comm_info[new_num_procs];
      hypre_MPI_Allgatherv ( f_data, n, HYPRE_MPI_REAL,
                          b_vec, info, displs,
                          HYPRE_MPI_REAL, new_comm );

      A_tmp = hypre_CTAlloc (HYPRE_Real, n_global*n_global);
      for (i=0; i < n_global*n_global; i++)
         A_tmp[i] = A_mat[i];

      if (relax_type == 9)
      {
         error_flag = gselim(A_tmp,b_vec,n_global);
      }
      else if (relax_type == 99) /* use pivoting */
      {
         HYPRE_Int *piv;
         piv = hypre_CTAlloc(HYPRE_Int, n_global);

         /* write over A with LU */
#ifdef HYPRE_USING_ESSL
         dgetrf(n_global, n_global, A_tmp, n_global, piv, &my_info);

#else
         hypre_F90_NAME_LAPACK(dgetrf, DGETRF)(&n_global, &n_global, 
                                      A_tmp, &n_global, piv, &my_info);
#endif
            
         /*now b_vec = inv(A)*b_vec  */
#ifdef HYPRE_USING_ESSL
         dgetrs("N", n_global, &one_i, A_tmp, 
                     n_global, piv, b_vec, 
                     n_global, &my_info);

#else
         hypre_F90_NAME_LAPACK(dgetrs, DGETRS)("N", &n_global, &one_i, A_tmp, 
                                             &n_global, piv, b_vec, 
                                             &n_global, &my_info);
#endif
         hypre_TFree(piv);
      }
      for (i = 0; i < n; i++)
      {
         u_data[i] = b_vec[first_index+i];
      }
      hypre_TFree(A_tmp);
   }
   if (error_flag) hypre_error(HYPRE_ERROR_GENERIC);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_GS_ELIM_SOLVE] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}


HYPRE_Int gselim(HYPRE_Real *A,
                 HYPRE_Real *x,
                 HYPRE_Int n)
{
   HYPRE_Int    err_flag = 0;
   HYPRE_Int    j,k,m;
   HYPRE_Real factor;
   HYPRE_Real divA;
   
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
             divA = 1.0/A[k*n+k];
             for (j = k+1; j < n; j++)
             {
                 if (A[j*n+k] != 0.0)
                 {
                    factor = A[j*n+k]*divA;
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
          if (A[k*n+k] != 0.0)
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
       }
       if (A[0] != 0.0) x[0] /= A[0];
       return(err_flag);
    }
}
 

         



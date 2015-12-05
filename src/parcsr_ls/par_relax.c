/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.16 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * Relaxation scheme
 *
 *****************************************************************************/

#include "headers.h"
#include "Common.h"

#ifdef HYPRE_USING_ESSL
#include <essl.h>
#else
HYPRE_Int hypre_F90_NAME_LAPACK(dgetrf, DGETRF) (HYPRE_Int *, HYPRE_Int *, double *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *);
HYPRE_Int hypre_F90_NAME_LAPACK(dgetrs, DGETRS) (char *, HYPRE_Int *, HYPRE_Int *, double *, HYPRE_Int *, HYPRE_Int *, double *b, HYPRE_Int*, HYPRE_Int *);
#endif

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int  hypre_BoomerAMGRelax( hypre_ParCSRMatrix *A,
                           hypre_ParVector    *f,
                           HYPRE_Int                *cf_marker,
                           HYPRE_Int                 relax_type,
                           HYPRE_Int                 relax_points,
                           double              relax_weight,
                           double              omega,
                           double             *l1_norms,
                           hypre_ParVector    *u,
                           hypre_ParVector    *Vtemp,
                           hypre_ParVector    *Ztemp )
{
   MPI_Comm	   comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   double         *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int            *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int            *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int            *A_offd_i     = hypre_CSRMatrixI(A_offd);
   double         *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int            *A_offd_j     = hypre_CSRMatrixJ(A_offd);
   hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;

   HYPRE_Int             n_global= hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int             n       = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int             num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int	      	   first_index = hypre_ParVectorFirstIndex(u);
   
   hypre_Vector   *u_local = hypre_ParVectorLocalVector(u);
   double         *u_data  = hypre_VectorData(u_local);

   hypre_Vector   *f_local = hypre_ParVectorLocalVector(f);
   double         *f_data  = hypre_VectorData(f_local);

   hypre_Vector   *Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   double         *Vtemp_data = hypre_VectorData(Vtemp_local);
   double 	  *Vext_data;
   double 	  *v_buf_data;
   double 	  *tmp_data;

   hypre_Vector   *Ztemp_local;
   double         *Ztemp_data;

   hypre_CSRMatrix *A_CSR;
   HYPRE_Int		   *A_CSR_i;   
   HYPRE_Int		   *A_CSR_j;
   double	   *A_CSR_data;
   
   hypre_Vector    *f_vector;
   double	   *f_vector_data;

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

   double         *A_mat;
   double         *b_vec;

   double          zero = 0.0;
   double	   res, res0, res2;
   double          one_minus_weight;
   double          one_minus_omega;
   double          prod;

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
    *     relax_type = 9 -> Direct Solve
    *     relax_type = 99-> Direct solve: use gaussian elimination & BLAS (with pivoting)
    *-----------------------------------------------------------------------*/
   switch (relax_type)
   {            
      case 0: /* Weighted Jacobi */
      {
	if (num_procs > 1)
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
 
   	comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, 
        	Vext_data);
	}
         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
         
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i,ii,jj,res
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i,ii,jj,res
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i,ii,jj,res
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i,ii,jj,res
#include "../utilities/hypre_smp_forloop.h"
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
         
         
         if (num_procs > 1)
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
        if (num_procs > 1)
        {
	   hypre_TFree(Vext_data);
	   hypre_TFree(v_buf_data);
        }
      }
      break;

      case 1: /* Gauss-Seidel VERY SLOW */
      {
        if (num_procs > 1)
        {
   	num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   	num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);

   	v_buf_data = hypre_CTAlloc(double, 
			hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

	Vext_data = hypre_CTAlloc(double,num_cols_offd);
        
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
	       hypre_MPI_Isend(&v_buf_data[vec_start], vec_len, hypre_MPI_DOUBLE,
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
             hypre_MPI_Irecv(&Vext_data[vec_start], vec_len, hypre_MPI_DOUBLE,
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

   	v_buf_data = hypre_CTAlloc(double, 
			hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

	Vext_data = hypre_CTAlloc(double,num_cols_offd);
        
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
	       hypre_MPI_Isend(&v_buf_data[vec_start], vec_len, hypre_MPI_DOUBLE,
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
             hypre_MPI_Irecv(&Vext_data[vec_start], vec_len, hypre_MPI_DOUBLE,
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
	   tmp_data = hypre_CTAlloc(double,n);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
	   tmp_data = hypre_CTAlloc(double,n);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
         for (i = 0; i < n; i++)
         {
            Vtemp_data[i] = u_data[i];
         }
         prod = (1.0-relax_weight*omega);
         if (relax_points == 0)
         {
	  if (num_threads > 1)
          {
	   tmp_data = hypre_CTAlloc(double,n);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
	   tmp_data = hypre_CTAlloc(double,n);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
	      tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
           for (i = 0; i < n; i++)
              tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
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

      case 9: /* Direct solve: use gaussian elimination */
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
            /* use version with pivoting */
            /* relax_error = gselim_piv(A_mat,b_vec,n_global);*/

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
      case 99: /* Direct solve: use gaussian elimination & BLAS (with pivoting) */
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

            A_mat = hypre_CTAlloc(double, n_global*n_global);
            b_vec = hypre_CTAlloc(double, n_global);    

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

HYPRE_Int gselim(A,x,n)
double *A;
double *x;
HYPRE_Int n;
{
   HYPRE_Int    err_flag = 0;
   HYPRE_Int    j,k,m;
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
 

         



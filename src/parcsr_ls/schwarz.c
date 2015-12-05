/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.41 $
 ***********************************************************************EHEADER*/

#include "headers.h"
#include "Common.h"


/* -------------------------------------------------------------------------
   dof_domain: for each dof defines neighborhood to build interpolation,
                               using 
	       domain_diagmat (for cut--off scaling) and
	       i_domain_dof, j_dof_domain (for extracting the block of A);

   domain_matrixinverse: contains the inverse of subdomain matrix;

   B can be used to define strength matrix;
   ----------------------------------------------------------------------- */

#ifdef HYPRE_USING_ESSL
#include <essl.h>
#else
HYPRE_Int hypre_F90_NAME_LAPACK(dpotrf, DPOTRF)(char *, HYPRE_Int *, double *, HYPRE_Int *, HYPRE_Int *);
HYPRE_Int hypre_F90_NAME_LAPACK(dpotrs, DPOTRS)(char *, HYPRE_Int *, HYPRE_Int *, double *, HYPRE_Int *, double *, HYPRE_Int *, HYPRE_Int *);

HYPRE_Int hypre_F90_NAME_LAPACK(dgetrf, DGETRF) (HYPRE_Int *, HYPRE_Int *n, double *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *);
HYPRE_Int hypre_F90_NAME_LAPACK(dgetrs, DGETRS) (char *, HYPRE_Int *, HYPRE_Int *, double *, HYPRE_Int *, HYPRE_Int *, double *b, HYPRE_Int*, HYPRE_Int *);

#endif

/*--------------------------------------------------------------------------
 * hypre_AMGNodalSchwarzSmoother: (Not used currently)
 *--------------------------------------------------------------------------*/



HYPRE_Int
hypre_AMGNodalSchwarzSmoother( hypre_CSRMatrix    *A,
			       HYPRE_Int                 num_functions,
			       HYPRE_Int		   option,
			       hypre_CSRMatrix   **domain_structure_pointer)

{

  /*  option =      0: nodal symGS; 
		    1: next to nodal symGS (overlapping Schwarz) */
	

  HYPRE_Int *i_domain_dof, *j_domain_dof;
  double *domain_matrixinverse;
  HYPRE_Int num_domains;
  hypre_CSRMatrix *domain_structure;

  HYPRE_Int *i_dof_node, *j_dof_node;
  HYPRE_Int *i_node_dof, *j_node_dof;

  HYPRE_Int *i_node_dof_dof, *j_node_dof_dof;

  HYPRE_Int *i_node_node, *j_node_node;

  HYPRE_Int num_nodes;

  HYPRE_Int *i_dof_dof = hypre_CSRMatrixI(A);
  HYPRE_Int *j_dof_dof = hypre_CSRMatrixJ(A);
  double *a_dof_dof = hypre_CSRMatrixData(A);
  HYPRE_Int num_dofs = hypre_CSRMatrixNumRows(A);


  HYPRE_Int ierr = 0;
  HYPRE_Int i,j,k, l_loc, i_loc, j_loc;
  HYPRE_Int i_dof, j_dof;
  HYPRE_Int *i_local_to_global;
  HYPRE_Int *i_global_to_local;

  HYPRE_Int *i_int;
  HYPRE_Int *i_int_to_local;

  HYPRE_Int int_dof_counter, local_dof_counter, max_local_dof_counter=0; 

  HYPRE_Int domain_dof_counter = 0, domain_matrixinverse_counter = 0;


  double *AE;

#ifdef HYPRE_USING_ESSL
#else
  char uplo = 'L';
#endif

  HYPRE_Int cnt;

  /* PCG arrays: --------------------------------------------------- 
  double *x, *rhs, *v, *w, *d, *aux;

  HYPRE_Int max_iter;

  ------------------------------------------------------------------ */




  /* build dof_node graph: ----------------------------------------- */

  num_nodes = num_dofs / num_functions;

  /*hypre_printf("\nnum_nodes: %d, num_dofs: %d = %d x %d\n", num_nodes, num_dofs,
	 num_nodes, num_functions);*/

  i_dof_node = hypre_CTAlloc(HYPRE_Int, num_dofs+1);
  j_dof_node = hypre_CTAlloc(HYPRE_Int, num_dofs);

  for (i=0; i < num_dofs+1; i++)
    i_dof_node[i] = i;

  for (j = 0; j < num_nodes; j++)
    for (k = 0; k < num_functions; k++) 
      j_dof_node[j*num_functions+k] = j;

  /* build node_dof graph: ----------------------------------------- */

  ierr = transpose_matrix_create(&i_node_dof, &j_node_dof,
				 i_dof_node, j_dof_node,
				 
				 num_dofs, num_nodes);


  /* build node_node graph: ----------------------------------------- */

  ierr = matrix_matrix_product(&i_node_dof_dof,
			       &j_node_dof_dof,

			       i_node_dof, j_node_dof,
			       i_dof_dof, j_dof_dof,
			       
			       num_nodes, num_dofs, num_dofs);


  ierr = matrix_matrix_product(&i_node_node,
			       &j_node_node,

			       i_node_dof_dof,
			       j_node_dof_dof,

			       i_dof_node, j_dof_node,
			       
			       num_nodes, num_dofs, num_nodes);

  hypre_TFree(i_node_dof_dof);
  hypre_TFree(j_node_dof_dof);



  /* compute for each node the local information: -------------------- */

  i_global_to_local = i_dof_node; 

  for (i_dof =0; i_dof < num_dofs; i_dof++)
     i_global_to_local[i_dof] = -1;

  domain_matrixinverse_counter = 0;
  domain_dof_counter = 0;
  for (i=0; i < num_nodes; i++)
    {
      local_dof_counter = 0;

      for (j=i_node_node[i]; j < i_node_node[i+1]; j++)
	for (k=i_node_dof[j_node_node[j]];
	     k<i_node_dof[j_node_node[j]+1]; k++)
	  {
	    j_dof = j_node_dof[k];

	    if (i_global_to_local[j_dof] < 0)
	      {
		i_global_to_local[j_dof] = local_dof_counter;
		local_dof_counter++;
	      }

	  }
#ifdef HYPRE_USING_ESSL
      if (i == num_nodes-1)
	 domain_matrixinverse_counter += local_dof_counter*local_dof_counter;
      else
	 domain_matrixinverse_counter += 
		local_dof_counter*(local_dof_counter+1)/2;
#else
      domain_matrixinverse_counter += local_dof_counter*local_dof_counter;
#endif
      domain_dof_counter += local_dof_counter;

      if (local_dof_counter > max_local_dof_counter)
	max_local_dof_counter = local_dof_counter;

      for (j=i_node_node[i]; j < i_node_node[i+1]; j++)
	for (k=i_node_dof[j_node_node[j]];
	     k<i_node_dof[j_node_node[j]+1]; k++)
	  {
	    j_dof = j_node_dof[k];
	    i_global_to_local[j_dof] = -1;
	  }	       
    }


  num_domains = num_nodes;

  
  i_domain_dof = hypre_CTAlloc(HYPRE_Int, num_domains+1);

  if (option == 1)
    j_domain_dof = hypre_CTAlloc(HYPRE_Int, domain_dof_counter);
  else
    j_domain_dof = hypre_CTAlloc(HYPRE_Int, num_dofs);


  if (option == 1)
    domain_matrixinverse = hypre_CTAlloc(double, domain_matrixinverse_counter);
  else
    domain_matrixinverse = hypre_CTAlloc(double, num_dofs * num_functions);



  i_local_to_global = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);


  i_int_to_local = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);
  i_int          = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);



  for (l_loc=0; l_loc < max_local_dof_counter; l_loc++)
    i_int[l_loc] = -1;


  domain_dof_counter = 0;
  domain_matrixinverse_counter = 0;
  for (i=0; i < num_nodes; i++)
    {
      i_domain_dof[i] = domain_dof_counter;
      local_dof_counter = 0;

      for (j=i_node_node[i]; j < i_node_node[i+1]; j++)
	for (k=i_node_dof[j_node_node[j]];
	     k<i_node_dof[j_node_node[j]+1]; k++)
	  {
	    j_dof = j_node_dof[k];

	    if (i_global_to_local[j_dof] < 0)
	      {
		i_global_to_local[j_dof] = local_dof_counter;
		i_local_to_global[local_dof_counter] = j_dof;
		local_dof_counter++;
	      }

	  }

      for (j=i_node_dof[i]; j < i_node_dof[i+1]; j++)
	for (k=i_dof_dof[j_node_dof[j]]; k < i_dof_dof[j_node_dof[j]+1]; k++)
	  if (i_global_to_local[j_dof_dof[k]] < 0)
	    hypre_printf("WRONG local indexing: ====================== \n");


      int_dof_counter = 0;
      for (k=i_node_dof[i]; k < i_node_dof[i+1]; k++)
	{
	  i_dof = j_node_dof[k];
	  i_loc = i_global_to_local[i_dof];
	  i_int[i_loc] = int_dof_counter;
	  i_int_to_local[int_dof_counter] = i_loc;
	  int_dof_counter++;
	}

      /* get local matrix AE: ======================================== */

      if (option == 1)
	{
	  AE = &domain_matrixinverse[domain_matrixinverse_counter];
	  cnt = 0;
	  for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	    for (j_loc=0; j_loc < local_dof_counter; j_loc++)
	      AE[cnt++] = 0.e0;
	  
	  for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	    {
	      i_dof = i_local_to_global[i_loc];
	      for (j=i_dof_dof[i_dof]; j < i_dof_dof[i_dof+1]; j++)
		{
		  j_loc = i_global_to_local[j_dof_dof[j]];
		  if (j_loc >=0)
		    AE[i_loc + j_loc * local_dof_counter] = a_dof_dof[j];
		}
	    }


	  /* get block for Schwarz smoother: ============================= */
	  /* ierr = matinv(XE, AE, local_dof_counter); */
	  /* hypre_printf("ierr_AE_inv: %d\n", ierr); */
#ifdef HYPRE_USING_ESSL
 	cnt = local_dof_counter;
        for (j_loc=1; j_loc < local_dof_counter; j_loc++)
	   for (i_loc=j_loc; i_loc < local_dof_counter; i_loc++)
	      AE[cnt++] = AE[i_loc + j_loc * local_dof_counter];
	ierr = dppf(AE, local_dof_counter, 1);
 	if (ierr == 1) hypre_printf ("Error! Matrix not SPD\n");
#else  
	hypre_F90_NAME_LAPACK(dpotrf,DPOTRF)(&uplo, &local_dof_counter, AE,
		&local_dof_counter, &ierr); 
 	if (ierr) hypre_printf ("Error! Matrix not SPD\n");
#endif  

	for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	      j_domain_dof[domain_dof_counter+i_loc] 
		= i_local_to_global[i_loc]; 
      }


      if (option == 0)
	{

	  AE = &domain_matrixinverse[domain_matrixinverse_counter];
	  for (i_loc=0; i_loc < int_dof_counter; i_loc++)
	    for (j_loc=0; j_loc < int_dof_counter; j_loc++)
	      AE[i_loc + j_loc * int_dof_counter] = 0.e0;

      

	  for (l_loc=0; l_loc < int_dof_counter; l_loc++)
	    {
	      i_loc = i_int_to_local[l_loc];
	      i_dof = i_local_to_global[i_loc];
	      for (j=i_dof_dof[i_dof]; j < i_dof_dof[i_dof+1]; j++)
		{
		  j_loc = i_global_to_local[j_dof_dof[j]];
		  if (j_loc >=0)
		    if (i_int[j_loc] >=0)
		      AE[i_loc + i_int[j_loc] * int_dof_counter]
			= a_dof_dof[j];
		}
	    }

	  /* ierr = matinv(XE, AE, int_dof_counter); */
#ifdef HYPRE_USING_ESSL
 	cnt = local_dof_counter;
        for (j_loc=1; j_loc < local_dof_counter; j_loc++)
	   for (i_loc=j_loc; i_loc < local_dof_counter; i_loc++)
	      AE[cnt++] = AE[i_loc + j_loc * local_dof_counter];
	dppf(AE, local_dof_counter, 1);
#else
	hypre_F90_NAME_LAPACK(dpotrf,DPOTRF)(&uplo, &local_dof_counter, AE, 
		&local_dof_counter, &ierr);
/*	dpotrf_(&uplo, &local_dof_counter, AE, &local_dof_counter, &ierr);*/
#endif 

      if (ierr) hypre_printf (" error in dpotrf !!!\n");

	  for (i_loc=0; i_loc < int_dof_counter; i_loc++)
	    {
	      j_domain_dof[domain_dof_counter + i_loc] =
		i_local_to_global[i_int_to_local[i_loc]];

	      for (j_loc=0; j_loc < int_dof_counter; j_loc++)
		domain_matrixinverse[domain_matrixinverse_counter
				    + i_loc + j_loc * int_dof_counter]
		  = AE[i_loc + j_loc * int_dof_counter];
	    }

	  domain_dof_counter+=int_dof_counter;
	  domain_matrixinverse_counter+=int_dof_counter*int_dof_counter;
	}
      else
	{
	  domain_dof_counter+=local_dof_counter;
#ifdef HYPRE_USING_ESSL
	  domain_matrixinverse_counter+=local_dof_counter*(local_dof_counter+1)/2;
#else
	  domain_matrixinverse_counter+=local_dof_counter*local_dof_counter;
#endif
	}


      for (l_loc=0; l_loc < local_dof_counter; l_loc++)
	{
	  i_int[l_loc] = -1;
	  i_global_to_local[i_local_to_global[l_loc]] = -1;
	}


    }

  i_domain_dof[num_nodes] = domain_dof_counter;


  hypre_TFree(i_dof_node);
  hypre_TFree(j_dof_node);  

  hypre_TFree(i_node_dof);
  hypre_TFree(j_node_dof);
  hypre_TFree(i_node_node);
  hypre_TFree(j_node_node);

  hypre_TFree(i_int);
  hypre_TFree(i_int_to_local);

  hypre_TFree(i_local_to_global);

  domain_structure = hypre_CSRMatrixCreate(num_domains, max_local_dof_counter, 
		i_domain_dof[num_domains]);
  hypre_CSRMatrixI(domain_structure) = i_domain_dof;
  hypre_CSRMatrixJ(domain_structure) = j_domain_dof;
  hypre_CSRMatrixData(domain_structure) = domain_matrixinverse;

  *domain_structure_pointer = domain_structure;


   return hypre_error_flag;

}

HYPRE_Int hypre_ParMPSchwarzSolve(hypre_ParCSRMatrix *par_A,
                            hypre_CSRMatrix *A_boundary,
                            hypre_ParVector *rhs_vector,
                            hypre_CSRMatrix *domain_structure,
                            hypre_ParVector *par_x,
                            double relax_wt,
                            double *scale,
                            hypre_ParVector *Vtemp, HYPRE_Int *pivots, 
                            HYPRE_Int use_nonsymm)

{
  hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(par_A);
  HYPRE_Int num_sends = 0;
  HYPRE_Int *send_map_starts;
  HYPRE_Int *send_map_elmts;

  hypre_ParCSRCommHandle *comm_handle;

  HYPRE_Int ierr = 0;
  /* HYPRE_Int num_dofs; */
  hypre_CSRMatrix *A_diag;
  HYPRE_Int *A_diag_i;
  HYPRE_Int *A_diag_j;
  double *A_diag_data;
  hypre_CSRMatrix *A_offd;
  HYPRE_Int *A_offd_i;
  HYPRE_Int *A_offd_j;
  double *A_offd_data;
  double *x;
  double *x_ext;
  double *x_ext_old;
  double *rhs;
  double *rhs_ext;
  double *vtemp_data;
  double *aux;
  double *buf_data;
/*hypre_Vector *x_vector;*/
  MPI_Comm comm = hypre_ParCSRMatrixComm(par_A);
  HYPRE_Int num_domains = hypre_CSRMatrixNumRows(domain_structure);
  HYPRE_Int max_domain_size = hypre_CSRMatrixNumCols(domain_structure);
  HYPRE_Int *i_domain_dof = hypre_CSRMatrixI(domain_structure);
  HYPRE_Int *j_domain_dof = hypre_CSRMatrixJ(domain_structure);
  double *domain_matrixinverse = hypre_CSRMatrixData(domain_structure);
  HYPRE_Int *A_boundary_i;
  HYPRE_Int *A_boundary_j;
  double *A_boundary_data;
  HYPRE_Int num_variables;
  HYPRE_Int num_cols_offd;


  HYPRE_Int piv_counter = 0;
  HYPRE_Int one = 1;
  char uplo = 'L';

  HYPRE_Int jj,i,j,k, j_loc, k_loc;
  HYPRE_Int index;

  HYPRE_Int matrix_size, matrix_size_counter = 0;

  HYPRE_Int num_procs;

  hypre_MPI_Comm_size(comm,&num_procs);

  /* initiate:      ----------------------------------------------- */
  /* num_dofs = hypre_CSRMatrixNumRows(A); */

  A_diag = hypre_ParCSRMatrixDiag(par_A);
  A_offd = hypre_ParCSRMatrixOffd(par_A);
  num_variables = hypre_CSRMatrixNumRows(A_diag);
  num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
  x = hypre_VectorData(hypre_ParVectorLocalVector(par_x));
  vtemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));
  rhs = hypre_VectorData(hypre_ParVectorLocalVector(rhs_vector));

  if (use_nonsymm)
     uplo = 'N';

/*x_vector = hypre_ParVectorLocalVector(par_x);*/
  A_diag_i = hypre_CSRMatrixI(A_diag);
  A_diag_j = hypre_CSRMatrixJ(A_diag);
  A_diag_data = hypre_CSRMatrixData(A_diag);
  A_offd_i = hypre_CSRMatrixI(A_offd);
  if (num_cols_offd)
  {
     A_offd_j = hypre_CSRMatrixJ(A_offd);
     A_offd_data = hypre_CSRMatrixData(A_offd);
     A_boundary_i = hypre_CSRMatrixI(A_boundary);
     A_boundary_j = hypre_CSRMatrixJ(A_boundary);
     A_boundary_data = hypre_CSRMatrixData(A_boundary);
  }
  aux = hypre_CTAlloc(double, max_domain_size);

  hypre_ParVectorCopy(rhs_vector,Vtemp);
  hypre_ParCSRMatrixMatvec(-1.0,par_A,par_x,1.0,Vtemp);

  if (comm_pkg)
  {
     num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
     send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
     send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

     buf_data = hypre_CTAlloc(double, send_map_starts[num_sends]);
     x_ext = hypre_CTAlloc(double, num_cols_offd);
     x_ext_old = hypre_CTAlloc(double, num_cols_offd);
     rhs_ext = hypre_CTAlloc(double, num_cols_offd);

     index = 0;
     for (i=0; i < num_sends; i++)
     {
        for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
           buf_data[index++] = vtemp_data[send_map_elmts[j]];
     }

     comm_handle = hypre_ParCSRCommHandleCreate(1,comm_pkg,buf_data,
                rhs_ext);
     hypre_ParCSRCommHandleDestroy(comm_handle);
     comm_handle = NULL;

     index = 0;
     for (i=0; i < num_sends; i++)
     {
        for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
           buf_data[index++] = x[send_map_elmts[j]];
     }

     comm_handle = hypre_ParCSRCommHandleCreate(1,comm_pkg,buf_data,x_ext);
     hypre_ParCSRCommHandleDestroy(comm_handle);
     comm_handle = NULL;
  }

  /* correction of residual for exterior points to be updated locally */
   for (i=0; i < num_cols_offd; i++)
   {
      x_ext_old[i] = x_ext[i];
      for (j = A_boundary_i[i]; j < A_boundary_i[i+1]; j++)
      {
         k_loc = A_boundary_j[j];
	 if (k_loc < num_variables)
           rhs_ext[i] += A_boundary_data[j]*x[k_loc]; 
	 else
           rhs_ext[i] += A_boundary_data[j]*x_ext[k_loc-num_variables]; 
      }
   }
   /* forward solve: ----------------------------------------------- */

   matrix_size_counter = 0;
   for (i=0; i < num_domains; i++)
   {
      matrix_size = i_domain_dof[i+1] - i_domain_dof[i];

      /* compute residual: ---------------------------------------- */

      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  j_loc = j_domain_dof[j];
	  if (j_loc < num_variables) 
	  {
	     aux[jj] = rhs[j_loc];
	     for (k=A_diag_i[j_loc]; k<A_diag_i[j_loc+1]; k++)
	        aux[jj] -= A_diag_data[k] * x[A_diag_j[k]];
	     for (k=A_offd_i[j_loc]; k<A_offd_i[j_loc+1]; k++)
	        aux[jj] -= A_offd_data[k] * x_ext[A_offd_j[k]];
	  }
	  else
	  {
	     j_loc -= num_variables;
	     aux[jj] = rhs_ext[j_loc];
	     for (k=A_boundary_i[j_loc]; k<A_boundary_i[j_loc+1]; k++)
	        k_loc = A_boundary_j[k];
		if (k_loc < num_variables)
	           aux[jj] -= A_boundary_data[k] * x[k_loc];
		else
	           aux[jj] -= A_boundary_data[k] * x_ext[k_loc-num_variables];
	  }
	  jj++;
	}
      /* solve for correction: ------------------------------------- */
/* OLD - HYPRE_USING_ESSL
	dpps(&domain_matrixinverse[matrix_size_counter], matrix_size, aux, 1);
*/
      if (use_nonsymm)
      {
#ifdef HYPRE_USING_ESSL
         dgetrs(&uplo, matrix_size, one, 
                &domain_matrixinverse[matrix_size_counter], 
                matrix_size, &pivots[piv_counter], aux,
                matrix_size, &ierr); 
#else
          hypre_F90_NAME_LAPACK(dgetrs, DGETRS)(&uplo, &matrix_size, &one, 
                                                &domain_matrixinverse[matrix_size_counter], 
                                                &matrix_size, &pivots[piv_counter], aux,
                                                &matrix_size, &ierr); 
#endif
      }
      else
      {
#ifdef HYPRE_USING_ESSL
            dpotrs(&uplo, matrix_size, one, 
                   &domain_matrixinverse[matrix_size_counter], 
                   matrix_size, aux,
                   matrix_size, &ierr); 
#else
            hypre_F90_NAME_LAPACK(dpotrs, DPOTRS)(&uplo, &matrix_size, &one, 
                                                  &domain_matrixinverse[matrix_size_counter], 
                                                  &matrix_size, aux,
                                                  &matrix_size, &ierr); 
#endif
      }
      if (ierr)  hypre_error(HYPRE_ERROR_GENERIC);
      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  j_loc = j_domain_dof[j];
	  if (j_loc < num_variables) 
	     x[j_loc] +=  relax_wt*aux[jj++];
	  else
	     x_ext[j_loc-num_variables] +=  relax_wt*aux[jj++];
	}
/* OLD - HYPRE_USING_ESSL
      matrix_size_counter += matrix_size * (matrix_size+1)/2;  
*/
      matrix_size_counter += matrix_size * matrix_size;
      piv_counter += matrix_size;


   }
/*
   for (i=0; i < num_cols_offd; i++)
      x_ext[i] -= x_ext_old[i];  

   if (comm_pkg)
   {
      comm_handle=hypre_ParCSRCommHandleCreate (2,comm_pkg,x_ext,buf_data);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      index = 0;
      for (i=0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
            x[send_map_elmts[j]] += buf_data[index++];
      }
   }
   for (i=0; i < num_variables; i++)
      x[i] *= scale[i];

   hypre_ParVectorCopy(rhs_vector,Vtemp);
   hypre_ParCSRMatrixMatvec(-1.0,par_A,par_x,1.0,Vtemp);

   if (comm_pkg)
   {
      index = 0;
      for (i=0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
            buf_data[index++] = vtemp_data[send_map_elmts[j]];
      }

      comm_handle = hypre_ParCSRCommHandleCreate(1,comm_pkg,buf_data,
                rhs_ext);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      index = 0;
      for (i=0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
            buf_data[index++] = x[send_map_elmts[j]];
      }

      comm_handle = hypre_ParCSRCommHandleCreate(1,comm_pkg,buf_data,x_ext);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }
*/
   /* correction of residual for exterior points to be updated locally */
/*   for (i=0; i < num_cols_offd; i++)
   {
      x_ext_old[i] = x_ext[i];
      for (j = A_boundary_i[i]; j < A_boundary_i[i+1]; j++)
      {
         k_loc = A_boundary_j[j];
	 if (k_loc < num_variables)
           rhs_ext[i] += A_boundary_i[k]*x[k_loc]; 
	 else
           rhs_ext[i] += A_boundary_i[k]*x_ext[k_loc-num_variables]; 
      }
   }
*/
   /* backward solve: ------------------------------------------------ */
   for (i=num_domains-1; i > -1; i--)
   {
      matrix_size = i_domain_dof[i+1] - i_domain_dof[i];
/* OLD - HYPRE_USING_ESSL
      matrix_size_counter -= matrix_size * (matrix_size+1)/2;  
*/
      matrix_size_counter -= matrix_size * matrix_size;
      piv_counter -= matrix_size;
 
      /* compute residual: ---------------------------------------- */
      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  j_loc = j_domain_dof[j];
	  if (j_loc < num_variables) 
	  {
	     aux[jj] = rhs[j_loc];
	     for (k=A_diag_i[j_loc]; k<A_diag_i[j_loc+1]; k++)
	        aux[jj] -= A_diag_data[k] * x[A_diag_j[k]];
	     for (k=A_offd_i[j_loc]; k<A_offd_i[j_loc+1]; k++)
	        aux[jj] -= A_offd_data[k] * x_ext[A_offd_j[k]];
	  }
	  else
	  {
	     j_loc -= num_variables;
	     aux[jj] = rhs_ext[j_loc];
	     for (k=A_boundary_i[j_loc]; k<A_boundary_i[j_loc+1]; k++)
	        k_loc = A_boundary_j[k];
		if (k_loc < num_variables)
	           aux[jj] -= A_boundary_data[k] * x[k_loc];
		else
	           aux[jj] -= A_boundary_data[k] * x_ext[k_loc-num_variables];
	  }
	  jj++;
	}

      /* solve for correction: ------------------------------------- */
/* OLD - HYPRE_USING_ESSL
	dpps(&domain_matrixinverse[matrix_size_counter], matrix_size, aux, 1);
*/ 
      if (use_nonsymm)
      {
#ifdef HYPRE_USING_ESSL
         dgetrs(&uplo, matrix_size, one, 
                &domain_matrixinverse[matrix_size_counter], 
                matrix_size, &pivots[piv_counter], aux,
                matrix_size, &ierr);
#else
          hypre_F90_NAME_LAPACK(dgetrs, DGETRS)(&uplo, &matrix_size, &one, 
                                                &domain_matrixinverse[matrix_size_counter], 
                                                &matrix_size, &pivots[piv_counter], aux,
                                                &matrix_size, &ierr); 
#endif 
      }
      else
      {
#ifdef HYPRE_USING_ESSL
           dpotrs(&uplo, matrix_size, one, 
                  &domain_matrixinverse[matrix_size_counter], 
                  matrix_size, aux,
                  matrix_size, &ierr); 
#else
           hypre_F90_NAME_LAPACK(dpotrs, DPOTRS)(&uplo, &matrix_size, &one, 
                                                 &domain_matrixinverse[matrix_size_counter], 
                                                 &matrix_size, aux,
                                                 &matrix_size, &ierr); 
#endif
      }
      
   if (ierr)  hypre_error(HYPRE_ERROR_GENERIC);
      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  j_loc = j_domain_dof[j];
	  if (j_loc < num_variables) 
	     x[j_loc] +=  relax_wt*aux[jj++];
	  else
	     x_ext[j_loc-num_variables] +=  relax_wt*aux[jj++];
	}
    }			      

   for (i=0; i < num_cols_offd; i++)
      x_ext[i] -= x_ext_old[i];  

   if (comm_pkg)
   {
      comm_handle=hypre_ParCSRCommHandleCreate (2,comm_pkg,x_ext,buf_data);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      index = 0;
      for (i=0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
            x[send_map_elmts[j]] += buf_data[index++];
      }
      
      hypre_TFree(buf_data);
      hypre_TFree(x_ext);
      hypre_TFree(x_ext_old);
      hypre_TFree(rhs_ext);
   }
   for (i=0; i < num_variables; i++)
      x[i] *= scale[i];

  hypre_TFree(aux);

  return hypre_error_flag;

}

HYPRE_Int hypre_MPSchwarzSolve(hypre_ParCSRMatrix *par_A,
                         hypre_Vector *rhs_vector,
                         hypre_CSRMatrix *domain_structure,
                         hypre_ParVector *par_x,
                         double relax_wt,
                         hypre_Vector *aux_vector, HYPRE_Int *pivots, 
                         HYPRE_Int use_nonsymm)

{
  HYPRE_Int ierr = 0;
  /* HYPRE_Int num_dofs; */
  HYPRE_Int *i_dof_dof;
  HYPRE_Int *j_dof_dof;
  double *a_dof_dof;
  double *x;
  double *rhs;
  double *aux;
  hypre_CSRMatrix *A;
  hypre_Vector *x_vector;
  MPI_Comm comm = hypre_ParCSRMatrixComm(par_A);
  HYPRE_Int num_domains = hypre_CSRMatrixNumRows(domain_structure);
  HYPRE_Int *i_domain_dof = hypre_CSRMatrixI(domain_structure);
  HYPRE_Int *j_domain_dof = hypre_CSRMatrixJ(domain_structure);
  double *domain_matrixinverse = hypre_CSRMatrixData(domain_structure);

  HYPRE_Int piv_counter = 0;
  HYPRE_Int one = 1;
  char uplo = 'L';

  HYPRE_Int jj,i,j,k; /*, j_loc, k_loc;*/


  HYPRE_Int matrix_size, matrix_size_counter = 0;
   
  HYPRE_Int num_procs;
  
  hypre_MPI_Comm_size(comm,&num_procs);

  /* initiate:      ----------------------------------------------- */
  /* num_dofs = hypre_CSRMatrixNumRows(A); */
  x_vector = hypre_ParVectorLocalVector(par_x);
  A = hypre_ParCSRMatrixDiag(par_A);
  i_dof_dof = hypre_CSRMatrixI(A);
  j_dof_dof = hypre_CSRMatrixJ(A);
  a_dof_dof = hypre_CSRMatrixData(A);
  x = hypre_VectorData(x_vector);
  aux = hypre_VectorData(aux_vector);
  /* for (i=0; i < num_dofs; i++)
    x[i] = 0.e0; */

  if (use_nonsymm)
     uplo = 'N';


  if (num_procs > 1)
     hypre_parCorrRes(par_A,par_x,rhs_vector,&rhs);
  else 
     rhs = hypre_VectorData(rhs_vector);

  /* forward solve: ----------------------------------------------- */

  matrix_size_counter = 0;
  for (i=0; i < num_domains; i++)
    {
      matrix_size = i_domain_dof[i+1] - i_domain_dof[i];

      /* compute residual: ---------------------------------------- */

      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  aux[jj] = rhs[j_domain_dof[j]]; 
	  for (k=i_dof_dof[j_domain_dof[j]];
	       k<i_dof_dof[j_domain_dof[j]+1]; k++)
	    aux[jj] -= a_dof_dof[k] * x[j_dof_dof[k]];
	  jj++;
	}
      /* solve for correction: ------------------------------------- */
/* OLD - HYPRE_USING_ESSL
	dpps(&domain_matrixinverse[matrix_size_counter], matrix_size, aux, 1);
*/
     
      if (use_nonsymm)
      {
         
#ifdef HYPRE_USING_ESSL
         dgetrs(&uplo, matrix_size, one, 
                &domain_matrixinverse[matrix_size_counter], 
                matrix_size, &pivots[piv_counter], aux,
                matrix_size, &ierr); 
#else
         hypre_F90_NAME_LAPACK(dgetrs, DGETRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, &pivots[piv_counter], aux,
                                               &matrix_size, &ierr); 
#endif
      }
      else
      {
         
#ifdef HYPRE_USING_ESSL
            dpotrs(&uplo, matrix_size, one, 
                   &domain_matrixinverse[matrix_size_counter], 
                   matrix_size, aux,
                   matrix_size, &ierr); 
#else
            hypre_F90_NAME_LAPACK(dpotrs, DPOTRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, aux,
                                               &matrix_size, &ierr); 
#endif
      }
      
      if (ierr) hypre_error(HYPRE_ERROR_GENERIC);
      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	   x[j_domain_dof[j]]+=  relax_wt * aux[jj++];
	}
/* OLD - HYPRE_USING_ESSL
      matrix_size_counter += matrix_size * (matrix_size+1)/2;  
*/
      matrix_size_counter += matrix_size * matrix_size;
      piv_counter += matrix_size;


    }

/*  if (num_procs > 1)
  {
     hypre_TFree(rhs);
     hypre_parCorrRes(par_A,par_x,rhs_vector,&rhs);
  }
  else 
     rhs = hypre_VectorData(rhs_vector);
*/
  /* backward solve: ------------------------------------------------ */
    for (i=num_domains-1; i > -1; i--)
    {
      matrix_size = i_domain_dof[i+1] - i_domain_dof[i];
/* OLD - HYPRE_USING_ESSL
      matrix_size_counter -= matrix_size * (matrix_size+1)/2;  
*/
      matrix_size_counter -= matrix_size * matrix_size;
      piv_counter -= matrix_size;

 
      /* compute residual: ---------------------------------------- */
      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  aux[jj] = rhs[j_domain_dof[j]]; 
	  for (k=i_dof_dof[j_domain_dof[j]];
	       k<i_dof_dof[j_domain_dof[j]+1]; k++)
	    aux[jj] -= a_dof_dof[k] * x[j_dof_dof[k]];
	  jj++;
	}

      /* solve for correction: ------------------------------------- */
/* OLD - HYPRE_USING_ESSL
	dpps(&domain_matrixinverse[matrix_size_counter], matrix_size, aux, 1);
*/

      if (use_nonsymm)
      {
         
#ifdef HYPRE_USING_ESSL
           dgetrs(&uplo, matrix_size, one, 
                  &domain_matrixinverse[matrix_size_counter], 
                  matrix_size, &pivots[piv_counter], aux,
                  matrix_size, &ierr);
#else
        hypre_F90_NAME_LAPACK(dgetrs, DGETRS)(&uplo, &matrix_size, &one, 
                                                &domain_matrixinverse[matrix_size_counter], 
                                                &matrix_size, &pivots[piv_counter], aux,
                                                &matrix_size, &ierr); 
#endif 
      }
      else
      {
         
#ifdef HYPRE_USING_ESSL
           dpotrs(&uplo, matrix_size, one, 
                  &domain_matrixinverse[matrix_size_counter], 
                  matrix_size, aux,
                  matrix_size, &ierr); 
#else
           hypre_F90_NAME_LAPACK(dpotrs, DPOTRS)(&uplo, &matrix_size, &one, 
                                                 &domain_matrixinverse[matrix_size_counter], 
                                                 &matrix_size, aux,
                                                 &matrix_size, &ierr); 
#endif
      }
      
      if (ierr)  hypre_error(HYPRE_ERROR_GENERIC);
      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	   x[j_domain_dof[j]]+=  relax_wt*aux[jj++];
	}
    }			      

			      
  if (num_procs > 1) hypre_TFree(rhs);

  return hypre_error_flag;
}

HYPRE_Int hypre_MPSchwarzCFSolve(hypre_ParCSRMatrix *par_A,
                           hypre_Vector *rhs_vector,
                           hypre_CSRMatrix *domain_structure,
                           hypre_ParVector *par_x,
                           double relax_wt,
                           hypre_Vector *aux_vector,
                           HYPRE_Int *CF_marker,
                           HYPRE_Int rlx_pt, HYPRE_Int *pivots, 
                           HYPRE_Int use_nonsymm)

{
  HYPRE_Int ierr = 0;
  /* HYPRE_Int num_dofs; */
  HYPRE_Int *i_dof_dof;
  HYPRE_Int *j_dof_dof;
  double *a_dof_dof;
  double *x;
  double *rhs;
  double *aux;
  hypre_CSRMatrix *A;
  hypre_Vector *x_vector;
  MPI_Comm comm = hypre_ParCSRMatrixComm(par_A);
  HYPRE_Int num_domains = hypre_CSRMatrixNumRows(domain_structure);
  HYPRE_Int *i_domain_dof = hypre_CSRMatrixI(domain_structure);
  HYPRE_Int *j_domain_dof = hypre_CSRMatrixJ(domain_structure);
  double *domain_matrixinverse = hypre_CSRMatrixData(domain_structure);


  HYPRE_Int piv_counter = 0;
  HYPRE_Int one = 1;
  char uplo = 'L';
  HYPRE_Int jj,i,j,k; /*, j_loc, k_loc;*/


  HYPRE_Int matrix_size, matrix_size_counter = 0;

  HYPRE_Int num_procs;

  hypre_MPI_Comm_size(comm,&num_procs);

  /* initiate:      ----------------------------------------------- */
  /* num_dofs = hypre_CSRMatrixNumRows(A); */
  x_vector = hypre_ParVectorLocalVector(par_x);
  A = hypre_ParCSRMatrixDiag(par_A);
  i_dof_dof = hypre_CSRMatrixI(A);
  j_dof_dof = hypre_CSRMatrixJ(A);
  a_dof_dof = hypre_CSRMatrixData(A);
  x = hypre_VectorData(x_vector);
  aux = hypre_VectorData(aux_vector);
  /* for (i=0; i < num_dofs; i++)
    x[i] = 0.e0; */

  if (use_nonsymm)
     uplo = 'N';
  
  if (num_procs > 1)
     hypre_parCorrRes(par_A,par_x,rhs_vector,&rhs);
  else 
     rhs = hypre_VectorData(rhs_vector);

  /* forward solve: ----------------------------------------------- */

  matrix_size_counter = 0;
  for (i=0; i < num_domains; i++)
    {
      if (CF_marker[i] == rlx_pt)
      {
      matrix_size = i_domain_dof[i+1] - i_domain_dof[i];

      /* compute residual: ---------------------------------------- */

      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  aux[jj] = rhs[j_domain_dof[j]]; 
	  if (CF_marker[j_domain_dof[j]] == rlx_pt)
	  {
	  for (k=i_dof_dof[j_domain_dof[j]];
	       k<i_dof_dof[j_domain_dof[j]+1]; k++)
	    aux[jj] -= a_dof_dof[k] * x[j_dof_dof[k]];
	  }
	  jj++;
	}
      /* solve for correction: ------------------------------------- */
/* OLD HYPRE_USING_ESSL
	dpps(&domain_matrixinverse[matrix_size_counter], matrix_size, aux, 1);
*/
      if (use_nonsymm)
      {
         
#ifdef HYPRE_USING_ESSL
         dgetrs(&uplo, matrix_size, one, 
                &domain_matrixinverse[matrix_size_counter], 
                matrix_size, &pivots[piv_counter], aux,
                matrix_size, &ierr); 
#else
         hypre_F90_NAME_LAPACK(dgetrs, DGETRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, &pivots[piv_counter], aux,
                                               &matrix_size, &ierr); 
#endif
      }
      else
      {
         
#ifdef HYPRE_USING_ESSL
            dpotrs(&uplo, matrix_size, one, 
                   &domain_matrixinverse[matrix_size_counter], 
                   matrix_size, aux,
                   matrix_size, &ierr); 
#else
            hypre_F90_NAME_LAPACK(dpotrs, DPOTRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, aux,
                                               &matrix_size, &ierr); 
#endif
      }
      

      if (ierr)  hypre_error(HYPRE_ERROR_GENERIC);
      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	   x[j_domain_dof[j]]+=  relax_wt * aux[jj++];
	}
/* OLD - HYPRE_USING_ESSL
      matrix_size_counter += matrix_size * (matrix_size+1)/2;  
*/
      matrix_size_counter += matrix_size * matrix_size;
      piv_counter += matrix_size;

      }
    }

/*  if (num_procs > 1)
  {
     hypre_TFree(rhs);
     hypre_parCorrRes(par_A,par_x,rhs_vector,&rhs);
  }
  else 
     rhs = hypre_VectorData(rhs_vector);
*/
  /* backward solve: ------------------------------------------------ */
    for (i=num_domains-1; i > -1; i--)
    {
      if (CF_marker[i] == rlx_pt)
      {
      matrix_size = i_domain_dof[i+1] - i_domain_dof[i];
/* OLD -  HYPRE_USING_ESSL
      matrix_size_counter -= matrix_size * (matrix_size+1)/2;  
*/
      matrix_size_counter -= matrix_size * matrix_size;
      piv_counter -= matrix_size;

 
      /* compute residual: ---------------------------------------- */
      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  aux[jj] = rhs[j_domain_dof[j]]; 
	  if (CF_marker[j_domain_dof[j]] == rlx_pt)
	  {
	  for (k=i_dof_dof[j_domain_dof[j]];
	       k<i_dof_dof[j_domain_dof[j]+1]; k++)
	    aux[jj] -= a_dof_dof[k] * x[j_dof_dof[k]];
	  }
	  jj++;
	}

      /* solve for correction: ------------------------------------- */
/* OLD -  HYPRE_USING_ESSL
	dpps(&domain_matrixinverse[matrix_size_counter], matrix_size, aux, 1);
*/
      if (use_nonsymm)
      {
         
#ifdef HYPRE_USING_ESSL
           dgetrs(&uplo, matrix_size, one, 
                  &domain_matrixinverse[matrix_size_counter], 
                  matrix_size, &pivots[piv_counter], aux,
                  matrix_size, &ierr);
#else
        hypre_F90_NAME_LAPACK(dgetrs, DGETRS)(&uplo, &matrix_size, &one, 
                                                &domain_matrixinverse[matrix_size_counter], 
                                                &matrix_size, &pivots[piv_counter], aux,
                                                &matrix_size, &ierr); 
#endif 
      }
      else
      {
         
#ifdef HYPRE_USING_ESSL
           dpotrs(&uplo, matrix_size, one, 
                  &domain_matrixinverse[matrix_size_counter], 
                  matrix_size, aux,
                  matrix_size, &ierr); 
#else
           hypre_F90_NAME_LAPACK(dpotrs, DPOTRS)(&uplo, &matrix_size, &one, 
                                                 &domain_matrixinverse[matrix_size_counter], 
                                                 &matrix_size, aux,
                                                 &matrix_size, &ierr); 
#endif   
      }
      
      if (ierr)  hypre_error(HYPRE_ERROR_GENERIC);
      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	   x[j_domain_dof[j]]+=  relax_wt*aux[jj++];
	}
      }
    }			      

			      
  if (num_procs > 1) hypre_TFree(rhs);

  return hypre_error_flag;
}

HYPRE_Int hypre_MPSchwarzFWSolve(hypre_ParCSRMatrix *par_A,
                           hypre_Vector *rhs_vector,
                           hypre_CSRMatrix *domain_structure,
                           hypre_ParVector *par_x,
                           double relax_wt,
                           hypre_Vector *aux_vector, HYPRE_Int *pivots, 
                           HYPRE_Int use_nonsymm)

{
  HYPRE_Int ierr = 0;
  /* HYPRE_Int num_dofs; */
  HYPRE_Int *i_dof_dof;
  HYPRE_Int *j_dof_dof;
  double *a_dof_dof;
  double *x;
  double *rhs;
  double *aux;
  hypre_CSRMatrix *A;
  hypre_Vector *x_vector;
  MPI_Comm comm = hypre_ParCSRMatrixComm(par_A);
  HYPRE_Int num_domains = hypre_CSRMatrixNumRows(domain_structure);
  HYPRE_Int *i_domain_dof = hypre_CSRMatrixI(domain_structure);
  HYPRE_Int *j_domain_dof = hypre_CSRMatrixJ(domain_structure);
  double *domain_matrixinverse = hypre_CSRMatrixData(domain_structure);


  HYPRE_Int piv_counter = 0;
  HYPRE_Int one = 1;
  char uplo = 'L';
  HYPRE_Int jj,i,j,k; /*, j_loc, k_loc;*/


  HYPRE_Int matrix_size, matrix_size_counter = 0;

  HYPRE_Int num_procs;

  hypre_MPI_Comm_size(comm,&num_procs);

  /* initiate:      ----------------------------------------------- */
  /* num_dofs = hypre_CSRMatrixNumRows(A); */
  x_vector = hypre_ParVectorLocalVector(par_x);
  A = hypre_ParCSRMatrixDiag(par_A);
  i_dof_dof = hypre_CSRMatrixI(A);
  j_dof_dof = hypre_CSRMatrixJ(A);
  a_dof_dof = hypre_CSRMatrixData(A);
  x = hypre_VectorData(x_vector);
  aux = hypre_VectorData(aux_vector);
  /* for (i=0; i < num_dofs; i++)
    x[i] = 0.e0; */

  if (num_procs > 1)
     hypre_parCorrRes(par_A,par_x,rhs_vector,&rhs);
  else 
     rhs = hypre_VectorData(rhs_vector);

  /* forward solve: ----------------------------------------------- */

  matrix_size_counter = 0;
  for (i=0; i < num_domains; i++)
    {
      matrix_size = i_domain_dof[i+1] - i_domain_dof[i];

      /* compute residual: ---------------------------------------- */

      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  aux[jj] = rhs[j_domain_dof[j]]; 
	  for (k=i_dof_dof[j_domain_dof[j]];
	       k<i_dof_dof[j_domain_dof[j]+1]; k++)
	     aux[jj] -= a_dof_dof[k] * x[j_dof_dof[k]];
	  jj++;
	}
      /* solve for correction: ------------------------------------- */
/* OLD - HYPRE_USING_ESSL
	dpps(&domain_matrixinverse[matrix_size_counter], matrix_size, aux, 1);
*/
      if (use_nonsymm)
      {
         
#ifdef HYPRE_USING_ESSL
         dgetrs(&uplo, matrix_size, one, 
                &domain_matrixinverse[matrix_size_counter], 
                matrix_size, &pivots[piv_counter], aux,
                matrix_size, &ierr); 
#else
         hypre_F90_NAME_LAPACK(dgetrs, DGETRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, &pivots[piv_counter], aux,
                                               &matrix_size, &ierr); 
#endif
      }
      else
      {
         
#ifdef HYPRE_USING_ESSL
            dpotrs(&uplo, matrix_size, one, 
                   &domain_matrixinverse[matrix_size_counter], 
                   matrix_size, aux,
                   matrix_size, &ierr); 
#else
            hypre_F90_NAME_LAPACK(dpotrs, DPOTRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, aux,
                                               &matrix_size, &ierr); 
#endif
      }
      

      if (ierr)   hypre_error(HYPRE_ERROR_GENERIC);
      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	   x[j_domain_dof[j]]+=  relax_wt * aux[jj++];
	}
/* OLD - HYPRE_USING_ESSL
      matrix_size_counter += matrix_size * (matrix_size+1)/2;  
*/
      matrix_size_counter += matrix_size * matrix_size;
      piv_counter += matrix_size;

    }

  if (num_procs > 1) hypre_TFree(rhs);

  return hypre_error_flag;

}

HYPRE_Int hypre_MPSchwarzCFFWSolve(hypre_ParCSRMatrix *par_A,
                             hypre_Vector *rhs_vector,
                             hypre_CSRMatrix *domain_structure,
                             hypre_ParVector *par_x,
                             double relax_wt,
                             hypre_Vector *aux_vector,
                             HYPRE_Int *CF_marker,
                             HYPRE_Int rlx_pt, HYPRE_Int *pivots, 
                             HYPRE_Int use_nonsymm)

{
  HYPRE_Int ierr = 0;
  /* HYPRE_Int num_dofs; */
  HYPRE_Int *i_dof_dof;
  HYPRE_Int *j_dof_dof;
  double *a_dof_dof;
  double *x;
  double *rhs;
  double *aux;
  hypre_CSRMatrix *A;
  hypre_Vector *x_vector;
  MPI_Comm comm = hypre_ParCSRMatrixComm(par_A);
  HYPRE_Int num_domains = hypre_CSRMatrixNumRows(domain_structure);
  HYPRE_Int *i_domain_dof = hypre_CSRMatrixI(domain_structure);
  HYPRE_Int *j_domain_dof = hypre_CSRMatrixJ(domain_structure);
  double *domain_matrixinverse = hypre_CSRMatrixData(domain_structure);

  HYPRE_Int piv_counter = 0;
  HYPRE_Int one = 1;

  char uplo = 'L';
  HYPRE_Int jj,i,j,k; /*, j_loc, k_loc;*/


  HYPRE_Int matrix_size, matrix_size_counter = 0;

  HYPRE_Int num_procs;

  hypre_MPI_Comm_size(comm,&num_procs);

  /* initiate:      ----------------------------------------------- */
  /* num_dofs = hypre_CSRMatrixNumRows(A); */
  x_vector = hypre_ParVectorLocalVector(par_x);
  A = hypre_ParCSRMatrixDiag(par_A);
  i_dof_dof = hypre_CSRMatrixI(A);
  j_dof_dof = hypre_CSRMatrixJ(A);
  a_dof_dof = hypre_CSRMatrixData(A);
  x = hypre_VectorData(x_vector);
  aux = hypre_VectorData(aux_vector);
  /* for (i=0; i < num_dofs; i++)
    x[i] = 0.e0; */

  if (use_nonsymm)
     uplo = 'N';

  if (num_procs > 1)
     hypre_parCorrRes(par_A,par_x,rhs_vector,&rhs);
  else 
     rhs = hypre_VectorData(rhs_vector);

  /* forward solve: ----------------------------------------------- */

  matrix_size_counter = 0;
  for (i=0; i < num_domains; i++)
    {
      if (CF_marker[i] == rlx_pt)
      {
      matrix_size = i_domain_dof[i+1] - i_domain_dof[i];

      /* compute residual: ---------------------------------------- */

      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  aux[jj] = rhs[j_domain_dof[j]]; 
          if (CF_marker[j_domain_dof[j]] == rlx_pt)
          {
	  for (k=i_dof_dof[j_domain_dof[j]];
	       k<i_dof_dof[j_domain_dof[j]+1]; k++)
	    aux[jj] -= a_dof_dof[k] * x[j_dof_dof[k]];
          }
	  jj++;
	}
      /* solve for correction: ------------------------------------- */
/* OLD -  HYPRE_USING_ESSL
	dpps(&domain_matrixinverse[matrix_size_counter], matrix_size, aux, 1);
*/
      if (use_nonsymm)
      {
         
#ifdef HYPRE_USING_ESSL
         dgetrs(&uplo, matrix_size, one, 
                &domain_matrixinverse[matrix_size_counter], 
                matrix_size, &pivots[piv_counter], aux,
                matrix_size, &ierr); 
#else
         hypre_F90_NAME_LAPACK(dgetrs, DGETRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, &pivots[piv_counter], aux,
                                               &matrix_size, &ierr); 
#endif
      }
      
      else
      {
         
#ifdef HYPRE_USING_ESSL
            dpotrs(&uplo, matrix_size, one, 
                   &domain_matrixinverse[matrix_size_counter], 
                   matrix_size, aux,
                   matrix_size, &ierr); 
#else
            hypre_F90_NAME_LAPACK(dpotrs, DPOTRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, aux,
                                               &matrix_size, &ierr); 
#endif
      }
      
      if (ierr)  hypre_error(HYPRE_ERROR_GENERIC);
      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	   x[j_domain_dof[j]]+=  relax_wt * aux[jj++];
	}
/* OLD -  HYPRE_USING_ESSL
      matrix_size_counter += matrix_size * (matrix_size+1)/2;  
*/
      matrix_size_counter += matrix_size * matrix_size;
      piv_counter += matrix_size;


      }
    }

  if (num_procs > 1) hypre_TFree(rhs);

  return hypre_error_flag;

}

HYPRE_Int 
transpose_matrix_create(  HYPRE_Int **i_face_element_pointer,
			  HYPRE_Int **j_face_element_pointer,

			  HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,

			  HYPRE_Int num_elements, HYPRE_Int num_faces)

{
  /* FILE *f; */
  HYPRE_Int i, j;

  HYPRE_Int *i_face_element, *j_face_element;

  /* ======================================================================
     first create face_element graph: -------------------------------------
     ====================================================================== */

  i_face_element = (HYPRE_Int *) malloc((num_faces+1) * sizeof(HYPRE_Int));
  j_face_element = (HYPRE_Int *) malloc(i_element_face[num_elements] * sizeof(HYPRE_Int));


  for (i=0; i < num_faces; i++)
    i_face_element[i] = 0;

  for (i=0; i < num_elements; i++)
    for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
      i_face_element[j_element_face[j]]++;

  i_face_element[num_faces] = i_element_face[num_elements];

  for (i=num_faces-1; i > -1; i--)
    i_face_element[i] = i_face_element[i+1] - i_face_element[i];

  for (i=0; i < num_elements; i++)
    for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
      {
	j_face_element[i_face_element[j_element_face[j]]] = i;
	i_face_element[j_element_face[j]]++;
      }

  for (i=num_faces-1; i > -1; i--)
    i_face_element[i+1] = i_face_element[i];

  i_face_element[0] = 0;

  /* hypre_printf("end building face--element graph: ++++++++++++++++++\n"); */

  /* END building face_element graph; ================================ */

  *i_face_element_pointer = i_face_element;
  *j_face_element_pointer = j_face_element;

  return 0;

}
HYPRE_Int 
matrix_matrix_product(    HYPRE_Int **i_element_edge_pointer, 
			  HYPRE_Int **j_element_edge_pointer,

			  HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
			  HYPRE_Int *i_face_edge, HYPRE_Int *j_face_edge,

			  HYPRE_Int num_elements, HYPRE_Int num_faces, HYPRE_Int num_edges)

{
  /* FILE *f; */
  HYPRE_Int i, j, k, l, m;

  HYPRE_Int i_edge_on_local_list, i_edge_on_list;
  HYPRE_Int local_element_edge_counter = 0, element_edge_counter = 0;
  HYPRE_Int *j_local_element_edge;

  
  HYPRE_Int *i_element_edge, *j_element_edge;


  j_local_element_edge = (HYPRE_Int *) malloc((num_edges+1) * sizeof(HYPRE_Int));

  i_element_edge = (HYPRE_Int *) malloc((num_elements+1) * sizeof(HYPRE_Int));

  for (i=0; i < num_elements+1; i++)
    i_element_edge[i] = 0;

  for (i=0; i < num_elements; i++)
    {
      local_element_edge_counter = 0;
      for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
	{
	  k = j_element_face[j];

	  for (l=i_face_edge[k]; l < i_face_edge[k+1]; l++)
	    {
	      /* element i  and edge j_face_edge[l] are connected */
	    
	      /* hypre_printf("element %d  contains edge %d;\n",
		     i, j_face_edge[l]);  */

	      i_edge_on_local_list = -1;
	      for (m=0; m < local_element_edge_counter; m++)
		if (j_local_element_edge[m] == j_face_edge[l])
		  {
		    i_edge_on_local_list++;
		    break;
		  }

	      if (i_edge_on_local_list == -1)
		{
		  i_element_edge[i]++;
		  j_local_element_edge[local_element_edge_counter]=
		    j_face_edge[l];
		  local_element_edge_counter++;
		}
	    }
	}
    }

  free(j_local_element_edge);

  for (i=0; i < num_elements; i++)
    i_element_edge[i+1] += i_element_edge[i];

  for (i=num_elements; i>0; i--)
    i_element_edge[i] = i_element_edge[i-1];

  i_element_edge[0] = 0;

  j_element_edge = (HYPRE_Int *) malloc(i_element_edge[num_elements]
				     * sizeof(HYPRE_Int));

  /* fill--in the actual j_element_edge array: --------------------- */

  element_edge_counter = 0;
  for (i=0; i < num_elements; i++)
    {
      i_element_edge[i] = element_edge_counter;
      for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
	{
	  for (k=i_face_edge[j_element_face[j]];
	       k<i_face_edge[j_element_face[j]+1];k++)
	    {
	      /* check if edge j_face_edge[k] is already on list ***/

	      i_edge_on_list = -1;
	      for (l=i_element_edge[i];
		   l<element_edge_counter; l++)
		if (j_element_edge[l] == j_face_edge[k])
		  {
		    i_edge_on_list++;
		    break;
		  }

	      if (i_edge_on_list == -1) 
		{
		  if (element_edge_counter >= 
		      i_element_edge[num_elements])
		    {
		      hypre_printf("error in j_element_edge size: %d \n",
			     element_edge_counter);
		      break;
		    }

		  j_element_edge[element_edge_counter] =
		    j_face_edge[k];
		  element_edge_counter++;
		}
	    }
	}
		
    }

  i_element_edge[num_elements] = element_edge_counter;

  /*------------------------------------------------------------------
  f = fopen("element_edge", "w");
  for (i=0; i < num_elements; i++)
    {   
      hypre_printf("\nelement: %d has edges:\n", i);  
      for (j=i_element_edge[i]; j < i_element_edge[i+1]; j++)
	{
	  hypre_printf("%d ", j_element_edge[j]); 
	  hypre_fprintf(f, "%d %d\n", i, j_element_edge[j]);
	}
	  
      hypre_printf("\n"); 
    }

  fclose(f);
  */

  /* hypre_printf("end element_edge computation: ++++++++++++++++++++++++ \n");*/

  *i_element_edge_pointer = i_element_edge;
  *j_element_edge_pointer = j_element_edge;

  return 0;

}


/*--------------------------------------------------------------------------
 * hypre_AMGCreateDomainDof:  
 *--------------------------------------------------------------------------*/

/*****************************************************************************
 *
 * Routine for constructing graph domain_dof with minimal overlap
 *             and computing the respective matrix inverses to be
 *             used in an overlapping Schwarz procedure (like smoother
 *             in AMG); 
 *
 *****************************************************************************/
HYPRE_Int
hypre_AMGCreateDomainDof(hypre_CSRMatrix     *A,
			 HYPRE_Int domain_type, HYPRE_Int overlap,
			 HYPRE_Int num_functions, HYPRE_Int *dof_func,
			 hypre_CSRMatrix    **domain_structure_pointer, 
                         HYPRE_Int **piv_pointer, HYPRE_Int use_nonsymm)
{

  HYPRE_Int *i_domain_dof, *j_domain_dof;
  double *domain_matrixinverse;
  HYPRE_Int num_domains;
  hypre_CSRMatrix *domain_structure;  

  HYPRE_Int *i_dof_dof = hypre_CSRMatrixI(A);
  HYPRE_Int *j_dof_dof = hypre_CSRMatrixJ(A);
  double *a_dof_dof = hypre_CSRMatrixData(A);
  HYPRE_Int num_dofs = hypre_CSRMatrixNumRows(A);

  /* HYPRE_Int *i_dof_to_accept_weight; */
  HYPRE_Int *i_dof_to_prefer_weight,
    *w_dof_dof, *i_dof_weight;
  HYPRE_Int *i_dof_to_aggregate, *i_aggregate_dof, *j_aggregate_dof;
  
  HYPRE_Int *i_dof_index;

  HYPRE_Int ierr = 0;
  HYPRE_Int i,j,k,  l_loc, i_loc, j_loc;
  HYPRE_Int i_dof;
  HYPRE_Int *i_local_to_global;
  HYPRE_Int *i_global_to_local;


  HYPRE_Int local_dof_counter, max_local_dof_counter=0; 

  HYPRE_Int domain_dof_counter = 0, domain_matrixinverse_counter = 0;
  HYPRE_Int nf;

  double *AE;

  HYPRE_Int piv_counter = 0;
  HYPRE_Int *ipiv;
  HYPRE_Int *piv = NULL;
  char uplo = 'L';
  HYPRE_Int cnt;


  /* --------------------------------------------------------------------- */

  /*=======================================================================*/
  /*    create artificial domains by agglomeration;                        */
  /*=======================================================================*/

  /*hypre_printf("----------- create artificials domain by agglomeration;  ======\n");
*/


  i_aggregate_dof = hypre_CTAlloc(HYPRE_Int,num_dofs+1);
  j_aggregate_dof= hypre_CTAlloc(HYPRE_Int,num_dofs);

  if (domain_type == 2)
  {
     i_dof_to_prefer_weight = hypre_CTAlloc(HYPRE_Int,num_dofs);
     w_dof_dof = hypre_CTAlloc(HYPRE_Int,i_dof_dof[num_dofs]);
     i_dof_weight = hypre_CTAlloc(HYPRE_Int,num_dofs);

     for (i=0; i<num_dofs; i++)
        for (j=i_dof_dof[i]; j< i_dof_dof[i+1]; j++)
        {
	   if (j_dof_dof[j] == i) 
	      w_dof_dof[j]=0;
	   else
	      w_dof_dof[j]=1;
        }


     /*hypre_printf("end computing weights for agglomeration procedure: --------\n");
*/
     hypre_AMGeAgglomerate(i_aggregate_dof, j_aggregate_dof,

			       i_dof_dof, j_dof_dof, w_dof_dof,
		     
			       i_dof_dof, j_dof_dof,
			       i_dof_dof, j_dof_dof,

			       i_dof_to_prefer_weight,
			       i_dof_weight,

			       num_dofs, num_dofs,
			       &num_domains);

     hypre_TFree(i_dof_to_prefer_weight);
     hypre_TFree(i_dof_weight);
     hypre_TFree(w_dof_dof);

  }
  else 
  {
     nf = 1;
     if (domain_type == 1) nf = num_functions;

     num_domains = num_dofs/nf;
     for (i=0; i < num_domains+1; i++)
     {
        i_aggregate_dof[i] = nf*i;
     }
     for (i=0; i < num_dofs; i++)
     {
        j_aggregate_dof[i] = i;
     }
  }
  /*hypre_printf("num_dofs: %d, num_domains: %d\n", num_dofs, num_domains);*/


  /*
  hypre_printf("========================================================\n");
  hypre_printf("== artificial non--overlapping domains (aggregates): ===\n");
  hypre_printf("========================================================\n");


  for (i=0; i < num_domains; i++)
    {
      hypre_printf("\n aggregate %d:\n", i);
      for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
	hypre_printf("%d, ", j_aggregate_dof[j]);

      hypre_printf("\n");
    }
    */



  /* make domains from aggregates: *********************************/


  if (overlap == 1)
  {
     i_domain_dof = hypre_CTAlloc(HYPRE_Int, num_domains+1);

     i_dof_index = hypre_CTAlloc(HYPRE_Int, num_dofs);

     for (i=0; i < num_dofs; i++)
        i_dof_index[i] = -1;

     i_dof_to_aggregate = hypre_CTAlloc(HYPRE_Int, num_dofs);
     for (i=0; i < num_domains; i++)
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
           i_dof_to_aggregate[j_aggregate_dof[j]] = i;

     domain_dof_counter=0;
     for (i=0; i < num_domains; i++)
     {
        i_domain_dof[i] =  domain_dof_counter;
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
	   i_dof_index[j_aggregate_dof[j]]=-1;

        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
           for (k=i_dof_dof[j_aggregate_dof[j]];
	       k<i_dof_dof[j_aggregate_dof[j]+1]; k++)
	      if (i_dof_to_aggregate[j_dof_dof[k]] >= i 
	          && i_dof_index[j_dof_dof[k]]==-1) 
	      {
	         i_dof_index[j_dof_dof[k]]++;
	         domain_dof_counter++;
	      }

     }

     i_domain_dof[num_domains] =  domain_dof_counter;
     j_domain_dof = hypre_CTAlloc(HYPRE_Int,domain_dof_counter);

     for (i=0; i < num_dofs; i++)
        i_dof_index[i] = -1;

     domain_dof_counter=0;
     for (i=0; i < num_domains; i++)
     { 
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
	   i_dof_index[j_aggregate_dof[j]]=-1;

        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
	   for (k=i_dof_dof[j_aggregate_dof[j]];
	        k<i_dof_dof[j_aggregate_dof[j]+1]; k++)
	      if (i_dof_to_aggregate[j_dof_dof[k]] >= i
	          && i_dof_index[j_dof_dof[k]]==-1) 
	      {
	         i_dof_index[j_dof_dof[k]]++;
	         j_domain_dof[domain_dof_counter] = j_dof_dof[k];
	         domain_dof_counter++;
	      }

     }

     hypre_TFree(i_aggregate_dof);
     hypre_TFree(j_aggregate_dof);
     hypre_TFree(i_dof_to_aggregate);
     hypre_TFree(i_dof_index);
  }
  else if (overlap == 2)
  {	
     i_domain_dof = hypre_CTAlloc(HYPRE_Int, num_domains+1);

     i_dof_index = hypre_CTAlloc(HYPRE_Int, num_dofs);

     for (i=0; i < num_dofs; i++)
        i_dof_index[i] = -1;

     domain_dof_counter=0;
     for (i=0; i < num_domains; i++)
     {
        i_domain_dof[i] =  domain_dof_counter;
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
           for (k=i_dof_dof[j_aggregate_dof[j]];
	       k<i_dof_dof[j_aggregate_dof[j]+1]; k++)
	      if (i_dof_index[j_dof_dof[k]]==-1) 
	      {
	         i_dof_index[j_dof_dof[k]]++;
	         domain_dof_counter++;
	      }

        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
           for (k=i_dof_dof[j_aggregate_dof[j]];
	        k<i_dof_dof[j_aggregate_dof[j]+1]; k++)
	      i_dof_index[j_dof_dof[k]]=-1;

     }

     for (i=0; i < num_dofs; i++)
        i_dof_index[i] = -1;

     i_domain_dof[num_domains] =  domain_dof_counter;
     j_domain_dof = hypre_CTAlloc(HYPRE_Int,domain_dof_counter);

     domain_dof_counter=0;
     for (i=0; i < num_domains; i++)
     {
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
	   for (k=i_dof_dof[j_aggregate_dof[j]];
	        k<i_dof_dof[j_aggregate_dof[j]+1]; k++)
	      if ( i_dof_index[j_dof_dof[k]]==-1) 
	      {
	         i_dof_index[j_dof_dof[k]]++;
	         j_domain_dof[domain_dof_counter] = j_dof_dof[k];
	         domain_dof_counter++;
	      }

        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
	   for (k=i_dof_dof[j_aggregate_dof[j]];
	        k<i_dof_dof[j_aggregate_dof[j]+1]; k++)
	      i_dof_index[j_dof_dof[k]]=-1;

     }

     hypre_TFree(i_aggregate_dof);
     hypre_TFree(j_aggregate_dof);
     hypre_TFree(i_dof_index);
  }
  else
  {
     i_domain_dof = i_aggregate_dof;
     j_domain_dof = j_aggregate_dof;
  }

  /*hypre_printf("END domain_dof computations: =================================\n");
*/
  domain_matrixinverse_counter = 0;
  local_dof_counter = 0;
  piv_counter = 0;

 
  for (i=0; i < num_domains; i++)
    {
      local_dof_counter = i_domain_dof[i+1]-i_domain_dof[i];
      /*  OLD - HYPRE_USING_ESSL
          domain_matrixinverse_counter+= local_dof_counter*(local_dof_counter+1)/2; */
      domain_matrixinverse_counter+= local_dof_counter * local_dof_counter;
      piv_counter += local_dof_counter;


      if (local_dof_counter > max_local_dof_counter)
	max_local_dof_counter = local_dof_counter;
    }

  domain_matrixinverse = hypre_CTAlloc(double, domain_matrixinverse_counter);
  if (use_nonsymm)
     piv = hypre_CTAlloc(HYPRE_Int, piv_counter);




  /* OLD-HYPRE_USING_ESSL
  AE = hypre_CTAlloc(double, max_local_dof_counter*max_local_dof_counter);
  */

  i_local_to_global = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);


  i_global_to_local = hypre_CTAlloc(HYPRE_Int,num_dofs);

  for (i=0; i < num_dofs; i++)
    i_global_to_local[i] = -1;

  piv_counter = 0;
  domain_matrixinverse_counter = 0;
  for (i=0; i < num_domains; i++)
  {
      local_dof_counter = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
      {
	  i_global_to_local[j_domain_dof[j]] = local_dof_counter;
	  i_local_to_global[local_dof_counter] = j_domain_dof[j];
	  local_dof_counter++;
      }


      /* get local matrix in AE: ======================================== */
      cnt = 0;  

      /* OLD - HYPRE_USING_ESSL
      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	for (j_loc=0; j_loc < local_dof_counter; j_loc++)
	  AE[cnt++] = 0.e0;
      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
      {
  	 i_dof = i_local_to_global[i_loc];
	 for (j=i_dof_dof[i_dof]; j < i_dof_dof[i_dof+1]; j++)
	 {
	    j_loc = i_global_to_local[j_dof_dof[j]];
	    if (j_loc >=0)
		AE[i_loc + j_loc * local_dof_counter] = a_dof_dof[j];
	 }
      }
      cnt = local_dof_counter;
      for (j_loc=1; j_loc < local_dof_counter; j_loc++)
         for (i_loc=j_loc; i_loc < local_dof_counter; i_loc++)
	    AE[cnt++] = AE[i_loc + j_loc * local_dof_counter];
      dppf(AE, local_dof_counter, 1);
      cnt = domain_matrixinverse_counter;
      size = local_dof_counter*(local_dof_counter+1)/2;
      for (i_loc = 0; i_loc < size ; i_loc++)
         domain_matrixinverse[cnt++] = AE[i_loc];
      domain_matrixinverse_counter += size;
      */
      AE = &domain_matrixinverse[domain_matrixinverse_counter];
      ipiv = &piv[piv_counter];
      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	for (j_loc=0; j_loc < local_dof_counter; j_loc++)
	  AE[cnt++] = 0.e0;

      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
      {
	 i_dof = i_local_to_global[i_loc];
	 for (j=i_dof_dof[i_dof]; j < i_dof_dof[i_dof+1]; j++)
	 {
	    j_loc = i_global_to_local[j_dof_dof[j]];
	    if (j_loc >=0)
	       AE[i_loc + j_loc * local_dof_counter] = a_dof_dof[j];
	 }
      }

      if (use_nonsymm)
      {
#ifdef HYPRE_USING_ESSL
         dgetrf(local_dof_counter,local_dof_counter, AE,
                local_dof_counter, ipiv, &ierr);
#else
         hypre_F90_NAME_LAPACK(dgetrf,DGETRF)(&local_dof_counter,
                                              &local_dof_counter, AE,
                                              &local_dof_counter, ipiv, &ierr);
#endif
         piv_counter +=local_dof_counter;
      }
      
      else
      {
#ifdef HYPRE_USING_ESSL         
         dpotrf(&uplo,local_dof_counter, AE,
                local_dof_counter, &ierr);
#else
         hypre_F90_NAME_LAPACK(dpotrf,DPOTRF)(&uplo,&local_dof_counter, AE,
                                              &local_dof_counter, &ierr);
#endif
        
      }
      
      domain_matrixinverse_counter+=local_dof_counter*local_dof_counter;	  

      for (l_loc=0; l_loc < local_dof_counter; l_loc++)
	i_global_to_local[i_local_to_global[l_loc]] = -1;
	  
    }

  hypre_TFree(i_local_to_global);
  hypre_TFree(i_global_to_local);
/* OLD - HYPRE_USING_ESSL
  hypre_TFree(AE);
*/ 
  
  domain_structure = hypre_CSRMatrixCreate(num_domains, max_local_dof_counter,
			i_domain_dof[num_domains]);

  hypre_CSRMatrixI(domain_structure) = i_domain_dof;
  hypre_CSRMatrixJ(domain_structure) = j_domain_dof;
  hypre_CSRMatrixData(domain_structure) = domain_matrixinverse;

  *domain_structure_pointer = domain_structure;

  *piv_pointer = piv;

  return hypre_error_flag;

}


/* unacceptable faces: i_face_to_prefer_weight[] = -1; ------------------*/


HYPRE_Int hypre_AMGeAgglomerate(HYPRE_Int *i_AE_element, HYPRE_Int *j_AE_element,

			  HYPRE_Int *i_face_face, HYPRE_Int *j_face_face, HYPRE_Int *w_face_face,
		     
			  HYPRE_Int *i_face_element, HYPRE_Int *j_face_element,
			  HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,

			  HYPRE_Int *i_face_to_prefer_weight,
			  HYPRE_Int *i_face_weight,

			  HYPRE_Int num_faces, HYPRE_Int num_elements,
			  HYPRE_Int *num_AEs_pointer)
{

  HYPRE_Int ierr = 0;
  HYPRE_Int i, j, k, l;

  HYPRE_Int face_to_eliminate;
  HYPRE_Int max_weight_old, max_weight;

  HYPRE_Int AE_counter=0, AE_element_counter=0;

  /* HYPRE_Int i_element_face_counter; */

  HYPRE_Int *i_element_to_AE;

  HYPRE_Int *previous, *next, *first;
  HYPRE_Int head, tail, last;

  HYPRE_Int face_max_weight, face_local_max_weight, preferred_weight;

  HYPRE_Int weight, weight_max;

  max_weight = 1;
  for (i=0; i < num_faces; i++)
    {
      weight = 1;
      for (j=i_face_face[i]; j < i_face_face[i+1]; j++)
	weight+= w_face_face[j];
      if (max_weight < weight) max_weight = weight;
    }

  first = hypre_CTAlloc(HYPRE_Int, max_weight+1);



  next = hypre_CTAlloc(HYPRE_Int, num_faces);


  previous = hypre_CTAlloc(HYPRE_Int, num_faces+1);


  tail = num_faces;
  head = -1;

  for (i=0; i < num_faces; i++)
    {
      next[i] = i+1;
      previous[i] = i-1;
    }
  
  last = num_faces-1;
  previous[tail] = last;

  for (weight=1; weight <= max_weight; weight++)
    first[weight] = tail;

  i_element_to_AE = hypre_CTAlloc(HYPRE_Int, num_elements);

  /*=======================================================================
                     AGGLOMERATION PROCEDURE:
    ======================================================================= */

  for (k=0; k < num_elements; k++) 
    i_element_to_AE[k] = -1;

  for (k=0; k < num_faces; k++) 
    i_face_weight[k] = 1;


  first[0] = 0;
  first[1] = 0;

  last = previous[tail];
  weight_max = i_face_weight[last];


  k = last;
  face_max_weight = -1;
  while (k!= head)
    {
      if (i_face_to_prefer_weight[k] > -1)
	face_max_weight = k;
	  
      if (face_max_weight > -1) break;
	  
      k=previous[k];
    }


  /* this will be used if the faces have been sorted: *****************
  k = last;
  face_max_weight = -1;
  while (k != head)
    {
      if (i_face_to_prefer_weight[k] > -1)
	face_max_weight = k;


      if (face_max_weight > -1) 
	{
	  max_weight = i_face_weight[face_max_weight];
	  l = face_max_weight;

	  while (previous[l] != head)
	    {

	      if (i_face_weight[previous[l]] < max_weight) 
		break;
	      else
		if (i_face_to_prefer_weight[previous[l]] > 
		    i_face_to_prefer_weight[face_max_weight])
		  {
		    l = previous[l];
		    face_max_weight = l;
		  }
		else
		  l = previous[l];
	    }

	  break; 
	}


      l =previous[k];



      weight = i_face_weight[k];
      last = previous[tail];
      if (last == head) 
	weight_max = 0;
      else
	weight_max = i_face_weight[last];


      ierr = remove_entry(weight, &weight_max, 
			  previous, next, first, &last,
			  head, tail, 
			  k);

			  



      k=l;
    }
    */

  if (face_max_weight == -1)
    {
      hypre_printf("all faces are unacceptable, i.e., no faces to eliminate !\n");

      *num_AEs_pointer = 1;

      i_AE_element[0] = 0;
      for (i=0; i < num_elements; i++)
	{
	  i_element_to_AE[i] = 0;
	  j_AE_element[i] = i;
	}

      i_AE_element[1] = num_elements;

      return hypre_error_flag;
    }

  for (k=0; k < num_faces; k++)
    if (i_face_to_prefer_weight[k] > i_face_to_prefer_weight[face_max_weight])
      face_max_weight = k;

  max_weight = i_face_weight[face_max_weight];

  AE_counter=0;
  AE_element_counter=0;
   

  i_AE_element[AE_counter] = AE_element_counter;

  max_weight_old = -1;

  face_local_max_weight = face_max_weight;

eliminate_face:

  face_to_eliminate = face_local_max_weight;

  max_weight = i_face_weight[face_to_eliminate]; 

  last = previous[tail];
  if (last == head) 
    weight_max = 0;
  else
    weight_max = i_face_weight[last];

		   
  ierr = remove_entry(max_weight, &weight_max, 
		      previous, next, first, &last,
		      head, tail, 
		      face_to_eliminate);

  i_face_weight[face_to_eliminate] = 0;

  /*----------------------------------------------------------
   *  agglomeration step: 
   *
   *  put on AE_element -- list all elements 
   *  that share face "face_to_eliminate";
   *----------------------------------------------------------*/

  for (k = i_face_element[face_to_eliminate];
       k < i_face_element[face_to_eliminate+1]; k++)
    {
      /* check if element j_face_element[k] is already on the list: */

      if (j_face_element[k] < num_elements)
	{
	  if (i_element_to_AE[j_face_element[k]] == -1)
	    {
	      j_AE_element[AE_element_counter] = j_face_element[k];
	      i_element_to_AE[j_face_element[k]] = AE_counter;
	      AE_element_counter++;
	    }
	}
    }	  


  /* local update & search:==================================== */

  for (j=i_face_face[face_to_eliminate];
       j<i_face_face[face_to_eliminate+1]; j++)
    if (i_face_weight[j_face_face[j]] > 0)
      {
	weight = i_face_weight[j_face_face[j]];


	last = previous[tail];
	if (last == head) 
	  weight_max = 0;
	else
	  weight_max = i_face_weight[last];

	ierr = move_entry(weight, &weight_max, 
			  previous, next, first, &last,
			  head, tail, 
			  j_face_face[j]);

	i_face_weight[j_face_face[j]]+=w_face_face[j];

	weight = i_face_weight[j_face_face[j]];

	/* hypre_printf("update entry: %d\n", j_face_face[j]);  */

	last = previous[tail];
	if (last == head) 
	  weight_max = 0;
	else
	  weight_max = i_face_weight[last];

	ierr = update_entry(weight, &weight_max, 
			    previous, next, first, &last,
			    head, tail, 
			    j_face_face[j]);

	last = previous[tail];
	if (last == head) 
	  weight_max = 0;
	else
	  weight_max = i_face_weight[last];
		
      }

  /* find a face of the elements that have already been agglomerated
     with a maximal weight: ====================================== */
	  
  max_weight_old = max_weight;

  face_local_max_weight = -1; 
  preferred_weight = -1;

  for (l = i_AE_element[AE_counter];
       l < AE_element_counter; l++)
    {
      for (j=i_element_face[j_AE_element[l]];
	   j<i_element_face[j_AE_element[l]+1]; j++)
	{
	  i = j_element_face[j];

	  if (max_weight_old > 1 && i_face_weight[i] > 0 &&
	      i_face_to_prefer_weight[i] > -1)
	    {
	      if ( max_weight < i_face_weight[i])
		{
		  face_local_max_weight = i;
		  max_weight = i_face_weight[i];
		  preferred_weight = i_face_to_prefer_weight[i];
		}

	      if ( max_weight == i_face_weight[i]
		   && i_face_to_prefer_weight[i] > preferred_weight)
		{
		  face_local_max_weight = i;
		  preferred_weight = i_face_to_prefer_weight[i];
		}

	    }		
	}
    }  

  if (face_local_max_weight > -1) goto eliminate_face;

  /* ----------------------------------------------------------------
   * eliminate and label with i_face_weight[ ] = -1
   * "boundary faces of agglomerated elements";
   * those faces will be preferred for the next coarse spaces 
   * in case multiple coarse spaces are to be built;    
   * ---------------------------------------------------------------*/

  for (k = i_AE_element[AE_counter]; k < AE_element_counter; k++)
    {
      for (j = i_element_face[j_AE_element[k]];
	   j < i_element_face[j_AE_element[k]+1]; j++)
	{
	  if (i_face_weight[j_element_face[j]] > 0)
	    {
	      weight = i_face_weight[j_element_face[j]];
	      last = previous[tail];
	      if (last == head) 
		weight_max = 0;
	      else
		weight_max = i_face_weight[last];


	      ierr = remove_entry(weight, &weight_max, 
				  previous, next, first, &last,
				  head, tail, 
				  j_element_face[j]);

	      i_face_weight[j_element_face[j]] = -1;

	    }
	}
    }
      
  if (AE_element_counter > i_AE_element[AE_counter]) 
    {
      /* hypre_printf("completing agglomerated element: %d\n", 
		  AE_counter);   */ 
      AE_counter++;
    }

  i_AE_element[AE_counter] = AE_element_counter;
      

  /* find a face with maximal weight: ---------------------------*/


  last = previous[tail];
  if (last == head) goto end_agglomerate;

  weight_max = i_face_weight[last];

      
  /* hypre_printf("global search: ======================================\n"); */

  face_max_weight = -1;

  k = last;
  while (k != head)
    {
      if (i_face_to_prefer_weight[k] > -1)
	face_max_weight = k;


      if (face_max_weight > -1) 
	{
	  max_weight = i_face_weight[face_max_weight];
	  l = face_max_weight;

	  while (previous[l] != head)
	    {

	      if (i_face_weight[previous[l]] < max_weight) 
		break;
	      else
		if (i_face_to_prefer_weight[previous[l]] > 
		    i_face_to_prefer_weight[face_max_weight])
		  {
		    l = previous[l];
		    face_max_weight = l;
		  }
		else
		  l = previous[l];
	    }

	  break; 
	}


      l =previous[k];
      /* remove face k: ---------------------------------------*/


      weight = i_face_weight[k];
      last = previous[tail];
      if (last == head) 
	weight_max = 0;
      else
	weight_max = i_face_weight[last];


      ierr = remove_entry(weight, &weight_max, 
			  previous, next, first, &last,
			  head, tail, 
			  k);

			  
      /* i_face_weight[k] = -1; */


      k=l;
    }

  if (face_max_weight == -1) goto end_agglomerate;

  max_weight = i_face_weight[face_max_weight];

  face_local_max_weight = face_max_weight;

  goto eliminate_face;

end_agglomerate:


  /* eliminate isolated elements: ----------------------------------*/

  for (i=0; i<num_elements; i++)
    {

      if (i_element_to_AE[i] == -1)
	{
	  for (j=i_element_face[i]; j < i_element_face[i+1]
		 && i_element_to_AE[i] == -1; j++)
	    if (i_face_to_prefer_weight[j_element_face[j]] > -1)
	      for (k=i_face_element[j_element_face[j]];
		   k<i_face_element[j_element_face[j]+1]
		     && i_element_to_AE[i] == -1; k++)
		if (i_element_to_AE[j_face_element[k]] != -1)
		  i_element_to_AE[i] = i_element_to_AE[j_face_element[k]];
	}

      /*
      if (i_element_to_AE[i] == -1)
	{
	  i_element_face_counter = 0;
	  for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
	    if (i_face_to_prefer_weight[j_element_face[j]] > -1)
	      i_element_face_counter++;

	  if (i_element_face_counter == 1)
	    {
	      for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
		if (i_face_to_prefer_weight[j_element_face[j]] > -1)
		  for (k=i_face_element[j_element_face[j]];
		       k<i_face_element[j_element_face[j]+1]; k++)
		    if (i_element_to_AE[j_face_element[k]] != -1)
		      i_element_to_AE[i] = i_element_to_AE[j_face_element[k]];
	    }
	}
	*/

      if (i_element_to_AE[i] == -1)
	{
	  i_element_to_AE[i] = AE_counter;
	  AE_counter++;
	}
    }
	  
  num_AEs_pointer[0] = AE_counter;


  /* compute adjoint graph: -------------------------------------------*/

  for (i=0; i < AE_counter; i++)
    i_AE_element[i] = 0;

  for (i=0; i < num_elements; i++)
    i_AE_element[i_element_to_AE[i]]++;

  i_AE_element[AE_counter] = num_elements;

  for (i=AE_counter-1; i > -1; i--)
    i_AE_element[i] = i_AE_element[i+1] - i_AE_element[i];

  for (i=0; i < num_elements; i++)
    {
      j_AE_element[i_AE_element[i_element_to_AE[i]]] = i;
      i_AE_element[i_element_to_AE[i]]++;
    }

  for (i=AE_counter-1; i > -1; i--)
    i_AE_element[i+1] = i_AE_element[i];

  i_AE_element[0] = 0;

  /*--------------------------------------------------------------------*/
  for (i=0; i < num_faces; i++)
    if (i_face_to_prefer_weight[i] == -1) i_face_weight[i] = -1;


  hypre_TFree(i_element_to_AE);

  hypre_TFree(previous);
  hypre_TFree(next);
  hypre_TFree(first);

  return ierr;
}

HYPRE_Int update_entry(HYPRE_Int weight, HYPRE_Int *weight_max, 
		 HYPRE_Int *previous, HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last,
		 HYPRE_Int head, HYPRE_Int tail, 
		 HYPRE_Int i)

{
  HYPRE_Int weight0;

  if (previous[i] != head) next[previous[i]] = next[i];
  previous[next[i]] = previous[i];


  if (first[weight] == tail)
    {
      if (weight <= weight_max[0]) 
	{
	  hypre_printf("ERROR IN UPDATE_ENTRY: ===================\n");
	  hypre_printf("weight: %d, weight_max: %d\n",
		 weight, weight_max[0]);
	  return -1;
	}
      for (weight0=weight_max[0]+1; weight0 <= weight; weight0++)
	{
	  first[weight0] = i;
	  /* hypre_printf("create first[%d] = %d\n", weight0, i); */
	}

	  previous[i] = previous[tail];
	  next[i] = tail;
	  if (previous[tail] > head)
	     next[previous[tail]] = i;
	  previous[tail] = i;

    }
  else
    /* first[weight] already exists: =====================*/
    {
      previous[i] = previous[first[weight]];
      next[i] = first[weight];
      
      if (previous[first[weight]] != head)
	next[previous[first[weight]]] = i;

      previous[first[weight]] = i;

      for (weight0=1; weight0 <= weight; weight0++)
	if (first[weight0] == first[weight])
	  first[weight0] = i;

    }


  return 0;
    
}

HYPRE_Int remove_entry(HYPRE_Int weight, HYPRE_Int *weight_max, 
		 HYPRE_Int *previous, HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last,
		 HYPRE_Int head, HYPRE_Int tail, 
		 HYPRE_Int i)
{
  HYPRE_Int weight0;

  if (previous[i] != head) next[previous[i]] = next[i];
  previous[next[i]] = previous[i];

  for (weight0=1; weight0 <= weight_max[0]; weight0++)
    {
      /* hypre_printf("first[%d}: %d\n", weight0,  first[weight0]); */
      if (first[weight0] == i)
	{
	  first[weight0] = next[i];
	  /* hypre_printf("shift: first[%d]= %d to %d\n",
		 weight0, i, next[i]);
	  if (i == last[0]) 
	    hypre_printf("i= last[0]: %d\n", i); */
	}
    }

  next[i] = i;
  previous[i] = i;

  return 0;

}

HYPRE_Int move_entry(HYPRE_Int weight, HYPRE_Int *weight_max, 
	       HYPRE_Int *previous, HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last,
	       HYPRE_Int head, HYPRE_Int tail, 
	       HYPRE_Int i)
{
  HYPRE_Int  weight0;

  if (previous[i] != head) next[previous[i]] = next[i];
  previous[next[i]] = previous[i];

  for (weight0=1; weight0 <= weight_max[0]; weight0++)
    {
      if (first[weight0] == i)
	first[weight0] = next[i];
    }

  return 0;

}



/*---------------------------------------------------------------------
 matinv:  X <--  A**(-1) ;  A IS POSITIVE DEFINITE (non--symmetric);
 ---------------------------------------------------------------------*/
      
HYPRE_Int matinv(double *x, double *a, HYPRE_Int k)
{
  HYPRE_Int i,j,l, ierr =0;



  for (i=0; i < k; i++)
    {
      if (a[i+i*k] <= 0.e0)
        {
	  if (i < k-1)
	    {
	      /*********
	      hypre_printf("indefinite singular matrix in *** matinv ***:\n");
	      hypre_printf("i:%d;  diagonal entry: %e\n", i, a[i+k*i]);
	      */
	      ierr = -1;
	    }

            a[i+i*k] = 0.e0;
        }
         else
            a[i+k*i] = 1.0 / a[i+i*k];



      for (j=1; j < k-i; j++)
        {
          for (l=1; l < k-i; l++)
            {
              a[i+l+k*(i+j)] -= a[i+l+k*i] * a[i+k*i] * a[i+k*(i+j)];
            }
        }
      
      for (j=1; j < k-i; j++)
        {
          a[i+j+k*i] = a[i+j+k*i] * a[i+k*i];
          a[i+k*(i+j)] = a[i+k*(i+j)] * a[i+k*i];
        }
    }



  /* FULL INVERSION: --------------------------------------------*/
  



  x[k*k-1] = a[k*k-1];
  for (i=k-1; i > -1; i--)
    {
      for (j=1; j < k-i; j++)
        {
          x[i+j+k*i] =0;
          x[i+k*(i+j)] =0;



          for (l=1; l< k-i; l++)
            {
              x[i+j+k*i] -= x[i+j+k*(i+l)] * a[i+l+k*i];
              x[i+k*(i+j)] -= a[i+k*(i+l)] * x[i+l+k*(i+j)];
            }
        }



      x[i+k*i] = a[i+k*i];
      for (j=1; j<k-i; j++)
        {
          x[i+k*i] -= x[i+k*(i+j)] * a[i+j+k*i];
        }
    }



  return ierr;
}

HYPRE_Int
hypre_parCorrRes( hypre_ParCSRMatrix *A,
		  hypre_ParVector *x,
		  hypre_Vector *rhs,
		  double **tmp_ptr)
{
   HYPRE_Int i, j, index, start;
   HYPRE_Int num_sends, num_cols_offd;
   HYPRE_Int local_size;
   double *x_buf_data, *x_tmp_data, *x_local_data;

   hypre_ParCSRCommPkg *comm_pkg;
   hypre_CSRMatrix *offd;
   hypre_Vector *x_local, *x_tmp, *tmp_vector;
   hypre_ParCSRCommHandle *comm_handle;

   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   offd = hypre_ParCSRMatrixOffd(A);
   num_cols_offd = hypre_CSRMatrixNumCols(offd);

   x_local = hypre_ParVectorLocalVector(x);
   x_local_data = hypre_VectorData(x_local);
   local_size = hypre_VectorSize(x_local);

   if (num_cols_offd)
   {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      x_buf_data = hypre_CTAlloc(double, 
			hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            x_buf_data[index++]
                 = x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }

      x_tmp = hypre_SeqVectorCreate(num_cols_offd);
      hypre_SeqVectorInitialize(x_tmp);
      x_tmp_data = hypre_VectorData(x_tmp);

      comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, x_buf_data,
        		x_tmp_data);

      tmp_vector = hypre_SeqVectorCreate(local_size);
      hypre_SeqVectorInitialize(tmp_vector);
      hypre_SeqVectorCopy(rhs,tmp_vector);
   
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      hypre_CSRMatrixMatvec(-1.0, offd, x_tmp, 1.0, tmp_vector);

      hypre_SeqVectorDestroy(x_tmp);
      hypre_TFree(x_buf_data);
   }
   else
   {
      tmp_vector = hypre_SeqVectorCreate(local_size);
      hypre_SeqVectorInitialize(tmp_vector);
      hypre_SeqVectorCopy(rhs,tmp_vector);
   }

   *tmp_ptr = hypre_VectorData(tmp_vector);
   hypre_VectorOwnsData(tmp_vector) = 0;
   hypre_SeqVectorDestroy(tmp_vector);

   return 0;
}


HYPRE_Int hypre_AdSchwarzSolve(hypre_ParCSRMatrix *par_A,
                         hypre_ParVector *par_rhs,
                         hypre_CSRMatrix *domain_structure,
                         double *scale,
                         hypre_ParVector *par_x,
                         hypre_ParVector *par_aux, HYPRE_Int *pivots, 
                         HYPRE_Int use_nonsymm)

{
  HYPRE_Int ierr = 0;
  double *x;
  double *aux;
  double *tmp;
  hypre_Vector *x_vector;
  hypre_Vector *aux_vector;
  MPI_Comm comm = hypre_ParCSRMatrixComm(par_A);
  HYPRE_Int num_domains;
  HYPRE_Int max_domain_size;
  HYPRE_Int *i_domain_dof;
  HYPRE_Int *j_domain_dof;
  double *domain_matrixinverse;

  HYPRE_Int piv_counter = 0;
  HYPRE_Int one = 1;
  char uplo = 'L';

  HYPRE_Int jj,i,j; /*, j_loc, k_loc;*/


  HYPRE_Int matrix_size, matrix_size_counter = 0;

  HYPRE_Int num_procs;

  hypre_MPI_Comm_size(comm,&num_procs);

  /* initiate:      ----------------------------------------------- */
  x_vector = hypre_ParVectorLocalVector(par_x);
  aux_vector = hypre_ParVectorLocalVector(par_aux);
  x = hypre_VectorData(x_vector);
  aux = hypre_VectorData(aux_vector);
  num_domains = hypre_CSRMatrixNumRows(domain_structure);
  max_domain_size = hypre_CSRMatrixNumCols(domain_structure);
  i_domain_dof = hypre_CSRMatrixI(domain_structure);
  j_domain_dof = hypre_CSRMatrixJ(domain_structure);
  domain_matrixinverse = hypre_CSRMatrixData(domain_structure);

  if (use_nonsymm)
     uplo = 'N';

  hypre_ParVectorCopy(par_rhs,par_aux);
  hypre_ParCSRMatrixMatvec(-1.0,par_A,par_x,1.0,par_aux);
  tmp = hypre_CTAlloc(double,max_domain_size);

  /* forward solve: ----------------------------------------------- */

  matrix_size_counter = 0;
  for (i=0; i < num_domains; i++)
  {
     matrix_size = i_domain_dof[i+1] - i_domain_dof[i];

     /* compute residual: ---------------------------------------- */
     
     jj = 0;
     for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
     {
        tmp[jj] = aux[j_domain_dof[j]]; 
        jj++;
     }
     /* solve for correction: ------------------------------------- */
/* OLD -  HYPRE_USING_ESSL
     dpps(&domain_matrixinverse[matrix_size_counter], matrix_size, tmp, 1);
*/
     if (use_nonsymm)
     {
        
#ifdef HYPRE_USING_ESSL
         dgetrs(&uplo, matrix_size, one, 
                &domain_matrixinverse[matrix_size_counter], 
                matrix_size, &pivots[piv_counter], tmp,
                matrix_size, &ierr); 
#else
         hypre_F90_NAME_LAPACK(dgetrs, DGETRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, &pivots[piv_counter], tmp,
                                               &matrix_size, &ierr); 
#endif
     }
     else
     {
        
#ifdef HYPRE_USING_ESSL
            dpotrs(&uplo, matrix_size, one, 
                   &domain_matrixinverse[matrix_size_counter], 
                   matrix_size, tmp,
                   matrix_size, &ierr); 
#else
            hypre_F90_NAME_LAPACK(dpotrs, DPOTRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, tmp,
                                               &matrix_size, &ierr); 
#endif
     }
     



     if (ierr)  hypre_error(HYPRE_ERROR_GENERIC);
     jj = 0;
     for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
     {
        x[j_domain_dof[j]]+=  scale[j_domain_dof[j]]*tmp[jj++];
     }
/* OLD -  HYPRE_USING_ESSL
     matrix_size_counter += matrix_size * (matrix_size+1)/2;  
*/
     matrix_size_counter += matrix_size * matrix_size;
     piv_counter += matrix_size;
     
  }
  
  hypre_TFree(tmp);

  return hypre_error_flag;

}

HYPRE_Int hypre_AdSchwarzCFSolve(hypre_ParCSRMatrix *par_A,
		       hypre_ParVector *par_rhs,
		       hypre_CSRMatrix *domain_structure,
		       double *scale,
		       hypre_ParVector *par_x,
		       hypre_ParVector *par_aux,
		       HYPRE_Int *CF_marker,
		       HYPRE_Int rlx_pt, HYPRE_Int *pivots, 
                         HYPRE_Int use_nonsymm)

{
  HYPRE_Int ierr = 0;
  double *x;
  double *aux;
  double *tmp;
  hypre_Vector *x_vector;
  hypre_Vector *aux_vector;
  MPI_Comm comm = hypre_ParCSRMatrixComm(par_A);
  HYPRE_Int num_domains;
  HYPRE_Int max_domain_size;
  HYPRE_Int *i_domain_dof;
  HYPRE_Int *j_domain_dof;
  double *domain_matrixinverse;

  HYPRE_Int piv_counter = 0;
  HYPRE_Int one = 1;

  char uplo = 'L';
  HYPRE_Int jj,i,j; /*, j_loc, k_loc;*/


  HYPRE_Int matrix_size, matrix_size_counter = 0;

  HYPRE_Int num_procs;

  hypre_MPI_Comm_size(comm,&num_procs);

  /* initiate:      ----------------------------------------------- */
  x_vector = hypre_ParVectorLocalVector(par_x);
  aux_vector = hypre_ParVectorLocalVector(par_aux);
  x = hypre_VectorData(x_vector);
  aux = hypre_VectorData(aux_vector);
  num_domains = hypre_CSRMatrixNumRows(domain_structure);
  max_domain_size = hypre_CSRMatrixNumCols(domain_structure);
  i_domain_dof = hypre_CSRMatrixI(domain_structure);
  j_domain_dof = hypre_CSRMatrixJ(domain_structure);
  domain_matrixinverse = hypre_CSRMatrixData(domain_structure);

  if (use_nonsymm)
     uplo = 'N';
  

  hypre_ParVectorCopy(par_rhs,par_aux);
  hypre_ParCSRMatrixMatvec(-1.0,par_A,par_x,1.0,par_aux);
  tmp = hypre_CTAlloc(double,max_domain_size);

  /* forward solve: ----------------------------------------------- */

  matrix_size_counter = 0;
  for (i=0; i < num_domains; i++)
    {
      if (CF_marker[i] == rlx_pt)
      {
      matrix_size = i_domain_dof[i+1] - i_domain_dof[i];

      /* compute residual: ---------------------------------------- */

      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  tmp[jj] = aux[j_domain_dof[j]]; 
	  jj++;
	}
      /* solve for correction: ------------------------------------- */
/* OLD -  HYPRE_USING_ESSL
	dpps(&domain_matrixinverse[matrix_size_counter], matrix_size, tmp, 1);
*/
      if (use_nonsymm)
      {
         
#ifdef HYPRE_USING_ESSL
         dgetrs(&uplo, matrix_size, one, 
                &domain_matrixinverse[matrix_size_counter], 
                matrix_size, &pivots[piv_counter], tmp,
                matrix_size, &ierr); 
#else
         hypre_F90_NAME_LAPACK(dgetrs, DGETRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, &pivots[piv_counter], tmp,
                                               &matrix_size, &ierr); 
#endif
      }
      
      else
      {
         
#ifdef HYPRE_USING_ESSL
            dpotrs(&uplo, matrix_size, one, 
                   &domain_matrixinverse[matrix_size_counter], 
                   matrix_size, tmp,
                   matrix_size, &ierr); 
#else
            hypre_F90_NAME_LAPACK(dpotrs, DPOTRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, tmp,
                                               &matrix_size, &ierr); 
#endif 
      }
      


      if (ierr)  hypre_error(HYPRE_ERROR_GENERIC);
      jj = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	   x[j_domain_dof[j]]+=  scale[j_domain_dof[j]]*tmp[jj++];
	}
/* OLD -  HYPRE_USING_ESSL
      matrix_size_counter += matrix_size * (matrix_size+1)/2;  
*/
      matrix_size_counter += matrix_size * matrix_size;
      piv_counter += matrix_size;


      }
    }

			      
  hypre_TFree(tmp);

  return hypre_error_flag;

}

HYPRE_Int 
hypre_GenerateScale(hypre_CSRMatrix *domain_structure,
		    HYPRE_Int              num_variables,
		    double 	     relaxation_weight,
		    double         **scale_pointer)
{
   HYPRE_Int num_domains = hypre_CSRMatrixNumRows(domain_structure);
   HYPRE_Int *i_domain_dof = hypre_CSRMatrixI(domain_structure);
   HYPRE_Int *j_domain_dof = hypre_CSRMatrixJ(domain_structure);
   HYPRE_Int i, j;
   double *scale;

   scale = hypre_CTAlloc(double, num_variables);

   for (i=0; i < num_domains; i++)
      for (j = i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	 scale[j_domain_dof[j]] += 1.0;

   for (i=0; i < num_variables; i++)
      scale[i] = relaxation_weight/scale[i];

   *scale_pointer = scale;

   return hypre_error_flag;
}


HYPRE_Int hypre_ParAdSchwarzSolve(hypre_ParCSRMatrix *A,
                            hypre_ParVector *F,
                            hypre_CSRMatrix *domain_structure,
                            double *scale,
                            hypre_ParVector *X,
                            hypre_ParVector *Vtemp, 
                            HYPRE_Int *pivots, 
                            HYPRE_Int use_nonsymm)

{
  hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  HYPRE_Int num_sends = 0;
  HYPRE_Int *send_map_starts;
  HYPRE_Int *send_map_elmts;

  hypre_ParCSRCommHandle *comm_handle;

  HYPRE_Int ierr = 0;
  double *x_data;
  double *x_ext_data;
  double *aux;
  double *vtemp_data;
  double *vtemp_ext_data;
  HYPRE_Int num_domains, max_domain_size;
  HYPRE_Int *i_domain_dof;
  HYPRE_Int *j_domain_dof;
  double *domain_matrixinverse;
  hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
  HYPRE_Int num_variables;
  HYPRE_Int num_cols_offd;
  double *scale_ext;
  double *buf_data;
  HYPRE_Int index; 
 
  HYPRE_Int piv_counter = 0;
  HYPRE_Int one = 1;

  char uplo = 'L';
  HYPRE_Int jj,i,j, j_loc; /*, j_loc, k_loc;*/


  HYPRE_Int matrix_size, matrix_size_counter = 0;

  /* initiate:      ----------------------------------------------- */
  num_variables = hypre_CSRMatrixNumRows(A_diag); 
  num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
  x_data = hypre_VectorData(hypre_ParVectorLocalVector(X));
  vtemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

  if (use_nonsymm)
     uplo = 'N';
  
  hypre_ParVectorCopy(F,Vtemp);
  hypre_ParCSRMatrixMatvec(-1.0,A,X,1.0,Vtemp); 

  /* forward solve: ----------------------------------------------- */

  num_domains = hypre_CSRMatrixNumRows(domain_structure);
  max_domain_size = hypre_CSRMatrixNumCols(domain_structure);
  i_domain_dof = hypre_CSRMatrixI(domain_structure);
  j_domain_dof = hypre_CSRMatrixJ(domain_structure);
  domain_matrixinverse = hypre_CSRMatrixData(domain_structure);
  aux = hypre_CTAlloc(double, max_domain_size);

  if (comm_pkg)
  {
     num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
     send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
     send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

     buf_data = hypre_CTAlloc(double, send_map_starts[num_sends]);
     x_ext_data = hypre_CTAlloc(double, num_cols_offd);
     vtemp_ext_data = hypre_CTAlloc(double, num_cols_offd);
     scale_ext = hypre_CTAlloc(double, num_cols_offd);

     index = 0;
     for (i=0; i < num_sends; i++)
     {
        for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
           buf_data[index++] = vtemp_data[send_map_elmts[j]];
     }

     comm_handle = hypre_ParCSRCommHandleCreate(1,comm_pkg,buf_data,
		vtemp_ext_data);
     hypre_ParCSRCommHandleDestroy(comm_handle);
     comm_handle = NULL;

     index = 0;
     for (i=0; i < num_sends; i++)
     {
        for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
           buf_data[index++] = scale[send_map_elmts[j]];
     }


     comm_handle = hypre_ParCSRCommHandleCreate(1,comm_pkg,buf_data,scale_ext);
     hypre_ParCSRCommHandleDestroy(comm_handle);
     comm_handle = NULL;
  }

  for (i=0; i < num_cols_offd; i++)
     x_ext_data[i] = 0.0;

  matrix_size_counter = 0;
  for (i=0; i < num_domains; i++)
  {
     matrix_size = i_domain_dof[i+1] - i_domain_dof[i];
     
     /* copy data contiguously into aux  --------------------------- */
     
     jj = 0;
     for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
     {
        j_loc = j_domain_dof[j]; 
        if (j_loc < num_variables)
           aux[jj] = vtemp_data[j_loc];
        else
           aux[jj] = vtemp_ext_data[j_loc-num_variables];
        jj++;
     }
     /* solve for correction: ------------------------------------- */
/* OLD -  HYPRE_USING_ESSL
     dpps(&domain_matrixinverse[matrix_size_counter], matrix_size, aux, 1);
*/
     if (use_nonsymm)
     {
        
#ifdef HYPRE_USING_ESSL
         dgetrs(&uplo, matrix_size, one, 
                &domain_matrixinverse[matrix_size_counter], 
                matrix_size, &pivots[piv_counter], aux,
                matrix_size, &ierr); 
#else
         hypre_F90_NAME_LAPACK(dgetrs, DGETRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, &pivots[piv_counter], aux,
                                               &matrix_size, &ierr); 
#endif
     }
     
     else
     {
        
#ifdef HYPRE_USING_ESSL
            dpotrs(&uplo, matrix_size, one, 
                   &domain_matrixinverse[matrix_size_counter], 
                   matrix_size, aux,
                   matrix_size, &ierr); 
#else
            hypre_F90_NAME_LAPACK(dpotrs, DPOTRS)(&uplo, &matrix_size, &one, 
                                               &domain_matrixinverse[matrix_size_counter], 
                                               &matrix_size, aux,
                                               &matrix_size, &ierr); 
#endif
     }
     
     if (ierr)  hypre_error(HYPRE_ERROR_GENERIC);
     jj = 0;
     for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
     {
        j_loc = j_domain_dof[j]; 
        if (j_loc < num_variables)
           x_data[j_loc]+= scale[j_loc] * aux[jj++];
        else
        {
           j_loc -= num_variables;
           x_ext_data[j_loc] += scale_ext[j_loc] * aux[jj++];
        }
     }
/* OLD -  HYPRE_USING_ESSL
     matrix_size_counter += matrix_size * (matrix_size+1)/2;  
*/
     matrix_size_counter += matrix_size * matrix_size;
     piv_counter += matrix_size;

     
  }

  if (comm_pkg)
  {
     comm_handle=hypre_ParCSRCommHandleCreate (2,comm_pkg,x_ext_data,buf_data); 

     hypre_ParCSRCommHandleDestroy(comm_handle);
     comm_handle = NULL;

     index = 0;
     for (i=0; i < num_sends; i++)
     {
        for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
	   x_data[send_map_elmts[j]] += buf_data[index++];
     }

     hypre_TFree(buf_data);
     hypre_TFree(x_ext_data);
     hypre_TFree(vtemp_ext_data);
     hypre_TFree(scale_ext);
  }
  hypre_TFree(aux);

  return hypre_error_flag;

}

/*--------------------------------------------------------------------------
 * hypre_ParAMGCreateDomainDof:  
 *--------------------------------------------------------------------------*/

/*****************************************************************************
 *
 * Routine for constructing graph domain_dof with minimal overlap
 *             and computing the respective matrix inverses to be
 *             used in an overlapping additive Schwarz procedure (smoother
 *             in AMG); 
 *
 *****************************************************************************/
HYPRE_Int
hypre_ParAMGCreateDomainDof(hypre_ParCSRMatrix   *A,
                            HYPRE_Int domain_type, HYPRE_Int overlap,
                            HYPRE_Int num_functions, HYPRE_Int *dof_func,
                            hypre_CSRMatrix     **domain_structure_pointer,
                            HYPRE_Int **piv_pointer, HYPRE_Int use_nonsymm)

{
  hypre_CSRMatrix *domain_structure;
  HYPRE_Int *i_domain_dof, *j_domain_dof;
  double *domain_matrixinverse;
  HYPRE_Int num_domains;

  hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
  HYPRE_Int *a_diag_i = hypre_CSRMatrixI(A_diag);
  HYPRE_Int *a_diag_j = hypre_CSRMatrixJ(A_diag);
  double *a_diag_data = hypre_CSRMatrixData(A_diag);
  HYPRE_Int num_variables = hypre_CSRMatrixNumRows(A_diag);
  HYPRE_Int first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
  HYPRE_Int col_0 = first_col_diag -1 ;
  HYPRE_Int col_n = first_col_diag + num_variables;

  hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
  HYPRE_Int *a_offd_i = hypre_CSRMatrixI(A_offd);
  HYPRE_Int *a_offd_j = hypre_CSRMatrixJ(A_offd);
  double *a_offd_data = hypre_CSRMatrixData(A_offd);
  HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
  HYPRE_Int *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);

  hypre_CSRMatrix *A_ext;
  HYPRE_Int *a_ext_i;
  HYPRE_Int *a_ext_j;
  double *a_ext_data;

  /* HYPRE_Int *i_dof_to_accept_weight; */
  HYPRE_Int *i_dof_to_prefer_weight,
    *w_dof_dof, *i_dof_weight;
  HYPRE_Int *i_dof_to_aggregate, *i_aggregate_dof, *j_aggregate_dof;
  
  HYPRE_Int *i_dof_index;
  HYPRE_Int *i_dof_index_offd;
  HYPRE_Int *i_proc;
  /* HYPRE_Int *row_starts = hypre_ParCSRMatrixRowStarts(A);*/
  hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  HYPRE_Int num_recvs = 0;
  HYPRE_Int *recv_vec_starts = NULL;

  HYPRE_Int ierr = 0;
  HYPRE_Int i,j,k, jj,  l_loc, i_loc, j_loc;
  HYPRE_Int i_dof;
  HYPRE_Int nf;
  HYPRE_Int *i_local_to_global;
  HYPRE_Int *i_global_to_local;

  HYPRE_Int local_dof_counter, max_local_dof_counter=0; 

  HYPRE_Int domain_dof_counter = 0, domain_matrixinverse_counter = 0;

  double *AE;

 
  HYPRE_Int *ipiv;
  char uplo = 'L';
  HYPRE_Int piv_counter;
  HYPRE_Int *piv = NULL;


  HYPRE_Int cnt, indx;
  HYPRE_Int num_procs, my_id;




  hypre_MPI_Comm_size(hypre_ParCSRMatrixComm(A),&num_procs);
  hypre_MPI_Comm_size(hypre_ParCSRMatrixComm(A),&my_id);

  /* --------------------------------------------------------------------- */

  /*=======================================================================*/
  /*    create artificial domains by agglomeration;                        */
  /*=======================================================================*/

  /*hypre_printf("----------- create artificials domain by agglomeration;  ======\n");
*/
  i_aggregate_dof = hypre_CTAlloc(HYPRE_Int,num_variables+1);
  j_aggregate_dof = hypre_CTAlloc (HYPRE_Int,num_variables);

  if (domain_type == 2)
  {
     i_dof_to_prefer_weight = hypre_CTAlloc (HYPRE_Int, num_variables);
     w_dof_dof = hypre_CTAlloc (HYPRE_Int, a_diag_i[num_variables]);

     for (i=0; i<num_variables; i++)
        for (j=a_diag_i[i]; j< a_diag_i[i+1]; j++)
        {
	   if (a_diag_j[j] == i) 
	      w_dof_dof[j]=0;
	   else
	      w_dof_dof[j]=1;
        }


     /*hypre_printf("end computing weights for agglomeration procedure: --------\n");
*/

     i_dof_weight = hypre_CTAlloc (HYPRE_Int, num_variables);
     hypre_AMGeAgglomerate(i_aggregate_dof, j_aggregate_dof,
			       a_diag_i, a_diag_j, w_dof_dof,
			       a_diag_i, a_diag_j,
			       a_diag_i, a_diag_j,
			       i_dof_to_prefer_weight,
			       i_dof_weight,
			       num_variables, num_variables,
			       &num_domains);

     hypre_TFree(i_dof_to_prefer_weight);
     hypre_TFree(i_dof_weight);
     hypre_TFree(w_dof_dof);

  }
  else
  {
     nf = 1;
     if (domain_type == 1) nf = num_functions;

     num_domains = num_variables/nf;
     for (i=0; i < num_domains+1; i++)
     {
        i_aggregate_dof[i] = nf*i;
     }
     for (i=0; i < num_variables; i++)
     {
        j_aggregate_dof[i] = i;
     }
  }
  /*hypre_printf("num_variables: %d, num_domains: %d\n", num_variables, num_domains);
*/
  if (overlap == 1)
  {
     i_domain_dof = hypre_CTAlloc(HYPRE_Int, num_domains+1);

     i_dof_to_aggregate = hypre_CTAlloc (HYPRE_Int, num_variables);
     for (i=0; i < num_domains; i++)
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
           i_dof_to_aggregate[j_aggregate_dof[j]] = i;

     i_proc = hypre_CTAlloc (HYPRE_Int, num_cols_offd);

     for (i=0; i < num_cols_offd; i++)
	i_proc[i] = 0;

     if (comm_pkg)
     {
	num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);    
	recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);    
     }
     else if (num_procs > 1)
     {

        hypre_MatvecCommPkgCreate(A);

        comm_pkg = hypre_ParCSRMatrixCommPkg(A);
	num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);    
	recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);    
     }

     for (i=0; i < num_recvs; i++)
	for (indx=recv_vec_starts[i]; indx < recv_vec_starts[i+1]; indx++)
              i_proc[indx] = i;

     /* make domains from aggregates: *********************************/

     i_dof_index = hypre_CTAlloc (HYPRE_Int, num_variables);

     i_dof_index_offd = hypre_CTAlloc (HYPRE_Int, num_cols_offd);

     for (i=0; i < num_variables; i++)
        i_dof_index[i] = -1;

     for (i=0; i < num_cols_offd; i++)
        i_dof_index_offd[i] = -1;

     domain_dof_counter=0;
     for (i=0; i < num_domains; i++)
     {
        i_domain_dof[i] =  domain_dof_counter;
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
        {
	   i_dof_index[j_aggregate_dof[j]]=-1;
     	}
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
        {
           for (k=a_diag_i[j_aggregate_dof[j]];
	          	k<a_diag_i[j_aggregate_dof[j]+1]; k++)
	      if (i_dof_to_aggregate[a_diag_j[k]] >= i 
	     	&& i_dof_index[a_diag_j[k]]==-1)
	      {
	         i_dof_index[a_diag_j[k]]++;
	         domain_dof_counter++;
	      }
           for (k=a_offd_i[j_aggregate_dof[j]];
	    		k<a_offd_i[j_aggregate_dof[j]+1]; k++)
	      if (i_proc[a_offd_j[k]] > my_id
	     	&& i_dof_index_offd[a_offd_j[k]]==-1)
	      {
	         i_dof_index_offd[a_offd_j[k]]++;
	         domain_dof_counter++;
	      }
        }
     }

     for (i=0; i < num_variables; i++)
        i_dof_index[i] = -1;

     for (i=0; i < num_cols_offd; i++)
        i_dof_index_offd[i] = -1;

     i_domain_dof[num_domains] =  domain_dof_counter;
     j_domain_dof = hypre_CTAlloc (HYPRE_Int, domain_dof_counter);

     domain_dof_counter=0;
     for (i=0; i < num_domains; i++)
     {
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
        {
	   i_dof_index[j_aggregate_dof[j]]=-1;
     	}
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
        {
	   for (k=a_diag_i[j_aggregate_dof[j]];
	     		k<a_diag_i[j_aggregate_dof[j]+1]; k++)
	      if (i_dof_to_aggregate[a_diag_j[k]] >= i
	      		&& i_dof_index[a_diag_j[k]]==-1)
	      {
	         i_dof_index[a_diag_j[k]]++;
	         j_domain_dof[domain_dof_counter] = a_diag_j[k];
	         domain_dof_counter++;
	      }
	   for (k=a_offd_i[j_aggregate_dof[j]];
	     		k<a_offd_i[j_aggregate_dof[j]+1]; k++)
	      if (i_proc[a_offd_j[k]] > my_id
	      		&& i_dof_index_offd[a_offd_j[k]]==-1)
	      {
	         i_dof_index_offd[a_offd_j[k]]++;
	         j_domain_dof[domain_dof_counter] = a_offd_j[k]+num_variables;
	         domain_dof_counter++;
	      }
         }
     }

     hypre_TFree(i_aggregate_dof);
     hypre_TFree(j_aggregate_dof);
     hypre_TFree(i_dof_to_aggregate);
     hypre_TFree(i_dof_index);
     hypre_TFree(i_dof_index_offd);
     hypre_TFree(i_proc);
  }
  else if (overlap == 2)
  {
     i_domain_dof = hypre_CTAlloc(HYPRE_Int, num_domains+1);

     i_dof_to_aggregate = hypre_CTAlloc (HYPRE_Int, num_variables);
     for (i=0; i < num_domains; i++)
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
           i_dof_to_aggregate[j_aggregate_dof[j]] = i;

     /* make domains from aggregates: *********************************/

     i_dof_index = hypre_CTAlloc (HYPRE_Int, num_variables);

     i_dof_index_offd = hypre_CTAlloc (HYPRE_Int, num_cols_offd);

     for (i=0; i < num_variables; i++)
        i_dof_index[i] = -1;

     for (i=0; i < num_cols_offd; i++)
        i_dof_index_offd[i] = -1;

     domain_dof_counter=0;
     for (i=0; i < num_domains; i++)
     {
        i_domain_dof[i] =  domain_dof_counter;
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
        {
           for (k=a_diag_i[j_aggregate_dof[j]];
	          	k<a_diag_i[j_aggregate_dof[j]+1]; k++)
	      if ( i_dof_index[a_diag_j[k]]==-1)
	      {
	         i_dof_index[a_diag_j[k]]++;
	         domain_dof_counter++;
	      }
           for (k=a_offd_i[j_aggregate_dof[j]];
	    		k<a_offd_i[j_aggregate_dof[j]+1]; k++)
	      if ( i_dof_index_offd[a_offd_j[k]]==-1)
	      {
	         i_dof_index_offd[a_offd_j[k]]++;
	         domain_dof_counter++;
	      }
        }
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
        {
           for (k=a_diag_i[j_aggregate_dof[j]];
	    		k<a_diag_i[j_aggregate_dof[j]+1]; k++)
	      i_dof_index[a_diag_j[k]]=-1;
           for (k=a_offd_i[j_aggregate_dof[j]];
	    		k<a_offd_i[j_aggregate_dof[j]+1]; k++)
	      i_dof_index_offd[a_offd_j[k]]=-1;
     	}

     }

     for (i=0; i < num_variables; i++)
        i_dof_index[i] = -1;

     for (i=0; i < num_cols_offd; i++)
        i_dof_index_offd[i] = -1;

     i_domain_dof[num_domains] =  domain_dof_counter;
     j_domain_dof = hypre_CTAlloc (HYPRE_Int, domain_dof_counter);

     domain_dof_counter=0;
     for (i=0; i < num_domains; i++)
     {
        for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
        {
	   for (k=a_diag_i[j_aggregate_dof[j]];
	     		k<a_diag_i[j_aggregate_dof[j]+1]; k++)
	      if ( i_dof_index[a_diag_j[k]]==-1)
	      {
	         i_dof_index[a_diag_j[k]]++;
	         j_domain_dof[domain_dof_counter] = a_diag_j[k];
	         domain_dof_counter++;
	      }
	   for (k=a_offd_i[j_aggregate_dof[j]];
	     		k<a_offd_i[j_aggregate_dof[j]+1]; k++)
	      if ( i_dof_index_offd[a_offd_j[k]]==-1)
	      {
	         i_dof_index_offd[a_offd_j[k]]++;
	         j_domain_dof[domain_dof_counter] = a_offd_j[k]+num_variables;
	         domain_dof_counter++;
	      }
         }

         for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
         {
	    for (k=a_diag_i[j_aggregate_dof[j]];
	     	k<a_diag_i[j_aggregate_dof[j]+1]; k++)
	    i_dof_index[a_diag_j[k]]=-1;
	    for (k=a_offd_i[j_aggregate_dof[j]];
	     	k<a_offd_i[j_aggregate_dof[j]+1]; k++)
	       i_dof_index_offd[a_offd_j[k]]=-1;
         }

     }

     hypre_TFree(i_aggregate_dof);
     hypre_TFree(j_aggregate_dof);
     hypre_TFree(i_dof_to_aggregate);
     hypre_TFree(i_dof_index);
     hypre_TFree(i_dof_index_offd);
  }
  else
  {
     i_domain_dof = i_aggregate_dof;
     j_domain_dof = j_aggregate_dof;
  }

  /*hypre_printf("END domain_dof computations: =================================\n");
*/
  domain_matrixinverse_counter = 0;
  local_dof_counter = 0;
  piv_counter = 0;

  for (i=0; i < num_domains; i++)
  {
     local_dof_counter = i_domain_dof[i+1]-i_domain_dof[i];
/* OLD- HYPRE_USING_ESSL
      domain_matrixinverse_counter+= local_dof_counter*(local_dof_counter+1)/2;
*/
      domain_matrixinverse_counter+= local_dof_counter * local_dof_counter;
      piv_counter += local_dof_counter;

      if (local_dof_counter > max_local_dof_counter)
	max_local_dof_counter = local_dof_counter;
    }

  domain_matrixinverse = hypre_CTAlloc(double, domain_matrixinverse_counter);
  if (use_nonsymm)
     piv = hypre_CTAlloc(HYPRE_Int, piv_counter);

/* OLD- HYPRE_USING_ESSL
  AE = hypre_CTAlloc(double, max_local_dof_counter*max_local_dof_counter);
*/

  if (num_procs > 1)
  {
     A_ext = hypre_ParCSRMatrixExtractBExt(A,A,1);
     a_ext_i = hypre_CSRMatrixI(A_ext);
     a_ext_j = hypre_CSRMatrixJ(A_ext);
     a_ext_data = hypre_CSRMatrixData(A_ext);
  }
  else 
     A_ext = NULL;

  i_local_to_global = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);

  i_global_to_local = hypre_CTAlloc(HYPRE_Int, num_variables+num_cols_offd);

  for (i=0; i < num_variables+num_cols_offd; i++)
    i_global_to_local[i] = -1;

  piv_counter = 0;
  domain_matrixinverse_counter = 0;
  for (i=0; i < num_domains; i++)
    {
      local_dof_counter = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  i_global_to_local[j_domain_dof[j]] = local_dof_counter;
	  i_local_to_global[local_dof_counter] = j_domain_dof[j];
	  local_dof_counter++;
	}

      /* get local matrix in AE: ======================================== */

      AE = &domain_matrixinverse[domain_matrixinverse_counter];
      ipiv = &piv[piv_counter];

      cnt = 0;  
      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	for (j_loc=0; j_loc < local_dof_counter; j_loc++)
	  AE[cnt++] = 0.e0;

      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	{
	  i_dof = i_local_to_global[i_loc];
	  if (i_dof < num_variables)
	  {
	    for (j=a_diag_i[i_dof]; j < a_diag_i[i_dof+1]; j++)
	    {
	      j_loc = i_global_to_local[a_diag_j[j]];
	      if (j_loc >=0)
		AE[i_loc + j_loc * local_dof_counter] = a_diag_data[j];
	    }
	    for (j=a_offd_i[i_dof]; j < a_offd_i[i_dof+1]; j++)
	    {
	      j_loc = i_global_to_local[a_offd_j[j]+num_variables];
	      if (j_loc >=0)
		AE[i_loc + j_loc * local_dof_counter] = a_offd_data[j];
	    }
	  }
	  else
	  {
	    i_dof -= num_variables;
	    for (j=a_ext_i[i_dof]; j < a_ext_i[i_dof+1]; j++)
	    {
	      jj = a_ext_j[j];
	      if (jj > col_0 && jj < col_n)
	      {
	         jj = jj - first_col_diag;
	      }
	      else
	      {
		 jj = hypre_BinarySearch(col_map_offd,jj,num_cols_offd);
		 if (jj > -1) jj += num_variables;
	      }
	      if (jj > -1)
	      {
	         j_loc = i_global_to_local[jj];
	         if (j_loc >=0)
		    AE[i_loc + j_loc * local_dof_counter] = a_ext_data[j];
	      }
	    }
	  }
	}

/* OLD- HYPRE_USING_ESSL
 	cnt = local_dof_counter;
        for (j_loc=1; j_loc < local_dof_counter; j_loc++)
	   for (i_loc=j_loc; i_loc < local_dof_counter; i_loc++)
	      AE[cnt++] = AE[i_loc + j_loc * local_dof_counter];
   	dppf(AE, local_dof_counter, 1);
        size = local_dof_counter*(local_dof_counter+1)/2;
 	cnt = domain_matrixinverse_counter;
	for (i_loc = 0; i_loc < size; i_loc++)
	   domain_matrixinverse[cnt++] = AE[i_loc];
        domain_matrixinverse_counter += size;
*/
       
      if (use_nonsymm)
      {
#ifdef HYPRE_USING_ESSL
         dgetrf(local_dof_counter,local_dof_counter, AE,
                local_dof_counter, ipiv, &ierr);
#else
         hypre_F90_NAME_LAPACK(dgetrf,DGETRF)(&local_dof_counter,
                                              &local_dof_counter, AE,
                                              &local_dof_counter, ipiv, &ierr);
#endif
         piv_counter +=local_dof_counter;
      }
      
      else
      {
#ifdef HYPRE_USING_ESSL         
         dpotrf(&uplo,local_dof_counter, AE,
                local_dof_counter, &ierr);
#else
         hypre_F90_NAME_LAPACK(dpotrf,DPOTRF)(&uplo,&local_dof_counter, AE,
                                              &local_dof_counter, &ierr);
#endif
        
      }
      
      domain_matrixinverse_counter+=local_dof_counter*local_dof_counter;



      for (l_loc=0; l_loc < local_dof_counter; l_loc++)
	i_global_to_local[i_local_to_global[l_loc]] = -1;
	  
    }

  hypre_TFree(i_local_to_global);
  hypre_TFree(i_global_to_local);
  hypre_CSRMatrixDestroy(A_ext);

/* OLD -HYPRE_USING_ESSL
  hypre_TFree(AE);
*/  

  domain_structure = hypre_CSRMatrixCreate(num_domains, max_local_dof_counter,
			i_domain_dof[num_domains]);

  hypre_CSRMatrixI(domain_structure) = i_domain_dof;
  hypre_CSRMatrixJ(domain_structure) = j_domain_dof;
  hypre_CSRMatrixData(domain_structure) = domain_matrixinverse;

  *domain_structure_pointer = domain_structure;
  *piv_pointer = piv;
  return hypre_error_flag;

}

HYPRE_Int
hypre_ParGenerateScale(hypre_ParCSRMatrix *A,
		    hypre_CSRMatrix *domain_structure,
                    double           relaxation_weight,
                    double         **scale_pointer)
{
   HYPRE_Int num_domains = hypre_CSRMatrixNumRows(domain_structure);
   HYPRE_Int *i_domain_dof = hypre_CSRMatrixI(domain_structure);
   HYPRE_Int *j_domain_dof = hypre_CSRMatrixJ(domain_structure);
   HYPRE_Int i, j;
   double *scale;
   double *scale_ext;
   double *scale_int;

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int num_sends = 0;
   HYPRE_Int *send_map_starts;
   HYPRE_Int *send_map_elmts;

   HYPRE_Int num_variables = hypre_ParCSRMatrixNumRows(A);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
   HYPRE_Int j_loc, index, start;
  
   hypre_ParCSRCommHandle *comm_handle;

   if (comm_pkg)
   {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
   }

   scale = hypre_CTAlloc(double, num_variables);
   if (num_cols_offd) scale_ext = hypre_CTAlloc(double,num_cols_offd);

   for (i=0; i < num_domains; i++)
   {  
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
      {
	 j_loc = j_domain_dof[j];
	 if (j_loc < num_variables)
	    scale[j_loc] += 1.0;
	 else
	    scale_ext[j_loc-num_variables] += 1.0;
      }
   }
   if (comm_pkg)
   {
      scale_int = hypre_CTAlloc(double, send_map_starts[num_sends]);
      comm_handle = hypre_ParCSRCommHandleCreate (2,comm_pkg,scale_ext,scale_int); 

      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = send_map_starts[i];
      for (j=start; j < send_map_starts[i+1]; j++)
	 scale[send_map_elmts[j]] += scale_int[index++];
   }

   if (comm_pkg) hypre_TFree(scale_int);
   if (num_cols_offd) hypre_TFree(scale_ext);

   for (i=0; i < num_variables; i++)
      scale[i] = relaxation_weight/scale[i];

   *scale_pointer = scale;

   return hypre_error_flag;
}

HYPRE_Int
hypre_ParGenerateHybridScale(hypre_ParCSRMatrix *A,
		    hypre_CSRMatrix  *domain_structure,
                    hypre_CSRMatrix **A_boundary_pointer,
                    double          **scale_pointer)
{
   hypre_CSRMatrix *A_ext;
   HYPRE_Int *A_ext_i;
   HYPRE_Int *A_ext_j;
   double *A_ext_data;

   hypre_CSRMatrix *A_boundary;
   HYPRE_Int *A_boundary_i;
   HYPRE_Int *A_boundary_j;
   double *A_boundary_data;
 
   HYPRE_Int num_domains = hypre_CSRMatrixNumRows(domain_structure);
   HYPRE_Int *i_domain_dof = hypre_CSRMatrixI(domain_structure);
   HYPRE_Int *j_domain_dof = hypre_CSRMatrixJ(domain_structure);
   HYPRE_Int i, j, j_col;
   double *scale;
   double *scale_ext;
   double *scale_int;

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int num_sends = 0;
   HYPRE_Int *send_map_starts;
   HYPRE_Int *send_map_elmts;
   HYPRE_Int *index_ext;

   HYPRE_Int num_variables = hypre_ParCSRMatrixNumRows(A);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
   HYPRE_Int jj, j_loc, index, start;
   HYPRE_Int col_0, col_n;
   HYPRE_Int *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
  
   hypre_ParCSRCommHandle *comm_handle;

   col_0 = hypre_ParCSRMatrixFirstColDiag(A)-1;
   col_n = col_0+num_variables;

   A_boundary = NULL;

   if (comm_pkg)
   {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
   }

   scale = hypre_CTAlloc(double, num_variables);
   if (num_cols_offd) 
   {
      scale_ext = hypre_CTAlloc(double,num_cols_offd);
      index_ext = hypre_CTAlloc(HYPRE_Int,num_cols_offd);
   }

   for (i=0; i < num_variables; i++)
      scale[i] = 1;

   for (i=0; i < num_cols_offd; i++)
      index_ext[i] = -1;

   for (i=0; i < num_domains; i++)
   {  
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
      {
	 j_loc = j_domain_dof[j];
	 if (j_loc >= num_variables)
	 {
	    j_loc -= num_variables;
	    if (index_ext[j_loc] == -1)
	    {
	       scale_ext[j_loc] += 1.0;
	       index_ext[j_loc] ++;
	    }
	 }
      }
   }
   if (comm_pkg)
   {
      scale_int = hypre_CTAlloc(double, send_map_starts[num_sends]);
      comm_handle=hypre_ParCSRCommHandleCreate(2,comm_pkg,scale_ext,scale_int); 

      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
      A_ext = hypre_ParCSRMatrixExtractBExt(A,A,1);
      A_ext_i = hypre_CSRMatrixI(A_ext);
      A_boundary_i = hypre_CTAlloc(HYPRE_Int,num_cols_offd+1);
      A_ext_j = hypre_CSRMatrixJ(A_ext);
      A_ext_data = hypre_CSRMatrixData(A_ext);
      /* compress A_ext to contain only local data and 
         necessary boundary points*/
      index = 0;
      for (i=0; i < num_cols_offd; i++)
      {
	 A_boundary_i[i] = index;
         for (j = A_ext_i[i]; j < A_ext_i[i+1]; j++)
         {
	    j_col = A_ext_j[j];
	    if (j_col > col_0 && j_col < col_n)
	    {
	       A_ext_j[j] = j_col-col_0;
	       index++;
	    }
	    else  
	    {
	       jj = hypre_BinarySearch(col_map_offd,j_col,num_cols_offd);
	       if (jj > -1 && (scale_ext[jj] > 0))
	       {
		  A_ext_j[j] = num_variables+jj;
	          index++;
	       }
	       else
	       {
		  A_ext_j[j] = -1;
	       }
	    }
	 }
      }
      A_boundary_i[num_cols_offd] = index;

      A_boundary_j = NULL;
      A_boundary_data = NULL;

      if (index)
      {
         A_boundary_j = hypre_CTAlloc(HYPRE_Int,index);
         A_boundary_data = hypre_CTAlloc(double,index);
      }

      index = 0;
      for (i=0; i < A_ext_i[num_cols_offd]; i++)
      {
	 if (A_ext_j[i] > -1)
	 {
	    A_boundary_j[index] = A_ext_j[i];
	    A_boundary_data[index] = A_ext_data[i];
	    index++;
	 }
      }
      A_boundary = hypre_CSRMatrixCreate(num_cols_offd,num_variables, index);
      hypre_CSRMatrixI(A_boundary) = A_boundary_i;
      hypre_CSRMatrixJ(A_boundary) = A_boundary_j;
      hypre_CSRMatrixData(A_boundary) = A_boundary_data;
      hypre_CSRMatrixDestroy(A_ext);
   }

   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = send_map_starts[i];
      for (j=start; j < send_map_starts[i+1]; j++)
	 scale[send_map_elmts[j]] += scale_int[index++];
   }

   if (comm_pkg) hypre_TFree(scale_int);
   if (num_cols_offd) 
   {
      hypre_TFree(scale_ext);
      hypre_TFree(index_ext);
   }

   for (i=0; i < num_variables; i++)
      scale[i] = 1.0/scale[i];

   *scale_pointer = scale;
   *A_boundary_pointer = A_boundary;

   return hypre_error_flag;
}


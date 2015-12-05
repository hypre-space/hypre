/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/





#include "headers.h"

/* -------------------------------------------------------------------------
   dof_domain: for each dof defines neighborhood to build interpolation,
                               using 
	       domain_diagmat (for cut--off scaling) and
	       i_domain_dof, j_dof_domain (for extracting the block of A);

   domain_matrixinverse: contains the inverse of subdomain matrix;

   B can be used to define strength matrix;
   ----------------------------------------------------------------------- */
     


/*--------------------------------------------------------------------------
 * hypre_AMGNodalSchwarzSmoother:
 *--------------------------------------------------------------------------*/



HYPRE_Int
hypre_AMGNodalSchwarzSmoother( hypre_CSRMatrix    *A,

			       HYPRE_Int                *dof_func,
			       HYPRE_Int                 num_functions,

			       HYPRE_Int		   option,
			       HYPRE_Int               **i_domain_dof_pointer,
			       HYPRE_Int               **j_domain_dof_pointer,
			       double            **domain_matrixinverse_pointer,
			       HYPRE_Int                *num_domains_pointer)

{

  /*  option =      0: nodal symGS; 
		    1: next to nodal symGS (overlapping Schwarz) */
	

  HYPRE_Int *i_domain_dof, *j_domain_dof;
  double *domain_matrixinverse;
  HYPRE_Int num_domains;


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


  double *AE, *XE;


  /* PCG arrays: --------------------------------------------------- 
  double *x, *rhs, *v, *w, *d, *aux;

  HYPRE_Int max_iter;

  ------------------------------------------------------------------ */




  /* build dof_node graph: ----------------------------------------- */

  num_nodes = num_dofs / num_functions;

  hypre_printf("\nnum_nodes: %d, num_dofs: %d = %d x %d\n", num_nodes, num_dofs,
	 num_nodes, num_functions);

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
      domain_matrixinverse_counter += local_dof_counter*local_dof_counter;
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


  AE = hypre_CTAlloc(double, max_local_dof_counter *
		     max_local_dof_counter);



  XE = hypre_CTAlloc(double, max_local_dof_counter *
		     max_local_dof_counter);

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
	  for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	    for (j_loc=0; j_loc < local_dof_counter; j_loc++)
	      AE[i_loc + j_loc * local_dof_counter] = 0.e0;
	  
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
	  ierr = matinv(XE, AE, local_dof_counter); 

	  /* hypre_printf("ierr_AE_inv: %d\n", ierr); */
  
	}

      if (option == 1)
	for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	      j_domain_dof[domain_dof_counter+i_loc] 
		= i_local_to_global[i_loc]; 


      if (option == 1)
	for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	  for (j_loc=0; j_loc < local_dof_counter; j_loc++)
	    domain_matrixinverse[domain_matrixinverse_counter
				+ i_loc + j_loc * local_dof_counter]
	      = XE[i_loc + j_loc * local_dof_counter];

      if (option == 0)
	{

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

	  ierr = matinv(XE, AE, int_dof_counter);


	  for (i_loc=0; i_loc < int_dof_counter; i_loc++)
	    {
	      j_domain_dof[domain_dof_counter + i_loc] =
		i_local_to_global[i_int_to_local[i_loc]];

	      for (j_loc=0; j_loc < int_dof_counter; j_loc++)
		domain_matrixinverse[domain_matrixinverse_counter
				    + i_loc + j_loc * int_dof_counter]
		  = XE[i_loc + j_loc * int_dof_counter];
	    }

	  domain_dof_counter+=int_dof_counter;
	  domain_matrixinverse_counter+=int_dof_counter*int_dof_counter;
	}
      else
	{
	  domain_dof_counter+=local_dof_counter;
	  domain_matrixinverse_counter+=local_dof_counter*local_dof_counter;
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


  hypre_TFree(AE);
  hypre_TFree(XE);


  *i_domain_dof_pointer = i_domain_dof;
  *j_domain_dof_pointer = j_domain_dof;

  *num_domains_pointer = num_domains;

  *domain_matrixinverse_pointer = domain_matrixinverse;



  /* hypre_printf("exit *Schwarz*: ===============================\n\n"); */

  /* -----------------------------------------------------------------
   x   = hypre_CTAlloc(double, num_dofs); 
   rhs = hypre_CTAlloc(double, num_dofs);

   v   = hypre_CTAlloc(double, num_dofs);
   w   = hypre_CTAlloc(double, num_dofs);
   d   = hypre_CTAlloc(double, num_dofs);
   aux = hypre_CTAlloc(double, num_dofs);

   for (i=0; i < num_dofs; i++)
     x[i] = 0.e0;

   for (i=0; i < num_dofs; i++)
     rhs[i] = rand();


   max_iter = 1000;

   hypre_printf("\nenter SchwarzPCG: =======================================\n");

   ierr = hypre_Schwarzpcg(x, rhs,
			   a_dof_dof,
			   i_dof_dof, j_dof_dof,

			   i_domain_dof, j_domain_dof,
			   domain_matrixinverse, 

			   num_domains,

			   v, w, d, aux,

			   max_iter, 

			   num_dofs);


   hypre_printf("\n\n=======================================================\n");
   hypre_printf("             END test PCG solve:                           \n");
   hypre_printf("===========================================================\n");
 
   hypre_TFree(x);
   hypre_TFree(rhs);

   hypre_TFree(aux);
   hypre_TFree(v);
   hypre_TFree(w);
   hypre_TFree(d);


   ----------------------------------------------------------------------- */

   return ierr;

}

HYPRE_Int hypre_SchwarzSolve(hypre_CSRMatrix *A,
		       hypre_Vector *rhs_vector,
		       HYPRE_Int num_domains,
		       HYPRE_Int *i_domain_dof,
		       HYPRE_Int *j_domain_dof,
		       double *domain_matrixinverse,
		       hypre_Vector *x_vector,
		       hypre_Vector *aux_vector)

{
  HYPRE_Int ierr = 0;
  /* HYPRE_Int num_dofs; */
  HYPRE_Int *i_dof_dof;
  HYPRE_Int *j_dof_dof;
  double *a_dof_dof;
  double *x;
  double *rhs;
  double *aux;

  HYPRE_Int i,j,k, j_loc, k_loc;


  HYPRE_Int matrix_size, matrix_size_counter = 0;


  /* initiate:      ----------------------------------------------- */
  /* num_dofs = hypre_CSRMatrixNumRows(A); */
  i_dof_dof = hypre_CSRMatrixI(A);
  j_dof_dof = hypre_CSRMatrixJ(A);
  a_dof_dof = hypre_CSRMatrixData(A);
  x = hypre_VectorData(x_vector);
  rhs = hypre_VectorData(rhs_vector);
  aux = hypre_VectorData(aux_vector);
  /* for (i=0; i < num_dofs; i++)
    x[i] = 0.e0; */
  
  /* forward solve: ----------------------------------------------- */

  matrix_size_counter = 0;
  for (i=0; i < num_domains; i++)
    {
      matrix_size = i_domain_dof[i+1] - i_domain_dof[i];

      /* compute residual: ---------------------------------------- */

      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  aux[j_domain_dof[j]] = rhs[j_domain_dof[j]];
	  for (k=i_dof_dof[j_domain_dof[j]];
	       k<i_dof_dof[j_domain_dof[j]+1]; k++)
	    aux[j_domain_dof[j]] -= a_dof_dof[k] * x[j_dof_dof[k]];
	}

      /* solve for correction: ------------------------------------- */
      
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  j_loc = j-i_domain_dof[i];

	  for (k=i_domain_dof[i]; k < i_domain_dof[i+1]; k++)
	    {
	      k_loc = k-i_domain_dof[i];
	      x[j_domain_dof[j]]+= 
		domain_matrixinverse[matrix_size_counter 
				    + j_loc + k_loc * matrix_size]
		* aux[j_domain_dof[k]];
	    }
	}

      matrix_size_counter += matrix_size * matrix_size;  

    }

  /* backward solve: ------------------------------------------------ */
  for (i=num_domains-1; i > -1; i--)
    {
      matrix_size = i_domain_dof[i+1] - i_domain_dof[i];
      matrix_size_counter -= matrix_size * matrix_size;
      
      /* compute residual: ---------------------------------------- */

      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  aux[j_domain_dof[j]] = rhs[j_domain_dof[j]];
	  for (k=i_dof_dof[j_domain_dof[j]];
	       k<i_dof_dof[j_domain_dof[j]+1]; k++)
	    aux[j_domain_dof[j]] -= a_dof_dof[k] * x[j_dof_dof[k]];
	}


      /* solve for correction: ------------------------------------- */
      
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  j_loc = j-i_domain_dof[i];

	  for (k=i_domain_dof[i]; k < i_domain_dof[i+1]; k++)
	    {
	      k_loc = k-i_domain_dof[i];
	      x[j_domain_dof[j]]+= 
		domain_matrixinverse[matrix_size_counter 
				    + j_loc + k_loc * matrix_size]
		* aux[j_domain_dof[k]];
	    }
	}
      
    }			      

  return ierr;

}

HYPRE_Int 
transpose_matrix_create(  HYPRE_Int **i_face_element_pointer,
			  HYPRE_Int **j_face_element_pointer,

			  HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,

			  HYPRE_Int num_elements, HYPRE_Int num_faces)

{
  /* FILE *f; */
  HYPRE_Int ierr =0, i, j;

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

  return ierr;

}
HYPRE_Int 
matrix_matrix_product(    HYPRE_Int **i_element_edge_pointer, 
			  HYPRE_Int **j_element_edge_pointer,

			  HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
			  HYPRE_Int *i_face_edge, HYPRE_Int *j_face_edge,

			  HYPRE_Int num_elements, HYPRE_Int num_faces, HYPRE_Int num_edges)

{
  /* FILE *f; */
  HYPRE_Int ierr =0, i, j, k, l, m;

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

  return ierr;

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


			 HYPRE_Int                 **i_domain_dof_pointer,
			 HYPRE_Int                 **j_domain_dof_pointer,
			 double              **domain_matrixinverse_pointer,


			 HYPRE_Int                 *num_domains_pointer) 

{

  HYPRE_Int *i_domain_dof, *j_domain_dof;
  double *domain_matrixinverse;
  HYPRE_Int num_domains;
  

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


  double *AE, *XE;

  /* PCG arrays: --------------------------------------------------- */
  /* double *x, *rhs, *v, *w, *d, *aux;

  HYPRE_Int max_iter; */

  /* --------------------------------------------------------------------- */

  /*=======================================================================*/
  /*    create artificial domains by agglomeration;                        */
  /*=======================================================================*/

  hypre_printf("----------- create artificials domain by agglomeration;  ======\n");


  i_dof_to_prefer_weight = (HYPRE_Int *) malloc(num_dofs * sizeof(HYPRE_Int));
  w_dof_dof = (HYPRE_Int *) malloc(i_dof_dof[num_dofs] * sizeof(HYPRE_Int));

  for (i=0; i < num_dofs; i++)
    i_dof_to_prefer_weight[i] = 0;

  for (i=0; i<num_dofs; i++)
    for (j=i_dof_dof[i]; j< i_dof_dof[i+1]; j++)
      {
	if (j_dof_dof[j] == i) 
	  w_dof_dof[j]=0;
	else
	  w_dof_dof[j]=1;
      }


  hypre_printf("end computing weights for agglomeration procedure: --------\n");


  i_dof_weight = (HYPRE_Int *) malloc(num_dofs * sizeof(HYPRE_Int));
  i_aggregate_dof = (HYPRE_Int *) malloc(num_dofs * sizeof(HYPRE_Int));
  j_aggregate_dof= (HYPRE_Int *) malloc(num_dofs * sizeof(HYPRE_Int));
  ierr = hypre_AMGeAgglomerate(i_aggregate_dof, j_aggregate_dof,

			       i_dof_dof, j_dof_dof, w_dof_dof,
		     
			       i_dof_dof, j_dof_dof,
			       i_dof_dof, j_dof_dof,

			       i_dof_to_prefer_weight,
			       i_dof_weight,

			       num_dofs, num_dofs,
			       &num_domains);



  hypre_printf("num_dofs: %d, num_domains: %d\n", num_dofs, num_domains);

  i_dof_to_aggregate = (HYPRE_Int *) malloc(num_dofs * sizeof(HYPRE_Int));
  for (i=0; i < num_domains; i++)
    for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
      i_dof_to_aggregate[j_aggregate_dof[j]] = i;


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


  free(i_dof_to_prefer_weight);
  free(i_dof_weight);
  free(w_dof_dof);


  /* make domains from aggregates: *********************************/


  i_domain_dof = (HYPRE_Int *) malloc((num_domains+1) * sizeof(HYPRE_Int));

  i_dof_index = (HYPRE_Int *) malloc(num_dofs * sizeof(HYPRE_Int));

  for (i=0; i < num_dofs; i++)
    i_dof_index[i] = -1;

  domain_dof_counter=0;
  for (i=0; i < num_domains; i++)
    {
     i_domain_dof[i] =  domain_dof_counter;
     for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
       for (k=i_dof_dof[j_aggregate_dof[j]];
	    k<i_dof_dof[j_aggregate_dof[j]+1]; k++)
	 if (i_dof_to_aggregate[j_dof_dof[k]] >= i 
	     && i_dof_index[j_dof_dof[k]]==-1)
	   {
	     i_dof_index[j_dof_dof[k]]++;
	     domain_dof_counter++;
	   }

     for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
       for (k=i_dof_dof[j_aggregate_dof[j]];
	    k<i_dof_dof[j_aggregate_dof[j]+1]; k++)
	 i_dof_index[j_dof_dof[k]]=-1;

    }

  i_domain_dof[num_domains] =  domain_dof_counter;
  j_domain_dof = (HYPRE_Int *) malloc(domain_dof_counter * sizeof(HYPRE_Int));

  domain_dof_counter=0;
  for (i=0; i < num_domains; i++)
    {
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

      for (j=i_aggregate_dof[i]; j < i_aggregate_dof[i+1]; j++)
	for (k=i_dof_dof[j_aggregate_dof[j]];
	     k<i_dof_dof[j_aggregate_dof[j]+1]; k++)
	  i_dof_index[j_dof_dof[k]]=-1;

    }

  free(i_aggregate_dof);
  free(j_aggregate_dof);
  free(i_dof_to_aggregate);


  /*
  i_domain_dof = i_aggregate_dof;
  j_domain_dof = j_aggregate_dof;
  */
  hypre_printf("END domain_dof computations: =================================\n");

  domain_matrixinverse_counter = 0;
  local_dof_counter = 0;
  
  for (i=0; i < num_domains; i++)
    {
      local_dof_counter = i_domain_dof[i+1]-i_domain_dof[i];
      domain_matrixinverse_counter+= local_dof_counter * local_dof_counter;

      if (local_dof_counter > max_local_dof_counter)
	max_local_dof_counter = local_dof_counter;
    }

  domain_matrixinverse = hypre_CTAlloc(double, domain_matrixinverse_counter);


  i_local_to_global = hypre_CTAlloc(HYPRE_Int, max_local_dof_counter);


  AE = hypre_CTAlloc(double, max_local_dof_counter *
		     max_local_dof_counter);

  XE = hypre_CTAlloc(double, max_local_dof_counter *
		     max_local_dof_counter);

  /* i_dof_index = (HYPRE_Int *) malloc(num_dofs * sizeof(HYPRE_Int)); */
  i_global_to_local = i_dof_index;

  for (i=0; i < num_dofs; i++)
    i_global_to_local[i] = -1;

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
  
      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	for (j_loc=0; j_loc < local_dof_counter; j_loc++)
	  AE[i_loc + j_loc * local_dof_counter] = 0.e0;

      

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
      ierr = matinv(XE, AE, local_dof_counter);


      /* hypre_printf("ierr_AE_inv: %d\n", ierr); */
  

      for (i_loc=0; i_loc < local_dof_counter; i_loc++)
	for (j_loc=0; j_loc < local_dof_counter; j_loc++)
	  domain_matrixinverse[domain_matrixinverse_counter
			      + i_loc + j_loc * local_dof_counter]
	    = XE[i_loc + j_loc * local_dof_counter];


      domain_matrixinverse_counter+=local_dof_counter*local_dof_counter;


      for (l_loc=0; l_loc < local_dof_counter; l_loc++)
	i_global_to_local[i_local_to_global[l_loc]] = -1;
	  
    }

  hypre_TFree(i_local_to_global);


  hypre_TFree(AE);
  hypre_TFree(XE);


  hypre_TFree(i_dof_index);
  

  *i_domain_dof_pointer = i_domain_dof;
  *j_domain_dof_pointer = j_domain_dof;

  *num_domains_pointer = num_domains;

  *domain_matrixinverse_pointer = domain_matrixinverse;


  
  /*
   x   = hypre_CTAlloc(double, num_dofs); 
   rhs = hypre_CTAlloc(double, num_dofs);

   v   = hypre_CTAlloc(double, num_dofs);
   w   = hypre_CTAlloc(double, num_dofs);
   d   = hypre_CTAlloc(double, num_dofs);
   aux = hypre_CTAlloc(double, num_dofs);

   for (i=0; i < num_dofs; i++)
     x[i] = 0.e0;

   for (i=0; i < num_dofs; i++)
     rhs[i] = rand();


   max_iter = 1000;

   hypre_printf("\nenter SchwarzPCG: =======================================\n");

   ierr = hypre_Schwarzpcg(x, rhs,
			   a_dof_dof,
			   i_dof_dof, j_dof_dof,

			   i_domain_dof, j_domain_dof,
			   domain_matrixinverse, 

			   num_domains,

			   v, w, d, aux,

			   max_iter, 

			   num_dofs);


   hypre_printf("\n\n=======================================================\n");
   hypre_printf("             END test PCG solve:                           \n");
   hypre_printf("===========================================================\n");
 
   hypre_TFree(x);
   hypre_TFree(rhs);

   hypre_TFree(aux);
   hypre_TFree(v);
   hypre_TFree(w);
   hypre_TFree(d);

   hypre_TFree(i_domain_dof);
   hypre_TFree(j_domain_dof);
   hypre_TFree(domain_matrixinverse); 

   */

  return ierr;

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

      return ierr;
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
  HYPRE_Int ierr = 0, weight0;

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


  return ierr;
    
}

HYPRE_Int remove_entry(HYPRE_Int weight, HYPRE_Int *weight_max, 
		 HYPRE_Int *previous, HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last,
		 HYPRE_Int head, HYPRE_Int tail, 
		 HYPRE_Int i)
{
  HYPRE_Int ierr=0, weight0;

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

  return ierr;

}

HYPRE_Int move_entry(HYPRE_Int weight, HYPRE_Int *weight_max, 
	       HYPRE_Int *previous, HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last,
	       HYPRE_Int head, HYPRE_Int tail, 
	       HYPRE_Int i)
{
  HYPRE_Int ierr=0, weight0;

  if (previous[i] != head) next[previous[i]] = next[i];
  previous[next[i]] = previous[i];

  for (weight0=1; weight0 <= weight_max[0]; weight0++)
    {
      if (first[weight0] == i)
	first[weight0] = next[i];
    }

  return ierr;

}




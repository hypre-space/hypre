/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"

int 
matrix_matrix_product(    int **i_element_edge_pointer, 
			  int **j_element_edge_pointer,

			  int *i_element_face, int *j_element_face,
			  int *i_face_edge, int *j_face_edge,

			  int num_elements, int num_faces, int num_edges)

{
  /* FILE *f; */
  int ierr =0, i, j, k, l, m;

  int i_edge_on_local_list, i_edge_on_list;
  int local_element_edge_counter = 0, element_edge_counter = 0;
  int *j_local_element_edge;

  
  int *i_element_edge, *j_element_edge;


  j_local_element_edge = (int *) malloc((num_edges+1) * sizeof(int));

  i_element_edge = (int *) malloc((num_elements+1) * sizeof(int));

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
	    
	      /* printf("element %d  contains edge %d;\n",
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

  j_element_edge = (int *) malloc(i_element_edge[num_elements]
				     * sizeof(int));

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
		      printf("error in j_element_edge size: %d \n",
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
      printf("\nelement: %d has edges:\n", i);  
      for (j=i_element_edge[i]; j < i_element_edge[i+1]; j++)
	{
	  printf("%d ", j_element_edge[j]); 
	  fprintf(f, "%d %d\n", i, j_element_edge[j]);
	}
	  
      printf("\n"); 
    }

  fclose(f);
  */

  /* printf("end element_edge computation: ++++++++++++++++++++++++ \n");*/

  *i_element_edge_pointer = i_element_edge;
  *j_element_edge_pointer = j_element_edge;

  return ierr;

}
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



int
hypre_AMGNodalSchwarzSmoother( hypre_CSRMatrix    *A,
			       int                *dof_func,
			       int                 num_functions,
			       int		   option,
			       int               **i_domain_dof_pointer,
			       int               **j_domain_dof_pointer,
			       double            **domain_matrixinverse_pointer,
			       int                *num_domains_pointer)
{

  /*  option =      0: nodal symGS; 
		    1: next to nodal symGS (overlapping Schwarz) */
							

  int *i_domain_dof, *j_domain_dof;
  double *domain_matrixinverse;
  int num_domains;


  int *i_dof_node, *j_dof_node;
  int *i_node_dof, *j_node_dof;

  int *i_node_dof_dof, *j_node_dof_dof;

  int *i_node_node, *j_node_node;

  int num_nodes;

  int *i_dof_dof = hypre_CSRMatrixI(A);
  int *j_dof_dof = hypre_CSRMatrixJ(A);
  double *a_dof_dof = hypre_CSRMatrixData(A);
  int num_dofs = hypre_CSRMatrixNumRows(A);


  int ierr = 0;
  int i,j,k, l_loc, i_loc, j_loc;
  int i_dof, j_dof;
  int *i_local_to_global;
  int *i_global_to_local;

  int *i_int;
  int *i_int_to_local;
  int *i_ext_to_local;
  int *i_local_to_ext;
  int *dof_domain;

  int local_dof_counter, max_local_dof_counter=0; 
  int int_dof_counter;

  int domain_dof_counter = 0, domain_matrixinverse_counter = 0;


  double *AE, *XE;


  /* build dof_node graph: ----------------------------------------- */
  num_nodes = num_dofs / num_functions;

  printf("\nnum_nodes: %d, num_dofs: %d = %d x %d\n", num_nodes, num_dofs,
	 num_nodes, num_functions);

  i_dof_node = hypre_CTAlloc(int, num_dofs+1);
  j_dof_node = hypre_CTAlloc(int, num_dofs);

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

  
  i_domain_dof = hypre_CTAlloc(int, num_domains+1);

  if (option == 1)
    j_domain_dof = hypre_CTAlloc(int, domain_dof_counter);
  else
    j_domain_dof = hypre_CTAlloc(int, num_dofs);


  if (option == 1)
    domain_matrixinverse = hypre_CTAlloc(double, domain_matrixinverse_counter);
  else
    domain_matrixinverse = hypre_CTAlloc(double, num_dofs * num_functions);



  i_local_to_global = hypre_CTAlloc(int, max_local_dof_counter);


  AE = hypre_CTAlloc(double, max_local_dof_counter *
		     max_local_dof_counter);



  XE = hypre_CTAlloc(double, max_local_dof_counter *
		     max_local_dof_counter);

  i_int_to_local = hypre_CTAlloc(int, max_local_dof_counter);
  i_int_to_local = hypre_CTAlloc(int, max_local_dof_counter);
  i_ext_to_local = hypre_CTAlloc(int, max_local_dof_counter);

  i_local_to_ext = hypre_CTAlloc(int, max_local_dof_counter);


  


  dof_domain = hypre_CTAlloc(int, num_dofs);
  for (i=0; i < num_nodes; i++)
    {
      for (k=i_node_dof[i]; k<i_node_dof[i+1]; k++)
	  {
	    i_dof = j_node_dof[k];
	    dof_domain[i_dof] = i;
	  }
    }



  i_int = hypre_CTAlloc(int, max_local_dof_counter);
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
	    printf("WRONG local indexing: ====================== \n");


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

      /* printf("ierr_AE_inv: %d\n", ierr); */
  

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


  hypre_TFree(j_dof_node);

  hypre_TFree(i_node_dof);
  hypre_TFree(j_node_dof);
  hypre_TFree(i_node_node);
  hypre_TFree(j_node_node);

  hypre_TFree(i_int);


  hypre_TFree(i_int_to_local);
  hypre_TFree(i_local_to_ext);


  hypre_TFree(i_local_to_global);


  hypre_TFree(AE);
  hypre_TFree(XE);

   *i_domain_dof_pointer = i_domain_dof;
   *j_domain_dof_pointer = j_domain_dof;

   *num_domains_pointer = num_domains;

   *domain_matrixinverse_pointer = domain_matrixinverse;

   hypre_TFree(i_domain_dof);
   hypre_TFree(j_domain_dof);
   hypre_TFree(domain_matrixinverse); 

   /* printf("exit *Schwarz*: ===============================\n\n"); */
   return ierr;

}

int hypre_SchwarzSolve(hypre_CSRMatrix *A,
		       hypre_Vector *rhs_vector,
		       int num_domains,
		       int *i_domain_dof,
		       int *j_domain_dof,
		       double *domain_matrixinverse,
		       hypre_Vector *x_vector,
		       hypre_Vector *aux_vector)

{
  int ierr = 0;
  int num_dofs;
  int *i_dof_dof;
  int *j_dof_dof;
  double *a_dof_dof;
  double *x;
  double *rhs;
  double *aux;

  int i,j,k, j_loc, k_loc;


  int matrix_size, matrix_size_counter = 0;


  /* initiate:      ----------------------------------------------- */
  num_dofs = hypre_CSRMatrixNumRows(A);
  i_dof_dof = hypre_CSRMatrixI(A);
  j_dof_dof = hypre_CSRMatrixJ(A);
  a_dof_dof = hypre_CSRMatrixData(A);
  x = hypre_VectorData(x_vector);
  rhs = hypre_VectorData(rhs_vector);
  aux = hypre_VectorData(aux_vector);
  for (i=0; i < num_dofs; i++)
    x[i] = 0.e0;
  
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

int 
transpose_matrix_create(  int **i_face_element_pointer,
			  int **j_face_element_pointer,

			  int *i_element_face, int *j_element_face,

			  int num_elements, int num_faces)

{
  FILE *f;
  int ierr =0, i, j;

  int *i_face_element, *j_face_element;

  /* ======================================================================
     first create face_element graph: -------------------------------------
     ====================================================================== */

  i_face_element = (int *) malloc((num_faces+1) * sizeof(int));
  j_face_element = (int *) malloc(i_element_face[num_elements] * sizeof(int));


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

  /* printf("end building face--element graph: ++++++++++++++++++\n"); */

  /* END building face_element graph; ================================ */

  *i_face_element_pointer = i_face_element;
  *j_face_element_pointer = j_face_element;

  return ierr;

}

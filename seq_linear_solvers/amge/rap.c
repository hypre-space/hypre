/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/



/*****************************************************************************
 *    creates coarse matrix P^TAP from P (interpolation) and A (fine matrix)
 *    all in hypre_CSRMatrix format;
 ****************************************************************************/

#include "headers.h"  

int hypre_AMGeRAP(hypre_CSRMatrix **A_crs_pointer,
		  hypre_CSRMatrix *A, 
		  hypre_CSRMatrix *P)

{
  int ierr = 0;

  int i,j,k,l,m,n,p;
  hypre_CSRMatrix *A_crs;

  int *i_dof_dof_b, *j_dof_dof_b;

  int *i_dof_dof_c, *j_dof_dof_c;
  int *i_dof_dof_c_t, *j_dof_dof_c_t;

  int *i_dof_dof[2], *j_dof_dof[2];

  double *b, *c_dof_dof, *c_t_dof_dof, *sparse_matrix[2];

  int Ndofs[2];

  Ndofs[0] =  hypre_CSRMatrixNumRows(P);
  Ndofs[1] =  hypre_CSRMatrixNumCols(P);


  i_dof_dof_c = hypre_CSRMatrixI(P);
  j_dof_dof_c = hypre_CSRMatrixJ(P);
  c_dof_dof = hypre_CSRMatrixData(P);

  i_dof_dof[0] = hypre_CSRMatrixI(A);
  j_dof_dof[0] = hypre_CSRMatrixJ(A);
  sparse_matrix[0] = hypre_CSRMatrixData(A);


  ierr = matrix_matrix_product(&i_dof_dof_b, &j_dof_dof_b,
			       i_dof_dof[0], j_dof_dof[0], 

			       i_dof_dof_c, j_dof_dof_c, 

			       Ndofs[0], Ndofs[0], Ndofs[1]);



  ierr = transpose_matrix_create_with_data(&i_dof_dof_c_t, 
					   &j_dof_dof_c_t,
					   &c_t_dof_dof,

					   i_dof_dof_c, 
					   j_dof_dof_c,
					   c_dof_dof,
					   
					   Ndofs[0], 
					   Ndofs[1]);



  ierr = matrix_matrix_product(&i_dof_dof[1], &j_dof_dof[1],


			       i_dof_dof_c_t, j_dof_dof_c_t, 
					 

			       i_dof_dof_b, j_dof_dof_b,
					 

			       Ndofs[1], Ndofs[0], Ndofs[1]);	

  sparse_matrix[1] = hypre_CTAlloc(double, i_dof_dof[1][Ndofs[1]]);


  free(i_dof_dof_b);
  free(j_dof_dof_b);


  b = hypre_CTAlloc(double, Ndofs[1]);

  for (i=0; i < Ndofs[1]; i++)
    b[i] = 0.e0;
 

  for (i=0; i < Ndofs[1]; i++)
    {
      for (k=i_dof_dof_c_t[i]; k < i_dof_dof_c_t[i+1]; k++)
	{
	  l = j_dof_dof_c_t[k];         /* l is a fine dof -----------*/
	  for (m=i_dof_dof[0][l]; m<i_dof_dof[0][l+1]; m++)
	    {
	      n = j_dof_dof[0][m];      /* n is a fine dof -----------*/
	      for (p=i_dof_dof_c[n]; p < i_dof_dof_c[n+1]; p++)
		{
		  j = j_dof_dof_c[p];   /* j is a coarse dof ---------*/
		  b[j] +=c_t_dof_dof[k] * sparse_matrix[0][m]
		                      * c_dof_dof[p];
		}
	    }
	}

      for (j=i_dof_dof[1][i]; j < i_dof_dof[1][i+1]; j++)
	{
	  sparse_matrix[1][j] = b[j_dof_dof[1][j]];
	  b[j_dof_dof[1][j]] = 0.e0;
	}
    }


  free(i_dof_dof_c_t);
  free(j_dof_dof_c_t);
  free(c_t_dof_dof);
  free(b);

  A_crs = hypre_CSRMatrixCreate(Ndofs[1], Ndofs[1],
			    i_dof_dof[1][Ndofs[1]]);

  /* printf("coarse matrix nnz: %d\n", i_dof_dof[1][Ndofs[1]]); */

  hypre_CSRMatrixData(A_crs) = sparse_matrix[1];
  hypre_CSRMatrixI(A_crs) = i_dof_dof[1];
  hypre_CSRMatrixJ(A_crs) = j_dof_dof[1];

  *A_crs_pointer = A_crs;


  return ierr;

}

int 
transpose_matrix_create_with_data(  int **i_face_element_pointer,
				    int **j_face_element_pointer,
				    double **data_face_element_pointer,

				    int *i_element_face, 
				    int *j_element_face,
				    double *data_element_face,

				    int num_elements, 
				    int num_faces)

{
  FILE *f;
  int ierr =0, i, j;

  int *i_face_element, *j_face_element;
  double *data_face_element;

  /* ======================================================================
     first create face_element graph: -------------------------------------
     ====================================================================== */

  i_face_element  = (int *) malloc((num_faces+1) * sizeof(int));
  j_face_element  = (int *) malloc(i_element_face[num_elements] * sizeof(int));
  data_face_element= (double *) malloc(i_element_face[num_elements] 
				      * sizeof(double));

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
	data_face_element[i_face_element[j_element_face[j]]] = 
	  data_element_face[j];
	i_face_element[j_element_face[j]]++;
      }

  for (i=num_faces-1; i > -1; i--)
    i_face_element[i+1] = i_face_element[i];

  i_face_element[0] = 0;

  /* printf("end building face--element graph: ++++++++++++++++++\n"); */

  /* END building face_element graph; ================================ */

  *i_face_element_pointer = i_face_element;
  *j_face_element_pointer = j_face_element;
  *data_face_element_pointer = data_face_element;

  return ierr;

}
int 
matrix_matrix_product_with_data(    int **i_element_edge_pointer, 
				    int **j_element_edge_pointer,
				    double **data_element_edge_pointer,

				    int *i_element_face, 
				    int *j_element_face,
				    double *data_element_face,

				    int *i_face_edge, 
				    int *j_face_edge,
				    double *data_face_edge,

				    int num_elements, 
				    int num_faces, 
				    int num_edges)

{
  FILE *f;
  int ierr =0, i, j, k, l, m;

  int i_edge_on_local_list, i_edge_on_list;
  int local_element_edge_counter = 0, element_edge_counter = 0;
  int *j_local_element_edge;

  
  int *i_element_edge, *j_element_edge;
  double *data_element_edge;


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
  data_element_edge = (double *) malloc(i_element_edge[num_elements]
					* sizeof(double));

  for (i=0; i < i_element_edge[num_elements]; i++)
    data_element_edge[i] = 0.e0;


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

  for (i=0; i < num_elements; i++)
    for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
      for (k=i_face_edge[j_element_face[j]];
	   k<i_face_edge[j_element_face[j]+1];k++)
	for (l=i_element_edge[i]; l < i_element_edge[i+1]; l++) 
	  if (j_element_edge[l] == j_face_edge[k])
	    {
	      data_element_edge[l] += data_element_face[j] *
		data_face_edge[k];
	      break;
	    }


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


  /* printf("end element_edge computation: ++++++++++++++++++++++++ \n");*/

  *i_element_edge_pointer = i_element_edge;
  *j_element_edge_pointer = j_element_edge;
  *data_element_edge_pointer = data_element_edge;

  return ierr;

}
    


/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/


#include "headers.h"  

/*****************************************************************************
 *
 * builds interpolation matrices P, and stiffness matrices Matrix;
 *        
 *
 ****************************************************************************/




int hypre_AMGeInterpolationSetup(hypre_CSRMatrix ***P_pointer,

				 hypre_CSRMatrix ***Matrix_pointer,

				 hypre_AMGeMatrixTopology **A,

				 int *level_pointer,

				 /* ------ fine-grid element matrices ----- */
				 int *i_element_chord_0,
				 int *j_element_chord_0,
				 double *a_element_chord_0,

				 int *i_chord_dof_0,
				 int *j_chord_dof_0,

				 /* nnz: of the assembled matrices -------*/
				 int *Num_chords,

				 /* ----- coarse node information  ------ */
				 int **i_node_neighbor_coarsenode,
				 int **j_node_neighbor_coarsenode,

				 int **i_node_coarsenode,
				 int **j_node_coarsenode,


				 /* --------- Dirichlet b.c. ----------- */
				 int *i_dof_on_boundary, 

				 /* -------- PDEsystem information -------- */
				 int system_size,
				 int *i_dof_node_0, int *j_dof_node_0,
				 int *i_node_dof_0, int *j_node_dof_0,

				 int ***i_node_dof_pointer, 
				 int ***j_node_dof_pointer,

				 /* --------------------------------------- */

				 int *Num_elements,
				 int *Num_nodes,
				 int *Num_dofs)



{
  int ierr = 0;

  int i,j,k,l,m;


  int **i_dof_coarsedof, **j_dof_coarsedof;
  int **i_dof_neighbor_coarsedof, **j_dof_neighbor_coarsedof;


  int *i_dof_dof_a, *j_dof_dof_a;
  double *a_dof_dof;

  int *i_dof_index;


  int *i_AE_chord, *j_AE_chord;
  double *a_AE_chord;
					

  int **i_dof_node, **j_dof_node,
    **i_node_dof, **j_node_dof;

  int *i_element_node, *j_element_node;

  int *i_AE_node, *j_AE_node;
  int *i_node_AE, *j_node_AE;

  int *i_AE_element, *j_AE_element;

  int *i_AE_dof, *j_AE_dof;
  int *i_dof_AE, *j_dof_AE;

  int **i_element_chord, **j_element_chord;
  double **a_element_chord;

  int **i_chord_dof, **j_chord_dof;
 
  int num_chords = Num_chords[0];
  int num_elements = Num_elements[0], num_dofs = Num_dofs[0], 
    num_coarsedofs;

  hypre_CSRMatrix **Matrix, **P;

  int level = level_pointer[0];


  int dof_coarsedof_counter, dof_neighbor_coarsedof_counter;


  for (l=0; l < level+1; l++)
    Num_dofs[l] = Num_nodes[l] * system_size;


  i_element_chord = hypre_CTAlloc(int*, level+1);

  j_element_chord = hypre_CTAlloc(int*, level+1);  
  a_element_chord = hypre_CTAlloc(double*, level+1);  

  i_chord_dof = hypre_CTAlloc(int*, level+1);
  j_chord_dof = hypre_CTAlloc(int*, level+1);
 

  i_element_chord[0] = i_element_chord_0;
  j_element_chord[0] = j_element_chord_0;
  a_element_chord[0] = a_element_chord_0;

  i_chord_dof[0] = i_chord_dof_0; 
  j_chord_dof[0] = j_chord_dof_0; 


  i_dof_node = hypre_CTAlloc(int*, level+1);
  j_dof_node = hypre_CTAlloc(int*, level+1);

  i_node_dof = hypre_CTAlloc(int*, level+1);
  j_node_dof = hypre_CTAlloc(int*, level+1);


  i_dof_node[0] = i_dof_node_0;
  j_dof_node[0] = j_dof_node_0;

  i_node_dof[0] = i_node_dof_0;
  j_node_dof[0] = j_node_dof_0;
 
  i_dof_neighbor_coarsedof = hypre_CTAlloc(int*, level);
  j_dof_neighbor_coarsedof = hypre_CTAlloc(int*, level);

  i_dof_coarsedof = hypre_CTAlloc(int*, level);
  j_dof_coarsedof = hypre_CTAlloc(int*, level);


  Matrix = hypre_CTAlloc(hypre_CSRMatrix*, level+1);
  P = hypre_CTAlloc(hypre_CSRMatrix*, level);


  /* assemble initial fine matrix: * ------------------------------------- */
  ierr = hypre_AMGeMatrixAssemble(&Matrix[0],

				  i_element_chord[0],
				  j_element_chord[0],
				  a_element_chord[0],

				  i_chord_dof[0], 
				  j_chord_dof[0],

				  Num_elements[0], 
				  Num_chords[0],
				  Num_dofs[0]);

  printf("nnz[0]: %d\n", hypre_CSRMatrixI(Matrix[0])[Num_dofs[0]]);
  /* impose Dirichlet boundary conditions: -----------------*/
  printf("imposing Dirichlet boundary conditions:====================\n");

  i_dof_dof_a = hypre_CSRMatrixI(Matrix[0]);
  j_dof_dof_a = hypre_CSRMatrixJ(Matrix[0]);
  a_dof_dof   = hypre_CSRMatrixData(Matrix[0]);
  for (i=0; i < Num_dofs[0]; i++)
    for (j=i_dof_dof_a[i]; j < i_dof_dof_a[i+1]; j++)
      if (i_dof_on_boundary[j_dof_dof_a[j]] == 0 
	  &&j_dof_dof_a[j]!=i)
	a_dof_dof[j] = 0.e0;

  for (i=0; i < Num_dofs[0]; i++)
    for (j=i_dof_dof_a[i]; j < i_dof_dof_a[i+1]; j++)
      if (i_dof_on_boundary[i] == 0 &&  j_dof_dof_a[j] !=i)
	a_dof_dof[j] = 0.e0;

  /* hypre_TFree(i_dof_on_boundary); */

  i_dof_index = hypre_CTAlloc(int, num_dofs);

  l=0;
interpolation_step:

  i_element_node = hypre_AMGeMatrixTopologyIElementNode(A[l]);
  j_element_node = hypre_AMGeMatrixTopologyJElementNode(A[l]);
  i_AE_element =  hypre_AMGeMatrixTopologyIAEElement(A[l+1]);
  j_AE_element =  hypre_AMGeMatrixTopologyJAEElement(A[l+1]);


  ierr = matrix_matrix_product(&i_AE_node, &j_AE_node,

			       i_AE_element, j_AE_element,
			       i_element_node, j_element_node,

			       Num_elements[l+1], Num_elements[l], Num_nodes[l]);




  /* free element_node: */


  ierr = transpose_matrix_create(&i_node_AE,
				 &j_node_AE,

				 i_AE_node, j_AE_node,

				 Num_elements[l+1], Num_nodes[l]);

  num_dofs = Num_dofs[l];
  num_coarsedofs = Num_dofs[l+1];

  if (system_size == 1)
    {
      i_AE_dof = i_AE_node;
      j_AE_dof = j_AE_node;

      i_dof_AE = i_node_AE;
      j_dof_AE = j_node_AE;

      i_dof_neighbor_coarsedof[l] = i_node_neighbor_coarsenode[l];
      j_dof_neighbor_coarsedof[l] = j_node_neighbor_coarsenode[l];

      i_dof_coarsedof[l] = i_node_coarsenode[l];
      j_dof_coarsedof[l] = j_node_coarsenode[l];


    }
  else
    {
      ierr = matrix_matrix_product(&i_AE_dof, &j_AE_dof,

				   i_AE_node, j_AE_node,
				   i_node_dof[l], j_node_dof[l],

				   Num_elements[l+1], 
				   Num_nodes[l], num_dofs);

      ierr = matrix_matrix_product(&i_dof_AE, &j_dof_AE,

				   i_dof_node[l], j_dof_node[l],
				   i_node_AE,  j_node_AE,

				   num_dofs,   Num_nodes[l], 
				   Num_elements[l+1]);

      /* free:  node_AE, AE_node */


      i_dof_coarsedof[l] = hypre_CTAlloc(int, num_dofs+1);
      j_dof_coarsedof[l] = hypre_CTAlloc(int, num_coarsedofs);

      for (i=0; i < Num_nodes[l]; i++)
	if (i_node_coarsenode[l][i] == i_node_coarsenode[l][i+1])
	  for (j=i_node_dof[l][i]; j < i_node_dof[l][i+1]; j++)
	    i_dof_index[j_node_dof[l][j]] = -1;
	else
	  for (j=i_node_coarsenode[l][i]; j < i_node_coarsenode[l][i+1]; j++)
	    for (j=i_node_dof[l][i]; j < i_node_dof[l][i+1]; j++)
	      i_dof_index[j_node_dof[l][j]] = 0; 

      dof_coarsedof_counter=0;
      for (i=0; i < num_dofs; i++)
	{
	  i_dof_coarsedof[l][i] = dof_coarsedof_counter;
	  if (i_dof_index[i] == 0)
	    {
	      j_dof_coarsedof[l][dof_coarsedof_counter] = dof_coarsedof_counter;
	      dof_coarsedof_counter++;
	    }
	}

      i_dof_coarsedof[l][num_dofs] = dof_coarsedof_counter;
	
      i_dof_neighbor_coarsedof[l] = hypre_CTAlloc(int, num_dofs+1);
      j_dof_neighbor_coarsedof[l] = hypre_CTAlloc
	(int, system_size*system_size *i_node_neighbor_coarsenode[l][Num_nodes[l]]);
	 
      dof_neighbor_coarsedof_counter=0;
      for (i=0; i < num_dofs; i++)
	{
	  i_dof_neighbor_coarsedof[l][i] = dof_neighbor_coarsedof_counter;
	  if (i_dof_coarsedof[l][i+1] == i_dof_coarsedof[l][i])
	    {
	      j_dof_neighbor_coarsedof[l][dof_neighbor_coarsedof_counter] = i;
	      dof_neighbor_coarsedof_counter++;
	    }
	  else
	    {
	      for (j=i_dof_node[l][i]; j < i_dof_node[l][i+1]; j++)
		for (k=i_node_neighbor_coarsenode[l][j_dof_node[l][j]];
		     k<i_node_neighbor_coarsenode[l][j_dof_node[l][j]+1]; k++)
		  for (m=i_node_dof[l][j_node_neighbor_coarsenode[l][k]];
		       m<i_node_dof[l][j_node_neighbor_coarsenode[l][k]+1]; m++)
		    {
		      j_dof_neighbor_coarsedof[l][dof_neighbor_coarsedof_counter] 
			= j_node_dof[l][m];
		      dof_neighbor_coarsedof_counter++;
		    }
	    }
	

	}

      i_dof_neighbor_coarsedof[l][num_dofs] = dof_neighbor_coarsedof_counter;

    }

  /* assemble AE_matrices: -------------------------------------------- */


  ierr = hypre_AMGeDomainElementSparseAssemble(i_AE_element,
					       j_AE_element,
					       Num_elements[l+1],

					       i_element_chord[l],
					       j_element_chord[l],
					       a_element_chord[l],

					       i_chord_dof[l], 
					       j_chord_dof[l],

					       &i_AE_chord,
					       &j_AE_chord,
					       &a_AE_chord,

					       Num_elements[l], 
					       Num_chords[l],
					       num_dofs);


  if (l > 0)
    {
      hypre_TFree(i_element_chord[l]);
      hypre_TFree(j_element_chord[l]);
      hypre_TFree(a_element_chord[l]);
    }


  hypre_TFree(i_AE_element);
  hypre_TFree(j_AE_element);

  hypre_AMGeMatrixTopologyIAEElement(A[l+1]) = NULL;
  hypre_AMGeMatrixTopologyJAEElement(A[l+1]) = NULL;


  printf("\n\nB U I L D I N G  level[%d] I N T E R P O L A T I O N   M A T R I X\n", l);

  ierr = hypre_AMGeBuildInterpolation(&P[l],

				      &i_element_chord[l+1],
				      &j_element_chord[l+1],
				      &a_element_chord[l+1],

				      &i_chord_dof[l+1],
				      &j_chord_dof[l+1],

				      &Num_chords[l+1],

				      i_AE_dof, j_AE_dof,
				      i_dof_AE, j_dof_AE,

				      i_dof_neighbor_coarsedof[l],
				      j_dof_neighbor_coarsedof[l],

				      i_dof_coarsedof[l],
				      j_dof_coarsedof[l],

				      i_AE_chord,
				      j_AE_chord,
				      a_AE_chord,

				      i_chord_dof[l], 
				      j_chord_dof[l],

				      Num_chords[l],

				      Num_elements[l+1],
				      num_dofs,
				      num_coarsedofs);


  if (l > 0)
    {
      hypre_TFree(i_chord_dof[l]);
      hypre_TFree(j_chord_dof[l]);
    }

  if (system_size > 1)
    { 
      hypre_TFree(i_dof_coarsedof[l]);
      hypre_TFree(j_dof_coarsedof[l]);
    }


  printf("END building Interpolation [%d]: ------------------------------\n", l);

  printf("\nB U I L D I N G  level[%d]  S T I F F N E S S   M A T R I X\n", l+1);
  printf("\n  ==================         R A P       ===================\n");



  ierr = hypre_AMGeRAP(&Matrix[l+1], Matrix[l], P[l]);


  printf("END building coarse matrix; ----------- ------------------------------\n");

  printf("nnz[%d]: %d\n", l+1, hypre_CSRMatrixI(Matrix[l+1])[num_coarsedofs]);

  hypre_TFree(i_AE_chord);
  hypre_TFree(j_AE_chord);
  hypre_TFree(a_AE_chord);


  hypre_TFree(i_AE_dof);
  hypre_TFree(j_AE_dof);
  hypre_TFree(i_dof_AE);
  hypre_TFree(j_dof_AE);

  if (l > 0)
    {
      hypre_TFree(i_dof_node[l]);
      hypre_TFree(j_dof_node[l]);
    }

  ierr = compute_dof_node(&i_dof_node[l+1], &j_dof_node[l+1],
			  Num_nodes[l+1], system_size, &num_dofs);


  ierr = transpose_matrix_create(&i_node_dof[l+1],
				 &j_node_dof[l+1],

				 i_dof_node[l+1], j_dof_node[l+1],

				 num_dofs, Num_nodes[l+1]);

  l++;
  
  if (l < level && Num_nodes[l+1] > 0) goto interpolation_step;
 
  hypre_TFree(i_dof_index);

  if (l > 0)
    {
      hypre_TFree(i_dof_node[l]);
      hypre_TFree(j_dof_node[l]);
    }


  if (l > 0)
    {
      hypre_TFree(i_chord_dof[l]);
      hypre_TFree(j_chord_dof[l]);

      hypre_TFree(i_element_chord[l]);
      hypre_TFree(j_element_chord[l]);
      hypre_TFree(a_element_chord[l]);
    }


  for (k=l+1; k <level+1; k++)
    {
      i_node_dof[k] = NULL;
      j_node_dof[k] = NULL;
    }

  *level_pointer = l;


  *Matrix_pointer = Matrix;
  *P_pointer = P;
  

  *i_node_dof_pointer = i_node_dof;
  *j_node_dof_pointer = j_node_dof;

  hypre_TFree(i_dof_coarsedof);
  hypre_TFree(j_dof_coarsedof);

  hypre_TFree(i_dof_neighbor_coarsedof);
  hypre_TFree(j_dof_neighbor_coarsedof);

  hypre_TFree(i_dof_node);
  hypre_TFree(j_dof_node);

  hypre_TFree(i_chord_dof);
  hypre_TFree(j_chord_dof);

  hypre_TFree(i_element_chord);

  hypre_TFree(j_element_chord);
  hypre_TFree(a_element_chord);


  return ierr; 

}

#include "headers.h"

/*--------------------------------------------------------------------------
 * Test driver prepares files
 *      for testing unstructured matrix topology interface (csr storage)
 *--------------------------------------------------------------------------*/
 
void main()

{
  FILE *f;
  int Problem; /* number of file to read from */

  int ierr;
  int i,j,k,l,m;

  int level;


  /* visualization arrays: -------------------------------------------------*/
  float *red, *blue, *green;
  int num_colors, i_color, max_color, facesize, no_color;
  float *x1, *y1;

  int i_color_to_choose;

  double *x_coord, *y_coord;

  int *i_AE_color;
  int *i_element_element_0, *j_element_element_0;
  int num_elements_0;
  int *i_AE_element_0, *j_AE_element_0;

  int *i_face_to_prefer_weight, *i_face_weight;


  int *i_element_element, *j_element_element;
  int *i_element_face, *j_element_face;


  int num_AEs, num_AEfaces, num_elements, num_nodes, num_faces, 
    num_coarsenodes;

  int Max_level = 25; 

  int *i_ILUdof_to_dof[25];


  int *i_ILUdof_ILUdof_t[25], *j_ILUdof_ILUdof_t[25],
    *i_ILUdof_ILUdof[25], *j_ILUdof_ILUdof[25];

  double *LD_data[25], *U_data[25];

  hypre_CSRMatrix     *P[25];
  hypre_CSRMatrix     **Matrix;
  int *i_node_on_boundary;
  int *i_dof_on_boundary;
  int *i_dof_dof_a, *j_dof_dof_a;
  double *a_dof_dof;

  hypre_AMGeMatrixTopology **A;

  int *i_node_coarsenode[25], *j_node_coarsenode[25];
  int *i_node_neighbor_coarsenode[25], *j_node_neighbor_coarsenode[25];

  int Num_nodes[25], Num_elements[25], Ndofs[25];


  int *i_element_node_0, *j_element_node_0;
  int *i_boundarysurface_node, *j_boundarysurface_node;
  int num_boundarysurfaces;

  int *i_AE_element, *j_AE_element;
  int *i_element_node, *j_element_node;
  int *i_AE_node, *j_AE_node;
  int *i_node_AE, *j_node_AE;

  int *i_AEface_node, *j_AEface_node;
  int *i_face_node, *j_face_node;


  int *i_node_index;
  int *i_node_block[25], *j_node_block[25],
    *i_block_node[25], *j_block_node[25];

  int Num_blocks[25];
  int min_block, max_block; 


  int system_size;
  int num_dofs, num_coarsedofs;
  int *i_dof_index;
  int *i_dof_node, *j_dof_node;
  int *i_node_dof, *j_node_dof;

  int *i_element_dof_0, *j_element_dof_0;
  double *element_data;

  int *i_AE_dof, *j_AE_dof;
  int *i_dof_AE, *j_dof_AE;


  int *i_dof_coarsedof[25], *j_dof_coarsedof[25];
  int *i_dof_neighbor_coarsedof[25], *j_dof_neighbor_coarsedof[25];

  int *i_element_chord[25], *j_element_chord[25];
  double *a_element_chord[25];

  int *i_chord_dof[25], *j_chord_dof[25];
  int Num_chords[25];

  int *i_AE_chord, *j_AE_chord;
  double *a_AE_chord;
  
  int dof_coarsedof_counter, dof_neighbor_coarsedof_counter;


  /* ---------------------------------------------------------------------- */
  /*  PCG arrays:                                                           */
  /* ---------------------------------------------------------------------- */

  double *x, *rhs, *r, *v, *w[25], *d[25],
    *aux, *v_coarse, *w_coarse, *d_coarse, *v_fine, *w_fine, *d_fine;
  int max_iter = 1000;
  int coarse_level; 
  int nu = 1;
  double reduction_factor;

  /*------------------------------------------------------------------------
   * initializing
   *------------------------------------------------------------------------*/

  red = hypre_CTAlloc(float, 256);
  blue = hypre_CTAlloc(float, 256);
  green = hypre_CTAlloc(float, 256);

  red[0] =  1;
  blue[0] = 0;
  green[0] = 0;

  red[1] =  0.;
  blue [1] = 1;
  green[1] = 0;

  red[2] =  0;
  blue [2] = 0;
  green[2] = 1;

  red[3] =  0;
  blue [3] = 1;
  green[3] = 1;


  red[4] =  1;
  blue [4] =0;    
  green[4] =1 ;

  red[5] =  1;
  blue [5] =1;    
  green[5] =0;

  red[6] =   0.1;
  blue [6] = 0.1;    
  green[6] = 0.1;

  red[7] =  0.1;
  blue [7] =1;    
  green[7] = 0;

  red[8] =  0.1;
  blue [8] =0.3;    
  green[8] = 1;

  red[9] =  1;
  blue [9] =0;    
  green[9] = 1;

  red[10] =  0.5;
  blue [10] =0;    
  green[10] = 1;

  red[11] =  0;
  blue [11] = 0;
  green[11] = 0;

  max_color = 11;

  no_color = -1;


try_again:

  printf("Problem: ?; enter file number - 50,51,52,53,54\n");
  printf("                           or - 100,400,1600,6400\n");
  scanf("%d", &Problem);
  printf("Problem: %d\n", Problem);

  if (Problem != 100 && Problem != 400 &&Problem != 1600 &&Problem != 6400 &&
      Problem != 50 && Problem != 51 && Problem != 52 && Problem != 53 &&
      Problem != 54) goto try_again;




  ierr = hypre_AMGeInitialGraphs(&i_element_node_0,
				 &j_element_node_0,

				 &i_boundarysurface_node, 
				 &j_boundarysurface_node,


				 &num_elements,
				 &num_nodes,
				 &num_boundarysurfaces,

				 &i_node_on_boundary,

				 &x_coord,
				 &y_coord,

				 Problem); 

  printf("num_nodes: %d, num_elements: %d, num_boundarysurfaces: %d\n", 
	 num_nodes, num_elements, num_boundarysurfaces);

  Num_nodes[0] = num_nodes;
  Num_elements[0] = num_elements;

  x1 = hypre_CTAlloc(float, num_nodes);
  y1 = hypre_CTAlloc(float, num_nodes);

  /* A[0] = hypre_CTAlloc(hypre_AMGeMatrixTopology, 1); */

  /* ierr = hypre_CreateAMGeMatrixTopology(A[0]); */
  A = hypre_CTAlloc(hypre_AMGeMatrixTopology*, Max_level);

  ierr = hypre_BuildAMGeMatrixTopology(&A[0],
				       i_element_node_0,
				       j_element_node_0,

				       i_boundarysurface_node,
				       j_boundarysurface_node,

				       num_elements,
				       num_nodes,
				       num_boundarysurfaces);

  hypre_TFree(i_boundarysurface_node);
  hypre_TFree(j_boundarysurface_node);



  i_element_face = hypre_AMGeMatrixTopologyIElementFace(A[0]);
  j_element_face = hypre_AMGeMatrixTopologyJElementFace(A[0]);
  num_faces = hypre_AMGeMatrixTopologyNumFaces(A[0]);

  ierr = matrix_matrix_t_product(&i_element_element, &j_element_element,

				 i_element_face, j_element_face,

				 num_elements, num_faces);



  i_AE_color = hypre_CTAlloc(int, num_elements+1);

  num_colors = 10;

e0:
  for (i=0; i < num_elements; i++)
    i_AE_color[i] = -1;

  for (i=0; i < num_elements; i++)
    {
      for (i_color = 0; i_color < num_colors; i_color++)
	{
	  i_color_to_choose = -1;
	  for (j=i_element_element[i]; j < i_element_element[i+1]; j++)
	    if (i_AE_color[j_element_element[j]] == i_color)
	      {
		i_color_to_choose = 0;
		break;
	      }
	  if (i_color_to_choose == -1)
	    {
	      i_AE_color[i] = i_color;

	      break;
	    }
	}
      if (i_AE_color[i] == -1) 
	{
	  num_colors++;
	  printf("num_colors: %d\n", num_colors);
	  goto e0;
	}
    }

  /* printf("num_colors: %d\n", num_colors); */

  hypre_TFree(i_element_element);
  hypre_TFree(j_element_element);

  xutl0_(&num_colors, red, green, blue );  

  for (i=0; i < num_elements; i++)
    {
      i_color = i_AE_color[i]+1;
      if (i_color > 10) i_color = 1;
      facesize=0;
      for (j=i_element_node_0[i]; j < i_element_node_0[i+1]; j++)
	{
	  x1[facesize] = x_coord[j_element_node_0[j]];
	  y1[facesize] = y_coord[j_element_node_0[j]];

	  facesize++;
	}


      xfill0_(x1, y1, &facesize, &i_color); 
      x1[facesize] = x1[0];
      y1[facesize] = y1[0];
      facesize++;
      xline0_(x1, y1, &facesize, &max_color);
    }

  xutl0_(&no_color, red, green, blue );  

  printf("end displaying: ====================================\n\n");


  i_face_to_prefer_weight = hypre_CTAlloc(int, num_faces);
  i_face_weight = hypre_CTAlloc(int, num_faces);

  i_element_element_0 = hypre_CTAlloc(int, num_elements+1);
  j_element_element_0 = hypre_CTAlloc(int, num_elements);
  num_elements_0 = num_elements;

  for (i=0; i < num_elements_0; i++)
    {
      i_element_element_0[i] = i;
      j_element_element_0[i] = i;
    }

  i_element_element_0[num_elements_0] = num_elements_0;

  level = 0;
  
agglomerate:
  A[level+1] = hypre_CTAlloc(hypre_AMGeMatrixTopology, 1); 

  ierr = hypre_CoarsenAMGeMatrixTopology(A[level+1],

				    
					 A[level],
				    
					 i_element_element_0,
					 j_element_element_0,
					 &i_AE_element_0, &j_AE_element_0,

					 i_element_node_0, j_element_node_0,
					 num_elements_0,

					 i_AE_color, 
					 x_coord, y_coord,
					 red, blue, green,
					 x1, y1,

					 i_face_to_prefer_weight,
					 i_face_weight);

  hypre_TFree(i_element_element_0);
  hypre_TFree(j_element_element_0);

  i_element_element_0 = i_AE_element_0;
  j_element_element_0 = j_AE_element_0;

  num_AEs = hypre_AMGeMatrixTopologyNumElements(A[level+1]);
  printf("level %d num_AEs: %d\n\n\n", level+1, num_AEs);

  level++;
   
  if (num_AEs > 1 && level+1 < Max_level) 
    goto agglomerate;

  printf("\n================================================================\n");
  printf("number of grids: from 0 to %d\n\n\n", level);
  printf("================================================================\n\n");


  /* we here change to local grid ordering: --------------------------------------*/

  i_node_index = hypre_CTAlloc(int, Num_nodes[0]);

  l = 0;
coarsen_grid:

  i_AE_element = hypre_AMGeMatrixTopologyIAEElement(A[l+1]);
  j_AE_element = hypre_AMGeMatrixTopologyJAEElement(A[l+1]);

  num_AEs = hypre_AMGeMatrixTopologyNumElements(A[l+1]);
  num_elements = hypre_AMGeMatrixTopologyNumElements(A[l]);
  num_nodes = hypre_AMGeMatrixTopologyNumNodes(A[l]);

  i_element_node = hypre_AMGeMatrixTopologyIElementNode(A[l]);
  j_element_node = hypre_AMGeMatrixTopologyJElementNode(A[l]);

  ierr = matrix_matrix_product(&i_AE_node, &j_AE_node,

			       i_AE_element, j_AE_element,
			       i_element_node, j_element_node,

			       num_AEs, num_elements, num_nodes);




  ierr = transpose_matrix_create(&i_node_AE,
				 &j_node_AE,

				 i_AE_node, j_AE_node,

				 num_AEs, num_nodes);

  i_AEface_node = hypre_AMGeMatrixTopologyIFaceNode(A[l+1]);
  j_AEface_node = hypre_AMGeMatrixTopologyJFaceNode(A[l+1]);
  num_AEfaces =  hypre_AMGeMatrixTopologyNumFaces(A[l+1]);



  ierr = hypre_AMGeCoarseNodeSelection(i_AEface_node, j_AEface_node,
				       i_AE_node, j_AE_node, 
				       i_node_AE, j_node_AE,

				       num_AEfaces, num_nodes,

				       &i_node_neighbor_coarsenode[l],
				       &j_node_neighbor_coarsenode[l],

				       &i_node_coarsenode[l],
				       &j_node_coarsenode[l],

				       &num_coarsenodes);
  printf("level %d num_AEfaces %d, num_coarsenodes: %d \n", l, num_AEfaces,
	 num_coarsenodes);


  hypre_TFree(i_node_AE);
  hypre_TFree(j_node_AE);

  hypre_AMGeMatrixTopologyNumNodes(A[l+1]) = num_coarsenodes;

  i_node_block[l] = hypre_CTAlloc(int, num_nodes+1);

  for (i=0; i < num_nodes; i++)
    i_node_block[l][i] = 0;

  i_face_node = hypre_AMGeMatrixTopologyIFaceNode(A[l]);
  j_face_node = hypre_AMGeMatrixTopologyJFaceNode(A[l]);
  num_faces =  hypre_AMGeMatrixTopologyNumFaces(A[l]);

  for (i=0; i < num_faces; i++)
    for (j=i_face_node[i]; j < i_face_node[i+1]; j++)
      i_node_index[j_face_node[j]] = -1;

  for (i=0; i < num_faces; i++)
    for (j=i_face_node[i]; j < i_face_node[i+1]; j++)
      {
	if (i_node_index[j_face_node[j]] == -1)
	  {
	    i_node_block[l][j_face_node[j]]++;
	    i_node_index[j_face_node[j]] = 0;
	  }
      }

   for (k=l; k < level; k++)
    {
      i_AEface_node = hypre_AMGeMatrixTopologyIFaceNode(A[k+1]);
      j_AEface_node = hypre_AMGeMatrixTopologyJFaceNode(A[k+1]);
      num_AEfaces =  hypre_AMGeMatrixTopologyNumFaces(A[k+1]);

      ierr = matrix_matrix_product(&i_face_node, &j_face_node, 

				   i_AEface_node, j_AEface_node,
				   i_node_coarsenode[l], j_node_coarsenode[l],

				   num_AEfaces, num_nodes, num_coarsenodes);

      /*
      hypre_TFree(i_AEface_node);
      hypre_TFree(j_AEface_node);
      */

      hypre_TFree(hypre_AMGeMatrixTopologyIFaceNode(A[k+1]));
      hypre_AMGeMatrixTopologyIFaceNode(A[k+1]) = i_face_node;
      hypre_TFree(hypre_AMGeMatrixTopologyJFaceNode(A[k+1]));
      hypre_AMGeMatrixTopologyJFaceNode(A[k+1]) = j_face_node;

      num_faces = num_AEfaces;

      for (i=0; i < num_faces; i++)
	for (j=i_face_node[i]; j < i_face_node[i+1]; j++)
	  i_node_index[j_face_node[j]] = -1;

      for (i=0; i < num_faces; i++)
	for (j=i_face_node[i]; j < i_face_node[i+1]; j++)
	  {
	    if (i_node_index[j_face_node[j]] == -1)
	      {
		i_node_block[l][j_face_node[j]]++;
		i_node_index[j_face_node[j]] = 0;
	      }
	  }

    }

   j_node_block[l] = hypre_CTAlloc(int, num_nodes);

   max_block = 0;
   min_block = level;
   for (i=0; i < num_nodes; i++)
     {
       if (max_block < i_node_block[l][i]) 
	 max_block = i_node_block[l][i];

       if (min_block > i_node_block[l][i]) 
	 min_block = i_node_block[l][i];

     }

   /*
   printf("num_levels: %d, max_block: %d, min_block: %d\n", level-l, max_block,
	  min_block);
	  */
   for (i=0; i < num_nodes; i++)
     {
       j_node_block[l][i] = i_node_block[l][i]-min_block;
       i_node_block[l][i] = i;
    }

   i_node_block[l][num_nodes] = num_nodes;


   ierr = transpose_matrix_create(&i_block_node[l], &j_block_node[l],

				  i_node_block[l], j_node_block[l],
				  num_nodes, max_block-min_block+1);

   Num_blocks[l] = max_block-min_block+1;
   hypre_TFree(i_node_block[l]);
   hypre_TFree(j_node_block[l]);


   /*
   printf("\n================================================================\n");
   printf("\n level[%d]  n e s t e d   d i s s e c t i o n   o r d e r i n g:\n",l);
   printf("\n================================================================\n");


   for (k=0; k < max_block-min_block+1; k++)
     {
       printf("block: %d contains %d nodes: \n", k, 
	      i_block_node[l][k+1]-i_block_node[l][k]);

       for (m=i_block_node[l][k]; m < i_block_node[l][k+1]; m++)
	 printf(" %d, ", j_block_node[l][m]);

       printf("\n\n");
     }
   printf("\n================================================================\n");


   printf("num_nodes %d and num_nodes counted: %d\n\n\n",
	  num_nodes, i_block_node[l][max_block-min_block+1]);

	  */

  /* END nested dissection ordering: ---------------------------------------*/

  ierr = matrix_matrix_product(&i_element_node, &j_element_node,  

			       i_AE_node, j_AE_node,
			       i_node_coarsenode[l], j_node_coarsenode[l],

			       num_AEs, num_nodes, num_coarsenodes);

  /*
  printf("AE_dof[%d] =======================================================\n",l);
  for (i=0; i < num_AEs; i++)
    {
      printf("AE %d coantins nodes:\n", i);
      for (j=i_AE_node[i]; j < i_AE_node[i+1]; j++)
	printf("%d ", j_AE_node[j]);
      printf("\n");
    }
  printf("END AE_dof[%d] =======================================================\n",l);

  */

  hypre_TFree(i_AE_node);
  hypre_TFree(j_AE_node);

  
  hypre_AMGeMatrixTopologyIElementNode(A[l+1]) = i_element_node;
  hypre_AMGeMatrixTopologyJElementNode(A[l+1]) = j_element_node;
  

  if (num_coarsenodes == 0)
    goto e_next;

  l++;
  Num_nodes[l] = num_coarsenodes;
  Num_elements[l] = num_AEs;
  if (l < level) goto coarsen_grid;

  /* ELEMENT MATRICES READ: ============================================ */

e_next:
  level = l;

  system_size = 1;

  ierr = compute_dof_node(&i_dof_node, &j_dof_node,
			  Num_nodes[0], system_size, &num_dofs);

  if (system_size == 1)
    i_dof_on_boundary = i_node_on_boundary;
  else
    {
      ierr = compute_dof_on_boundary(&i_dof_on_boundary,
				     i_node_on_boundary,
				     
				     Num_nodes[0], system_size);
      free(i_node_on_boundary);
    }


  ierr = transpose_matrix_create(&i_node_dof,
				 &j_node_dof,

				 i_dof_node, j_dof_node,

				 num_dofs, Num_nodes[0]);


  if (system_size == 1)
    {
      i_element_dof_0 = i_element_node_0;
      j_element_dof_0 = j_element_node_0;
    }
  else
    ierr = matrix_matrix_product(&i_element_dof_0, &j_element_dof_0, 

				 i_element_node_0, j_element_node_0,
				 i_node_dof, j_node_dof,

				 Num_elements[0], Num_nodes[0], num_dofs);


  ierr = hypre_AMGeElmMatRead(&element_data, 

			      i_element_dof_0,
			      j_element_dof_0,
			      Num_elements[0],

			      Problem);


  printf("store element matrices in element_chord format: ----------------\n");

  ierr = hypre_AMGeElementMatrixDof(i_element_dof_0, j_element_dof_0,

				    element_data,

				    &i_element_chord[0],
				    &j_element_chord[0],
				    &a_element_chord[0],

				    &i_chord_dof[0],
				    &j_chord_dof[0],

				    &Num_chords[0],

				    Num_elements[0], num_dofs);

  printf("num_chords: %d\n", Num_chords[0]);


  /* assemble initial fine matrix: * ------------------------------------- */

  Matrix = hypre_CTAlloc(hypre_CSRMatrix*, Max_level);

  ierr = hypre_AMGeMatrixAssemble(&Matrix[0],

				  i_element_chord[0],
				  j_element_chord[0],
				  a_element_chord[0],

				  i_chord_dof[0], 
				  j_chord_dof[0],

				  Num_elements[0], 
				  Num_chords[0],
				  num_dofs);

  printf("nnz[0]: %d\n", hypre_CSRMatrixI(Matrix[0])[num_dofs]);
  /* impose Dirichlet boundary conditions: -----------------*/
  printf("imposing Dirichlet boundary conditions:====================\n");

  i_dof_dof_a = hypre_CSRMatrixI(Matrix[0]);
  j_dof_dof_a = hypre_CSRMatrixJ(Matrix[0]);
  a_dof_dof   = hypre_CSRMatrixData(Matrix[0]);
  for (i=0; i < num_dofs; i++)
    for (j=i_dof_dof_a[i]; j < i_dof_dof_a[i+1]; j++)
      if (i_dof_on_boundary[j_dof_dof_a[j]] == 0 
	  &&j_dof_dof_a[j]!=i)
	a_dof_dof[j] = 0.e0;

  for (i=0; i < num_dofs; i++)
    for (j=i_dof_dof_a[i]; j < i_dof_dof_a[i+1]; j++)
      if (i_dof_on_boundary[i] == 0 &&  j_dof_dof_a[j] !=i)
	a_dof_dof[j] = 0.e0;

  hypre_TFree(i_dof_on_boundary);

  printf("\n\nB U I L D I N G  level[0] ILU(1) FACTORIZATION  M A T R I X:\n");
  ierr = hypre_ILUfactor(&i_ILUdof_to_dof[0],


			 &i_ILUdof_ILUdof[0],
			 &j_ILUdof_ILUdof[0],
			 &LD_data[0],

			 &i_ILUdof_ILUdof_t[0],
			 &j_ILUdof_ILUdof_t[0],
			 &U_data[0],

			 Matrix[0],

			 i_node_dof, j_node_dof,

			 i_block_node[0], j_block_node[0],
			 Num_blocks[0], 
				    
			 num_dofs,
			 Num_nodes[0]);
  printf("LD_nnz: %d\n", i_ILUdof_ILUdof[0][num_dofs]);
  printf("U_nnz: %d\n", i_ILUdof_ILUdof_t[0][num_dofs]);
  printf("\n\n END building ILU(1)  FACTORIZATION  MATRIX; -------------------------\n");

  hypre_TFree(i_block_node[0]);
  hypre_TFree(j_block_node[0]);


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

  num_dofs = Num_nodes[l] * system_size; 
  num_coarsedofs = Num_nodes[l+1] * system_size; 

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
				   i_node_dof, j_node_dof,

				   Num_elements[l+1], 
				   Num_nodes[l], num_dofs);

      ierr = matrix_matrix_product(&i_dof_AE, &j_dof_AE,

				   i_dof_node, j_dof_node,
				   i_node_AE,  j_node_AE,

				   num_dofs,   Num_nodes[l], Num_elements[l+1]);

      /* free:  node_AE, AE_node */


      i_dof_coarsedof[l] = hypre_CTAlloc(int, num_dofs+1);
      j_dof_coarsedof[l] = hypre_CTAlloc(int, num_coarsedofs);

      for (i=0; i < Num_nodes[l]; i++)
	if (i_node_coarsenode[l][i] == i_node_coarsenode[l][i+1])
	  for (j=i_node_dof[i]; j < i_node_dof[i+1]; j++)
	    i_dof_index[j_node_dof[j]] = -1;
	else
	  for (j=i_node_coarsenode[l][i]; j < i_node_coarsenode[l][i+1]; j++)
	    for (j=i_node_dof[i]; j < i_node_dof[i+1]; j++)
	      i_dof_index[j_node_dof[j]] = 0; 

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
	      for (j=i_dof_node[i]; j < i_dof_node[i+1]; j++)
		for (k=i_node_neighbor_coarsenode[l][j_dof_node[j]];
		     k<i_node_neighbor_coarsenode[l][j_dof_node[j]+1]; k++)
		  for (m=i_node_dof[j_node_neighbor_coarsenode[l][k]];
		       m<i_node_dof[j_node_neighbor_coarsenode[l][k]+1]; m++)
		    {
		      j_dof_neighbor_coarsedof[l][dof_neighbor_coarsedof_counter] 
			= j_node_dof[m];
		      dof_neighbor_coarsedof_counter++;
		    }
	    }
	

	}

      i_dof_neighbor_coarsedof[l][num_dofs] = dof_neighbor_coarsedof_counter;


      /* free node_coarsenode[l], node_neighbor_coarsenode[l]; */

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



  hypre_TFree(i_element_chord[l]);
  hypre_TFree(j_element_chord[l]);
  hypre_TFree(a_element_chord[l]);



  hypre_TFree(i_AE_element);
  hypre_TFree(j_AE_element);



  printf("\n\nB U I L D I N G  level[%d] I N T E R P O L A T I O N   M A T R I X\n", l);

  /*
  printf("AE_dof[%d] =======================================================\n",l);
  for (i=0; i < Num_elements[l+1]; i++)
    {
      printf("AE %d coantins dofs:\n", i);
      for (j=i_AE_dof[i]; j < i_AE_dof[i+1]; j++)
	printf("%d ", j_AE_dof[j]);
      printf("\n");
    }
  printf("END AE_dof[%d] =======================================================\n",l);
  */

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


  hypre_TFree(i_dof_coarsedof[l]);
  hypre_TFree(i_dof_coarsedof[l]);

  printf("END building Interpolation [%d]: ------------------------------\n", l);

  printf("\nB U I L D I N G  level[%d]  S T I F F N E S S   M A T R I X\n", l+1);

  ierr = hypre_AMGeRAP(&Matrix[l+1], Matrix[l], P[l]);

  printf("nnz[%d]: %d\n", l+1, hypre_CSRMatrixI(Matrix[l+1])[num_coarsedofs]);
  printf("END building coarse matrix; ----------- ------------------------------\n");

  hypre_TFree(i_AE_chord);
  hypre_TFree(j_AE_chord);
  hypre_TFree(a_AE_chord);


  hypre_TFree(i_chord_dof[l]);
  hypre_TFree(j_chord_dof[l]);
 
  hypre_TFree(i_AE_dof);
  hypre_TFree(j_AE_dof);
  hypre_TFree(i_dof_AE);
  hypre_TFree(j_dof_AE);



  hypre_TFree(i_dof_node);
  hypre_TFree(j_dof_node);

  hypre_TFree(i_node_dof);
  hypre_TFree(j_node_dof);

  ierr = compute_dof_node(&i_dof_node, &j_dof_node,
			  Num_nodes[l+1], system_size, &num_dofs);


  ierr = transpose_matrix_create(&i_node_dof,
				 &j_node_dof,

				 i_dof_node, j_dof_node,

				 num_dofs, Num_nodes[l+1]);

  if (l+1 < level)
    {
      printf("\n\nB U I L D I N G  level[%d] ILU(1)  FACTORIZATION  M A T R I X\n",l+1);
      ierr = hypre_ILUfactor(&i_ILUdof_to_dof[l+1],


			     &i_ILUdof_ILUdof[l+1],
			     &j_ILUdof_ILUdof[l+1],
			     &LD_data[l+1],
			     
			     &i_ILUdof_ILUdof_t[l+1],
			     &j_ILUdof_ILUdof_t[l+1],
			     &U_data[l+1],

			     Matrix[l+1],

			     i_node_dof, j_node_dof,

			     i_block_node[l+1], j_block_node[l+1],
			     Num_blocks[l+1], 
				    
			     num_dofs,
			     Num_nodes[l+1]);


      printf("LD_nnz: %d\n", i_ILUdof_ILUdof[l+1][num_dofs]);
      printf("U_nnz: %d\n", i_ILUdof_ILUdof_t[l+1][num_dofs]);
      printf("\n\n END building ILU(1) FACTORIZATION  MATRIX;------------------------\n");
    

      hypre_TFree(i_block_node[l+1]);
      hypre_TFree(j_block_node[l+1]);
    }
  
  l++;
  
  if (l < level && Num_nodes[l+1] > 0) goto interpolation_step;
 
  hypre_TFree(i_dof_index);

  level = l;

  /* ========================================================================== */
  /* ======================== S O L U T I O N   P A R T: ====================== */
  /* ========================================================================== */

  /* one V(1,1) --cycle as preconditioner in PCG: ============================== */
  /* LD ILU solve pre--smoothing, U solve post--smoothing; ===================== */

  num_dofs = Num_nodes[0] * system_size;
  x = hypre_CTAlloc(double, num_dofs); 
  rhs = hypre_CTAlloc(double, num_dofs);

  r = hypre_CTAlloc(double, num_dofs); 
  aux = hypre_CTAlloc(double, num_dofs);


  for (l=0; l < level+1; l++)
    {
      Ndofs[l] = Num_nodes[l] * system_size;
      if (Ndofs[l] > 0)
	{
	  w[l] = hypre_CTAlloc(double, Ndofs[l]);
	  d[l] = hypre_CTAlloc(double, Ndofs[l]);
	}
      else
	{
	  level = l-1;
	  break;
	}
    }

  v_fine = hypre_CTAlloc(double, num_dofs);
  w_fine = hypre_CTAlloc(double, num_dofs);
  d_fine = hypre_CTAlloc(double, num_dofs);

  coarse_level = level;
  v_coarse = hypre_CTAlloc(double, Ndofs[coarse_level]);
  w_coarse = hypre_CTAlloc(double, Ndofs[coarse_level]);
  d_coarse = hypre_CTAlloc(double, Ndofs[coarse_level]);

  for (l=0; l < level; l++)
    {
      printf("\n\n=======================================================\n");
      printf("             Testing level[%d] PCG solve:                  \n",l);
      printf("===========================================================\n");
 
      for (i=0; i < Ndofs[l]; i++)
	x[i] = 0.e0;

      for (i=0; i < Ndofs[l]; i++)
	rhs[i] = rand();

      i_dof_dof_a = hypre_CSRMatrixI(Matrix[l]);
      j_dof_dof_a = hypre_CSRMatrixJ(Matrix[l]);
      a_dof_dof   = hypre_CSRMatrixData(Matrix[l]);


      ierr = hypre_ILUsolve(x,

			    i_ILUdof_to_dof[l],
					 
			    i_ILUdof_ILUdof[l],
			    j_ILUdof_ILUdof[l],
			    LD_data[l],

			    i_ILUdof_ILUdof_t[l],
			    j_ILUdof_ILUdof_t[l],
			    U_data[l],

			    rhs, 

			    Ndofs[l]);


      ierr = hypre_ILUpcg(x, rhs,
			  a_dof_dof,
			  i_dof_dof_a, j_dof_dof_a,

			  i_ILUdof_to_dof[l],

			  i_ILUdof_ILUdof[l], 
			  j_ILUdof_ILUdof[l],
			  LD_data[l],

			  i_ILUdof_ILUdof_t[l], 
			  j_ILUdof_ILUdof_t[l],
			  U_data[l],

			  v_fine, w_fine, d_fine, max_iter, 

			  Ndofs[l]);


      printf("\n\n=======================================================\n");
      printf("             END test PCG solve:                           \n");
      printf("===========================================================\n");
 
    }
  printf("\n\n===============================================================\n");
  printf("                      Problem: %d \n", Problem);
  printf(" -------- V_cycle & nested dissection ILU smoothing: ----------\n");
  printf("================================================================\n");

  num_dofs = Ndofs[0];

  for (i=0; i < num_dofs; i++)
    rhs[i] = rand();
  
  
  ierr = hypre_VcycleILUpcg(x, rhs,
			    w, d,

			    &reduction_factor,
			       
			    Matrix,
			    i_ILUdof_to_dof,

			    i_ILUdof_ILUdof, 
			    j_ILUdof_ILUdof,
			    LD_data,

			    i_ILUdof_ILUdof_t, 
			    j_ILUdof_ILUdof_t,
			    U_data,

			    P,

			    aux, r, 
			    v_fine, w_fine, d_fine, max_iter, 

			    v_coarse, w_coarse, d_coarse, 

			    nu, 
			    level, coarse_level, 
			    Ndofs);


}

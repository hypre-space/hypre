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
 * builds node_coarsenode, node_neighbor_coarsenode, and 
 *        block_node graphs for Interpolation, and Smoother Matrices;
 *
 ****************************************************************************/

int hypre_AMGeCoarsenodeSetup(hypre_AMGeMatrixTopology **A,
			      int *level_pointer,

			      int ***i_node_neighbor_coarsenode_pointer,
			      int ***j_node_neighbor_coarsenode_pointer,

			      int ***i_node_coarsenode_pointer,
			      int ***j_node_coarsenode_pointer,

			      int ***i_block_node_pointer,
			      int ***j_block_node_pointer,

			      int *Num_blocks,
			      int *Num_elements,
			      int *Num_nodes)



{

  int ierr = 0;

  int i,j,k,l; 

  int min_block, max_block;

  int level = level_pointer[0];


  int **i_node_block, **j_node_block;
  int **i_block_node, **j_block_node;

  int **i_node_coarsenode, **j_node_coarsenode;
  int **i_node_neighbor_coarsenode, **j_node_neighbor_coarsenode;


  int *i_element_node, *j_element_node;
  int *i_face_node, *j_face_node;
  int *i_AE_element, *j_AE_element;

  int *i_AEface_node, *j_AEface_node;
  int *i_AE_node, *j_AE_node;
  int *i_node_AE, *j_node_AE;


  int *i_node_index;


  int num_nodes, num_elements, num_faces, num_AEs, num_AEfaces;
  int num_coarsenodes;


  i_node_coarsenode = hypre_CTAlloc(int*, level_pointer[0]);
  j_node_coarsenode = hypre_CTAlloc(int*, level_pointer[0]);

  i_node_neighbor_coarsenode = hypre_CTAlloc(int*, level_pointer[0]);
  j_node_neighbor_coarsenode = hypre_CTAlloc(int*, level_pointer[0]);

  i_node_block = hypre_CTAlloc(int*, level_pointer[0]+1);
  j_node_block = hypre_CTAlloc(int*, level_pointer[0]+1);
  i_block_node = hypre_CTAlloc(int*, level_pointer[0]+1);
  j_block_node = hypre_CTAlloc(int*, level_pointer[0]+1);




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


e_next: 
   hypre_TFree(i_node_index);

   for (k=l+1; k <level+1; k++)
     {
       i_block_node[k] = NULL;
       j_block_node[k] = NULL;
     }

  *level_pointer = l;

  *i_node_coarsenode_pointer = i_node_coarsenode;
  *j_node_coarsenode_pointer = j_node_coarsenode;

  *i_node_neighbor_coarsenode_pointer = i_node_neighbor_coarsenode;
  *j_node_neighbor_coarsenode_pointer = j_node_neighbor_coarsenode;

  *i_block_node_pointer = i_block_node;
  *j_block_node_pointer = j_block_node;

  hypre_TFree(i_node_block);
  hypre_TFree(j_node_block);

  return ierr;
}


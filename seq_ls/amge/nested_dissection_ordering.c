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
 * builds nested dissection ordering based on "node_level"
 *
 * i_node_level[i] = # faces
 * the node "i" belongs to;
 *
 * at all levels created throughout the AMGeMatrixTopology process; 
 *
 * if i_node_level = i_node_HB_level as input, the output
 *
 *           "level_node" 
 *
 *    is the HB node ordering;
 *
 ****************************************************************************/

int AMGeNestedDissectionOrdering(int *i_node_level,
				 int num_nodes, 
				 int level, 

				 int **j_node_level_pointer,

				 int **i_level_node_pointer, 
				 int **j_level_node_pointer, 

				 int *num_levels_pointer)
{
  int ierr = 0;
  int i,j,k,l;  

  int *j_node_level, *i_level_node, *j_level_node;

  int min_level, max_level;

  j_node_level = hypre_CTAlloc(int, num_nodes);

  max_level = 0;
  min_level = level;
  for (i=0; i < num_nodes; i++)
    {
      if (max_level < i_node_level[i]) 
	max_level = i_node_level[i];

      if (min_level > i_node_level[i]) 
	min_level = i_node_level[i];

    }

  printf("level: %d, max_level: %d, min_level: %d\n", level, max_level,
	 min_level);

  for (i=0; i < num_nodes; i++)
    {
      j_node_level[i] = i_node_level[i]-min_level;
      i_node_level[i] = i;
    }

  i_node_level[num_nodes] = num_nodes;


  ierr = transpose_matrix_create(&i_level_node, &j_level_node,

				 i_node_level, j_node_level,
				 num_nodes, max_level-min_level+1);


  *num_levels_pointer = max_level-min_level+1;
  *i_level_node_pointer = i_level_node;
  *j_level_node_pointer = j_level_node;

  *j_node_level_pointer = j_node_level; 


  printf("\n==============================================================\n");
  printf("\n     n e s t e d   d i s s e c t i o n   o r d e r i n g:     \n");
  printf("\n==============================================================\n");


  for (l=0; l < max_level-min_level+1; l++)
    {
      printf("level: %d contains %d nodes: \n", l, 
	     i_level_node[l+1]-i_level_node[l]);
      /*
      for (k=i_level_node[l]; k < i_level_node[l+1]; k++)
	printf(" %d, ", j_level_node[k]);
	*/
      printf("\n\n");
    }
  printf("\n==============================================================\n");
  printf("num_nodes %d and num_nodes counted: %d\n\n\n",
	 num_nodes, i_level_node[max_level-min_level+1]);


  return ierr;

}

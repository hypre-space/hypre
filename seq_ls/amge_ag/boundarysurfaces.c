/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/
/*****************************************************************************
 *
 * creates graph boundarysurface_node from node_node graph;
 *
 ****************************************************************************/

#include "headers.h"  



int hypre_AMGeCreateBoundarysurfaces(int **i_boundarysurface_node_pointer,
				     int **j_boundarysurface_node_pointer,

				     int *num_boundarysurfaces_pointer,

				     int *i_element_node,
				     int *j_element_node,

				     int num_elements, 
				     int num_nodes)

{

  int ierr = 0;
  int i,j,k,l;
  int *i_node_node, *j_node_node; 
  int *i_node_element, *j_node_element;
  int *i_boundarysurface_node, *j_boundarysurface_node;

  int *i_levelset_node, *j_levelset_node;
  int *i_node_index;
  int num_levelsets;
  int level_0, level_1;

  int levelset_node_counter;
  int num_boundarysurfaces;
  int boundarysurface_node_counter;

  ierr = transpose_matrix_create(&i_node_element,
				 &j_node_element,

				 i_element_node, j_element_node,

				 num_elements, num_nodes);


   ierr = matrix_matrix_product(&i_node_node, &j_node_node,

			       i_node_element, j_node_element,
			       i_element_node, j_element_node,

			       num_nodes, num_elements, num_nodes);

   hypre_TFree(i_node_element);
   hypre_TFree(j_node_element);
  

  i_levelset_node = hypre_CTAlloc(int, num_nodes+1);
  j_levelset_node = hypre_CTAlloc(int, num_nodes);

  i_node_index = hypre_CTAlloc(int, num_nodes);
  for (i=0; i < num_nodes; i++)
    i_node_index[i] = 0;

  num_levelsets = 1;
  i_levelset_node[0] = 0;
  i_levelset_node[1] = 1;

  j_levelset_node[0] = num_nodes/2;
  i_node_index[num_nodes/2] = 1;

  levelset_node_counter = 1;

new_level:  
  l=num_levelsets-1; 

  num_levelsets++;
  i_levelset_node[num_levelsets] = levelset_node_counter;

  for (j=i_levelset_node[l]; j < i_levelset_node[l+1]; j++)
    {
      for (k=i_node_node[j_levelset_node[j]];
	   k<i_node_node[j_levelset_node[j]+1]; k++)
	{
	  if (i_node_index[j_node_node[k]] == 0)
	    {
	      i_node_index[j_node_node[k]] = num_levelsets;
	      j_levelset_node[levelset_node_counter] = j_node_node[k];
	      levelset_node_counter++;
	    }
	}
    }

  if (i_levelset_node[num_levelsets] < levelset_node_counter)
    {
      i_levelset_node[num_levelsets] = levelset_node_counter;
      goto new_level;
      
    }

  num_levelsets--;
  level_0 = num_levelsets;

  /*
  printf("num_levelsets: %d\n", num_levelsets);
  */

  num_levelsets = 0;
  levelset_node_counter = 0;
  i_levelset_node[num_levelsets] = 0;
  num_levelsets++;

  for (i=0; i< num_nodes; i++)
    if (i_node_index[i] < level_0-1) 
      i_node_index[i] = 0;
    else
      {
	i_node_index[i] = 1;
	j_levelset_node[levelset_node_counter] = i;
	levelset_node_counter++;
      }
  
  i_levelset_node[num_levelsets] = levelset_node_counter;
	

new_level_again:  
  l=num_levelsets-1; 


  for (j=i_levelset_node[l]; j < i_levelset_node[l+1]; j++)
    {
      for (k=i_node_node[j_levelset_node[j]];
	   k<i_node_node[j_levelset_node[j]+1]; k++)
	{
	  if (i_node_index[j_node_node[k]] == 0)
	    {
	      i_node_index[j_node_node[k]] = num_levelsets+1;
	      j_levelset_node[levelset_node_counter] = j_node_node[k];
	      levelset_node_counter++;
	    }
	}
    }

  if (i_levelset_node[num_levelsets] < levelset_node_counter)
    {
      num_levelsets++;
      i_levelset_node[num_levelsets] = levelset_node_counter;
      goto new_level_again;
      
    }


  hypre_TFree(i_node_node);
  hypre_TFree(j_node_node);

  level_1 = num_levelsets-1;

  if (i_levelset_node[level_1+1] -i_levelset_node[level_1] == 1)
    i_levelset_node[level_1+1] = i_levelset_node[level_1];



  i_boundarysurface_node = hypre_CTAlloc(int, 3);
  j_boundarysurface_node = hypre_CTAlloc(int, 
					 i_levelset_node[1]-i_levelset_node[0]
					 +i_levelset_node[level_1+1]
					 -i_levelset_node[level_1]);
  num_boundarysurfaces = 0;
  boundarysurface_node_counter= 0;
  for (l=0; l < level_1+1; l+=level_1)
    {
      i_boundarysurface_node[num_boundarysurfaces] = 
	boundarysurface_node_counter;
      num_boundarysurfaces++;
    for (j=i_levelset_node[l]; j < i_levelset_node[l+1]; j++)
      {
	j_boundarysurface_node[boundarysurface_node_counter] =
	  j_levelset_node[j];
	boundarysurface_node_counter++;
      }
    }
  
  i_boundarysurface_node[num_boundarysurfaces] = 
    boundarysurface_node_counter;

  /*
  printf("GRAPH boundarysurfaces: =================================\n");
  for (i=0; i < num_boundarysurfaces; i++)
    {
      printf("boundarysurface %d contains nodes: \n", i);
      for (j=i_boundarysurface_node[i]; j < i_boundarysurface_node[i+1]; j++)
	printf("%d ", j_boundarysurface_node[j]);
      
      printf("\n");
    }
    */

  *num_boundarysurfaces_pointer = num_boundarysurfaces;


  *i_boundarysurface_node_pointer = i_boundarysurface_node;
  *j_boundarysurface_node_pointer = j_boundarysurface_node;

  hypre_TFree(i_node_index);
  hypre_TFree(i_levelset_node);
  hypre_TFree(j_levelset_node);
    

  return ierr;

}

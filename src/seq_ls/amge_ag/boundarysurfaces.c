/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




/*****************************************************************************
 *
 * creates graph boundarysurface_node from node_node graph;
 *
 ****************************************************************************/

#include "headers.h"  



HYPRE_Int hypre_AMGeCreateBoundarysurfaces(HYPRE_Int **i_boundarysurface_node_pointer,
				     HYPRE_Int **j_boundarysurface_node_pointer,

				     HYPRE_Int *num_boundarysurfaces_pointer,

				     HYPRE_Int *i_element_node,
				     HYPRE_Int *j_element_node,

				     HYPRE_Int num_elements, 
				     HYPRE_Int num_nodes)

{

  HYPRE_Int ierr = 0;
  HYPRE_Int i,j,k,l;
  HYPRE_Int *i_node_node, *j_node_node; 
  HYPRE_Int *i_node_element, *j_node_element;
  HYPRE_Int *i_boundarysurface_node, *j_boundarysurface_node;

  HYPRE_Int *i_levelset_node, *j_levelset_node;
  HYPRE_Int *i_node_index;
  HYPRE_Int num_levelsets;
  HYPRE_Int level_0, level_1;

  HYPRE_Int levelset_node_counter;
  HYPRE_Int num_boundarysurfaces;
  HYPRE_Int boundarysurface_node_counter;

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
  

  i_levelset_node = hypre_CTAlloc(HYPRE_Int, num_nodes+1);
  j_levelset_node = hypre_CTAlloc(HYPRE_Int, num_nodes);

  i_node_index = hypre_CTAlloc(HYPRE_Int, num_nodes);
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
  hypre_printf("num_levelsets: %d\n", num_levelsets);
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



  i_boundarysurface_node = hypre_CTAlloc(HYPRE_Int, 3);
  j_boundarysurface_node = hypre_CTAlloc(HYPRE_Int, 
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
  hypre_printf("GRAPH boundarysurfaces: =================================\n");
  for (i=0; i < num_boundarysurfaces; i++)
    {
      hypre_printf("boundarysurface %d contains nodes: \n", i);
      for (j=i_boundarysurface_node[i]; j < i_boundarysurface_node[i+1]; j++)
	hypre_printf("%d ", j_boundarysurface_node[j]);
      
      hypre_printf("\n");
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

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






#include "headers.h"  

/*****************************************************************************
 *
 * builds (minimal) coarse nodal set and node coarse node neighbors;
 * coarse node neighbors are in terms of fine node ordering;
 *
 * input information AEface_node, and AE_node, node_AE graphs;
 *
 ****************************************************************************/

HYPRE_Int hypre_AMGeCoarseNodeSelection(HYPRE_Int *i_AEface_node, HYPRE_Int *j_AEface_node,
				  HYPRE_Int *i_AE_node, HYPRE_Int *j_AE_node, 
				  HYPRE_Int *i_node_AE, HYPRE_Int *j_node_AE,

				  HYPRE_Int num_AEfaces, HYPRE_Int num_nodes,

				  HYPRE_Int **i_node_coarsenode_neighbor_pointer,
				  HYPRE_Int **j_node_coarsenode_neighbor_pointer,

				  HYPRE_Int **i_node_coarsenode_pointer,
				  HYPRE_Int **j_node_coarsenode_pointer,

				  HYPRE_Int *num_coarsenodes)

{

  HYPRE_Int ierr = 0;
  HYPRE_Int i,j,k;

 
  HYPRE_Int *i_node_coarsenode, *j_node_coarsenode;
  HYPRE_Int *i_node_coarsenode_neighbor, *j_node_coarsenode_neighbor;


  HYPRE_Int *i_node_index;
  HYPRE_Int coarsenode_counter, node_coarsenode_neighbor_counter;
  


  i_node_coarsenode = hypre_CTAlloc(HYPRE_Int, num_nodes+1);
  for (i=0; i < num_nodes+1; i++)
    i_node_coarsenode[i] = 0;

  for (i=0; i < num_AEfaces; i++)
    for (j=i_AEface_node[i]; j < i_AEface_node[i+1]; j++)
      i_node_coarsenode[j_AEface_node[j]]++;    


  coarsenode_counter = 0;
  for (i=0; i < num_nodes; i++)
    if (i_node_coarsenode[i] > 1) /*  || num_AEfaces < 3)  */
      coarsenode_counter++;


  j_node_coarsenode = hypre_CTAlloc(HYPRE_Int, coarsenode_counter);

  coarsenode_counter = 0;
  for (i=0; i < num_nodes; i++)
    if (i_node_coarsenode[i] > 1) /* || num_AEfaces < 3 )  */
      {
	i_node_coarsenode[i] = 1;
	j_node_coarsenode[coarsenode_counter] = coarsenode_counter;
	coarsenode_counter++;
      }
    else 
      i_node_coarsenode[i] = 0;

  *num_coarsenodes = coarsenode_counter;

  for (i=0; i < num_nodes; i++)
    i_node_coarsenode[i+1] += i_node_coarsenode[i];

  for (i=num_nodes; i>0; i--)
    i_node_coarsenode[i] = i_node_coarsenode[i-1];

  i_node_coarsenode[0] = 0;


  i_node_index = hypre_CTAlloc(HYPRE_Int, num_nodes+1);

  for (i=0; i < num_nodes; i++)
    i_node_index[i] = 0;

  node_coarsenode_neighbor_counter = 0;
  for (i=0; i < num_nodes; i++)
    if (i_node_coarsenode[i+1] > i_node_coarsenode[i])
      node_coarsenode_neighbor_counter++;
    else
      if (i_node_AE[i+1] == i_node_AE[i]+1)
	{
	  for (j=i_AE_node[j_node_AE[i_node_AE[i]]];
	       j<i_AE_node[j_node_AE[i_node_AE[i]] + 1]; j++)
	    if (i_node_coarsenode[j_AE_node[j]+1] > 
		i_node_coarsenode[j_AE_node[j]])
	      node_coarsenode_neighbor_counter++;
	}
      else
	{
	  for (j=i_node_AE[i]; j < i_node_AE[i+1]; j++)
	    for (k=i_AE_node[j_node_AE[j]]; k < i_AE_node[j_node_AE[j]+1]; k++)
	      i_node_index[j_AE_node[k]]++;

	  for (j=i_AE_node[j_node_AE[i_node_AE[i]]]; 
	       j < i_AE_node[j_node_AE[i_node_AE[i]]+1]; j++)
	    if (i_node_index[j_AE_node[j]] == i_node_AE[i+1]- i_node_AE[i] &&
		i_node_coarsenode[j_AE_node[j]+1] > 
		i_node_coarsenode[j_AE_node[j]])
	      node_coarsenode_neighbor_counter++;

	  for (j=i_node_AE[i]; j < i_node_AE[i+1]; j++)
	    for (k=i_AE_node[j_node_AE[j]]; k < i_AE_node[j_node_AE[j]+1]; k++)
	      i_node_index[j_AE_node[k]] = 0;
	}
    

  j_node_coarsenode_neighbor = hypre_CTAlloc(HYPRE_Int,  
					     node_coarsenode_neighbor_counter);


  i_node_coarsenode_neighbor = hypre_CTAlloc(HYPRE_Int, num_nodes+1);
	
  node_coarsenode_neighbor_counter = 0;
  for (i=0; i < num_nodes; i++)
    {
      i_node_coarsenode_neighbor[i] = node_coarsenode_neighbor_counter;
      if (i_node_coarsenode[i+1] > i_node_coarsenode[i])
	{
	  j_node_coarsenode_neighbor[node_coarsenode_neighbor_counter]=i;
	  node_coarsenode_neighbor_counter++;
	}
      else
	if (i_node_AE[i+1] == i_node_AE[i]+1)
	  {
	    for (j=i_AE_node[j_node_AE[i_node_AE[i]]];
		 j<i_AE_node[j_node_AE[i_node_AE[i]] + 1]; j++)
	      if (i_node_coarsenode[j_AE_node[j]+1] > 
		  i_node_coarsenode[j_AE_node[j]])
		{
		  j_node_coarsenode_neighbor[node_coarsenode_neighbor_counter]
		    =j_AE_node[j];
		  node_coarsenode_neighbor_counter++;
		}
	  }
	else
	  {
	    for (j=i_node_AE[i]; j < i_node_AE[i+1]; j++)
	      for (k=i_AE_node[j_node_AE[j]]; k<i_AE_node[j_node_AE[j]+1]; k++)
		i_node_index[j_AE_node[k]]++;

	    for (j=i_AE_node[j_node_AE[i_node_AE[i]]]; 
		 j < i_AE_node[j_node_AE[i_node_AE[i]]+1]; j++)
	      if (i_node_index[j_AE_node[j]] == i_node_AE[i+1]- i_node_AE[i] &&
		  i_node_coarsenode[j_AE_node[j]+1] > 
		  i_node_coarsenode[j_AE_node[j]])
		{
		  j_node_coarsenode_neighbor[node_coarsenode_neighbor_counter]
		    =j_AE_node[j];
		  node_coarsenode_neighbor_counter++;
		}

	    for (j=i_node_AE[i]; j < i_node_AE[i+1]; j++)
	      for (k=i_AE_node[j_node_AE[j]]; k < i_AE_node[j_node_AE[j]+1]; k++)
		i_node_index[j_AE_node[k]] = 0;
	  }
    }
  i_node_coarsenode_neighbor[num_nodes] = node_coarsenode_neighbor_counter;

  hypre_TFree(i_node_index);    


  *i_node_coarsenode_neighbor_pointer = i_node_coarsenode_neighbor;
  *j_node_coarsenode_neighbor_pointer = j_node_coarsenode_neighbor;

  *i_node_coarsenode_pointer = i_node_coarsenode;
  *j_node_coarsenode_pointer = j_node_coarsenode;

  /*

  hypre_printf("============ begin coarse node neighbors =======================\n");
  for (i=0; i < num_nodes; i++)
    {
      if (i_node_coarsenode[i+1] == i_node_coarsenode[i])
	hypre_printf("fine node %d has coarse neighbors:\n", i);
      else
	hypre_printf("coarse node[%d]:%d has coarse neighbors:\n", i,
	       j_node_coarsenode[i_node_coarsenode[i]]);
      for (j = i_node_coarsenode_neighbor[i]; 
	   j < i_node_coarsenode_neighbor[i+1]; j++)
	hypre_printf("%d ", j_node_coarsenode_neighbor[j]);

      hypre_printf("\n");
    }
	   
  hypre_printf("============ end coarse node neighbors ========================\n");

  */

  return ierr;
}



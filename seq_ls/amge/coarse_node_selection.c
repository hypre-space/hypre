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
 * builds (minimal) coarse nodal set and node coarse node neighbors;
 * coarse node neighbors are in terms of fine node ordering;
 *
 * input information AEface_node, and AE_node, node_AE graphs;
 *
 ****************************************************************************/

int hypre_AMGeCoarseNodeSelection(int *i_AEface_node, int *j_AEface_node,
				  int *i_AE_node, int *j_AE_node, 
				  int *i_node_AE, int *j_node_AE,

				  int num_AEfaces, int num_nodes,

				  int **i_node_coarsenode_neighbor_pointer,
				  int **j_node_coarsenode_neighbor_pointer,

				  int **i_node_coarsenode_pointer,
				  int **j_node_coarsenode_pointer,

				  int *num_coarsenodes)

{

  int ierr = 0;
  int i,j,k;

 
  int *i_node_coarsenode, *j_node_coarsenode;
  int *i_node_coarsenode_neighbor, *j_node_coarsenode_neighbor;


  int *i_node_index;
  int coarsenode_counter, node_coarsenode_neighbor_counter;
  


  i_node_coarsenode = hypre_CTAlloc(int, num_nodes+1);
  for (i=0; i < num_nodes+1; i++)
    i_node_coarsenode[i] = 0;

  for (i=0; i < num_AEfaces; i++)
    for (j=i_AEface_node[i]; j < i_AEface_node[i+1]; j++)
      i_node_coarsenode[j_AEface_node[j]]++;    


  coarsenode_counter = 0;
  for (i=0; i < num_nodes; i++)
    if (i_node_coarsenode[i] > 1)
      coarsenode_counter++;


  j_node_coarsenode = hypre_CTAlloc(int, coarsenode_counter);

  coarsenode_counter = 0;
  for (i=0; i < num_nodes; i++)
    if (i_node_coarsenode[i] > 1)
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


  i_node_index = hypre_CTAlloc(int, num_nodes+1);

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
    

  j_node_coarsenode_neighbor = hypre_CTAlloc(int,  
					     node_coarsenode_neighbor_counter);


  i_node_coarsenode_neighbor = hypre_CTAlloc(int, num_nodes+1);
	
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

  printf("============ begin coarse node neighbors =======================\n");
  for (i=0; i < num_nodes; i++)
    {
      if (i_node_coarsenode[i+1] == i_node_coarsenode[i])
	printf("fine node %d has coarse neighbors:\n", i);
      else
	printf("coarse node[%d]:%d has coarse neighbors:\n", i,
	       j_node_coarsenode[i_node_coarsenode[i]]);
      for (j = i_node_coarsenode_neighbor[i]; 
	   j < i_node_coarsenode_neighbor[i+1]; j++)
	printf("%d ", j_node_coarsenode_neighbor[j]);

      printf("\n");
    }
	   
  printf("============ end coarse node neighbors ========================\n");

  */

  return ierr;
}



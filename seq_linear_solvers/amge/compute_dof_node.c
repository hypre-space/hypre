#include "headers.h"  

/*-------------------------------------------------------------------------*/

int compute_dof_node(int **i_dof_node_pointer, int **j_dof_node_pointer,
		     int num_nodes, int system_size,
		     int *num_dofs_pointer)

{
  int ierr = 0, i,j, k;
  int num_dofs = system_size *num_nodes;
  
  int *i_dof_node, *j_dof_node;

  int dof_counter = 0, dof_node_counter=0;

  i_dof_node = hypre_CTAlloc(int, num_dofs+1);
  j_dof_node = hypre_CTAlloc(int, num_dofs);

  for (i=0; i<num_nodes; i++)
    {
      for (j=0; j < system_size; j++)
	{
	  i_dof_node[dof_counter] = dof_node_counter;
	  dof_counter++;
	  j_dof_node[dof_node_counter] = i;
	  dof_node_counter++;
	}
    }
  i_dof_node[dof_counter] = dof_node_counter;

  *i_dof_node_pointer = i_dof_node;
  *j_dof_node_pointer = j_dof_node;

  num_dofs_pointer[0] = dof_counter;

  return ierr;
}
    

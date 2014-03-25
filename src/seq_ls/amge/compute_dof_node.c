#include "headers.h"  

/*-------------------------------------------------------------------------*/

HYPRE_Int compute_dof_node(HYPRE_Int **i_dof_node_pointer, HYPRE_Int **j_dof_node_pointer,
		     HYPRE_Int num_nodes, HYPRE_Int system_size,
		     HYPRE_Int *num_dofs_pointer)

{
  HYPRE_Int ierr = 0, i,j, k;
  HYPRE_Int num_dofs = system_size *num_nodes;
  
  HYPRE_Int *i_dof_node, *j_dof_node;

  HYPRE_Int dof_counter = 0, dof_node_counter=0;

  i_dof_node = hypre_CTAlloc(HYPRE_Int, num_dofs+1);
  j_dof_node = hypre_CTAlloc(HYPRE_Int, num_dofs);

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
    

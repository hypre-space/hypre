/*-------------------------------------------------------------------------*/

int compute_dof_on_boundary(int **i_dof_on_boundary_pointer, 
			    int *i_node_on_boundary,
			    int num_nodes, int system_size)

{
  int ierr = 0, i,j;
  int num_dofs = system_size *num_nodes;
  int dof_counter = 0;

  int *i_dof_on_boundary;

  i_dof_on_boundary = (int *) malloc(num_dofs * sizeof(int));

  for (i=0; i<num_nodes; i++)
    {
      for (j=0; j < system_size; j++)
	{
	  i_dof_on_boundary[dof_counter] = i_node_on_boundary[i];
	  dof_counter++;
	}
    }

  *i_dof_on_boundary_pointer = i_dof_on_boundary;

  return ierr;
}
    

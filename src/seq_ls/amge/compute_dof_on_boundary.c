/*-------------------------------------------------------------------------*/

HYPRE_Int compute_dof_on_boundary(HYPRE_Int **i_dof_on_boundary_pointer, 
			    HYPRE_Int *i_node_on_boundary,
			    HYPRE_Int num_nodes, HYPRE_Int system_size)

{
  HYPRE_Int ierr = 0, i,j;
  HYPRE_Int num_dofs = system_size *num_nodes;
  HYPRE_Int dof_counter = 0;

  HYPRE_Int *i_dof_on_boundary;

  i_dof_on_boundary = (HYPRE_Int *) malloc(num_dofs * sizeof(HYPRE_Int));

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
    

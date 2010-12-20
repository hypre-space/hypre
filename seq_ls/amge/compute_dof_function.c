#include "headers.h"  

/*-------------------------------------------------------------------------*/
/* compute dof_function based on standard node_dof distribution; ----------*/
/*-------------------------------------------------------------------------*/

HYPRE_Int hypre_DofFunction(HYPRE_Int **dof_function_pointer,

		      HYPRE_Int num_dofs, HYPRE_Int num_functions)

{
  HYPRE_Int ierr = 0;
  HYPRE_Int i,j; 

  HYPRE_Int num_nodes;

  HYPRE_Int *dof_function;
  HYPRE_Int dof_counter = 0;

  num_nodes = num_dofs / num_functions;

  if (num_nodes * num_functions != num_dofs)
    {
      hypre_printf("WRONG num_dofs: %d, num_nodes: %d, OR num_functions: %d\n",
	     num_dofs, num_nodes, num_functions);

      return -1; 
    }

  dof_function = hypre_CTAlloc(HYPRE_Int, num_dofs);
  
  for (i=0; i < num_nodes; i++)
    for (j=0; j < num_functions; j++)
      {
	dof_function[dof_counter] = j;
	dof_counter++;
      }

  *dof_function_pointer = dof_function;

  return ierr;
}




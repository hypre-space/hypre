#include "headers.h"  

/*-------------------------------------------------------------------------*/
/* compute dof_function based on standard node_dof distribution; ----------*/
/*-------------------------------------------------------------------------*/

int hypre_DofFunction(int **dof_function_pointer,

		      int num_dofs, int num_functions)

{
  int ierr = 0;
  int i,j; 

  int num_nodes;

  int *dof_function;
  int dof_counter = 0;

  num_nodes = num_dofs / num_functions;

  if (num_nodes * num_functions != num_dofs)
    {
      printf("WRONG num_dofs: %d, num_nodes: %d, OR num_functions: %d\n",
	     num_dofs, num_nodes, num_functions);

      return -1; 
    }

  dof_function = hypre_CTAlloc(int, num_dofs);
  
  for (i=0; i < num_nodes; i++)
    for (j=0; j < num_functions; j++)
      {
	dof_function[dof_counter] = j;
	dof_counter++;
      }

  *dof_function_pointer = dof_function;

  return ierr;
}




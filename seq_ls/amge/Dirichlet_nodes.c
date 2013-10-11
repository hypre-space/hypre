
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void main()

{
  FILE *f;

  HYPRE_Int i;

  HYPRE_Int num_nodes, num_elements;
  HYPRE_Int *i_node_on_boundary;

  HYPRE_Real *x_coord, *y_coord;
  HYPRE_Real eps = 1.e-4;


  f = fopen("element_node", "r");
  hypre_fscanf(f, "%d %d", &num_elements, &num_nodes);

  fclose(f);
 

  x_coord = (HYPRE_Real *) malloc(num_nodes * sizeof(HYPRE_Real));
  y_coord = (HYPRE_Real *) malloc(num_nodes * sizeof(HYPRE_Real));

  f = fopen("coordinates", "r");

  for( i = 0; i < num_nodes; i++ )
    hypre_fscanf(f, "%le %le\n", &x_coord[i], &y_coord[i]);

  fclose(f);

  i_node_on_boundary = (HYPRE_Int *) malloc(num_nodes * sizeof(HYPRE_Int));
  for (i=0; i<num_nodes; i++)
    i_node_on_boundary[i] = -1;

  for (i=0; i<num_nodes; i++)
    if (x_coord[i] < eps || x_coord[i] > 1.e0-eps || y_coord[i] < eps
	|| y_coord[i] > 1.e0-eps)
      i_node_on_boundary[i] = 0;

  f = fopen("node_on_boundary", "w");

  for (i=0; i<num_nodes; i++)
    hypre_fprintf(f, "%d\n", i_node_on_boundary[i]);

  fclose(f);

  free(i_node_on_boundary);
  free(x_coord);
  free(y_coord);

}
  



  

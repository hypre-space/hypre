
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void main()

{
  FILE *f;

  int i;

  int num_nodes, num_elements;
  int *i_node_on_boundary;

  double *x_coord, *y_coord;
  double eps = 1.e-4;


  f = fopen("element_node", "r");
  fscanf(f, "%d %d", &num_elements, &num_nodes);

  fclose(f);
 

  x_coord = (double *) malloc(num_nodes * sizeof(double));
  y_coord = (double *) malloc(num_nodes * sizeof(double));

  f = fopen("coordinates", "r");

  for( i = 0; i < num_nodes; i++ )
    fscanf(f, "%le %le\n", &x_coord[i], &y_coord[i]);

  fclose(f);

  i_node_on_boundary = (int *) malloc(num_nodes * sizeof(int));
  for (i=0; i<num_nodes; i++)
    i_node_on_boundary[i] = -1;

  for (i=0; i<num_nodes; i++)
    if (x_coord[i] < eps || x_coord[i] > 1.e0-eps || y_coord[i] < eps
	|| y_coord[i] > 1.e0-eps)
      i_node_on_boundary[i] = 0;

  f = fopen("node_on_boundary", "w");

  for (i=0; i<num_nodes; i++)
    fprintf(f, "%d\n", i_node_on_boundary[i]);

  fclose(f);

  free(i_node_on_boundary);
  free(x_coord);
  free(y_coord);

}
  



  

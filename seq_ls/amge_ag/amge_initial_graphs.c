/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/
/*****************************************************************************
 * reads from file and creates graphs:
 * element_node, element_face, face_element, boundarysurface_face, and
 * face_node;
 ****************************************************************************/

#include "headers.h"  


int hypre_AMGeInitialGraphs(int **i_element_node_pointer,
			    int **j_element_node_pointer,


			    int **i_boundarysurface_node_pointer,
			    int **j_boundarysurface_node_pointer,


			    int *num_elements_pointer,
			    int *num_nodes_pointer,
			    int *num_boundarysurfaces_pointer,

			    int **i_node_on_boundary_pointer,


			    char *element_node_file)

{
  int ierr = 0;
  FILE *f;

  int i,j,k,l;
  int num_elements, num_faces, num_boundarysurfaces;
  int *i_element_face, *j_element_face;
  int *i_face_face, *j_face_face;


  int *i_boundarysurface_node, *j_boundarysurface_node;

  int *i_node_on_boundary;

  int num_nodes;
  int *i_element_node, *j_element_node;

  int element_node_counter, boundarysurface_node_counter;

  double eps = 1.e-4;

  f = fopen(element_node_file, "r");

  fscanf(f, "%d %d", &num_elements, &num_nodes);


  i_element_node = hypre_CTAlloc(int, num_elements+1);

  /* j_element_node = hypre_CTAlloc(int, 3*num_elements); */

  j_element_node = hypre_CTAlloc(int, 24*num_elements); 

  element_node_counter = 0;
  for (i=0; i < num_elements; i++)
    {
      i_element_node[i] = element_node_counter;
      /*      fscanf(f, "%d %d %d\n", */
      fscanf(f, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
	     &j_element_node[element_node_counter],
	     &j_element_node[element_node_counter+1],
	     &j_element_node[element_node_counter+2],
	     &j_element_node[element_node_counter+3],
	     &j_element_node[element_node_counter+4],
	     &j_element_node[element_node_counter+5],
	     &j_element_node[element_node_counter+6],
	     &j_element_node[element_node_counter+7],
	     &j_element_node[element_node_counter+8],
	     &j_element_node[element_node_counter+9],
	     &j_element_node[element_node_counter+10],
	     &j_element_node[element_node_counter+11],
	     &j_element_node[element_node_counter+12],
	     &j_element_node[element_node_counter+13],
	     &j_element_node[element_node_counter+14],
	     &j_element_node[element_node_counter+15],
	     &j_element_node[element_node_counter+16],
	     &j_element_node[element_node_counter+17],
	     &j_element_node[element_node_counter+18],
	     &j_element_node[element_node_counter+19],
	     &j_element_node[element_node_counter+20],
	     &j_element_node[element_node_counter+21],
	     &j_element_node[element_node_counter+22],
	     &j_element_node[element_node_counter+23]);
      /* element_node_counter+=3; */
      element_node_counter+=24;
    }

  i_element_node[num_elements] = element_node_counter;

  fclose(f); 


  for (i=0; i < num_elements; i++)
    for (j=i_element_node[i]; j < i_element_node[i+1]; j++)
      j_element_node[j]--;
  
  ierr = hypre_AMGeCreateBoundarysurfaces(&i_boundarysurface_node,
					  &j_boundarysurface_node,

					  &num_boundarysurfaces,

					  i_element_node,
					  j_element_node,

					  num_elements, 
					  num_nodes);
					  

  /*
  printf("GRAPH boundarysurfaces: =================================\n");
  for (i=0; i < num_boundarysurfaces; i++)
    {
      printf("boundarysurface %d contains nodes: \n", i);
      for (j=i_boundarysurface_node[i]; j < i_boundarysurface_node[i+1]; j++)
	printf("%d: \n", j_boundarysurface_node[j]);
      
      printf("\n");
    }

    */

  i_node_on_boundary = hypre_CTAlloc(int, num_nodes);
  for (i=0; i<num_nodes; i++)
    i_node_on_boundary[i] = -1;

  for (i=0; i < num_boundarysurfaces; i++)
    {
      for (j=i_boundarysurface_node[i]; j < i_boundarysurface_node[i+1]; j++)
	i_node_on_boundary[j_boundarysurface_node[j]] = 0;
    }


  num_boundarysurfaces = 0; 
  /* ================================================================*/
  /*                                                                 */
  /* build element topology:                                         */
  /* ================================================================*/

  *i_node_on_boundary_pointer = i_node_on_boundary;

  *i_element_node_pointer = i_element_node;
  *j_element_node_pointer = j_element_node;

  *i_boundarysurface_node_pointer = i_boundarysurface_node;
  *j_boundarysurface_node_pointer = j_boundarysurface_node;

  *num_elements_pointer = num_elements;
  *num_nodes_pointer = num_nodes;
  *num_boundarysurfaces_pointer = num_boundarysurfaces;


  return ierr;


}

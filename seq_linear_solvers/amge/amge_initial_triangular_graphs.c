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


int hypre_AMGeTriangInitialGraphs(int **i_element_node_pointer,
				  int **j_element_node_pointer,




				  int *num_elements_pointer,
				  int *num_nodes_pointer,

				  int **i_node_on_boundary_pointer,


				  char *element_node_file,
				  char *node_on_boundary_file)

{
  int ierr = 0;
  FILE *f;

  int i,j,k,l;
  int num_elements, num_faces;
  int *i_element_face, *j_element_face;
  int *i_face_face, *j_face_face;


  int num_boundarynodes;
  int *i_node_on_boundary;

  int num_nodes;
  int *i_element_node, *j_element_node;

  int element_node_counter;

  double eps = 1.e-4;

  f = fopen(element_node_file, "r");

  fscanf(f, "%d %d", &num_elements, &num_nodes);


  i_element_node = hypre_CTAlloc(int, num_elements+1);

  j_element_node = hypre_CTAlloc(int, 3*num_elements); 

  element_node_counter = 0;
  for (i=0; i < num_elements; i++)
    {
      i_element_node[i] = element_node_counter;
      fscanf(f, "%d %d %d\n",
	     &j_element_node[element_node_counter],
	     &j_element_node[element_node_counter+1],
	     &j_element_node[element_node_counter+2]);
      element_node_counter+=3; 
    }

  i_element_node[num_elements] = element_node_counter;

  fclose(f); 


  i_node_on_boundary = hypre_CTAlloc(int, num_nodes);
  for (i=0; i<num_nodes; i++)
    i_node_on_boundary[i] = -1;

  f = fopen(node_on_boundary_file, "r");

  fscanf(f, "%d ", &num_boundarynodes);

  for (i=0; i < num_boundarynodes; i++)
    {
      fscanf(f, "%d \n", &j);
      i_node_on_boundary[j] = 0;
    }

  fclose(f);

  /* ================================================================*/
  /*                                                                 */
  /* build element topology:                                         */
  /* ================================================================*/

  *i_node_on_boundary_pointer = i_node_on_boundary;

  *i_element_node_pointer = i_element_node;
  *j_element_node_pointer = j_element_node;


  *num_elements_pointer = num_elements;
  *num_nodes_pointer = num_nodes;


  return ierr;


}

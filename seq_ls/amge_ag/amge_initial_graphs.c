/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




/*****************************************************************************
 * reads from file and creates graphs:
 * element_node, element_face, face_element, boundarysurface_face, and
 * face_node;
 ****************************************************************************/

#include "headers.h"  


HYPRE_Int hypre_AMGeInitialGraphs(HYPRE_Int **i_element_node_pointer,
			    HYPRE_Int **j_element_node_pointer,


			    HYPRE_Int **i_boundarysurface_node_pointer,
			    HYPRE_Int **j_boundarysurface_node_pointer,


			    HYPRE_Int *num_elements_pointer,
			    HYPRE_Int *num_nodes_pointer,
			    HYPRE_Int *num_boundarysurfaces_pointer,

			    HYPRE_Int **i_node_on_boundary_pointer,


			    char *element_node_file)

{
  HYPRE_Int ierr = 0;
  FILE *f;

  HYPRE_Int i,j,k,l;
  HYPRE_Int num_elements, num_faces, num_boundarysurfaces;
  HYPRE_Int *i_element_face, *j_element_face;
  HYPRE_Int *i_face_face, *j_face_face;


  HYPRE_Int *i_boundarysurface_node, *j_boundarysurface_node;

  HYPRE_Int *i_node_on_boundary;

  HYPRE_Int num_nodes;
  HYPRE_Int *i_element_node, *j_element_node;

  HYPRE_Int element_node_counter, boundarysurface_node_counter;

  HYPRE_Real eps = 1.e-4;

  f = fopen(element_node_file, "r");

  hypre_fscanf(f, "%d %d", &num_elements, &num_nodes);


  i_element_node = hypre_CTAlloc(HYPRE_Int, num_elements+1);

  /* j_element_node = hypre_CTAlloc(HYPRE_Int, 3*num_elements); */

  j_element_node = hypre_CTAlloc(HYPRE_Int, 24*num_elements); 

  element_node_counter = 0;
  for (i=0; i < num_elements; i++)
    {
      i_element_node[i] = element_node_counter;
      /*      hypre_fscanf(f, "%d %d %d\n", */
      hypre_fscanf(f, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
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
  hypre_printf("GRAPH boundarysurfaces: =================================\n");
  for (i=0; i < num_boundarysurfaces; i++)
    {
      hypre_printf("boundarysurface %d contains nodes: \n", i);
      for (j=i_boundarysurface_node[i]; j < i_boundarysurface_node[i+1]; j++)
	hypre_printf("%d: \n", j_boundarysurface_node[j]);
      
      hypre_printf("\n");
    }

    */

  i_node_on_boundary = hypre_CTAlloc(HYPRE_Int, num_nodes);
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

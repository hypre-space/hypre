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
 *
 * builds AMGe matrix topology from graphs:
 *
 * element_node and boundarysurface_node 
 *
 ****************************************************************************/


#include "headers.h"  

HYPRE_Int hypre_BuildAMGeMatrixTopology(hypre_AMGeMatrixTopology **A_pointer,
				  HYPRE_Int *i_element_node,
				  HYPRE_Int *j_element_node,

				  HYPRE_Int *i_boundarysurface_node,
				  HYPRE_Int *j_boundarysurface_node,

				  HYPRE_Int num_elements,
				  HYPRE_Int num_nodes,
				  HYPRE_Int num_boundarysurfaces)

{
  
  HYPRE_Int ierr = 0;
  HYPRE_Int i,j,k,l,m,n;

  hypre_AMGeMatrixTopology *A; 

  HYPRE_Int *i_element_face, *j_element_face;
  HYPRE_Int *i_face_face, *j_face_face;
  HYPRE_Int *i_boundarysurface_face, *j_boundarysurface_face;

  HYPRE_Int *i_face_node, *j_face_node;
  HYPRE_Int *i_face_element, *j_face_element;
  HYPRE_Int *i_face_index, *i_node_index;

  HYPRE_Int *i_face_boundarysurface, *j_face_boundarysurface;

  HYPRE_Int *i_node_face, *j_node_face;
  HYPRE_Int *i_node_element, *j_node_element;
  HYPRE_Int *i_element_element, *j_element_element;
  HYPRE_Int *i_element_boundarysurface, *j_element_boundarysurface;

  HYPRE_Int *i_boundarysurface_element, *j_boundarysurface_element;

  HYPRE_Int num_faces;

  HYPRE_Int num_boundary_faces, num_bfaces;

  HYPRE_Int element_node_counter, boundarysurface_node_counter=0;

    
  HYPRE_Int i_face_on_surface;

  HYPRE_Int *i_element_element_0, *j_element_element_0;

  /* A = hypre_CreateAMGeMatrixTopology( );  */

  A = hypre_CTAlloc(hypre_AMGeMatrixTopology, 1); 
  hypre_CreateAMGeMatrixTopology(A);
  
  /* ================================================================*/
  /*                                                                 */
  /* build element topology:                                         */
  /* ================================================================*/




  ierr = transpose_matrix_create(&i_node_element, &j_node_element,

				 i_element_node, j_element_node,
				 num_elements, num_nodes);




  ierr = matrix_matrix_product(&i_element_element, &j_element_element,
			       i_element_node, j_element_node,
			       i_node_element, j_node_element,
			       num_elements, num_nodes, num_elements);





  if (num_boundarysurfaces > 0)
  ierr = matrix_matrix_product(&i_boundarysurface_element,
			       &j_boundarysurface_element,

			       i_boundarysurface_node, j_boundarysurface_node,
			       
			       i_node_element, j_node_element,
			       
			       num_boundarysurfaces, num_nodes, num_elements);



  /* compute adjoint graph: -----------------------------------------*/
  if (num_boundarysurfaces > 0)
  ierr = transpose_matrix_create(&i_element_boundarysurface, 
				 &j_element_boundarysurface,

				 i_boundarysurface_element, 
				 j_boundarysurface_element,

				 num_boundarysurfaces, num_elements);



  if (num_boundarysurfaces > 0)
    {
      hypre_TFree(i_boundarysurface_element);
      hypre_TFree(j_boundarysurface_element);
    }


  /* ==================================================================== */

  ierr = create_maximal_intersection_sets(i_element_node, j_element_node,
					  i_node_element, j_node_element,
					  i_element_element, j_element_element,

					  i_element_boundarysurface, 
					  j_element_boundarysurface,

					  i_boundarysurface_node, 
					  j_boundarysurface_node,

					  num_boundarysurfaces, 
					  num_elements, num_nodes,

					  &i_face_element, 
					  &j_face_element,

					  &i_face_node, &j_face_node,

					  &num_faces);


  hypre_TFree(i_element_element);
  hypre_TFree(j_element_element);

  ierr = transpose_matrix_create(&i_node_face, &j_node_face,

				 i_face_node, j_face_node,
				 num_faces, num_nodes);




  num_boundary_faces=0;
  for (i=0; i < num_faces; i++)
    if (i_face_element[i]+1 == i_face_element[i+1])
      num_boundary_faces++;







  hypre_TFree(i_node_element);
  hypre_TFree(j_node_element);
  if (num_boundarysurfaces >0)
    {
      hypre_TFree(i_element_boundarysurface);
      hypre_TFree(j_element_boundarysurface);
    }
  /* ================================================================*/
  /*                                                                 */
  /* build face topology:                                            */
  /* ================================================================*/

  ierr = matrix_matrix_t_product(&i_face_face, &j_face_face,

				 i_face_node, j_face_node,

				 num_faces, num_nodes);

 

  /* label faces that are on domain boundary; ============================= */

  /* i_face_index = (HYPRE_Int *) malloc(num_faces * sizeof(HYPRE_Int)); */
  i_face_index = hypre_CTAlloc(HYPRE_Int, num_faces);


  for (i=0; i < num_faces; i++)
    i_face_index[i] = -1;

  i_boundarysurface_face = hypre_CTAlloc(HYPRE_Int, num_boundarysurfaces+1);


  i_node_index = hypre_CTAlloc(HYPRE_Int, num_nodes);
  for (i=0; i <num_nodes; i++)
    i_node_index[i] = -1;

  for (i=0; i < num_boundarysurfaces; i++)
    {
      for (j=i_boundarysurface_node[i]; j<i_boundarysurface_node[i+1]; j++)
	i_node_index[j_boundarysurface_node[j]] = 0;

      for (j=i_boundarysurface_node[i]; j<i_boundarysurface_node[i+1]; j++)
	for (k=i_node_face[j_boundarysurface_node[j]];
	     k<i_node_face[j_boundarysurface_node[j]+1]; k++)
	  {
	    i_face_on_surface = -1;
	    for (l = i_face_node[j_node_face[k]]; 
		 l < i_face_node[j_node_face[k]+1]; l++)
	      if (i_node_index[j_face_node[l]] == -1) i_face_on_surface=0;

	    if (i_face_on_surface == -1)   
	      i_face_index[j_node_face[k]] = 0;
	  }

      for (j=i_boundarysurface_node[i]; 
	   j<i_boundarysurface_node[i+1]; j++)
	i_node_index[j_boundarysurface_node[j]] = -1;

    }


  i_boundarysurface_face[num_boundarysurfaces] = 0;
  for (i=0; i <num_faces; i++)
    if (i_face_index[i] == 0) 
      i_boundarysurface_face[num_boundarysurfaces]++;




  if (num_boundarysurfaces > 0)
  j_boundarysurface_face = hypre_CTAlloc(HYPRE_Int,i_boundarysurface_face
					   [num_boundarysurfaces]);

  num_bfaces=0;
  for (i=0; i < num_boundarysurfaces; i++)
    {
      i_boundarysurface_face[i] = num_bfaces;
      for (j=i_boundarysurface_node[i]; j<i_boundarysurface_node[i+1]; j++)
	i_node_index[j_boundarysurface_node[j]] = 0;

      for (j=i_boundarysurface_node[i]; j<i_boundarysurface_node[i+1]; j++)
	for (k=i_node_face[j_boundarysurface_node[j]];
	     k<i_node_face[j_boundarysurface_node[j]+1]; k++)
	  {
	    i_face_on_surface = -1;
	    for (l = i_face_node[j_node_face[k]]; 
		 l < i_face_node[j_node_face[k]+1]; l++)
	      if (i_node_index[j_face_node[l]] == -1) i_face_on_surface=0;

	    if (i_face_on_surface == -1)   
	      {
		if (i_face_index[j_node_face[k]] == 0)
		  {
		    j_boundarysurface_face[num_bfaces] = j_node_face[k];
		    num_bfaces++;
		    i_face_index[j_node_face[k]]=-1;
		  }
	      }
	  }

      for (j=i_boundarysurface_node[i]; 
	   j<i_boundarysurface_node[i+1]; j++)
	i_node_index[j_boundarysurface_node[j]] = -1;

    }

  /*
  if (num_boundarysurfaces > 0)
    {
      hypre_TFree(i_boundarysurface_node);
      hypre_TFree(j_boundarysurface_node);
    }

    */


  hypre_TFree(i_node_index);
  hypre_TFree(i_face_index);

  hypre_TFree(i_node_face);
  hypre_TFree(j_node_face);

  /*
  hypre_printf("num_boundary_faces: %d, num_bfaces: %d\n",
	 i_boundarysurface_face[num_boundarysurfaces], num_bfaces);
	 */

  i_boundarysurface_face[num_boundarysurfaces] = num_bfaces;


  for (i=0; i < num_boundarysurfaces; i++)
    {
      hypre_printf("boundarysurface_face %d contains faces:\n", i);
      for (j=i_boundarysurface_face[i]; 
	   j<i_boundarysurface_face[i+1]; j++)
	hypre_printf("%d, ", j_boundarysurface_face[j]);

      hypre_printf("\n ");
    }



  ierr = transpose_matrix_create(&i_element_face, &j_element_face,
				 i_face_element, j_face_element,
				 num_faces, num_elements);



  i_element_element_0 = hypre_CTAlloc(HYPRE_Int, num_elements+1);
  j_element_element_0 = hypre_CTAlloc(HYPRE_Int, num_elements);


  for (i=0; i < num_elements; i++)
    {
      i_element_element_0[i] = i;
      j_element_element_0[i] = i;
    }

  i_element_element_0[num_elements] = num_elements;

  hypre_AMGeMatrixTopologyIAEElement(A) = i_element_element_0;
  hypre_AMGeMatrixTopologyJAEElement(A) = j_element_element_0;
  

  hypre_AMGeMatrixTopologyIElementNode(A) = i_element_node;
  hypre_AMGeMatrixTopologyJElementNode(A) = j_element_node;

  hypre_AMGeMatrixTopologyNumElements(A) = num_elements;
  hypre_AMGeMatrixTopologyNumNodes(A) = num_nodes;
  hypre_AMGeMatrixTopologyNumFaces(A) = num_faces;

  hypre_AMGeMatrixTopologyNumBoundarysurfaces(A) = num_boundarysurfaces;

  hypre_AMGeMatrixTopologyIElementFace(A) = i_element_face;
  hypre_AMGeMatrixTopologyJElementFace(A) = j_element_face;


  hypre_AMGeMatrixTopologyIFaceElement(A) = i_face_element;
  hypre_AMGeMatrixTopologyJFaceElement(A) = j_face_element;

  hypre_AMGeMatrixTopologyIFaceFace(A) = i_face_face;
  hypre_AMGeMatrixTopologyJFaceFace(A) = j_face_face;


  hypre_AMGeMatrixTopologyIFaceNode(A) = i_face_node;
  hypre_AMGeMatrixTopologyJFaceNode(A) = j_face_node;

  if (num_boundarysurfaces > 0)
    {
      hypre_AMGeMatrixTopologyIBoundarysurfaceFace(A) = i_boundarysurface_face;
      hypre_AMGeMatrixTopologyJBoundarysurfaceFace(A) = j_boundarysurface_face;
    }
  else
    {
      hypre_AMGeMatrixTopologyIBoundarysurfaceFace(A) = NULL;
      hypre_AMGeMatrixTopologyJBoundarysurfaceFace(A) = NULL;

      hypre_TFree(i_boundarysurface_face);
    }

  *A_pointer = A;

  return ierr;
}

/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/


/*****************************************************************************
 *
 * builds AMGe matrix topology from graphs:
 *
 * element_node and boundarysurface_node 
 *
 ****************************************************************************/


#include "headers.h"  

int hypre_BuildAMGeMatrixTopology(hypre_AMGeMatrixTopology **A_pointer,
				  int *i_element_node,
				  int *j_element_node,

				  int *i_boundarysurface_node,
				  int *j_boundarysurface_node,

				  int num_elements,
				  int num_nodes,
				  int num_boundarysurfaces)

{
  
  int ierr = 0;
  int i,j,k,l,m,n;

  hypre_AMGeMatrixTopology *A; 

  int *i_element_face, *j_element_face;
  int *i_face_face, *j_face_face;
  int *i_boundarysurface_face, *j_boundarysurface_face;

  int *i_face_node, *j_face_node;
  int *i_face_element, *j_face_element;
  int *i_face_index, *i_node_index;

  int *i_face_boundarysurface, *j_face_boundarysurface;

  int *i_node_face, *j_node_face;
  int *i_node_element, *j_node_element;
  int *i_element_element, *j_element_element;
  int *i_element_boundarysurface, *j_element_boundarysurface;

  int *i_boundarysurface_element, *j_boundarysurface_element;

  int num_faces;

  int num_boundary_faces, num_bfaces;

  int element_node_counter, boundarysurface_node_counter=0;

    
  int i_face_on_surface;

  int *i_element_element_0, *j_element_element_0;

  /* A = hypre_CreateAMGeMatrixTopology( ); */

  A = hypre_CTAlloc(hypre_AMGeMatrixTopology, 1);

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

  /* i_face_index = (int *) malloc(num_faces * sizeof(int)); */
  i_face_index = hypre_CTAlloc(int, num_faces);


  for (i=0; i < num_faces; i++)
    i_face_index[i] = -1;

  i_boundarysurface_face = hypre_CTAlloc(int, num_boundarysurfaces+1);


  i_node_index = hypre_CTAlloc(int, num_nodes);
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
  j_boundarysurface_face = hypre_CTAlloc(int,i_boundarysurface_face
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

  printf("num_boundary_faces: %d, num_bfaces: %d\n",
	 i_boundarysurface_face[num_boundarysurfaces], num_bfaces);

  i_boundarysurface_face[num_boundarysurfaces] = num_bfaces;


  for (i=0; i < num_boundarysurfaces; i++)
    {
      printf("boundarysurface_face %d contains faces:\n", i);
      for (j=i_boundarysurface_face[i]; 
	   j<i_boundarysurface_face[i+1]; j++)
	printf("%d, ", j_boundarysurface_face[j]);

      printf("\n ");
    }



  ierr = transpose_matrix_create(&i_element_face, &j_element_face,
				 i_face_element, j_face_element,
				 num_faces, num_elements);



  i_element_element_0 = hypre_CTAlloc(int, num_elements+1);
  j_element_element_0 = hypre_CTAlloc(int, num_elements);


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

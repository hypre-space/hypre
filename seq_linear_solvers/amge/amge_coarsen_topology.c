/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/


#include "headers.h"  

/*****************************************************************************
 *
 * builds coarse AMGe matrix topology from fine AMGe matrix topology
 *
 *
 ****************************************************************************/

int hypre_CoarsenAMGeMatrixTopology(hypre_AMGeMatrixTopology *A_crs,

				    
				    hypre_AMGeMatrixTopology *A,

				    int *i_face_to_prefer_weight,
				    int *i_face_weight)

{
  int ierr = 0;
  int i,j,k,l,m;

  int num_elements, num_faces, num_boundarysurfaces;


  int *i_element_face, *j_element_face;
  int *i_face_face, *j_face_face;
  int *i_boundarysurface_face, *j_boundarysurface_face;


  int *i_face_boundarysurface, *j_face_boundarysurface;

  int num_nodes;


  /*------------------------------------------------------------------------
   * arrays to be created: 
   *
   *        AE_AEface, AEface_AEface, boundarysurface_AEface;
   *------------------------------------------------------------------------*/

  int num_AEs, num_AEfaces;
  int *i_AE_AEface, *j_AE_AEface;
  
  int *i_AEface_AEface, *j_AEface_AEface;

  int *i_boundarysurface_AEface, *j_boundarysurface_AEface;

  /*------------------------------------------------------------------------
   * temporary arrays:
   *        AE_AE, AE_node, 
   *        AE_element, AE_face, ... 
   *------------------------------------------------------------------------*/
  
  int *i_AE_AE, *j_AE_AE;
  int *i_AE_element, *j_AE_element;
  int *i_AE_face, *j_AE_face;

  int *i_face_AE, *j_face_AE;

  int *i_AEface_AE, *j_AEface_AE;
  int *i_AEface_face, *j_AEface_face;
  int *i_face_AEface, *j_face_AEface;
  int *i_face_face_AEface, *j_face_face_AEface;

  int *i_AEface_node, *j_AEface_node;


  int *i_boundarysurface_AE, *j_boundarysurface_AE;
  int *i_AE_boundarysurface, *j_AE_boundarysurface;

  int *i_node_face, *j_node_face;
  int *i_node_element, *j_node_element;
  int *i_element_element, *j_element_element;


  int *i_face_node, *j_face_node;
  int *i_face_element, *j_face_element;

  int *w_face_face;



  num_elements = hypre_AMGeMatrixTopologyNumElements(A);
  num_nodes = hypre_AMGeMatrixTopologyNumNodes(A);
  num_faces = hypre_AMGeMatrixTopologyNumFaces(A);

  num_boundarysurfaces = hypre_AMGeMatrixTopologyNumBoundarysurfaces(A);

  i_element_face = hypre_AMGeMatrixTopologyIElementFace(A);
  j_element_face = hypre_AMGeMatrixTopologyJElementFace(A);


  i_face_element = hypre_AMGeMatrixTopologyIFaceElement(A);
  j_face_element = hypre_AMGeMatrixTopologyJFaceElement(A);

  i_face_face = hypre_AMGeMatrixTopologyIFaceFace(A);
  j_face_face = hypre_AMGeMatrixTopologyJFaceFace(A);

  i_face_node = hypre_AMGeMatrixTopologyIFaceNode(A);
  j_face_node = hypre_AMGeMatrixTopologyJFaceNode(A);

  i_boundarysurface_face = hypre_AMGeMatrixTopologyIBoundarysurfaceFace(A);
  j_boundarysurface_face = hypre_AMGeMatrixTopologyJBoundarysurfaceFace(A);



  for (i=0; i < num_faces; i++)
    i_face_to_prefer_weight[i] = i_face_node[i+1] - i_face_node[i];

  /* check for boundary faces and make them unacceptable: ******/

  for (i=0; i < num_faces; i++)
    if (i_face_element[i]+1 == i_face_element[i+1])
      i_face_to_prefer_weight[i] = -1;

  /* if two faces share a common element w = 2 -------------------------*/
  
  w_face_face = hypre_CTAlloc(int, i_face_face[num_faces]);

  /* w_face_face = (int *) malloc(sizeof(int) *i_face_face[num_faces]);  */

  for (i=0; i<num_faces; i++)
    for (j=i_face_face[i]; j< i_face_face[i+1]; j++)
      {
	k= j_face_face[j];
	w_face_face[j]=1;
	for (l=i_face_element[k]; l < i_face_element[k+1]; l++)
	  {
	    for (m=i_element_face[j_face_element[l]];
		 m<i_element_face[j_face_element[l]+1]; m++)
	      if (j_element_face[m] == i) 
		{
		  w_face_face[j] = 3;
		  break;
		}
	  }
      }

  /* weights on the diagonal are zero, i.e., w = 0 ---------------------*/

  for (i=0; i<num_faces; i++)
    for (j=i_face_face[i]; j< i_face_face[i+1]; j++)
      if (j_face_face[j] == i) w_face_face[j] = 0;


  i_AE_element = hypre_CTAlloc(int, num_elements +1);
  j_AE_element = hypre_CTAlloc(int, num_elements);



  ierr = hypre_AMGeAgglomerate(i_AE_element, j_AE_element, 

			       i_face_face, j_face_face, 
			       w_face_face,
		     
			       i_face_element, j_face_element,
			       i_element_face, j_element_face,

			       i_face_to_prefer_weight,

			       i_face_weight,

			       num_faces, num_elements,
			       &num_AEs);

  /*
  printf("num_AEs: %d\n\n", num_AEs);
  printf("END agglomeration step: -----------------------------------\n");
  */

  ierr = matrix_matrix_product(&i_AE_face, &j_AE_face,

			       i_AE_element, j_AE_element,
			       i_element_face, j_element_face,
			       num_AEs, num_elements, num_faces);



  ierr = transpose_matrix_create(&i_face_AE, 
				 &j_face_AE,

				 i_AE_face,
				 j_AE_face,

				 num_AEs, num_faces);




  if (num_boundarysurfaces > 0)
  ierr = transpose_matrix_create(&i_face_boundarysurface, 
				 &j_face_boundarysurface,

				 i_boundarysurface_face, 
				 j_boundarysurface_face,
				 num_boundarysurfaces, num_faces);

  if (num_boundarysurfaces > 0)
  ierr = matrix_matrix_product(&i_boundarysurface_AE,
			       &j_boundarysurface_AE,

			       i_boundarysurface_face, j_boundarysurface_face,
			       
			       i_face_AE, j_face_AE,
			       
			       num_boundarysurfaces, num_faces, num_AEs);



  /* compute adjoint graph: -----------------------------------------*/

  if (num_boundarysurfaces > 0)
  ierr = transpose_matrix_create(&i_AE_boundarysurface, 
				 &j_AE_boundarysurface,

				 i_boundarysurface_AE,
				 j_boundarysurface_AE,

				 num_boundarysurfaces, num_AEs);



  if (num_boundarysurfaces > 0)
    {
      hypre_TFree(i_boundarysurface_AE);
      hypre_TFree(j_boundarysurface_AE);
    
    }

  ierr = matrix_matrix_t_product(&i_AE_AE, &j_AE_AE,

				 i_AE_face, j_AE_face,

				 num_AEs, num_faces);




  ierr = create_maximal_intersection_sets(i_AE_face, j_AE_face,
					  i_face_AE, j_face_AE,
					  i_AE_AE, j_AE_AE,

					  i_AE_boundarysurface, 
					  j_AE_boundarysurface,

					  i_boundarysurface_face, 
					  j_boundarysurface_face,

					  num_boundarysurfaces, 
					  num_AEs, num_faces,

					  &i_AEface_AE, 
					  &j_AEface_AE,

					  &i_AEface_face, &j_AEface_face,

					  &num_AEfaces);

  hypre_TFree(i_AE_face);
  hypre_TFree(j_AE_face);

  hypre_TFree(i_face_AE);
  hypre_TFree(j_face_AE);

  hypre_TFree(i_AE_AE);
  hypre_TFree(j_AE_AE);
  
  ierr = transpose_matrix_create(&i_face_AEface, &j_face_AEface,

				 i_AEface_face, j_AEface_face,
				 num_AEfaces, num_faces);



  ierr = matrix_matrix_product(&i_face_face_AEface,
			       &j_face_face_AEface,

			       i_face_face, j_face_face,
			       
			       i_face_AEface, j_face_AEface,
			       
			       num_faces, num_faces, num_AEfaces);

  ierr = matrix_matrix_product(&i_AEface_AEface,
			       &j_AEface_AEface,

			       i_AEface_face, j_AEface_face,
			       
			       i_face_face_AEface, j_face_face_AEface,
			       
			       num_AEfaces, num_faces, num_AEfaces);

  hypre_TFree(i_face_face_AEface);
  hypre_TFree(j_face_face_AEface);


  if (num_boundarysurfaces > 0)
  ierr = matrix_matrix_product(&i_boundarysurface_AEface,
			       &j_boundarysurface_AEface,

			       i_boundarysurface_face, j_boundarysurface_face,
			       
			       i_face_AEface, j_face_AEface,
			       
			       num_boundarysurfaces, num_faces, num_AEfaces);

  ierr = transpose_matrix_create(&i_AE_AEface, &j_AE_AEface,

				 i_AEface_AE, j_AEface_AE,
				 num_AEfaces, num_AEs);


  ierr = matrix_matrix_product(&i_AEface_node,
			       &j_AEface_node,

			       i_AEface_face, j_AEface_face,
			       
			       i_face_node, j_face_node,
			       
			       num_AEfaces, num_faces, num_nodes);

  hypre_TFree(i_face_AEface);
  hypre_TFree(j_face_AEface);
  hypre_TFree(i_AEface_face);
  hypre_TFree(j_AEface_face);

  hypre_TFree(w_face_face);

  hypre_AMGeMatrixTopologyIAEElement(A_crs) = i_AE_element;
  hypre_AMGeMatrixTopologyJAEElement(A_crs) = j_AE_element;
 
  hypre_AMGeMatrixTopologyIElementNode(A_crs) = NULL;
  hypre_AMGeMatrixTopologyJElementNode(A_crs) = NULL;

  hypre_AMGeMatrixTopologyNumElements(A_crs) = num_AEs;
  hypre_AMGeMatrixTopologyNumNodes(A_crs) = num_nodes;
  hypre_AMGeMatrixTopologyNumFaces(A_crs) = num_AEfaces;

  hypre_AMGeMatrixTopologyNumBoundarysurfaces(A_crs) = num_boundarysurfaces;

  hypre_AMGeMatrixTopologyIElementFace(A_crs) = i_AE_AEface;
  hypre_AMGeMatrixTopologyJElementFace(A_crs) = j_AE_AEface;


  hypre_AMGeMatrixTopologyIFaceElement(A_crs) = i_AEface_AE;
  hypre_AMGeMatrixTopologyJFaceElement(A_crs) = j_AEface_AE;


  hypre_AMGeMatrixTopologyIFaceFace(A_crs) = i_AEface_AEface;
  hypre_AMGeMatrixTopologyJFaceFace(A_crs) = j_AEface_AEface;

  hypre_AMGeMatrixTopologyIFaceNode(A_crs) = i_AEface_node;
  hypre_AMGeMatrixTopologyJFaceNode(A_crs) = j_AEface_node;



  hypre_AMGeMatrixTopologyIBoundarysurfaceFace(A_crs) = i_boundarysurface_AEface;
  hypre_AMGeMatrixTopologyJBoundarysurfaceFace(A_crs) = j_boundarysurface_AEface;


  return ierr;


}
